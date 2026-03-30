#!/usr/bin/env python3
"""Download VGGSound videos and extract DINOv2+Whisper embeddings incrementally.

Downloads videos in small batches, extracts embeddings on CPU, deletes videos
immediately, and appends to the safetensors embedding cache.

Usage:
    python scripts/expand_embeddings.py --target 20000 --batch 100
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("expand_embeddings")

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("/opt/brain/data/vggsound")
CSV_PATH = DATA_DIR / "vggsound.csv"
CACHE_DIR = DATA_DIR / ".embed_cache"
# We'll write to a NEW file to avoid corrupting the existing one
OUTPUT_PATH = CACHE_DIR / "expanded_embeddings.safetensors"
EXISTING_PATH = CACHE_DIR / "dinov2_vits14__whisper__clip4.0__f3a4230e3a07c57d.safetensors"
PROGRESS_PATH = CACHE_DIR / "expand_progress.txt"


def load_existing_ids():
    """Load set of video IDs that are already embedded."""
    if PROGRESS_PATH.exists():
        return set(PROGRESS_PATH.read_text().strip().split("\n"))
    return set()


def save_progress(embedded_ids):
    """Save set of embedded video IDs."""
    PROGRESS_PATH.write_text("\n".join(sorted(embedded_ids)))


def parse_csv():
    """Parse vggsound.csv and return list of (youtube_id, start_sec, label, split)."""
    entries = []
    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                entries.append((row[0], int(row[1]), row[2], row[3]))
    return entries


def download_clip(youtube_id, start_sec, duration=4.0, tmp_dir="/tmp/vggsound_dl"):
    """Download a single clip from YouTube. Returns path or None on failure."""
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f"{youtube_id}_{start_sec:06d}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Download the actual clip (skip simulate check — just let download fail fast)
    cmd = [
        "yt-dlp",
        "--quiet", "--no-warnings",
        "--download-sections", f"*{start_sec}-{start_sec + duration}",
        "-f", "worst[ext=mp4]/worst",
        "--no-playlist",
        "--socket-timeout", "10",
        "--retries", "1",
        "-o", out_path,
        url,
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return out_path
    except (subprocess.TimeoutExpired, Exception):
        pass

    # Cleanup partial files
    for f in [out_path, out_path + ".part"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    return None


def load_models():
    """Load DINOv2, Whisper, wav2vec2, and face emotion models."""
    import torchvision.transforms as T

    log.info("Loading DINOv2-ViTS14...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2.eval()
    log.info(f"DINOv2 loaded, dim={dinov2.embed_dim}")

    dinov2_transform = T.Compose([
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    log.info("Loading Whisper-base encoder...")
    from transformers import WhisperModel, WhisperFeatureExtractor
    whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
    whisper_model.eval()
    whisper_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    log.info(f"Whisper loaded, dim={whisper_model.config.d_model}")

    # Load emotion models (audio + face)
    log.info("Loading wav2vec2-base for audio emotion...")
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model.eval()
    wav2vec_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    log.info("Loading face emotion models...")
    import cv2
    face_net = None
    hsemotion = None
    face_models_dir = CACHE_DIR / "face_models"
    face_models_dir.mkdir(exist_ok=True)
    prototxt = face_models_dir / "deploy.prototxt"
    caffemodel = face_models_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    if caffemodel.exists() and prototxt.exists():
        face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        log.info("OpenCV face detector loaded")
    else:
        log.info("Face detector model not found, downloading...")
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                str(caffemodel),
            )
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                str(prototxt),
            )
            face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
            log.info("Face detector downloaded and loaded")
        except Exception as e:
            log.warning(f"Failed to download face detector: {e}")

    if face_net is not None:
        try:
            import timm
            hsemotion = timm.create_model("hsemotion_enet_b0_8", pretrained=True, num_classes=0)
            hsemotion.eval()
            log.info(f"HSEmotion loaded (dim={hsemotion.num_features})")
        except Exception:
            log.info("HSEmotion not available, using ResNet-18 for face features")
            from torchvision.models import ResNet18_Weights, resnet18
            hsemotion = resnet18(weights=ResNet18_Weights.DEFAULT)
            hsemotion.fc = torch.nn.Identity()
            hsemotion.eval()
            log.info(f"ResNet-18 face encoder loaded (dim=512)")

    return (dinov2, dinov2_transform, whisper_model, whisper_processor,
            wav2vec_model, wav2vec_proc, face_net, hsemotion)


def extract_frame_ffmpeg(video_path):
    """Extract a single middle frame from video using ffmpeg → numpy."""
    try:
        # Get duration first
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(probe.stdout.strip() or "2.0")
        mid = duration / 2

        # Extract single frame as raw RGB
        cmd = [
            "ffmpeg", "-ss", str(mid), "-i", video_path,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-vf", "scale=256:256",
            "-loglevel", "quiet", "-y", "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0 or len(result.stdout) == 0:
            return None

        frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(256, 256, 3)
        return frame
    except Exception as e:
        log.debug(f"Frame extraction failed: {e}")
        return None


def extract_audio_ffmpeg(video_path):
    """Extract audio as 16kHz mono PCM from video using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", "16000",
            "-loglevel", "quiet", "-y", "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0 or len(result.stdout) < 1600:
            return None

        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    except Exception as e:
        log.debug(f"Audio extraction failed: {e}")
        return None


def extract_visual_embedding(dinov2, video_path, transform):
    """Extract DINOv2 embedding from a video frame."""
    frame = extract_frame_ffmpeg(video_path)
    if frame is None:
        return None

    try:
        # Convert numpy [H, W, C] to tensor [C, H, W]
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_t = transform(frame_t).unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            emb = dinov2(frame_t)  # [1, 384]
        return emb.squeeze(0).numpy()
    except Exception as e:
        log.debug(f"Visual embedding failed: {e}")
        return None


def extract_audio_embedding(whisper_model, whisper_processor, video_path):
    """Extract Whisper embedding from video audio via ffmpeg."""
    audio = extract_audio_ffmpeg(video_path)
    if audio is None:
        return None

    try:
        inputs = whisper_processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features  # [1, 80, 3000]

        with torch.no_grad():
            outputs = whisper_model.encoder(input_features)
            emb = outputs.last_hidden_state.mean(dim=1)  # [1, 512]
        return emb.squeeze(0).numpy()
    except Exception as e:
        log.debug(f"Audio embedding failed: {e}")
        return None


# ─── Emotion Extraction (audio + face) ───────────────────────────────────────

AUDIO_EMO_DIM = 768    # wav2vec2-base hidden size
FACE_EMO_DIM = 512     # ResNet-18 features (HSEmotion=288 if available, else ResNet-18=512)
TOTAL_EMO_DIM = AUDIO_EMO_DIM + FACE_EMO_DIM  # 1280


@torch.no_grad()
def extract_audio_emotion(wav2vec_model, wav2vec_proc, video_path):
    """Extract audio emotion features using wav2vec2 hidden states (768-dim)."""
    audio = extract_audio_ffmpeg(video_path)
    if audio is None:
        return None
    try:
        inputs = wav2vec_proc(
            audio, sampling_rate=16000, return_tensors="pt",
            padding=True, max_length=16000 * 10, truncation=True,
        )
        outputs = wav2vec_model(inputs.input_values)
        hidden = outputs.last_hidden_state  # [1, T, 768]
        emb = hidden.mean(dim=1).squeeze(0)  # [768]
        return emb.numpy()
    except Exception as e:
        log.debug(f"Audio emotion failed: {e}")
        return None


@torch.no_grad()
def extract_face_emotion(face_net, hsemotion, video_path):
    """Extract facial emotion features using OpenCV face detection + HSEmotion."""
    if face_net is None or hsemotion is None:
        return None

    import cv2

    # Extract 3 frames from the video
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(probe.stdout.strip() or "2.0")
    except Exception:
        duration = 2.0

    for i in range(3):
        t = duration * (i + 1) / 4
        try:
            cmd = [
                "ffmpeg", "-ss", str(t), "-i", video_path,
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-vf", "scale=300:300",
                "-loglevel", "quiet", "-y", "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode != 0 or len(result.stdout) != 300 * 300 * 3:
                continue

            frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(300, 300, 3)

            # Detect face
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            for j in range(detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence < 0.5:
                    continue

                x1 = max(0, int(detections[0, 0, j, 3] * w))
                y1 = max(0, int(detections[0, 0, j, 4] * h))
                x2 = min(w, int(detections[0, 0, j, 5] * w))
                y2 = min(h, int(detections[0, 0, j, 6] * h))

                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                face_crop = frame[y1:y2, x1:x2]
                face_resized = cv2.resize(face_crop, (224, 224))
                face_t = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                face_t = (face_t - mean) / std
                face_t = face_t.unsqueeze(0)

                emb = hsemotion(face_t)  # [1, D]
                return emb.squeeze(0).numpy()
        except Exception:
            continue

    return None


def extract_emotion_embedding(wav2vec_model, wav2vec_proc, face_net, hsemotion, video_path):
    """Extract fused emotion embedding [audio_768, face_288] = 1056-dim."""
    emo = np.zeros(TOTAL_EMO_DIM, dtype=np.float32)

    # Audio emotion (always attempted)
    audio_emo = extract_audio_emotion(wav2vec_model, wav2vec_proc, video_path)
    if audio_emo is not None:
        emo[:AUDIO_EMO_DIM] = audio_emo[:AUDIO_EMO_DIM]

    # Face emotion (when models available)
    face_emo = extract_face_emotion(face_net, hsemotion, video_path)
    if face_emo is not None:
        d = min(len(face_emo), FACE_EMO_DIM)
        emo[AUDIO_EMO_DIM:AUDIO_EMO_DIM + d] = face_emo[:d]

    return emo


def save_embeddings(v_list, a_list, e_list, existing_v, existing_a, existing_e):
    """Save combined embeddings to safetensors."""
    from safetensors.numpy import save_file

    # Stack new embeddings
    if v_list:
        new_v = np.stack(v_list)
        new_a = np.stack(a_list)
        new_e = np.stack(e_list)

        # Combine with existing
        all_v = np.concatenate([existing_v, new_v], axis=0) if existing_v is not None else new_v
        all_a = np.concatenate([existing_a, new_a], axis=0) if existing_a is not None else new_a
        all_e = np.concatenate([existing_e, new_e], axis=0) if existing_e is not None else new_e
    else:
        all_v, all_a, all_e = existing_v, existing_a, existing_e

    save_file(
        {"v_emb": all_v, "a_emb": all_a, "e_emb": all_e},
        str(OUTPUT_PATH),
    )
    # Update data_inventory in DB to match
    try:
        import sqlite3
        db_path = Path("/opt/brain/outputs/cortex/knowledge.db")
        conn = sqlite3.connect(str(db_path))
        current_db = conn.execute("SELECT COUNT(*) FROM data_inventory").fetchone()[0]
        total = all_v.shape[0]
        if total > current_db:
            for i in range(total - current_db):
                conn.execute(
                    "INSERT INTO data_inventory (dataset, clip_id, path, size_bytes, status) VALUES (?, ?, ?, ?, ?)",
                    ("vggsound", f"expanded_{current_db + i}", "embedded", 0, "downloaded"),
                )
            conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Failed to update data_inventory: {e}")
    log.info(f"Saved embeddings: v={all_v.shape}, a={all_a.shape}, e={all_e.shape}")
    return all_v, all_a, all_e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=20000,
                        help="Target total number of clips")
    parser.add_argument("--batch", type=int, default=50,
                        help="Download batch size (save after each batch)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers")
    args = parser.parse_args()

    # Load existing embeddings
    from safetensors.numpy import load_file
    if OUTPUT_PATH.exists():
        log.info(f"Loading expanded embeddings from {OUTPUT_PATH}")
        data = load_file(str(OUTPUT_PATH))
    elif EXISTING_PATH.exists():
        log.info(f"Loading original embeddings from {EXISTING_PATH}")
        data = load_file(str(EXISTING_PATH))
    else:
        data = {}

    existing_v = data.get("v_emb")
    existing_a = data.get("a_emb")
    existing_e = data.get("e_emb")
    current_count = existing_v.shape[0] if existing_v is not None else 0
    log.info(f"Starting with {current_count} embeddings, target={args.target}")

    if current_count >= args.target:
        log.info("Already at target, nothing to do")
        return

    # Load progress
    embedded_ids = load_existing_ids()
    # Add existing video IDs (from original dataset)
    existing_video_dir = DATA_DIR / "video"
    for f in existing_video_dir.glob("*.mp4"):
        embedded_ids.add(f.stem)
    # Mark the original 5080 as done
    log.info(f"Already embedded: {len(embedded_ids)} clips")

    # Parse CSV and filter
    entries = parse_csv()
    candidates = []
    for yt_id, start, label, split in entries:
        clip_id = f"{yt_id}_{start:06d}"
        if clip_id not in embedded_ids:
            candidates.append((yt_id, start, label, split, clip_id))

    log.info(f"Candidates to download: {len(candidates)}")
    needed = args.target - current_count

    # Load models
    (dinov2, dinov2_transform, whisper_model, whisper_processor,
     wav2vec_model, wav2vec_proc, face_net, hsemotion) = load_models()

    # Process in batches
    new_v, new_a, new_e = [], [], []
    total_downloaded = 0
    total_failed = 0
    total_embedded = 0
    batch_start = time.monotonic()
    tmp_dir = "/tmp/vggsound_dl"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    i = 0
    while total_embedded < needed and i < len(candidates):
        batch = candidates[i:i + args.batch]
        i += args.batch

        # Parallel download
        downloaded = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for yt_id, start, label, split, clip_id in batch:
                f = pool.submit(download_clip, yt_id, start)
                futures[f] = (yt_id, start, label, split, clip_id)

            for f in as_completed(futures):
                yt_id, start, label, split, clip_id = futures[f]
                path = f.result()
                if path:
                    downloaded.append((path, clip_id))
                    total_downloaded += 1
                else:
                    total_failed += 1
                    embedded_ids.add(clip_id)  # Mark as attempted

        # Extract embeddings from downloaded clips
        for path, clip_id in downloaded:
            v_emb = extract_visual_embedding(dinov2, path, dinov2_transform)
            a_emb = extract_audio_embedding(whisper_model, whisper_processor, path)

            if v_emb is not None and a_emb is not None:
                new_v.append(v_emb)
                new_a.append(a_emb)
                # Extract emotion (audio + face) before deleting video
                e_emb = extract_emotion_embedding(
                    wav2vec_model, wav2vec_proc, face_net, hsemotion, path
                )
                new_e.append(e_emb)
                embedded_ids.add(clip_id)
                total_embedded += 1
            else:
                embedded_ids.add(clip_id)

            # Delete video after all extractions
            try:
                os.remove(path)
            except OSError:
                pass

        # Save after each batch (always save when we have new embeddings)
        if new_v:
            existing_v, existing_a, existing_e = save_embeddings(
                new_v, new_a, new_e, existing_v, existing_a, existing_e
            )
            new_v, new_a, new_e = [], [], []
            save_progress(embedded_ids)

        elapsed = time.monotonic() - batch_start
        rate = total_embedded / elapsed if elapsed > 0 else 0
        eta = (needed - total_embedded) / rate / 60 if rate > 0 else float("inf")
        log.info(
            f"Progress: {current_count + total_embedded}/{args.target} "
            f"(+{total_embedded} new, {total_failed} failed, "
            f"{rate:.1f} clips/sec, ETA {eta:.0f}min)"
        )

    # Final save
    if new_v:
        save_embeddings(new_v, new_a, new_e, existing_v, existing_a, existing_e)
        save_progress(embedded_ids)

    # Cleanup tmp dir
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.monotonic() - batch_start
    log.info(
        f"Done! Embedded {total_embedded} new clips in {elapsed/60:.1f}min "
        f"({total_failed} download failures). "
        f"Total: {current_count + total_embedded} clips."
    )


if __name__ == "__main__":
    main()
