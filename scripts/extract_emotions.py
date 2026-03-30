#!/usr/bin/env python3
"""Extract emotion embeddings (audio + face) for all clips.

Re-downloads VGGSound clips, extracts:
  - Audio emotion: wav2vec2-base hidden states (768-dim) from audio waveform
  - Face emotion: HSEmotion EfficientNet-B0 features (288-dim) from detected faces
  - Fused: concatenated [audio_768, face_288] = 1056-dim

Updates e_emb in the safetensors embedding cache.

Usage:
    python scripts/extract_emotions.py --workers 6 --batch 50
    python scripts/extract_emotions.py --start-idx 5000  # resume from index
"""

import argparse
import csv
import logging
import os
import subprocess
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
log = logging.getLogger("extract_emotions")

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/opt/brain/data/vggsound")
CSV_PATH = DATA_DIR / "vggsound.csv"
CACHE_DIR = DATA_DIR / ".embed_cache"
EMBED_PATH = CACHE_DIR / "expanded_embeddings.safetensors"
PROGRESS_PATH = CACHE_DIR / "emotion_progress.txt"
TMP_DIR = Path("/tmp/emotion_dl")

# Emotion embedding dimensions
AUDIO_EMO_DIM = 768    # wav2vec2-base hidden size
FACE_EMO_DIM = 512     # ResNet-18 features (HSEmotion=288 if available, else ResNet-18=512)
TOTAL_EMO_DIM = AUDIO_EMO_DIM + FACE_EMO_DIM  # 1280


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_wav2vec2():
    """Load wav2vec2-base for audio emotion feature extraction."""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    log.info("Loading wav2vec2-base for audio emotion...")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    log.info("wav2vec2-base loaded (768-dim hidden states)")
    return model, processor


def load_face_models():
    """Load OpenCV face detector + HSEmotion for facial emotion features."""
    # OpenCV DNN face detector (SSD with ResNet-10 backbone)
    # Download the model files if not present
    model_dir = CACHE_DIR / "face_models"
    model_dir.mkdir(exist_ok=True)

    prototxt = model_dir / "deploy.prototxt"
    caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    if not caffemodel.exists():
        log.info("Downloading OpenCV face detector model...")
        import urllib.request
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector"
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            str(caffemodel),
        )
        # Write prototxt inline (smaller file)
        urllib.request.urlretrieve(
            f"{base_url}/deploy.prototxt",
            str(prototxt),
        )
        log.info("Face detector model downloaded")

    face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    log.info("OpenCV DNN face detector loaded")

    # Face emotion encoder: try HSEmotion, fallback to ResNet-18
    try:
        import timm
        hsemotion = timm.create_model("hsemotion_enet_b0_8", pretrained=True, num_classes=0)
        hsemotion.eval()
        emo_dim = hsemotion.num_features
        log.info(f"HSEmotion loaded (dim={emo_dim})")
    except Exception as e:
        log.info(f"HSEmotion not available ({e}), using ResNet-18 face features")
        from torchvision.models import ResNet18_Weights, resnet18
        hsemotion = resnet18(weights=ResNet18_Weights.DEFAULT)
        emo_dim = hsemotion.fc.in_features  # 512
        hsemotion.fc = torch.nn.Identity()
        hsemotion.eval()
        log.info(f"ResNet-18 face encoder loaded (dim={emo_dim})")

    return face_net, hsemotion, emo_dim


# ─── Audio Emotion Extraction ────────────────────────────────────────────────

def extract_audio_pcm(video_path):
    """Extract 16kHz mono audio from video via ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", "16000",
            "-loglevel", "quiet", "-y", "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0 or len(result.stdout) < 1600:
            return None
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    except Exception:
        return None


@torch.no_grad()
def extract_audio_emotion(wav2vec_model, wav2vec_proc, audio_pcm):
    """Extract audio emotion features using wav2vec2 hidden states.

    Returns 768-dim mean-pooled hidden states capturing prosody, tone,
    rhythm, and emotional acoustic properties.
    """
    try:
        inputs = wav2vec_proc(
            audio_pcm, sampling_rate=16000, return_tensors="pt",
            padding=True, max_length=16000 * 10, truncation=True,
        )
        outputs = wav2vec_model(inputs.input_values)
        # Mean pool over time dimension → [768]
        hidden = outputs.last_hidden_state  # [1, T, 768]
        emb = hidden.mean(dim=1).squeeze(0)  # [768]
        return emb.numpy()
    except Exception as e:
        log.debug(f"Audio emotion failed: {e}")
        return None


# ─── Face Emotion Extraction ─────────────────────────────────────────────────

def extract_frames_ffmpeg(video_path, n_frames=3):
    """Extract multiple frames from video for face detection."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(probe.stdout.strip() or "2.0")

        frames = []
        for i in range(n_frames):
            t = duration * (i + 1) / (n_frames + 1)
            cmd = [
                "ffmpeg", "-ss", str(t), "-i", str(video_path),
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-vf", "scale=300:300",
                "-loglevel", "quiet", "-y", "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0 and len(result.stdout) == 300 * 300 * 3:
                frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(300, 300, 3)
                frames.append(frame)
        return frames
    except Exception:
        return []


def detect_face(face_net, frame, min_confidence=0.5):
    """Detect face using OpenCV DNN and return largest face crop.

    Returns face crop as [H, W, 3] RGB numpy array or None.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    best_face = None
    best_area = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))

        area = (x2 - x1) * (y2 - y1)
        if area > best_area and (x2 - x1) > 20 and (y2 - y1) > 20:
            best_area = area
            best_face = frame[y1:y2, x1:x2]

    return best_face


@torch.no_grad()
def extract_face_emotion(face_net, hsemotion, frames):
    """Extract facial emotion features from video frames.

    Runs face detection on each frame, takes the best face,
    and passes through HSEmotion to get emotion features.
    Returns D_face-dim features or None if no face found.
    """
    for frame in frames:
        face_crop = detect_face(face_net, frame)
        if face_crop is not None:
            try:
                # Resize to 224x224 for HSEmotion
                face_resized = cv2.resize(face_crop, (224, 224))
                # Convert BGR→RGB, normalize
                face_t = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
                # ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                face_t = (face_t - mean) / std
                face_t = face_t.unsqueeze(0)  # [1, 3, 224, 224]

                emb = hsemotion(face_t)  # [1, D]
                return emb.squeeze(0).numpy()
            except Exception as e:
                log.debug(f"Face emotion extraction failed: {e}")
                continue

    return None


# ─── Video Download ──────────────────────────────────────────────────────────

def download_clip(youtube_id, start_sec, duration=4.0):
    """Download a VGGSound clip. Returns path or None."""
    TMP_DIR.mkdir(exist_ok=True)
    out_path = TMP_DIR / f"{youtube_id}_{start_sec:06d}.mp4"

    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        "yt-dlp",
        "--quiet", "--no-warnings",
        "--download-sections", f"*{start_sec}-{start_sec + duration}",
        "-f", "worst[ext=mp4]/worst",
        "--no-playlist",
        "--socket-timeout", "10",
        "--retries", "1",
        "-o", str(out_path),
        url,
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        if out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
    except Exception:
        pass

    # Cleanup
    for f in [out_path, Path(str(out_path) + ".part")]:
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass
    return None


# ─── Clip ID → VGGSound Metadata ─────────────────────────────────────────────

def build_clip_metadata():
    """Build clip_id → (yt_id, start_sec, label) mapping from VGGSound CSV."""
    metadata = {}
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    yt_id, start, label = row[0], int(row[1]), row[2]
                    clip_id = f"{yt_id}_{start:06d}"
                    metadata[clip_id] = (yt_id, start, label)
    log.info(f"VGGSound metadata: {len(metadata)} entries")
    return metadata


def get_clip_ids_from_db(n_clips):
    """Get ordered clip_ids from data_inventory."""
    import sqlite3
    db_path = Path("/opt/brain/outputs/cortex/knowledge.db")
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT clip_id FROM data_inventory ORDER BY rowid LIMIT ?", (n_clips,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        log.warning(f"Failed to read clip_ids from DB: {e}")
        return []


# ─── Progress Tracking ───────────────────────────────────────────────────────

def load_progress():
    """Load set of indices already processed."""
    if PROGRESS_PATH.exists():
        return set(int(x) for x in PROGRESS_PATH.read_text().strip().split("\n") if x.strip())
    return set()


def save_progress(done_indices):
    """Save processed indices."""
    PROGRESS_PATH.write_text("\n".join(str(i) for i in sorted(done_indices)))


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract emotion embeddings for all clips")
    parser.add_argument("--batch", type=int, default=50, help="Batch size for saving")
    parser.add_argument("--workers", type=int, default=6, help="Download workers")
    parser.add_argument("--start-idx", type=int, default=0, help="Start from this clip index")
    args = parser.parse_args()

    # Load existing embeddings
    from safetensors.numpy import load_file, save_file

    if not EMBED_PATH.exists():
        log.error(f"Embeddings not found at {EMBED_PATH}")
        return

    data = load_file(str(EMBED_PATH))
    v_emb = data["v_emb"]
    a_emb = data["a_emb"]
    old_e_emb = data["e_emb"]
    n_clips = v_emb.shape[0]
    log.info(f"Loaded {n_clips} clips, v={v_emb.shape}, a={a_emb.shape}, old_e={old_e_emb.shape}")

    # Initialize new emotion embeddings
    e_emb = np.zeros((n_clips, TOTAL_EMO_DIM), dtype=np.float32)

    # Load progress
    done_indices = load_progress()
    log.info(f"Already processed: {len(done_indices)} clips")

    # Build clip metadata
    vggsound_meta = build_clip_metadata()
    clip_ids = get_clip_ids_from_db(n_clips)
    if len(clip_ids) < n_clips:
        # Pad with generated IDs
        clip_ids.extend(f"unknown_{i}" for i in range(len(clip_ids), n_clips))

    # Load models
    wav2vec_model, wav2vec_proc = load_wav2vec2()
    face_net, hsemotion, face_emo_dim = load_face_models()
    log.info(f"Face emotion dim: {face_emo_dim} (will pad to {FACE_EMO_DIM})")

    # Stats
    total_processed = 0
    total_audio_ok = 0
    total_face_ok = 0
    total_download_fail = 0
    start_time = time.monotonic()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Process all clips
    idx = args.start_idx
    while idx < n_clips:
        batch_end = min(idx + args.batch, n_clips)
        batch_indices = [i for i in range(idx, batch_end) if i not in done_indices]

        if not batch_indices:
            idx = batch_end
            continue

        # Prepare download tasks
        download_tasks = []
        for i in batch_indices:
            cid = clip_ids[i] if i < len(clip_ids) else f"unknown_{i}"

            # Try to find VGGSound metadata
            if cid in vggsound_meta:
                yt_id, start, label = vggsound_meta[cid]
                download_tasks.append((i, yt_id, start, "vggsound"))
            elif cid.startswith("expanded_"):
                # Expanded clips: we lost the original clip_id mapping
                # Try to find it by searching CSV in order (slow but one-time)
                download_tasks.append((i, None, None, "expanded"))
            elif cid.startswith("ravdess_"):
                # RAVDESS clips: need special handling
                download_tasks.append((i, None, None, "ravdess"))
            else:
                # Try parsing as VGGSound clip_id
                parts = cid.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    yt_id = parts[0]
                    start = int(parts[1])
                    download_tasks.append((i, yt_id, start, "vggsound"))
                else:
                    download_tasks.append((i, None, None, "unknown"))

        # Parallel download VGGSound clips
        downloaded = {}
        vggsound_tasks = [(i, yt, st) for i, yt, st, src in download_tasks
                          if src == "vggsound" and yt is not None]

        if vggsound_tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for i, yt_id, start in vggsound_tasks:
                    f = pool.submit(download_clip, yt_id, start)
                    futures[f] = i

                for f in as_completed(futures):
                    i = futures[f]
                    path = f.result()
                    if path:
                        downloaded[i] = path
                    else:
                        total_download_fail += 1

        # Process each clip
        for i, yt_id, start, src in download_tasks:
            video_path = downloaded.get(i)

            audio_emo = None
            face_emo = None

            if video_path:
                # Extract audio
                audio_pcm = extract_audio_pcm(video_path)
                if audio_pcm is not None:
                    audio_emo = extract_audio_emotion(wav2vec_model, wav2vec_proc, audio_pcm)
                    if audio_emo is not None:
                        total_audio_ok += 1

                # Extract face emotion
                frames = extract_frames_ffmpeg(video_path, n_frames=3)
                if frames:
                    face_emo = extract_face_emotion(face_net, hsemotion, frames)
                    if face_emo is not None:
                        total_face_ok += 1

                # Delete video
                try:
                    video_path.unlink()
                except OSError:
                    pass

            # Build fused emotion embedding [audio_768, face_288]
            emo = np.zeros(TOTAL_EMO_DIM, dtype=np.float32)
            if audio_emo is not None:
                emo[:AUDIO_EMO_DIM] = audio_emo[:AUDIO_EMO_DIM]
            if face_emo is not None:
                # Pad/truncate face emotion to FACE_EMO_DIM
                d = min(len(face_emo), FACE_EMO_DIM)
                emo[AUDIO_EMO_DIM:AUDIO_EMO_DIM + d] = face_emo[:d]

            e_emb[i] = emo
            done_indices.add(i)
            total_processed += 1

        # Save progress periodically
        if total_processed % args.batch == 0 or batch_end >= n_clips:
            save_file(
                {"v_emb": v_emb, "a_emb": a_emb, "e_emb": e_emb},
                str(EMBED_PATH),
            )
            save_progress(done_indices)

            elapsed = time.monotonic() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining = n_clips - len(done_indices)
            eta = remaining / rate / 3600 if rate > 0 else float("inf")
            log.info(
                f"Progress: {len(done_indices)}/{n_clips} "
                f"(audio={total_audio_ok}, face={total_face_ok}, "
                f"dl_fail={total_download_fail}, "
                f"{rate:.1f}/sec, ETA {eta:.1f}h)"
            )

        idx = batch_end

    # Final save
    save_file(
        {"v_emb": v_emb, "a_emb": a_emb, "e_emb": e_emb},
        str(EMBED_PATH),
    )
    save_progress(done_indices)

    elapsed = time.monotonic() - start_time
    log.info(
        f"Done! Processed {total_processed} clips in {elapsed/3600:.1f}h. "
        f"Audio emotion: {total_audio_ok}/{total_processed} ({total_audio_ok/max(1,total_processed)*100:.0f}%). "
        f"Face emotion: {total_face_ok}/{total_processed} ({total_face_ok/max(1,total_processed)*100:.0f}%). "
        f"Download failures: {total_download_fail}."
    )


if __name__ == "__main__":
    main()
