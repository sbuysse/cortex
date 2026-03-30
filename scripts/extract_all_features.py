#!/usr/bin/env python3
"""Extract ALL features from video clips in a single download pass.

For each clip, extracts 3 new embedding tensors:
  e_emb: Emotion [audio_emotion_768 + face_emotion_512] = 1280-dim
  s_emb: Speech [Whisper ASR → sentence embedding] = 384-dim
  p_emb: Properties [color_48 + motion_32 + edges_16 + loudness_16 + pitch_32 + tempo_8] = 152-dim

Downloads each clip once, extracts everything, deletes immediately.

Usage:
    python scripts/extract_all_features.py --batch 50 --workers 6
    python scripts/extract_all_features.py --start-idx 5000  # resume
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("extract_all")

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/opt/brain/data/vggsound")
CSV_PATH = DATA_DIR / "vggsound.csv"
CACHE_DIR = DATA_DIR / ".embed_cache"
EMBED_PATH = CACHE_DIR / "expanded_embeddings.safetensors"
PROGRESS_PATH = CACHE_DIR / "allfeatures_progress.txt"
TMP_DIR = Path("/tmp/allfeatures_dl")

# ─── Dimensions ───────────────────────────────────────────────────────────────

AUDIO_EMO_DIM = 768
FACE_EMO_DIM = 512
E_DIM = AUDIO_EMO_DIM + FACE_EMO_DIM  # 1280

S_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2

# Properties: color(48) + motion(32) + edges(16) + loudness(16) + pitch(32) + tempo(8) = 152
COLOR_DIM = 48       # 16 bins × 3 channels
MOTION_DIM = 32      # 4×4 grid × 2 (mean + std)
EDGES_DIM = 16       # 4×4 grid edge density
LOUDNESS_DIM = 16    # RMS energy in 16 time windows
PITCH_DIM = 32       # spectral centroid(16) + zero crossing rate(16)
TEMPO_DIM = 8        # onset strength in 8 windows
P_DIM = COLOR_DIM + MOTION_DIM + EDGES_DIM + LOUDNESS_DIM + PITCH_DIM + TEMPO_DIM  # 152


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class Models:
    """Lazy-loaded model container."""

    def __init__(self):
        self.wav2vec = None
        self.wav2vec_proc = None
        self.face_net = None
        self.face_model = None
        self.whisper_asr = None
        self.whisper_proc = None
        self.sent_tokenizer = None
        self.sent_model = None

    def load_all(self):
        self._load_wav2vec()
        self._load_face()
        self._load_whisper_asr()
        self._load_sentence_model()

    def _load_wav2vec(self):
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        log.info("Loading wav2vec2-base (audio emotion)...")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec.eval()
        self.wav2vec_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def _load_face(self):
        # OpenCV DNN face detector
        model_dir = CACHE_DIR / "face_models"
        model_dir.mkdir(exist_ok=True)
        prototxt = model_dir / "deploy.prototxt"
        caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if not caffemodel.exists():
            log.info("Downloading face detector...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                str(caffemodel))
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                str(prototxt))

        self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

        # Face emotion encoder (ResNet-18)
        from torchvision.models import ResNet18_Weights, resnet18
        self.face_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.face_model.fc = torch.nn.Identity()
        self.face_model.eval()
        log.info("Face detection + ResNet-18 loaded")

    def _load_whisper_asr(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        log.info("Loading Whisper-base (ASR)...")
        self.whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper_asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        self.whisper_asr.eval()

    def _load_sentence_model(self):
        from transformers import AutoTokenizer, AutoModel
        log.info("Loading all-MiniLM-L6-v2 (sentence embeddings)...")
        self.sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.sent_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.sent_model.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA EXTRACTION (ffmpeg)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio_pcm(video_path):
    """Extract 16kHz mono PCM from video."""
    try:
        cmd = ["ffmpeg", "-i", str(video_path),
               "-f", "s16le", "-acodec", "pcm_s16le",
               "-ac", "1", "-ar", "16000",
               "-loglevel", "quiet", "-y", "pipe:1"]
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if r.returncode != 0 or len(r.stdout) < 1600:
            return None
        return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None


def extract_frames(video_path, n_frames=5):
    """Extract N evenly-spaced frames as 300×300 RGB numpy arrays."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=10)
        duration = float(probe.stdout.strip() or "2.0")
    except Exception:
        duration = 2.0

    frames = []
    for i in range(n_frames):
        t = duration * (i + 1) / (n_frames + 1)
        try:
            cmd = ["ffmpeg", "-ss", str(t), "-i", str(video_path),
                   "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-vf", "scale=300:300", "-loglevel", "quiet", "-y", "pipe:1"]
            r = subprocess.run(cmd, capture_output=True, timeout=10)
            if r.returncode == 0 and len(r.stdout) == 300 * 300 * 3:
                frames.append(np.frombuffer(r.stdout, dtype=np.uint8).reshape(300, 300, 3))
        except Exception:
            continue
    return frames


def download_clip(youtube_id, start_sec, duration=4.0):
    """Download a clip from YouTube. Returns path or None."""
    TMP_DIR.mkdir(exist_ok=True)
    out_path = TMP_DIR / f"{youtube_id}_{start_sec:06d}.mp4"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = ["yt-dlp", "--quiet", "--no-warnings",
           "--download-sections", f"*{start_sec}-{start_sec + duration}",
           "-f", "worst[ext=mp4]/worst",
           "--no-playlist", "--socket-timeout", "10", "--retries", "1",
           "-o", str(out_path), url]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        if out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
    except Exception:
        pass
    for f in [out_path, Path(str(out_path) + ".part")]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════

# --- Emotion (e_emb) ---

@torch.no_grad()
def feat_audio_emotion(models, audio_pcm):
    """wav2vec2 hidden states → 768-dim audio emotion."""
    if audio_pcm is None:
        return None
    try:
        inputs = models.wav2vec_proc(
            audio_pcm, sampling_rate=16000, return_tensors="pt",
            padding=True, max_length=16000 * 10, truncation=True)
        out = models.wav2vec(inputs.input_values)
        return out.last_hidden_state.mean(dim=1).squeeze(0).numpy()
    except Exception:
        return None


@torch.no_grad()
def feat_face_emotion(models, frames):
    """Detect face → ResNet-18 features → 512-dim face emotion."""
    if not frames or models.face_net is None:
        return None
    for frame in frames:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        models.face_net.setInput(blob)
        dets = models.face_net.forward()
        for j in range(dets.shape[2]):
            if dets[0, 0, j, 2] < 0.5:
                continue
            x1, y1 = max(0, int(dets[0, 0, j, 3] * w)), max(0, int(dets[0, 0, j, 4] * h))
            x2, y2 = min(w, int(dets[0, 0, j, 5] * w)), min(h, int(dets[0, 0, j, 6] * h))
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            face = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
            face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
            face_t = (face_t - torch.tensor([.485, .456, .406]).view(3, 1, 1)) / torch.tensor([.229, .224, .225]).view(3, 1, 1)
            return models.face_model(face_t.unsqueeze(0)).squeeze(0).numpy()
    return None


def build_emotion(audio_emo, face_emo):
    """Fuse into [audio_768, face_512] = 1280-dim."""
    e = np.zeros(E_DIM, dtype=np.float32)
    if audio_emo is not None:
        e[:AUDIO_EMO_DIM] = audio_emo[:AUDIO_EMO_DIM]
    if face_emo is not None:
        d = min(len(face_emo), FACE_EMO_DIM)
        e[AUDIO_EMO_DIM:AUDIO_EMO_DIM + d] = face_emo[:d]
    return e


# --- Speech (s_emb) ---

@torch.no_grad()
def feat_speech(models, audio_pcm):
    """Whisper ASR → text → sentence embedding (384-dim)."""
    if audio_pcm is None:
        return np.zeros(S_DIM, dtype=np.float32)
    try:
        # ASR: audio → text
        inputs = models.whisper_proc(audio_pcm, sampling_rate=16000, return_tensors="pt")
        ids = models.whisper_asr.generate(
            inputs.input_features,
            max_new_tokens=80,
            language="en",
            task="transcribe",
        )
        text = models.whisper_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

        if not text or len(text) < 3:
            return np.zeros(S_DIM, dtype=np.float32)

        # Text → sentence embedding
        tok_inputs = models.sent_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        out = models.sent_model(**tok_inputs)
        # Mean pooling with attention mask
        mask = tok_inputs["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return emb.squeeze(0).numpy()
    except Exception as ex:
        log.debug(f"Speech extraction failed: {ex}")
        return np.zeros(S_DIM, dtype=np.float32)


# --- Properties (p_emb) ---

def feat_color(frames):
    """Color histogram: 16 bins × 3 channels = 48-dim."""
    if not frames:
        return np.zeros(COLOR_DIM, dtype=np.float32)
    # Average histograms across frames
    hist = np.zeros(COLOR_DIM, dtype=np.float32)
    for frame in frames:
        for c in range(3):
            h, _ = np.histogram(frame[:, :, c], bins=16, range=(0, 256))
            hist[c * 16:(c + 1) * 16] += h.astype(np.float32)
    hist /= (len(frames) * 300 * 300)  # normalize
    return hist


def feat_motion(frames):
    """Temporal motion: mean + std of frame differences in 4×4 grid = 32-dim."""
    if len(frames) < 2:
        return np.zeros(MOTION_DIM, dtype=np.float32)
    # Convert to grayscale
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    diffs = [np.abs(grays[i + 1] - grays[i]) for i in range(len(grays) - 1)]
    if not diffs:
        return np.zeros(MOTION_DIM, dtype=np.float32)
    avg_diff = np.mean(diffs, axis=0)  # [300, 300]
    # Split into 4×4 grid
    result = np.zeros(MOTION_DIM, dtype=np.float32)
    h, w = avg_diff.shape
    gh, gw = h // 4, w // 4
    for r in range(4):
        for c in range(4):
            patch = avg_diff[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            idx = r * 4 + c
            result[idx] = patch.mean() / 255.0
            result[16 + idx] = patch.std() / 255.0
    return result


def feat_edges(frames):
    """Edge density in 4×4 grid = 16-dim."""
    if not frames:
        return np.zeros(EDGES_DIM, dtype=np.float32)
    # Average edge maps
    edge_sum = np.zeros((300, 300), dtype=np.float32)
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edge_sum += edges
    edge_avg = edge_sum / len(frames)
    # 4×4 grid density
    result = np.zeros(EDGES_DIM, dtype=np.float32)
    gh, gw = 75, 75
    for r in range(4):
        for c in range(4):
            result[r * 4 + c] = edge_avg[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw].mean()
    return result


def feat_loudness(audio_pcm):
    """RMS energy in 16 time windows = 16-dim."""
    if audio_pcm is None or len(audio_pcm) < 160:
        return np.zeros(LOUDNESS_DIM, dtype=np.float32)
    result = np.zeros(LOUDNESS_DIM, dtype=np.float32)
    win_size = len(audio_pcm) // LOUDNESS_DIM
    for i in range(LOUDNESS_DIM):
        chunk = audio_pcm[i * win_size:(i + 1) * win_size]
        result[i] = np.sqrt(np.mean(chunk ** 2))
    return result


def feat_pitch(audio_pcm):
    """Spectral features: centroid(16) + zero-crossing rate(16) = 32-dim."""
    if audio_pcm is None or len(audio_pcm) < 1600:
        return np.zeros(PITCH_DIM, dtype=np.float32)
    result = np.zeros(PITCH_DIM, dtype=np.float32)
    n_windows = 16
    win_size = len(audio_pcm) // n_windows
    for i in range(n_windows):
        chunk = audio_pcm[i * win_size:(i + 1) * win_size]
        # Spectral centroid
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / 16000)
        total = fft.sum()
        if total > 1e-10:
            result[i] = (freqs * fft).sum() / total / 8000.0  # normalize to [0, 1]
        # Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(chunk)))) / (2 * len(chunk))
        result[16 + i] = zcr
    return result


def feat_tempo(audio_pcm):
    """Onset strength in 8 windows = 8-dim."""
    if audio_pcm is None or len(audio_pcm) < 1600:
        return np.zeros(TEMPO_DIM, dtype=np.float32)
    result = np.zeros(TEMPO_DIM, dtype=np.float32)
    # Compute spectral flux as onset proxy
    hop = 512
    n_frames = len(audio_pcm) // hop
    if n_frames < 2:
        return result
    prev_fft = None
    flux = []
    for j in range(n_frames):
        chunk = audio_pcm[j * hop:(j + 1) * hop]
        if len(chunk) < hop:
            break
        fft = np.abs(np.fft.rfft(chunk))
        if prev_fft is not None:
            diff = np.maximum(fft - prev_fft, 0)
            flux.append(diff.sum())
        prev_fft = fft
    if not flux:
        return result
    flux = np.array(flux, dtype=np.float32)
    # Split into 8 windows
    win = len(flux) // TEMPO_DIM
    if win < 1:
        result[:len(flux)] = flux / (flux.max() + 1e-10)
    else:
        for i in range(TEMPO_DIM):
            result[i] = flux[i * win:(i + 1) * win].mean()
        mx = result.max()
        if mx > 1e-10:
            result /= mx
    return result


def build_properties(frames, audio_pcm):
    """Concatenate all property features → 152-dim."""
    parts = [
        feat_color(frames),
        feat_motion(frames),
        feat_edges(frames),
        feat_loudness(audio_pcm),
        feat_pitch(audio_pcm),
        feat_tempo(audio_pcm),
    ]
    return np.concatenate(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_clip_metadata():
    """clip_id → (yt_id, start_sec, label) from VGGSound CSV."""
    meta = {}
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    yt_id, start, label = row[0], int(row[1]), row[2]
                    meta[f"{yt_id}_{start:06d}"] = (yt_id, start, label)
    log.info(f"VGGSound metadata: {len(meta)} entries")
    return meta


def get_clip_ids_from_db(n_clips):
    import sqlite3
    try:
        conn = sqlite3.connect(str(Path("/opt/brain/outputs/cortex/knowledge.db")))
        rows = conn.execute("SELECT clip_id FROM data_inventory ORDER BY rowid LIMIT ?", (n_clips,)).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def load_progress():
    if PROGRESS_PATH.exists():
        return set(int(x) for x in PROGRESS_PATH.read_text().strip().split("\n") if x.strip())
    return set()


def save_progress(done):
    PROGRESS_PATH.write_text("\n".join(str(i) for i in sorted(done)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()

    from safetensors.numpy import load_file, save_file

    if not EMBED_PATH.exists():
        log.error(f"No embeddings at {EMBED_PATH}")
        return

    data = load_file(str(EMBED_PATH))
    v_emb = data["v_emb"]
    a_emb = data["a_emb"]
    n_clips = v_emb.shape[0]
    log.info(f"Loaded {n_clips} clips")

    # Initialize new tensors (or load existing partial results)
    e_emb = data.get("e_emb")
    if e_emb is None or e_emb.shape[1] != E_DIM:
        log.info(f"Creating new e_emb [{n_clips}, {E_DIM}]")
        e_emb = np.zeros((n_clips, E_DIM), dtype=np.float32)
    else:
        log.info(f"Loaded existing e_emb {e_emb.shape}")

    s_emb = data.get("s_emb")
    if s_emb is None or s_emb.shape != (n_clips, S_DIM):
        log.info(f"Creating new s_emb [{n_clips}, {S_DIM}]")
        s_emb = np.zeros((n_clips, S_DIM), dtype=np.float32)
    else:
        log.info(f"Loaded existing s_emb {s_emb.shape}")

    p_emb = data.get("p_emb")
    if p_emb is None or p_emb.shape != (n_clips, P_DIM):
        log.info(f"Creating new p_emb [{n_clips}, {P_DIM}]")
        p_emb = np.zeros((n_clips, P_DIM), dtype=np.float32)
    else:
        log.info(f"Loaded existing p_emb {p_emb.shape}")

    # Progress
    done = load_progress()
    log.info(f"Already processed: {len(done)} clips")

    # Metadata
    meta = build_clip_metadata()
    clip_ids = get_clip_ids_from_db(n_clips)
    clip_ids.extend(f"unknown_{i}" for i in range(len(clip_ids), n_clips))

    # Load models
    models = Models()
    models.load_all()
    log.info("All models loaded, starting extraction")

    # Stats
    stats = {"processed": 0, "audio_emo": 0, "face_emo": 0,
             "speech": 0, "dl_fail": 0, "dl_skip": 0}
    t0 = time.monotonic()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    idx = args.start_idx
    while idx < n_clips:
        batch_end = min(idx + args.batch, n_clips)
        batch_indices = [i for i in range(idx, batch_end) if i not in done]
        if not batch_indices:
            idx = batch_end
            continue

        # Prepare download tasks
        tasks = []
        for i in batch_indices:
            cid = clip_ids[i]
            if cid in meta:
                yt_id, start, _ = meta[cid]
                tasks.append((i, yt_id, start))
            else:
                parts = cid.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit() and not cid.startswith(("expanded_", "ravdess_", "unknown_")):
                    tasks.append((i, parts[0], int(parts[1])))
                else:
                    # Can't download (expanded/ravdess/unknown) — process with no video
                    tasks.append((i, None, None))

        # Parallel download
        downloaded = {}
        dl_tasks = [(i, yt, st) for i, yt, st in tasks if yt is not None]
        if dl_tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(download_clip, yt, st): i for i, yt, st in dl_tasks}
                for f in as_completed(futures):
                    i = futures[f]
                    path = f.result()
                    if path:
                        downloaded[i] = path
                    else:
                        stats["dl_fail"] += 1

        # Extract features for each clip
        for i, yt_id, start in tasks:
            video_path = downloaded.get(i)

            audio_pcm = None
            frames = []

            if video_path:
                audio_pcm = extract_audio_pcm(video_path)
                frames = extract_frames(video_path, n_frames=5)
            else:
                stats["dl_skip"] += 1

            # 1) Emotion
            audio_emo = feat_audio_emotion(models, audio_pcm)
            face_emo = feat_face_emotion(models, frames)
            e_emb[i] = build_emotion(audio_emo, face_emo)
            if audio_emo is not None:
                stats["audio_emo"] += 1
            if face_emo is not None:
                stats["face_emo"] += 1

            # 2) Speech (ASR → sentence embedding)
            s = feat_speech(models, audio_pcm)
            s_emb[i] = s
            if np.linalg.norm(s) > 0.1:
                stats["speech"] += 1

            # 3) Properties (color, motion, edges, loudness, pitch, tempo)
            p_emb[i] = build_properties(frames, audio_pcm)

            # Cleanup video
            if video_path:
                try:
                    video_path.unlink()
                except Exception:
                    pass

            done.add(i)
            stats["processed"] += 1

        # Save periodically
        save_file(
            {"v_emb": v_emb, "a_emb": a_emb, "e_emb": e_emb, "s_emb": s_emb, "p_emb": p_emb},
            str(EMBED_PATH),
        )
        save_progress(done)

        elapsed = time.monotonic() - t0
        rate = stats["processed"] / elapsed if elapsed > 0 else 0
        remaining = n_clips - len(done)
        eta = remaining / rate / 3600 if rate > 0 else float("inf")
        log.info(
            f"[{len(done)}/{n_clips}] "
            f"emo_a={stats['audio_emo']} emo_f={stats['face_emo']} "
            f"speech={stats['speech']} "
            f"dl_fail={stats['dl_fail']} skip={stats['dl_skip']} "
            f"| {rate:.2f}/s ETA {eta:.1f}h"
        )

        idx = batch_end

    # Final save
    save_file(
        {"v_emb": v_emb, "a_emb": a_emb, "e_emb": e_emb, "s_emb": s_emb, "p_emb": p_emb},
        str(EMBED_PATH),
    )
    save_progress(done)

    elapsed = time.monotonic() - t0
    log.info(
        f"DONE in {elapsed/3600:.1f}h — {stats['processed']} clips | "
        f"Audio emotion: {stats['audio_emo']} | Face emotion: {stats['face_emo']} | "
        f"Speech: {stats['speech']} | DL fail: {stats['dl_fail']} | Skip: {stats['dl_skip']}"
    )


if __name__ == "__main__":
    main()
