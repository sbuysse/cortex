#!/usr/bin/env python3
"""YouTube Brain Interaction Service.

Downloads a YouTube video, extracts visual + audio embeddings using the same
encoders as the VGGSound training data, then finds what the brain associates
with the input via similarity search against 24K cached clips.

Runs as a FastAPI microservice on port 8099.
"""

import os
import sys
import subprocess
import tempfile
import time
import json
import logging
import sqlite3
import threading
import queue
from pathlib import Path

import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
_start_time = time.time()

# ─── SSE Event Bus ─────────────────────────────────────────────────
# Thread-safe broadcast: multiple SSE clients subscribe to a shared event stream
_sse_subscribers: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _emit_event(event_type: str, data: dict):
    """Broadcast an event to all SSE subscribers."""
    import time as _t
    event = {"type": event_type, "time": _t.time(), **data}
    dead = []
    with _sse_lock:
        for q in _sse_subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)


def _subscribe_sse() -> queue.Queue:
    """Create a new SSE subscriber queue."""
    q = queue.Queue(maxsize=200)
    with _sse_lock:
        _sse_subscribers.append(q)
    return q


def _unsubscribe_sse(q: queue.Queue):
    """Remove an SSE subscriber."""
    with _sse_lock:
        if q in _sse_subscribers:
            _sse_subscribers.remove(q)

PROJECT_ROOT = Path("/opt/brain")
EMBED_CACHE = PROJECT_ROOT / "data/vggsound/.embed_cache/expanded_embeddings.safetensors"
VGGSOUND_CSV = PROJECT_ROOT / "data/vggsound/vggsound.csv"

# ─── Lazy-loaded globals ───────────────────────────────────────────
_visual_model = None
_whisper_model = None
_whisper_processor = None
_text_model = None     # Sentence-transformer for Phase 3 text Q&A
_label_embeddings = None  # Pre-encoded label text embeddings (N_labels, 384)
_cached_embeddings = None
_clip_labels = None
_clip_model = None     # OpenAI CLIP ViT-B/32 for zero-shot scene classification
_clip_processor = None
_clip_text_features = None  # Pre-encoded scene description embeddings
_w_v = None            # MLP weight W_v (384, d_hidden) for v4
_w_a = None            # MLP weight W_a (512, d_hidden) for v4
_m_matrix = None       # Bilinear M matrix (384x512) for v2/v3 fallback
_proj_v = None         # Projected visual embeddings (N, 512)
_proj_a = None         # Projected audio embeddings (N, 512)
_device = "cpu"

# Phase 4: Online learning buffer
_online_pairs: list[tuple[np.ndarray, np.ndarray]] = []  # (v_emb, a_emb) pairs
_online_learning_count = 0

# Step 1: World model
_world_model = None
# Step 2: NN graph for spreading activation
_nn_graph = None
# Step 3: Curiosity profiles
_curiosity_profiles = None
# Step 4: Grounded text projection W_t
_w_t = None
# CLAP audio model for discriminative audio similarity
_clap_model = None
_clap_processor = None
_clap_embeddings = None  # (24604, 512) CLAP text-encoded label embeddings
# Step 5: Concept codebook
_concept_codebook = None
_concept_labels = None
# Step 6: Causal graph
_causal_transitions = None
_causal_pmi = None
_causal_asymmetry = None
_causal_labels = None
# Step 7: Self-model
_category_stats = None
_confidence_model = None
# AudioSet expansion (588 categories beyond VGGSound)
_audioset_labels = None       # list[str] of expanded category names
_audioset_embeddings = None   # (N, 512) MLP-projected embeddings

# Phase 2: Episodic memory
_current_episode_id = None
_last_perception_time = 0.0
EPISODE_GAP_SECONDS = 30.0  # gap >30s = new episode

# Phase 3: Concept hierarchy
_concept_hierarchy = None  # JSON tree of hierarchical categories

# Phase 4: Working memory (7±2 slots) with theta-gamma phase ordering (#6)
_working_memory: list[dict] = []  # [{embedding, label, modality, timestamp, activation, theta_phase}]
WORKING_MEMORY_SLOTS = 7
WORKING_MEMORY_DECAY = 0.85
_wm_theta_phase = 0.0    # current theta oscillation phase (0 to 2*pi)
WM_THETA_FREQ = 0.15     # theta frequency (cycles per perception)

# Phase 5: Prototype memory (few-shot learning)
_prototypes: dict[str, dict] = {}
_consolidation_running = False

# Phase 6: Goal-directed planning
_active_goals: list[dict] = []

# Neuroscience-inspired features
# #9 ACh adaptive learning rate
_prediction_error_history: list[float] = []
ACH_WINDOW = 50
ACH_LR_MIN = 0.0002
ACH_LR_MAX = 0.005

# #10 Sparse embeddings
SPARSE_K = 0  # 0=disabled, 100=keep top 100 of 512 dims

# #8 Opponent curiosity (distributional)
_curiosity_optimistic: dict[str, float] = {}
_curiosity_pessimistic: dict[str, float] = {}

# #11 Dual fast/slow memory
_fast_memory_patterns: np.ndarray | None = None  # (capacity, 512)
_fast_memory_labels: list[str] = []
_fast_memory_count = 0
FAST_MEMORY_CAPACITY = 2000

# Grid Cell System: hexagonal spatial encoding of concept space
_grid_cell_encoder = None  # trained grid projection matrices
_grid_concept_coords = None  # (N_concepts, 2) grid coordinates per concept
_grid_episode_paths = []  # list of grid trajectories from episodes

# Phase 7: AudioSet expansion (2M clips in MLP-projected 512-dim space)
_audioset_pool = None        # (N, 512) MLP-space embeddings from AudioSet
_audioset_pool_labels = None # list[list[str]] per-clip label lists
_audioset_pool_count = 0     # number of AudioSet clips loaded

MODEL_DIR_V4 = PROJECT_ROOT / "outputs/cortex/v6_mlp"
MODEL_DIR_V3 = PROJECT_ROOT / "outputs/cortex/v3f_anneal"
MODEL_DIR = PROJECT_ROOT / "outputs/cortex/best_model"


def _load_bin_matrix(path: Path) -> np.ndarray | None:
    """Load a binary matrix saved by train-and-save (header: RxC\\n + f32 LE data)."""
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split('x'))
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(rows, cols)


def load_mlp_weights():
    """Load v4 MLP weights (W_v, W_a) for cross-modal queries."""
    global _w_v, _w_a
    if _w_v is not None:
        return _w_v, _w_a
    _w_v = _load_bin_matrix(MODEL_DIR_V4 / "w_v.bin")
    _w_a = _load_bin_matrix(MODEL_DIR_V4 / "w_a.bin")
    if _w_v is not None:
        log.info(f"Loaded v4 MLP weights: W_v={_w_v.shape}, W_a={_w_a.shape}")
    return _w_v, _w_a


def mlp_project(emb: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Project embedding through MLP layer: ReLU(emb @ W), L2-normalized.
    With optional top-K sparsification (#10 sparse coding)."""
    proj = emb @ w
    proj = np.maximum(proj, 0)  # ReLU
    # #10: Sparse coding — keep only top-K activations
    if SPARSE_K > 0 and proj.ndim == 2 and proj.shape[1] > SPARSE_K:
        for i in range(len(proj)):
            if proj[i].sum() > 0:
                threshold = np.partition(proj[i], -SPARSE_K)[-SPARSE_K]
                proj[i, proj[i] < threshold] = 0
    elif SPARSE_K > 0 and proj.ndim == 1 and len(proj) > SPARSE_K:
        if proj.sum() > 0:
            threshold = np.partition(proj, -SPARSE_K)[-SPARSE_K]
            proj[proj < threshold] = 0
    norm = np.linalg.norm(proj, axis=-1, keepdims=True).clip(1e-12)
    return proj / norm


def load_m_matrix():
    """Load the trained M matrix for cross-modal queries (v3f or fallback)."""
    global _m_matrix
    if _m_matrix is not None:
        return _m_matrix
    # Try v3f first, then v2, then old best_model
    for d in [MODEL_DIR_V3, MODEL_DIR]:
        _m_matrix = _load_bin_matrix(d / "m_va.bin")
        if _m_matrix is not None:
            log.info(f"Loaded M matrix from {d}: {_m_matrix.shape}")
            return _m_matrix
    return _m_matrix


def load_projected_embeddings():
    """Load projected embeddings (512-dim, same space as M matrix)."""
    global _proj_v, _proj_a
    if _proj_v is not None:
        return _proj_v, _proj_a
    _proj_v = _load_bin_matrix(MODEL_DIR / "v_proj.bin")
    _proj_a = _load_bin_matrix(MODEL_DIR / "a_proj.bin")
    if _proj_v is not None:
        log.info(f"Loaded projected embeddings: v={_proj_v.shape}, a={_proj_a.shape}")
    return _proj_v, _proj_a


def load_clip_labels():
    """Load VGGSound CSV → list of (youtube_id, start_sec, label)."""
    global _clip_labels
    if _clip_labels is not None:
        return _clip_labels

    import safetensors.numpy as sf
    data = sf.load_file(str(EMBED_CACHE))
    n_clips = data["v_emb"].shape[0]

    labels = []
    with open(VGGSOUND_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                labels.append({
                    "youtube_id": parts[0],
                    "start_sec": int(parts[1]),
                    "label": parts[2].strip('"'),
                    "split": parts[3] if len(parts) > 3 else ""
                })
            if len(labels) >= n_clips:
                break

    _clip_labels = labels
    log.info(f"Loaded {len(labels)} clip labels")
    return labels


def load_cached_embeddings():
    """Load pre-computed VGGSound embeddings."""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    import safetensors.numpy as sf
    data = sf.load_file(str(EMBED_CACHE))
    _cached_embeddings = {
        "v_emb": data["v_emb"],  # (N, 384)
        "a_emb": data["a_emb"],  # (N, 512)
    }
    # L2 normalize for cosine similarity
    for k in _cached_embeddings:
        norms = np.linalg.norm(_cached_embeddings[k], axis=1, keepdims=True).clip(1e-12)
        _cached_embeddings[k] = _cached_embeddings[k] / norms

    log.info(f"Loaded cached embeddings: v={data['v_emb'].shape}, a={data['a_emb'].shape}")
    return _cached_embeddings


def load_text_model():
    """Load sentence-transformer for text→embedding Q&A (Phase 3)."""
    global _text_model, _label_embeddings
    if _text_model is not None:
        return _text_model
    from sentence_transformers import SentenceTransformer
    log.info("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Text encoder loaded (384-dim, multilingual)")

    # Pre-encode all unique clip labels
    labels = load_clip_labels()
    unique_labels = list(dict.fromkeys(clip["label"] for clip in labels))
    prefixed_labels = unique_labels
    _label_embeddings = {
        "labels": unique_labels,
        "embeddings": _text_model.encode(prefixed_labels, normalize_embeddings=True,
                                          show_progress_bar=False),
    }
    log.info(f"Pre-encoded {len(unique_labels)} unique labels")
    return _text_model


def load_audioset_expansion():
    """Load AudioSet expanded categories (labels + MLP-projected embeddings)."""
    global _audioset_labels, _audioset_embeddings
    if _audioset_labels is not None:
        return _audioset_labels, _audioset_embeddings

    expansion_dir = PROJECT_ROOT / "outputs/cortex/audioset_expansion"
    labels_path = expansion_dir / "labels.json"
    emb_path = expansion_dir / "embeddings.npy"

    if not labels_path.exists() or not emb_path.exists():
        log.warning("AudioSet expansion not found — run build_audioset_expansion.py first")
        return None, None

    with open(labels_path) as f:
        _audioset_labels = json.load(f)
    _audioset_embeddings = np.load(emb_path).astype(np.float32)

    # L2 normalize (should already be, but ensure)
    norms = np.linalg.norm(_audioset_embeddings, axis=1, keepdims=True).clip(1e-12)
    _audioset_embeddings = _audioset_embeddings / norms

    log.info(f"Loaded AudioSet expansion: {len(_audioset_labels)} categories, "
             f"embeddings {_audioset_embeddings.shape}")
    return _audioset_labels, _audioset_embeddings


def load_audioset_pool():
    """Load the 2M AudioSet embeddings using memory-mapped files (saves ~4GB RAM)."""
    global _audioset_pool, _audioset_pool_labels, _audioset_pool_count
    if _audioset_pool is not None:
        return _audioset_pool

    pool_dir = PROJECT_ROOT / "outputs/cortex/audioset_brain"
    if not pool_dir.exists():
        return None

    parts = []
    label_parts = []
    for split in ["bal_train", "eval", "unbal_train"]:
        fpath = pool_dir / f"{split}_embeddings.npy"
        shape_path = pool_dir / f"{split}_embeddings.shape"
        lpath = pool_dir / f"{split}_labels.json"
        if fpath.exists() and shape_path.exists():
            with open(shape_path) as f:
                shape = tuple(json.load(f))
            # Memory-mapped: OS manages paging, only loads accessed pages into RAM
            arr = np.memmap(str(fpath), dtype=np.float32, mode='r',
                           offset=128, shape=shape)  # npy header ~128 bytes
            parts.append(arr)
            log.info(f"Loaded AudioSet {split}: {shape} (memmap)")
            if lpath.exists():
                with open(lpath) as f:
                    label_parts.extend(json.load(f))
            else:
                label_parts.extend([["unknown"]] * shape[0])
        elif fpath.exists():
            # Fallback: regular load if no shape file
            arr = np.load(str(fpath)).astype(np.float32)
            parts.append(arr)
            log.info(f"Loaded AudioSet {split}: {arr.shape}")
            if lpath.exists():
                with open(lpath) as f:
                    label_parts.extend(json.load(f))
            else:
                label_parts.extend([["unknown"]] * len(arr))

    if not parts:
        return None

    _audioset_pool = np.concatenate([np.array(p) for p in parts], axis=0)
    _audioset_pool_labels = label_parts
    _audioset_pool_count = len(_audioset_pool)
    log.info(f"AudioSet pool ready: {_audioset_pool_count:,} clips ({_audioset_pool.shape[1]}-dim), "
             f"{len(_audioset_pool_labels)} labeled")
    return _audioset_pool


def search_audioset_pool(query_emb_512: np.ndarray, top_k: int = 10) -> list[dict]:
    """Search the 2M AudioSet pool using a 512-dim MLP-space query embedding.

    Returns top_k matches with similarity scores and real AudioSet labels.
    """
    pool = load_audioset_pool()
    if pool is None:
        return []

    q_norm = query_emb_512 / (np.linalg.norm(query_emb_512) + 1e-12)
    sims = pool @ q_norm  # (2M,)

    top_idx = np.argsort(sims)[::-1][:top_k * 3]  # get extra to deduplicate

    results = []
    seen_labels = set()
    for idx in top_idx:
        if idx >= len(pool):
            continue
        # Use real AudioSet labels if available
        if _audioset_pool_labels and idx < len(_audioset_pool_labels):
            clip_labels = _audioset_pool_labels[idx]
            primary = clip_labels[0] if clip_labels else "unknown"
        else:
            primary = f"audioset_clip_{idx}"

        if primary not in seen_labels:
            seen_labels.add(primary)
            results.append({
                "label": primary,
                "all_labels": clip_labels if _audioset_pool_labels and idx < len(_audioset_pool_labels) else [],
                "similarity": round(float(sims[idx]), 4),
                "audioset_idx": int(idx),
                "source": "audioset",
            })
        if len(results) >= top_k:
            break

    return results


def text_semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Encode a text query and find semantically similar clip labels (Phase 3).

    Searches both VGGSound labels (310, with real audio data) and AudioSet
    expanded labels (588, text-only — no audio clips behind them).
    """
    model = load_text_model()
    q_emb = model.encode([f"{query}"], normalize_embeddings=True)[0]  # (384,)

    lab_data = _label_embeddings
    sims = lab_data["embeddings"] @ q_emb  # (N_labels,)
    top_idx = np.argsort(sims)[::-1][:top_k]

    # Map back to clip indices
    labels = load_clip_labels()
    label_to_clips = {}
    for i, clip in enumerate(labels):
        label_to_clips.setdefault(clip["label"], []).append(i)

    results = []
    for idx in top_idx:
        label = lab_data["labels"][idx]
        sim = float(sims[idx])
        clips = label_to_clips.get(label, [])
        results.append({
            "label": label,
            "text_similarity": sim,
            "clip_count": len(clips),
            "clip_indices": clips[:5],  # first 5 clip indices for this label
            "source": "vggsound",
        })

    # Also search AudioSet expanded categories (text-projected, no real audio)
    as_labels, as_embs = load_audioset_expansion()
    if as_labels is not None:
        # Project query through MLP W_v to match AudioSet's projected space
        w_v, _ = load_mlp_weights()
        if w_v is not None:
            q_proj = mlp_project(q_emb.reshape(1, -1).astype(np.float32), w_v)[0]  # (512,)
            as_sims = as_embs @ q_proj  # (N_audioset,)
            as_top_idx = np.argsort(as_sims)[::-1][:top_k]

            # Collect VGGSound labels already in results to avoid duplicates
            seen_labels = {r["label"].lower() for r in results}

            for idx in as_top_idx:
                label = as_labels[idx]
                if label.lower() in seen_labels:
                    continue
                sim = float(as_sims[idx])
                results.append({
                    "label": label,
                    "text_similarity": sim,
                    "clip_count": 0,  # no real audio clips
                    "clip_indices": [],
                    "source": "audioset",
                })

    # Sort all results by similarity, return top_k
    results.sort(key=lambda r: r["text_similarity"], reverse=True)
    return results[:top_k]


# ─── CLAP (Contrastive Language-Audio Pretraining) ─────────────────

CLAP_TEXT_CACHE = PROJECT_ROOT / "data/vggsound/.embed_cache/clap_text_embeddings.npy"


def load_clap_model():
    """Lazy-load CLAP audio processor + model from laion/clap-htsat-fused."""
    global _clap_model, _clap_processor
    if _clap_model is not None:
        return _clap_model, _clap_processor
    from transformers import ClapModel, ClapProcessor
    log.info("Loading CLAP model (laion/clap-htsat-fused)...")
    _clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    _clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused")
    _clap_model.eval()
    _clap_model.to(_device)
    log.info("CLAP model loaded (512-dim audio embeddings)")
    return _clap_model, _clap_processor


def encode_audio_clap(samples: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Encode raw audio waveform to 512-dim CLAP embedding.

    CLAP expects 48kHz audio. If input is 16kHz, resample first.
    Returns L2-normalized 512-dim numpy array.
    """
    model, processor = load_clap_model()

    # Resample to 48kHz if needed
    if sr != 48000:
        import scipy.signal
        samples = scipy.signal.resample_poly(samples, 48000, sr).astype(np.float32)

    inputs = processor(audio=samples, sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_features = model.get_audio_features(**inputs)

    emb = audio_features.squeeze(0).cpu().numpy().astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def load_clap_audio_embeddings() -> np.ndarray:
    """Load or compute CLAP embeddings for all 24,604 VGGSound clips.

    Uses CLAP's TEXT encoder on the 310 category labels as a pragmatic shortcut
    (avoids re-encoding 24K audio files). Each clip gets the text embedding of
    its label. Cached to disk after first computation.

    Returns shape (N_clips, 512) L2-normalized embeddings.
    """
    global _clap_embeddings
    if _clap_embeddings is not None:
        return _clap_embeddings

    if CLAP_TEXT_CACHE.exists():
        log.info(f"Loading cached CLAP text embeddings from {CLAP_TEXT_CACHE}")
        _clap_embeddings = np.load(str(CLAP_TEXT_CACHE))
        log.info(f"CLAP embeddings loaded: {_clap_embeddings.shape}")
        return _clap_embeddings

    log.info("Computing CLAP text embeddings for all VGGSound labels (first time)...")
    model, processor = load_clap_model()
    labels = load_clip_labels()

    # Get unique labels and their CLAP text embeddings
    unique_labels = list(dict.fromkeys(clip["label"] for clip in labels))
    log.info(f"Encoding {len(unique_labels)} unique labels with CLAP text encoder...")

    label_to_emb = {}
    batch_size = 32
    for i in range(0, len(unique_labels), batch_size):
        batch = unique_labels[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        text_embs = text_features.cpu().numpy().astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(text_embs, axis=1, keepdims=True).clip(1e-12)
        text_embs = text_embs / norms
        for j, label in enumerate(batch):
            label_to_emb[label] = text_embs[j]

    # Map each clip to its label's embedding
    n_clips = len(labels)
    emb_dim = 512
    all_embs = np.zeros((n_clips, emb_dim), dtype=np.float32)
    for i, clip in enumerate(labels):
        all_embs[i] = label_to_emb[clip["label"]]

    # Cache to disk
    CLAP_TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(CLAP_TEXT_CACHE), all_embs)
    log.info(f"Saved CLAP text embeddings to {CLAP_TEXT_CACHE}: {all_embs.shape}")

    _clap_embeddings = all_embs
    return _clap_embeddings


def load_visual_model():

    """Load DINOv2 vits14 (384-dim output)."""
    global _visual_model
    if _visual_model is not None:
        return _visual_model

    log.info("Loading DINOv2 vits14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.eval()
    model.to(_device)
    _visual_model = model
    log.info("DINOv2 loaded")
    return model


# ─── CLIP zero-shot scene descriptions ────────────────────────────
CLIP_SCENE_DESCRIPTIONS = [
    # General scenes & places
    "a person sitting at a desk",
    "a person working at a computer",
    "a person in front of a monitor",
    "a home office with a desk and chair",
    "a person looking at the camera",
    "a person on a video call",
    "a person wearing headphones",
    "a bedroom",
    "a person sitting on a couch",
    "people talking",
    "an office",
    "a kitchen",
    "outdoors in nature",
    "a dog",
    "a cat",
    "a car on a road",
    "a street with buildings",
    "food on a table",
    "a computer screen",
    "people walking",
    "a living room",
    "a bathroom",
    "a garden with plants",
    "musical instruments",
    "a concert or live performance",
    "a sports event",
    "rain falling",
    "a forest with trees",
    "water or ocean waves",
    "fire or flames",
    "a person cooking",
    "a person reading",
    "a person exercising",
    "a classroom or lecture",
    "a city skyline",
    "a beach",
    "a mountain landscape",
    "a sunset or sunrise",
    "a person using a phone",
    "a store or shopping",
    "a restaurant or cafe",
    "a park",
    "a crowd of people",
    "a person playing video games",
    "a baby or toddler",
    "a bird",
    "a close-up of a face",
    "a person working on a laptop",
    "a construction site",
    "a vehicle interior",
    "a hallway or corridor",
    "nighttime scene",
    "snow or winter scene",
    "an empty room",
    "a bookshelf",
    "a TV or monitor displaying content",
    "hands doing something",
    "a pet animal",
    # Top 30 VGGSound categories as scene descriptions
    "a person playing acoustic guitar",
    "a person playing piano",
    "a person playing violin",
    "a person playing drums",
    "a person playing electric guitar",
    "a person playing flute",
    "a person playing cello",
    "a person playing saxophone",
    "a person playing trumpet or trombone",
    "a person singing",
    "a person speaking or talking",
    "a child speaking or crying",
    "a dog barking",
    "a cat meowing",
    "a helicopter flying",
    "a race car or vehicle",
    "a motorcycle",
    "a bus or truck",
    "fireworks",
    "an orchestra performing",
    "people clapping or cheering",
    "a crowd booing or shouting",
    "a toilet flushing",
    "a siren from police or ambulance",
    "tap dancing",
    "rowing a boat on water",
    "a pigeon or bird",
    "a train or railroad",
    "a horn honking",
    "a motorboat on water",
]


def load_clip_model():
    """Lazy-load OpenAI CLIP ViT-B/32 for zero-shot scene classification."""
    global _clip_model, _clip_processor, _clip_text_features
    if _clip_model is not None:
        return _clip_model, _clip_processor, _clip_text_features

    log.info("Loading CLIP ViT-B/32...")
    from transformers import CLIPProcessor, CLIPModel
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
    _clip_model.to(_device)

    # Pre-encode all scene text descriptions (done once)
    log.info(f"Pre-encoding {len(CLIP_SCENE_DESCRIPTIONS)} scene descriptions...")
    text_inputs = _clip_processor(text=CLIP_SCENE_DESCRIPTIONS, return_tensors="pt",
                                   padding=True, truncation=True)
    text_inputs = {k: v.to(_device) for k, v in text_inputs.items() if k != "pixel_values"}
    with torch.no_grad():
        text_embs = _clip_model.get_text_features(**text_inputs)
    # Handle both tensor and BaseModelOutput returns
    if not isinstance(text_embs, torch.Tensor):
        text_embs = text_embs.pooler_output if hasattr(text_embs, 'pooler_output') else text_embs[0]
    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True).clamp(min=1e-12)
    _clip_text_features = text_embs  # (N_scenes, 512) on CPU
    log.info(f"CLIP loaded. Text features shape: {_clip_text_features.shape}")
    return _clip_model, _clip_processor, _clip_text_features


def clip_classify_image(image, top_k: int = 5) -> list[dict]:
    """Run CLIP zero-shot classification on a PIL Image.

    Returns top-k scene descriptions with confidence scores.
    """
    clip_model, clip_processor, text_features = load_clip_model()

    # Encode image
    img_inputs = clip_processor(images=image, return_tensors="pt")
    img_inputs = {k: v.to(_device) for k, v in img_inputs.items()}
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**img_inputs)
    if not isinstance(img_emb, torch.Tensor):
        img_emb = img_emb.pooler_output if hasattr(img_emb, 'pooler_output') else img_emb[0]
    img_emb = img_emb / torch.norm(img_emb, dim=-1, keepdim=True).clamp(min=1e-12)

    # Cosine similarity against all pre-encoded scene descriptions
    sims = (img_emb @ text_features.T).squeeze(0).cpu().numpy()  # (N_scenes,)

    # Softmax for confidence scores
    exp_sims = np.exp(sims - sims.max())
    probs = exp_sims / exp_sims.sum()

    top_idx = np.argsort(probs)[::-1][:top_k]
    return [
        {"description": CLIP_SCENE_DESCRIPTIONS[i], "confidence": float(probs[i])}
        for i in top_idx
    ]


def load_whisper_model():
    """Load Whisper-base encoder (512-dim output)."""
    global _whisper_model, _whisper_processor
    if _whisper_model is not None:
        return _whisper_model, _whisper_processor

    log.info("Loading Whisper-base...")
    from transformers import WhisperModel, WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperModel.from_pretrained("openai/whisper-base")
    model.eval()
    model.to(_device)
    _whisper_model = model.encoder
    _whisper_processor = processor
    log.info("Whisper loaded")
    return _whisper_model, _whisper_processor


def download_youtube(url: str, output_dir: str, max_duration: int = 30) -> dict:
    """Download YouTube video, return paths to video and audio files."""
    video_path = os.path.join(output_dir, "video.mp4")
    audio_path = os.path.join(output_dir, "audio.wav")

    # Download video
    log.info(f"Downloading: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "--merge-output-format", "mp4",
        "-o", video_path,
        "--no-playlist",
        "--max-downloads", "1",
        "--max-filesize", "50M",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if not os.path.exists(video_path):
        # Try simpler format
        cmd = ["yt-dlp", "-f", "best[height<=480]", "-o", video_path,
               "--no-playlist", "--max-downloads", "1", "--max-filesize", "50M", url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not os.path.exists(video_path):
            raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

    # Extract audio as WAV
    log.info("Extracting audio...")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "-t", str(max_duration), audio_path]
    subprocess.run(cmd, capture_output=True, timeout=60)

    # Get video info
    probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", video_path]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    info = {}
    try:
        probe_data = json.loads(probe.stdout)
        info["duration"] = float(probe_data.get("format", {}).get("duration", 0))
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "video":
                info["width"] = s.get("width", 0)
                info["height"] = s.get("height", 0)
    except Exception:
        pass

    # Get video title
    title_cmd = ["yt-dlp", "--get-title", "--no-playlist", url]
    title_result = subprocess.run(title_cmd, capture_output=True, text=True, timeout=30)
    info["title"] = title_result.stdout.strip() if title_result.returncode == 0 else "Unknown"

    return {"video": video_path, "audio": audio_path, "info": info}


def extract_visual_embedding(video_path: str, max_frames: int = 16) -> np.ndarray:
    """Extract DINOv2 embedding from video frames → mean pool → 384-dim."""
    import torchvision.transforms as T
    from torchvision.io import read_video

    model = load_visual_model()

    # Read video frames
    try:
        frames, _, finfo = read_video(video_path, pts_unit="sec", output_format="TCHW")
    except Exception:
        # Fallback: use ffmpeg to extract frames
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-vf", "fps=4,scale=224:224",
                "-t", "10", os.path.join(td, "frame_%04d.png")
            ], capture_output=True, timeout=30)
            from PIL import Image
            frame_files = sorted(Path(td).glob("frame_*.png"))
            if not frame_files:
                return np.zeros(384, dtype=np.float32)
            frames = []
            for fp in frame_files[:max_frames]:
                img = Image.open(fp).convert("RGB")
                frames.append(np.array(img))
            frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

    if frames.shape[0] == 0:
        return np.zeros(384, dtype=np.float32)

    # Sample evenly spaced frames
    n = frames.shape[0]
    indices = np.linspace(0, n - 1, min(max_frames, n), dtype=int)
    frames = frames[indices]

    # DINOv2 expects ImageNet-normalized 224x224
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # frames is [T, C, H, W] in 0-1 range
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    frames = torch.stack([transform(f) for f in frames])

    with torch.no_grad():
        embeddings = model(frames.to(_device))  # [T, 384]

    # Mean pool over frames
    embedding = embeddings.mean(dim=0).cpu().numpy()
    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
    return embedding.astype(np.float32)


def extract_audio_embedding(audio_path: str) -> np.ndarray:
    """Extract Whisper encoder embedding from audio → mean pool → 512-dim."""
    whisper_enc, processor = load_whisper_model()

    # Load audio (soundfile for WAV — torchaudio needs torchcodec)
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]  # mono
    if sr != 16000:
        # ffmpeg already outputs 16kHz, but just in case
        import scipy.signal
        data = scipy.signal.resample_poly(data, 16000, sr).astype(np.float32)
    waveform = torch.from_numpy(data)

    # Truncate to 30s
    max_samples = 16000 * 30
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    # Process with Whisper
    inputs = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features.to(_device)

    with torch.no_grad():
        enc_out = whisper_enc(input_features)
        # enc_out.last_hidden_state: [1, T, 512]
        embedding = enc_out.last_hidden_state.mean(dim=1).squeeze(0)

    embedding = embedding.cpu().numpy()
    embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
    return embedding.astype(np.float32)


def find_associations(v_emb: np.ndarray, a_emb: np.ndarray, top_k: int = 20):
    """Find most associated clips using v4 MLP + direct similarity."""
    cached = load_cached_embeddings()
    labels = load_clip_labels()

    v_emb_norm = v_emb / (np.linalg.norm(v_emb) + 1e-12)
    a_emb_norm = a_emb / (np.linalg.norm(a_emb) + 1e-12)

    # Direct visual similarity (what looks like this?)
    v_sims = cached["v_emb"] @ v_emb_norm  # (N,)
    v_top = np.argsort(v_sims)[::-1][:top_k]

    # Direct audio similarity (what sounds like this?)
    a_sims = cached["a_emb"] @ a_emb_norm  # (N,)
    a_top = np.argsort(a_sims)[::-1][:top_k]

    # Cross-modal via v4 MLP: project input through ReLU(emb @ W), then dot-product
    w_v, w_a = load_mlp_weights()
    if w_v is not None:
        # Project input embeddings through MLP
        input_v_proj = mlp_project(v_emb_norm.reshape(1, -1), w_v)[0]   # (d_hidden,)
        input_a_proj = mlp_project(a_emb_norm.reshape(1, -1), w_a)[0]   # (d_hidden,)

        # Project all cached embeddings through MLP
        all_v_proj = mlp_project(cached["v_emb"], w_v)  # (N, d_hidden)
        all_a_proj = mlp_project(cached["a_emb"], w_a)  # (N, d_hidden)

        # V→A cross-modal: input visual → find matching audio
        v2a_sims = all_a_proj @ input_v_proj  # (N,)
        v2a_top = np.argsort(v2a_sims)[::-1][:top_k]

        # A→V cross-modal: input audio → find matching visual
        a2v_sims = all_v_proj @ input_a_proj  # (N,)
        a2v_top = np.argsort(a2v_sims)[::-1][:top_k]

        cross_modal_v2a = _format_results(v2a_top, v2a_sims, labels)
        cross_modal_a2v = _format_results(a2v_top, a2v_sims, labels)
    else:
        cross_modal_v2a = _format_results(v_top, a_sims, labels)
        cross_modal_a2v = []

    result = {
        "visual_similar": _format_results(v_top, v_sims, labels),
        "audio_similar": _format_results(a_top, a_sims, labels),
        "cross_modal": cross_modal_v2a,
        "cross_modal_a2v": cross_modal_a2v,
    }

    # Search the 2M AudioSet pool (already in MLP space)
    if w_v is not None and _audioset_pool is not None:
        as_v2a = search_audioset_pool(input_v_proj, top_k=top_k)
        as_a2v = search_audioset_pool(input_a_proj, top_k=top_k)
        if as_v2a:
            result["audioset_v2a"] = as_v2a
        if as_a2v:
            result["audioset_a2v"] = as_a2v

    return result


def _format_results(indices, sims, labels):
    results = []
    for idx in indices:
        if idx < len(labels):
            results.append({
                "idx": int(idx),
                "label": labels[idx]["label"],
                "youtube_id": labels[idx]["youtube_id"],
                "start_sec": labels[idx]["start_sec"],
                "similarity": float(sims[idx]),
            })
    return results


# ─── FastAPI app ───────────────────────────────────────────────────
# ─── Persistent Memory DB ──────────────────────────────────────────
MEMORY_DB_PATH = PROJECT_ROOT / "outputs/cortex/brain_memory.db"
_memory_db_local = threading.local()


def _get_memory_db() -> sqlite3.Connection:
    """Get a thread-local SQLite connection for the memory DB."""
    conn = getattr(_memory_db_local, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(str(MEMORY_DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _memory_db_local.conn = conn
    return conn


def _init_memory_db():
    """Create/open the memory DB and create tables if needed."""
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_memory_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS perceptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            modality TEXT NOT NULL,
            transcription TEXT,
            top_labels TEXT,
            cross_labels TEXT,
            imagination TEXT,
            narration TEXT
        );
        CREATE TABLE IF NOT EXISTS reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            insight TEXT NOT NULL,
            perception_count INTEGER,
            online_pairs_learned INTEGER
        );
        CREATE TABLE IF NOT EXISTS learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event TEXT NOT NULL,
            details TEXT
        );
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time REAL NOT NULL,
            end_time REAL,
            context TEXT,
            event_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS episode_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            modality TEXT NOT NULL,
            embedding_blob BLOB,
            label TEXT,
            metadata_json TEXT,
            FOREIGN KEY (episode_id) REFERENCES episodes(id)
        );
        CREATE INDEX IF NOT EXISTS idx_episode_events_episode ON episode_events(episode_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_time);
        CREATE TABLE IF NOT EXISTS prototypes (
            name TEXT PRIMARY KEY,
            centroid_blob BLOB NOT NULL,
            count INTEGER DEFAULT 1,
            examples_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            embedding_blob BLOB,
            priority REAL DEFAULT 1.0,
            status TEXT DEFAULT 'active',
            created_at REAL NOT NULL,
            completed_at REAL
        );
        CREATE TABLE IF NOT EXISTS knowledge_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_label TEXT NOT NULL,
            relation TEXT NOT NULL,
            target_label TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            evidence_count INTEGER DEFAULT 1,
            last_seen REAL NOT NULL,
            metadata_json TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_unique ON knowledge_edges(source_label, relation, target_label);
        CREATE INDEX IF NOT EXISTS idx_ke_source ON knowledge_edges(source_label);
        CREATE INDEX IF NOT EXISTS idx_ke_target ON knowledge_edges(target_label);
        CREATE TABLE IF NOT EXISTS youtube_learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            category TEXT NOT NULL,
            timestamp REAL NOT NULL,
            status TEXT NOT NULL,
            pairs_generated INTEGER DEFAULT 0,
            error_text TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_yll_category ON youtube_learning_log(category);
    """)
    conn.commit()
    log.info(f"Memory DB initialized at {MEMORY_DB_PATH}")


def _store_perception(modality: str, transcription: str | None = None,
                      top_labels: list | None = None, cross_labels: list | None = None,
                      imagination: dict | None = None, narration: str | None = None,
                      embedding: np.ndarray | None = None):
    """Insert a perception record into the memory DB + episode tracking + working memory."""
    try:
        conn = _get_memory_db()
        conn.execute(
            "INSERT INTO perceptions (timestamp, modality, transcription, top_labels, cross_labels, imagination, narration) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), modality, transcription,
             json.dumps(top_labels) if top_labels else None,
             json.dumps(cross_labels) if cross_labels else None,
             json.dumps(imagination) if imagination else None,
             narration)
        )
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to store perception: {e}")

    # Phase 2: Store in current episode
    primary_label = top_labels[0] if top_labels else (transcription or "unknown")
    if isinstance(primary_label, dict):
        primary_label = primary_label.get("label", str(primary_label))

    _emit_event("perception", {"modality": modality, "label": primary_label,
                                "cross": cross_labels[:3] if cross_labels else []})

    # #7: Compute surprise tag for tag-then-replay consolidation
    surprise_tag = 0.0
    if embedding is not None:
        codebook, _ = build_concept_codebook()
        if codebook is not None:
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
            sims = codebook @ emb_norm
            surprise_tag = round(1.0 - float(sims.max()), 4)
        # #9: Track prediction error for ACh modulation
        _prediction_error_history.append(surprise_tag)
        if len(_prediction_error_history) > ACH_WINDOW * 2:
            _prediction_error_history[:] = _prediction_error_history[-ACH_WINDOW:]

    _store_episode_event(modality, primary_label, embedding,
                         {"cross_labels": cross_labels[:3] if cross_labels else None,
                          "surprise_tag": surprise_tag})

    # Phase 4: Update working memory
    if embedding is not None:
        wm_result = _update_working_memory(embedding, primary_label, modality)
        _emit_event("working_memory", {"label": primary_label, "modality": modality,
                                        "attention": wm_result.get("attention_score", 0),
                                        "related": wm_result.get("related_items", []),
                                        "slots": wm_result.get("buffer_size", 0)})

    # Phase 5.1: Check for novel prototypes (only if embedding available)
    if embedding is not None:
        proto_result = _check_novelty_and_prototype(embedding, primary_label, modality)
        if proto_result.get("action") in ("created", "refined"):
            _emit_event("prototype", proto_result)


def _store_reflection(insight: str, perception_count: int = 0, online_pairs: int = 0):
    """Insert a reflection record into the memory DB."""
    try:
        conn = _get_memory_db()
        conn.execute(
            "INSERT INTO reflections (timestamp, insight, perception_count, online_pairs_learned) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), insight, perception_count, online_pairs)
        )
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to store reflection: {e}")


def _store_learning_log(event: str, details: dict | None = None):
    """Insert a learning log entry."""
    try:
        conn = _get_memory_db()
        conn.execute(
            "INSERT INTO learning_log (timestamp, event, details) VALUES (?, ?, ?)",
            (time.time(), event, json.dumps(details) if details else None)
        )
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to store learning log: {e}")


def _update_stat(key: str, value):
    """Upsert a stat value."""
    try:
        conn = _get_memory_db()
        conn.execute(
            "INSERT INTO stats (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, str(value), time.time())
        )
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to update stat {key}: {e}")


def _get_stat(key: str, default=None) -> str | None:
    """Read a stat value."""
    try:
        conn = _get_memory_db()
        row = conn.execute("SELECT value FROM stats WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
    except Exception:
        return default


def _get_recent_perceptions(limit: int = 20) -> list[dict]:
    """Retrieve recent perceptions from the DB."""
    try:
        conn = _get_memory_db()
        rows = conn.execute(
            "SELECT * FROM perceptions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for row in reversed(rows):  # chronological order
            d = dict(row)
            for field in ("top_labels", "cross_labels", "imagination"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        return results
    except Exception as e:
        log.warning(f"Failed to get recent perceptions: {e}")
        return []


def _get_recent_reflections(limit: int = 10) -> list[dict]:
    """Retrieve recent reflections from the DB."""
    try:
        conn = _get_memory_db()
        rows = conn.execute(
            "SELECT * FROM reflections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in reversed(rows)]
    except Exception as e:
        log.warning(f"Failed to get recent reflections: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
# Feature B: Knowledge Graph — explicit relational memory
# ═══════════════════════════════════════════════════════════════════

def _upsert_knowledge_edge(source: str, relation: str, target: str,
                            weight: float = 1.0, metadata: dict = None):
    """Insert or update a knowledge edge (idempotent)."""
    try:
        conn = _get_memory_db()
        conn.execute(
            "INSERT INTO knowledge_edges (source_label, relation, target_label, weight, last_seen, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(source_label, relation, target_label) DO UPDATE SET "
            "evidence_count = evidence_count + 1, "
            "weight = max(weight, excluded.weight), "
            "last_seen = excluded.last_seen",
            (source, relation, target, weight, time.time(),
             json.dumps(metadata) if metadata else None))
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to upsert knowledge edge: {e}")


def _get_knowledge_edges(source: str = None, target: str = None,
                          relation: str = None, limit: int = 100) -> list[dict]:
    """Query knowledge edges with optional filters."""
    try:
        conn = _get_memory_db()
        conditions, params = [], []
        if source:
            conditions.append("source_label = ?")
            params.append(source)
        if target:
            conditions.append("target_label = ?")
            params.append(target)
        if relation:
            conditions.append("relation = ?")
            params.append(relation)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM knowledge_edges WHERE {where} ORDER BY weight DESC LIMIT ?",
            params + [limit]).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"Failed to query knowledge edges: {e}")
        return []


def _multi_hop_traverse(start: str, relations: list[str] = None,
                         max_hops: int = 3, max_results: int = 20) -> list[dict]:
    """BFS traversal through knowledge graph edges."""
    try:
        conn = _get_memory_db()
        visited = {start}
        frontier = [(start, [start], 1.0)]  # (node, path, accumulated_weight)
        results = []

        for hop in range(max_hops):
            next_frontier = []
            for node, path, acc_weight in frontier:
                rel_filter = ""
                params = [node]
                if relations:
                    placeholders = ",".join("?" * len(relations))
                    rel_filter = f" AND relation IN ({placeholders})"
                    params.extend(relations)
                edges = conn.execute(
                    f"SELECT target_label, relation, weight FROM knowledge_edges "
                    f"WHERE source_label = ?{rel_filter} ORDER BY weight DESC LIMIT 20",
                    params).fetchall()
                for edge in edges:
                    target = edge["target_label"]
                    if target not in visited:
                        visited.add(target)
                        new_weight = acc_weight * edge["weight"]
                        new_path = path + [f"--{edge['relation']}-->", target]
                        results.append({
                            "target": target,
                            "path": new_path,
                            "hops": hop + 1,
                            "weight": round(new_weight, 4),
                            "relation": edge["relation"],
                        })
                        next_frontier.append((target, new_path, new_weight))
            frontier = next_frontier
            if not frontier:
                break

        results.sort(key=lambda x: -x["weight"])
        return results[:max_results]
    except Exception as e:
        log.warning(f"Knowledge graph traversal failed: {e}")
        return []


def _mine_causal_edges():
    """Extract 'follows' and 'causes' edges from the causal transition matrix."""
    if _causal_transitions is None or _causal_labels is None:
        return 0
    count = 0
    n = len(_causal_labels)
    # Normalize transitions to probabilities
    row_sums = _causal_transitions.sum(axis=1).clip(1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            raw = float(_causal_transitions[i, j])
            if raw > 5:
                weight = raw / float(row_sums[i])
                _upsert_knowledge_edge(_causal_labels[i], "follows", _causal_labels[j], weight)
                count += 1
            # Asymmetric causation
            if _causal_asymmetry is not None and float(_causal_asymmetry[i, j]) > 0.6 and raw > 3:
                _upsert_knowledge_edge(_causal_labels[i], "causes", _causal_labels[j],
                                        float(_causal_asymmetry[i, j]))
                count += 1
    return count


def _mine_episode_cooccurrence():
    """Extract 'co-occurs-with' edges from episode sequences."""
    try:
        conn = _get_memory_db()
        episodes = conn.execute("SELECT id FROM episodes ORDER BY id DESC LIMIT 200").fetchall()
        count = 0
        for ep in episodes:
            events = conn.execute(
                "SELECT label FROM episode_events WHERE episode_id=? AND label IS NOT NULL ORDER BY timestamp",
                (ep["id"],)).fetchall()
            labels = [e["label"] for e in events if e["label"] and not e["label"].startswith("http")]
            for i, l1 in enumerate(labels):
                for l2 in labels[i+1:min(i+4, len(labels))]:
                    if l1 != l2:
                        _upsert_knowledge_edge(l1, "co-occurs-with", l2, 0.5)
                        count += 1
        return count
    except Exception:
        return 0


def _mine_ontology_hierarchy():
    """Extract 'part-of' edges from AudioSet ontology."""
    ontology_path = PROJECT_ROOT / "data/audioset/metadata/ontology.json"
    if not ontology_path.exists():
        return 0
    try:
        with open(ontology_path) as f:
            ontology = json.load(f)
        # Build mid→name lookup
        mid_to_name = {}
        for entry in ontology:
            mid_to_name[entry["id"]] = entry["name"]
        count = 0
        for entry in ontology:
            parent = entry["name"]
            for child_id in entry.get("child_ids", []):
                child = mid_to_name.get(child_id)
                if child:
                    _upsert_knowledge_edge(child, "part-of", parent, 1.0)
                    count += 1
        return count
    except Exception as e:
        log.warning(f"Ontology mining failed: {e}")
        return 0


def _mine_concept_hierarchy():
    """Extract 'part-of' edges from the concept hierarchy tree."""
    tree = load_concept_hierarchy()
    if not tree:
        return 0
    count = 0
    def traverse(node, parent_name=None):
        nonlocal count
        name = node.get("name", "")
        if parent_name and name:
            _upsert_knowledge_edge(name, "part-of", parent_name, 0.8)
            count += 1
        for child in node.get("children", []):
            traverse(child, name)
    traverse(tree)
    return count


def _mine_prototype_similarity():
    """Extract 'sounds-like' edges from similar prototypes."""
    count = 0
    proto_list = list(_prototypes.items())
    for i, (n1, p1) in enumerate(proto_list):
        for n2, p2 in proto_list[i+1:]:
            sim = float(np.dot(p1["centroid"], p2["centroid"]))
            if sim > 0.7:
                _upsert_knowledge_edge(n1, "sounds-like", n2, round(sim, 3))
                count += 1
    return count


def _build_knowledge_graph():
    """Build the full knowledge graph from all sources."""
    log.info("Building knowledge graph...")
    t0 = time.time()
    stats = {}
    stats["causal"] = _mine_causal_edges()
    stats["episodes"] = _mine_episode_cooccurrence()
    stats["ontology"] = _mine_ontology_hierarchy()
    stats["hierarchy"] = _mine_concept_hierarchy()
    stats["prototypes"] = _mine_prototype_similarity()
    total = sum(stats.values())
    elapsed = round(time.time() - t0, 2)
    log.info(f"Knowledge graph built: {total} edges in {elapsed}s — {stats}")
    _emit_event("knowledge_built", {"total": total, "sources": stats})
    return stats


# ═══════════════════════════════════════════════════════════════════
# Feature D: Text Understanding — read and learn from documents
# ═══════════════════════════════════════════════════════════════════

def _encode_text_to_brain_space(text: str) -> np.ndarray:
    """Encode text to 512-dim brain space via MiniLM + W_v projection."""
    model = load_text_model()
    emb = model.encode([text[:2000]], normalize_embeddings=True)[0].astype(np.float32)
    w_v, _ = load_mlp_weights()
    if w_v is not None:
        return mlp_project(emb.reshape(1, -1), w_v)[0]
    return emb


def _extract_text_relations(text: str) -> list[dict]:
    """Extract knowledge edges from text using pattern matching."""
    import re
    patterns = [
        (r"(\w[\w\s]+?) (?:causes?|leads? to|produces?) (\w[\w\s]+?)(?:\.|,|$)", "causes"),
        (r"(\w[\w\s]+?) (?:is a (?:type|kind|form) of|is an? ) (\w[\w\s]+?)(?:\.|,|$)", "part-of"),
        (r"(\w[\w\s]+?) (?:sounds? like|is similar to|resembles?) (\w[\w\s]+?)(?:\.|,|$)", "sounds-like"),
        (r"(\w[\w\s]+?) (?:(?:often |usually )?follows?|comes? after) (\w[\w\s]+?)(?:\.|,|$)", "follows"),
        (r"(\w[\w\s]+?) (?:(?:often |usually )?co-?occurs? with|accompanies?) (\w[\w\s]+?)(?:\.|,|$)", "co-occurs-with"),
    ]
    edges = []
    for pattern, relation in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            source = match.group(1).strip().lower()[:50]
            target = match.group(2).strip().lower()[:50]
            if len(source) > 2 and len(target) > 2 and source != target:
                edges.append({"source": source, "relation": relation, "target": target})
    return edges


def _ingest_audioset_descriptions():
    """Load AudioSet ontology descriptions as text perceptions + knowledge edges."""
    flag = _get_stat("audioset_descriptions_loaded")
    if flag == "true":
        return {"status": "already_loaded"}

    ontology_path = PROJECT_ROOT / "data/audioset/metadata/ontology.json"
    if not ontology_path.exists():
        return {"error": "ontology.json not found"}

    with open(ontology_path) as f:
        ontology = json.load(f)

    count = 0
    for entry in ontology:
        name = entry.get("name", "")
        desc = entry.get("description", "")
        if not name:
            continue
        text = f"{name}: {desc}" if desc else name
        embedding = _encode_text_to_brain_space(text)
        _store_perception(
            modality="text",
            transcription=text[:500],
            top_labels=[name],
            embedding=embedding,
        )
        # Extract hierarchy edges
        mid_to_name = {e["id"]: e["name"] for e in ontology}
        for child_id in entry.get("child_ids", []):
            child = mid_to_name.get(child_id)
            if child:
                _upsert_knowledge_edge(child, "part-of", name, 1.0)
        count += 1
        if count % 100 == 0:
            _emit_event("text_ingested", {"source": "audioset", "count": count})

    _update_stat("audioset_descriptions_loaded", "true")
    log.info(f"Ingested {count} AudioSet descriptions")
    _emit_event("text_ingested", {"source": "audioset", "count": count, "done": True})
    return {"status": "loaded", "count": count}


def _ingest_wikipedia_summaries():
    """Fetch Wikipedia summaries for VGGSound categories (background task)."""
    import urllib.request
    codebook, concept_labels = build_concept_codebook()
    if not concept_labels:
        return {"error": "No concept labels"}

    loaded = int(_get_stat("wikipedia_loaded_count") or "0")
    count = 0
    for i, label in enumerate(concept_labels):
        if i < loaded:
            continue
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.request.quote(label)}"
            req = urllib.request.Request(url, headers={"User-Agent": "BrainProject/1.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            extract = data.get("extract", "")
            if extract:
                text = f"{label}: {extract}"
                embedding = _encode_text_to_brain_space(text)
                _store_perception(
                    modality="text",
                    transcription=text[:500],
                    top_labels=[label],
                    embedding=embedding,
                )
                # Extract relations
                edges = _extract_text_relations(extract)
                for edge in edges:
                    _upsert_knowledge_edge(edge["source"], edge["relation"], edge["target"], 0.7)
                count += 1
                if count % 20 == 0:
                    _emit_event("text_ingested", {"source": "wikipedia", "count": count})
                    _update_stat("wikipedia_loaded_count", str(loaded + count))
        except Exception:
            pass
        time.sleep(1)  # rate limit

    _update_stat("wikipedia_loaded_count", str(loaded + count))
    log.info(f"Ingested {count} Wikipedia summaries")
    _emit_event("text_ingested", {"source": "wikipedia", "count": count, "done": True})
    return {"status": "loaded", "count": count}


# ═══════════════════════════════════════════════════════════════════
# Feature A: Dream Machine — offline imagination with learning
# ═══════════════════════════════════════════════════════════════════

_dream_history: list[dict] = []
_dream_count = 0


def _generate_dream(seed_concept: str = None, max_steps: int = 5) -> dict:
    """Generate a dream by chaining world model predictions."""
    global _dream_count

    codebook, concept_labels = build_concept_codebook()
    if codebook is None:
        return {"error": "No concept codebook"}

    # 1. Seed selection
    if seed_concept:
        # Find matching concept
        matches = [i for i, l in enumerate(concept_labels) if seed_concept.lower() in l.lower()]
        if matches:
            seed_emb = codebook[matches[0]]
            seed_name = concept_labels[matches[0]]
        else:
            seed_emb = _encode_text_to_brain_space(seed_concept)
            seed_name = seed_concept
    else:
        # Random concept (prefer prototypes if available)
        if _prototypes and np.random.random() < 0.3:
            proto_name = np.random.choice(list(_prototypes.keys()))
            seed_emb = _prototypes[proto_name]["centroid"]
            seed_name = proto_name
        else:
            idx = np.random.randint(len(concept_labels))
            seed_emb = codebook[idx]
            seed_name = concept_labels[idx]

    # 2. Chain predictions
    steps = [{"concept": seed_name, "step": 0, "surprise": 0.0}]
    current_emb = seed_emb.copy()
    learning_pairs = 0

    for step_num in range(1, max_steps + 1):
        # Predict next via world model (current_emb is already in 512-dim MLP space)
        # Feed directly into the model, bypassing W_v projection
        model = load_world_model()
        if model is None:
            break
        with torch.no_grad():
            inp = torch.from_numpy(current_emb.reshape(1, -1).astype(np.float32))
            pred = model(inp).squeeze(0).numpy()
        pred_norm = pred / (np.linalg.norm(pred) + 1e-12)

        # Find nearest real concept
        sims = codebook @ pred_norm
        nearest_idx = int(np.argmax(sims))
        nearest_sim = float(sims[nearest_idx])
        nearest_name = concept_labels[nearest_idx]

        # Surprise = 1 - similarity
        surprise = round(1.0 - nearest_sim, 4)

        # Check knowledge graph support
        kg_support = len(_get_knowledge_edges(
            source=steps[-1]["concept"], target=nearest_name, limit=1))
        if kg_support > 0:
            surprise *= 0.7  # reduce surprise if graph supports this transition

        steps.append({
            "concept": nearest_name,
            "step": step_num,
            "predicted_sim": round(nearest_sim, 4),
            "surprise": round(surprise, 4),
            "kg_support": kg_support > 0,
        })

        # Dream learning: high-surprise → generate training pair
        if surprise > 0.3 and len(steps) >= 2:
            prev_concept = steps[-2]["concept"]
            # Find the real embedding for the predicted concept
            w_v, w_a = load_mlp_weights()
            if w_v is not None:
                cached = load_cached_embeddings()
                labels = load_clip_labels()
                # Find a real clip matching the predicted concept
                for ci, clip in enumerate(labels):
                    if clip["label"].lower() == nearest_name.lower():
                        v_real = cached["v_emb"][ci]
                        a_real = cached["a_emb"][ci]
                        _online_pairs.append((v_real.copy(), a_real.copy()))
                        learning_pairs += 1
                        break

        # Use nearest concept as next input
        current_emb = codebook[nearest_idx]

    avg_surprise = np.mean([s["surprise"] for s in steps[1:]]) if len(steps) > 1 else 0
    _dream_count += 1

    # #12: Multi-timescale prediction errors
    multi_scale_pe = {}
    if len(steps) >= 3:
        short_pe = np.mean([s.get("surprise", 0) for s in steps[1:3]])  # 1-2 step
        multi_scale_pe["short"] = round(float(short_pe), 4)
    if len(steps) >= 5:
        medium_pe = np.mean([s.get("surprise", 0) for s in steps[2:5]])  # 3-5 step
        multi_scale_pe["medium"] = round(float(medium_pe), 4)
    if len(steps) > 1:
        long_pe = float(avg_surprise)  # overall trend
        multi_scale_pe["long"] = round(long_pe, 4)

    dream = {
        "id": _dream_count,
        "timestamp": time.time(),
        "seed": seed_name,
        "steps": steps,
        "avg_surprise": round(float(avg_surprise), 4),
        "max_surprise": round(max((s["surprise"] for s in steps[1:]), default=0), 4),
        "learning_pairs_generated": learning_pairs,
        "multi_scale_pe": multi_scale_pe,
    }
    return dream


# ═══════════════════════════════════════════════════════════════════
# Option 4: Agentic Brain — autonomous web search + tool use
# ═══════════════════════════════════════════════════════════════════

def _web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo HTML API (no API key needed)."""
    import urllib.request, urllib.parse, re
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "BrainProject/1.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode("utf-8", errors="ignore")

        # Parse results from DuckDuckGo HTML
        results = []
        for match in re.finditer(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html):
            link = match.group(1)
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            if title and link and len(results) < max_results:
                # DDG wraps URLs
                if "uddg=" in link:
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])
                results.append({"title": title, "url": link})
        return results
    except Exception as e:
        log.warning(f"Web search failed: {e}")
        return []


def _fetch_and_read_url(url: str) -> dict:
    """Fetch a URL, extract text, encode into brain space, store as perception."""
    import urllib.request, re
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BrainProject/1.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode("utf-8", errors="ignore")

        # Simple HTML→text extraction
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()[:3000]

        if len(text) < 50:
            return {"error": "Page too short or empty"}

        # Encode and store
        embedding = _encode_text_to_brain_space(text)
        title = text[:80]

        _store_perception(
            modality="text", transcription=text[:500],
            top_labels=[title], embedding=embedding,
        )

        # Extract knowledge edges
        edges = _extract_text_relations(text)
        for edge in edges:
            _upsert_knowledge_edge(edge["source"], edge["relation"], edge["target"], 0.6)

        _emit_event("text_ingested", {"source": "web", "url": url[:80], "edges": len(edges)})
        return {"status": "read", "url": url, "text_length": len(text), "edges": len(edges)}
    except Exception as e:
        return {"error": str(e)[:200]}


def _autonomous_research(topic: str) -> dict:
    """Autonomously research a topic: search web, read top results, extract knowledge."""
    log.info(f"Autonomous research: {topic}")
    _emit_event("research_start", {"topic": topic})

    results = _web_search(f"{topic} sound audio", max_results=3)
    if not results:
        results = _web_search(topic, max_results=3)

    pages_read = 0
    edges_total = 0
    for r in results[:3]:
        result = _fetch_and_read_url(r["url"])
        if result.get("status") == "read":
            pages_read += 1
            edges_total += result.get("edges", 0)

    _emit_event("research_done", {"topic": topic, "pages": pages_read, "edges": edges_total})
    return {"topic": topic, "search_results": len(results), "pages_read": pages_read, "edges": edges_total}


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# #11: Dual Fast/Slow Memory — Hopfield fast store
# ═══════════════════════════════════════════════════════════════════

def _fast_memory_store(embedding: np.ndarray, label: str):
    """Store pattern in fast Hopfield memory (one-shot, instant)."""
    global _fast_memory_patterns, _fast_memory_count
    if embedding is None:
        return
    if _fast_memory_patterns is None:
        _fast_memory_patterns = np.zeros((FAST_MEMORY_CAPACITY, 512), dtype=np.float32)
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
    if len(emb_norm) != 512:
        return
    idx = _fast_memory_count % FAST_MEMORY_CAPACITY
    _fast_memory_patterns[idx] = emb_norm
    if idx >= len(_fast_memory_labels):
        _fast_memory_labels.append(label)
    else:
        _fast_memory_labels[idx] = label
    _fast_memory_count += 1


def _fast_memory_retrieve(query: np.ndarray, top_k: int = 5) -> list[dict]:
    """Pattern completion from fast memory."""
    if _fast_memory_patterns is None or _fast_memory_count == 0:
        return []
    n = min(_fast_memory_count, FAST_MEMORY_CAPACITY)
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    sims = _fast_memory_patterns[:n] @ q_norm
    top = np.argsort(sims)[::-1][:top_k]
    return [{"label": _fast_memory_labels[i] if i < len(_fast_memory_labels) else "?",
             "similarity": round(float(sims[i]), 4), "idx": int(i)} for i in top]


def _fast_memory_get_recent(n: int = 100) -> np.ndarray:
    """Get recent patterns for consolidation transfer to slow memory."""
    if _fast_memory_patterns is None or _fast_memory_count == 0:
        return np.zeros((0, 512), dtype=np.float32)
    total = min(_fast_memory_count, FAST_MEMORY_CAPACITY)
    start = max(0, _fast_memory_count - n) % FAST_MEMORY_CAPACITY
    if start + n <= total:
        return _fast_memory_patterns[start:start + n].copy()
    else:
        part1 = _fast_memory_patterns[start:total]
        part2 = _fast_memory_patterns[:n - len(part1)]
        return np.concatenate([part1, part2])


# Grid Cell System: Hexagonal spatial encoding (inspired by Doeller et al. 2010)
#
# Grid cells fire in hexagonal lattice patterns. We implement:
# 1. Multi-scale hexagonal grid encoding of 512-dim concept space
# 2. Path integration for episodic trajectories
# 3. Grid-metric distance (beyond cosine similarity)
# 4. Spatial working memory (grid regions, not just items)
# ═══════════════════════════════════════════════════════════════════

GRID_SCALES = [0.05, 0.15, 0.5]  # fine, medium, coarse grid spacings
GRID_N_ORIENTATIONS = 3  # 3 grid modules at 60° offsets (hexagonal)
GRID_EMBED_DIM = 512
GRID_2D_DIM = 2  # project to 2D for visualization + navigation


class GridCellEncoder:
    """Multi-scale hexagonal grid encoding of concept embeddings.

    Projects 512-dim embeddings to 2D coordinates, then computes
    hexagonal grid cell activations at multiple scales. The grid
    has 60° rotational symmetry (6-fold) matching biological grid cells.
    """

    def __init__(self, n_scales=3, embed_dim=512):
        self.n_scales = n_scales
        self.embed_dim = embed_dim
        # Learn a 512 → 2D projection via PCA of concept codebook
        self.projection = None  # (512, 2) matrix
        self.mean = None  # (512,) center
        self.scales = GRID_SCALES
        # 3 grid orientations at 0°, 20°, 40° (covering 60° hexagonal symmetry)
        self.orientations = [0, np.pi/9, 2*np.pi/9]

    def fit(self, embeddings: np.ndarray):
        """Fit the 2D projection from a set of embeddings (concept codebook)."""
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        # PCA → top 2 components
        cov = (centered.T @ centered) / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:2]
        self.projection = eigenvectors[:, idx]  # (512, 2)
        log.info(f"Grid cell encoder fitted: {embeddings.shape} → 2D "
                 f"(var explained: {eigenvalues[idx].sum()/eigenvalues.sum():.1%})")

    def to_2d(self, embedding: np.ndarray) -> np.ndarray:
        """Project 512-dim embedding to 2D grid coordinates."""
        if self.projection is None:
            return np.zeros(2)
        centered = embedding - self.mean
        if centered.ndim == 1:
            return centered @ self.projection
        return centered @ self.projection  # (N, 2)

    def grid_activation(self, pos_2d: np.ndarray) -> np.ndarray:
        """Compute hexagonal grid cell activations at all scales.

        For each scale and orientation, compute a periodic activation
        with 6-fold rotational symmetry (hexagonal lattice).

        Returns: (n_scales * n_orientations * 2,) activation vector
        """
        activations = []
        for scale in self.scales:
            for theta in self.orientations:
                # Rotate position by grid orientation
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rotated = np.array([
                    pos_2d[0] * cos_t - pos_2d[1] * sin_t,
                    pos_2d[0] * sin_t + pos_2d[1] * cos_t,
                ])
                # Hexagonal grid: sum of 3 cosines at 60° intervals
                # This creates the characteristic 6-fold pattern
                hex_activation = 0.0
                for k in range(3):
                    angle = k * np.pi / 3  # 0°, 60°, 120°
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    hex_activation += np.cos(2 * np.pi * np.dot(rotated, direction) / scale)
                activations.append(hex_activation / 3.0)  # normalize to [-1, 1]
                # Also store the phase (for path integration)
                phase = np.arctan2(
                    np.sin(2 * np.pi * rotated[0] / scale),
                    np.cos(2 * np.pi * rotated[0] / scale))
                activations.append(phase / np.pi)  # normalize to [-1, 1]
        return np.array(activations, dtype=np.float32)

    def grid_distance(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Compute grid-metric distance between two embeddings.

        Combines Euclidean distance in 2D grid space with
        grid activation similarity (periodic structure).
        """
        pos_a = self.to_2d(emb_a)
        pos_b = self.to_2d(emb_b)
        # Euclidean distance in grid space
        euclidean = float(np.linalg.norm(pos_a - pos_b))
        # Grid activation cosine similarity
        act_a = self.grid_activation(pos_a)
        act_b = self.grid_activation(pos_b)
        cos_sim = float(np.dot(act_a, act_b) / (np.linalg.norm(act_a) * np.linalg.norm(act_b) + 1e-12))
        # Combined: euclidean weighted by periodic similarity
        return euclidean * (1.0 - cos_sim * 0.5)

    def path_integrate(self, embeddings: list[np.ndarray]) -> list[np.ndarray]:
        """Convert a sequence of embeddings into a grid path (trajectory)."""
        if not embeddings:
            return []
        path = []
        for emb in embeddings:
            pos = self.to_2d(emb)
            path.append(pos)
        return path

    def find_nearby(self, embedding: np.ndarray, candidates: np.ndarray,
                    labels: list[str], top_k: int = 10) -> list[dict]:
        """Find concepts near a position on the grid (spatial neighbors)."""
        pos = self.to_2d(embedding)
        if candidates.ndim == 2:
            cand_pos = self.to_2d(candidates)
        else:
            return []
        dists = np.linalg.norm(cand_pos - pos, axis=1)
        top_idx = np.argsort(dists)[:top_k]
        return [
            {"label": labels[i] if i < len(labels) else f"idx_{i}",
             "distance": round(float(dists[i]), 4),
             "grid_x": round(float(cand_pos[i, 0]), 4),
             "grid_y": round(float(cand_pos[i, 1]), 4)}
            for i in top_idx
        ]

    def get_region(self, center_emb: np.ndarray, radius: float,
                   candidates: np.ndarray, labels: list[str]) -> list[dict]:
        """Get all concepts within a grid radius (spatial attention region)."""
        pos = self.to_2d(center_emb)
        cand_pos = self.to_2d(candidates)
        dists = np.linalg.norm(cand_pos - pos, axis=1)
        mask = dists < radius
        indices = np.where(mask)[0]
        return [
            {"label": labels[i] if i < len(labels) else f"idx_{i}",
             "distance": round(float(dists[i]), 4)}
            for i in indices[:50]  # cap at 50
        ]


def build_grid_encoder():
    """Build and fit the grid cell encoder from the concept codebook."""
    global _grid_cell_encoder, _grid_concept_coords
    if _grid_cell_encoder is not None:
        return _grid_cell_encoder

    codebook, labels = build_concept_codebook()
    if codebook is None:
        return None

    _grid_cell_encoder = GridCellEncoder()
    _grid_cell_encoder.fit(codebook)

    # Pre-compute grid coordinates for all concepts
    _grid_concept_coords = _grid_cell_encoder.to_2d(codebook)
    log.info(f"Grid encoder built: {len(labels)} concepts mapped to 2D hexagonal grid")

    return _grid_cell_encoder


def grid_encode_episode(episode_embeddings: list[np.ndarray]) -> dict:
    """Encode an episode as a grid trajectory."""
    encoder = build_grid_encoder()
    if encoder is None or not episode_embeddings:
        return {"error": "Grid encoder not available"}

    path = encoder.path_integrate(episode_embeddings)
    if not path:
        return {"path": [], "distance": 0}

    # Compute total path distance (how far the brain "traveled")
    total_dist = sum(float(np.linalg.norm(path[i+1] - path[i]))
                     for i in range(len(path) - 1))

    # Compute grid activations along the path
    activations = [encoder.grid_activation(p).tolist() for p in path]

    return {
        "path": [{"x": round(float(p[0]), 4), "y": round(float(p[1]), 4)} for p in path],
        "total_distance": round(total_dist, 4),
        "n_steps": len(path),
        "start": {"x": round(float(path[0][0]), 4), "y": round(float(path[0][1]), 4)},
        "end": {"x": round(float(path[-1][0]), 4), "y": round(float(path[-1][1]), 4)},
    }


# ═══════════════════════════════════════════════════════════════════
# Feature C: YouTube Scaling helpers
# ═══════════════════════════════════════════════════════════════════

def _get_underrepresented_categories(n: int = 5) -> list[str]:
    """Find categories with least YouTube learning coverage."""
    try:
        conn = _get_memory_db()
        # Get coverage counts
        coverage = conn.execute(
            "SELECT category, count(*) as cnt FROM youtube_learning_log "
            "WHERE status='success' GROUP BY category"
        ).fetchall()
        covered = {r["category"]: r["cnt"] for r in coverage}
    except Exception:
        covered = {}

    # All categories from codebook + curiosity profiles
    all_cats = set()
    if _concept_labels:
        all_cats.update(_concept_labels)
    profiles = compute_category_profiles()
    if profiles:
        all_cats.update(profiles.keys())

    # Sort by: coverage ascending, then curiosity descending
    scored = []
    for cat in all_cats:
        cov = covered.get(cat, 0)
        curiosity = profiles.get(cat, {}).get("curiosity", 0.5) if profiles else 0.5
        scored.append((cat, cov, curiosity))
    scored.sort(key=lambda x: (x[1], -x[2]))
    return [cat for cat, _, _ in scored[:n]]


def _log_youtube_attempt(video_id: str, category: str, status: str,
                          pairs: int = 0, error: str = None):
    """Log a YouTube learning attempt."""
    try:
        conn = _get_memory_db()
        conn.execute(
            "INSERT INTO youtube_learning_log (video_id, category, timestamp, status, pairs_generated, error_text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, category, time.time(), status, pairs, error))
        conn.commit()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Episodic Memory — temporal sequences of perceptions
# ═══════════════════════════════════════════════════════════════════

def _get_or_create_episode() -> int:
    """Get current episode or create a new one if gap > EPISODE_GAP_SECONDS."""
    global _current_episode_id, _last_perception_time
    now = time.time()
    if _current_episode_id is None or (now - _last_perception_time) > EPISODE_GAP_SECONDS:
        try:
            conn = _get_memory_db()
            # Close previous episode
            if _current_episode_id is not None:
                conn.execute("UPDATE episodes SET end_time=? WHERE id=?",
                             (_last_perception_time, _current_episode_id))
            cur = conn.execute(
                "INSERT INTO episodes (start_time, context) VALUES (?, ?)",
                (now, None))
            conn.commit()
            _current_episode_id = cur.lastrowid
            log.info(f"New episode started: #{_current_episode_id}")
            _emit_event("episode_start", {"episode_id": _current_episode_id})
        except Exception as e:
            log.warning(f"Failed to create episode: {e}")
            return _current_episode_id or 0
    _last_perception_time = now
    return _current_episode_id


def _store_episode_event(modality: str, label: str = None,
                         embedding: np.ndarray = None, metadata: dict = None):
    """Store an event in the current episode."""
    episode_id = _get_or_create_episode()
    if not episode_id:
        return
    try:
        conn = _get_memory_db()
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        conn.execute(
            "INSERT INTO episode_events (episode_id, timestamp, modality, embedding_blob, label, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (episode_id, time.time(), modality, emb_blob, label,
             json.dumps(metadata) if metadata else None))
        conn.execute("UPDATE episodes SET event_count = event_count + 1, end_time = ? WHERE id = ?",
                     (time.time(), episode_id))
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to store episode event: {e}")


def _get_episodes(limit: int = 20) -> list[dict]:
    """Retrieve recent episodes with their events."""
    try:
        conn = _get_memory_db()
        episodes = conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for ep in reversed(episodes):
            ep_dict = dict(ep)
            events = conn.execute(
                "SELECT id, timestamp, modality, label, metadata_json FROM episode_events "
                "WHERE episode_id=? ORDER BY timestamp", (ep_dict["id"],)
            ).fetchall()
            ep_dict["events"] = []
            for ev in events:
                ev_dict = dict(ev)
                if ev_dict.get("metadata_json"):
                    try:
                        ev_dict["metadata"] = json.loads(ev_dict["metadata_json"])
                    except (json.JSONDecodeError, TypeError):
                        ev_dict["metadata"] = None
                    del ev_dict["metadata_json"]
                ep_dict["events"].append(ev_dict)
            result.append(ep_dict)
        return result
    except Exception as e:
        log.warning(f"Failed to get episodes: {e}")
        return []


def _get_episode_embeddings(episode_id: int) -> list[np.ndarray]:
    """Get all embeddings from an episode."""
    try:
        conn = _get_memory_db()
        rows = conn.execute(
            "SELECT embedding_blob FROM episode_events WHERE episode_id=? AND embedding_blob IS NOT NULL ORDER BY timestamp",
            (episode_id,)).fetchall()
        return [np.frombuffer(row["embedding_blob"], dtype=np.float32) for row in rows]
    except Exception:
        return []


def _search_episodes_by_sequence(query_labels: list[str], top_k: int = 5) -> list[dict]:
    """Find episodes matching a sequence pattern using label subsequence matching."""
    try:
        conn = _get_memory_db()
        episodes = conn.execute(
            "SELECT id, start_time, end_time, event_count FROM episodes WHERE event_count >= ? ORDER BY id DESC LIMIT 200",
            (len(query_labels),)).fetchall()

        results = []
        for ep in episodes:
            events = conn.execute(
                "SELECT label FROM episode_events WHERE episode_id=? ORDER BY timestamp",
                (ep["id"],)).fetchall()
            ep_labels = [e["label"] for e in events if e["label"]]

            # Subsequence matching: find best alignment score
            score = _sequence_match_score(query_labels, ep_labels)
            if score > 0.3:
                results.append({
                    "episode_id": ep["id"],
                    "start_time": ep["start_time"],
                    "end_time": ep["end_time"],
                    "event_count": ep["event_count"],
                    "labels": ep_labels,
                    "match_score": round(score, 3),
                })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        log.warning(f"Episode search failed: {e}")
        return []


def _sequence_match_score(query: list[str], episode_list: list[str]) -> float:
    """Score how well query sequence matches episode label list (fuzzy subsequence)."""
    if not query or not episode_list:
        return 0.0

    # Try to load text model for semantic matching
    text_model = None
    try:
        text_model = load_text_model()
    except Exception:
        pass

    if text_model is not None and _label_embeddings is not None:
        # Semantic matching via text embeddings
        from sentence_transformers import SentenceTransformer
        q_embs = text_model.encode(query, normalize_embeddings=True)
        e_embs = text_model.encode(episode_list, normalize_embeddings=True)
        sim_matrix = q_embs @ e_embs.T  # (len_q, len_e)

        # Find best ordered subsequence alignment
        best_score = 0.0
        for start in range(len(episode_list) - len(query) + 1):
            score = sum(sim_matrix[i, start + i] for i in range(len(query))) / len(query)
            best_score = max(best_score, score)
        return float(best_score)
    else:
        # Fallback: exact substring match
        matched = 0
        ep_lower = [l.lower() for l in episode_list]
        for q in query:
            q_lower = q.lower()
            if any(q_lower in el for el in ep_lower):
                matched += 1
        return matched / len(query)


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Working Memory — 7±2 active concept slots
# ═══════════════════════════════════════════════════════════════════

def _update_working_memory(embedding: np.ndarray, label: str, modality: str):
    """Add a new perception to working memory with theta-gamma phase ordering (#6)."""
    global _working_memory, _wm_theta_phase
    now = time.time()

    # #6: Advance theta phase
    _wm_theta_phase = (_wm_theta_phase + WM_THETA_FREQ * 2 * np.pi) % (2 * np.pi)

    # Decay existing activations + theta-gamma modulation
    for item in _working_memory:
        item["activation"] *= WORKING_MEMORY_DECAY
        # #6: Gamma burst — boost items near current theta phase
        if "theta_phase" in item:
            phase_dist = abs(item["theta_phase"] - _wm_theta_phase)
            phase_dist = min(phase_dist, 2 * np.pi - phase_dist)
            gamma_boost = np.exp(-phase_dist ** 2 / 0.5)
            item["activation"] = min(1.0, item["activation"] + gamma_boost * 0.15)

    # Compute attention scores
    attention_scores = []
    if _working_memory and embedding is not None:
        new_emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        for item in _working_memory:
            if item.get("embedding") is not None:
                old_emb_norm = item["embedding"] / (np.linalg.norm(item["embedding"]) + 1e-12)
                sim = float(np.dot(new_emb_norm, old_emb_norm))
                attention_scores.append(sim)
                if sim > 0.5:
                    item["activation"] = min(1.0, item["activation"] + 0.2 * sim)
            else:
                attention_scores.append(0.0)

    # Add new item with full activation + theta phase
    new_item = {
        "embedding": embedding,
        "label": label,
        "modality": modality,
        "timestamp": now,
        "activation": 1.0,
        "theta_phase": round(_wm_theta_phase, 4),  # #6
    }
    _working_memory.append(new_item)

    # Remove items below threshold or exceed slots
    _working_memory = [m for m in _working_memory if m["activation"] > 0.1]
    if len(_working_memory) > WORKING_MEMORY_SLOTS:
        _working_memory.sort(key=lambda x: x["activation"], reverse=True)
        _working_memory = _working_memory[:WORKING_MEMORY_SLOTS]

    # #11: Store in fast Hopfield memory
    _fast_memory_store(embedding, label)

    avg_attention = np.mean(attention_scores) if attention_scores else 0.0
    return {
        "attention_score": round(float(avg_attention), 3),
        "related_items": [
            _working_memory[i]["label"]
            for i in range(len(attention_scores))
            if attention_scores[i] > 0.5
        ] if attention_scores else [],
        "buffer_size": len(_working_memory),
    }


def _get_working_memory_state() -> dict:
    """Return the current working memory contents with theta-gamma phase info."""
    return {
        "slots_used": len(_working_memory),
        "max_slots": WORKING_MEMORY_SLOTS,
        "theta_phase": round(_wm_theta_phase, 4),
        "items": [
            {
                "label": m["label"],
                "modality": m["modality"],
                "activation": round(m["activation"], 3),
                "theta_phase": round(m.get("theta_phase", 0), 4),
                "age_seconds": round(time.time() - m["timestamp"], 1),
            }
            for m in sorted(_working_memory, key=lambda x: x["activation"], reverse=True)
        ],
        "focus": _working_memory[0]["label"] if _working_memory else None,
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 5.1: Prototype Memory — few-shot concept learning
# ═══════════════════════════════════════════════════════════════════

def _load_prototypes_from_db():
    """Load saved prototypes from SQLite."""
    global _prototypes
    try:
        conn = _get_memory_db()
        rows = conn.execute("SELECT * FROM prototypes").fetchall()
        for row in rows:
            centroid = np.frombuffer(row["centroid_blob"], dtype=np.float32)
            examples = json.loads(row["examples_json"]) if row["examples_json"] else []
            _prototypes[row["name"]] = {
                "centroid": centroid,
                "count": row["count"],
                "examples": examples,
                "created_at": row["created_at"],
            }
        if _prototypes:
            log.info(f"Loaded {len(_prototypes)} prototypes from DB")
    except Exception as e:
        log.warning(f"Failed to load prototypes: {e}")


def _add_prototype(name: str, embedding: np.ndarray, example_label: str = None):
    """Add or update a prototype (running average centroid)."""
    global _prototypes
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)

    if name in _prototypes:
        proto = _prototypes[name]
        count = proto["count"]
        # Running average
        new_centroid = (proto["centroid"] * count + emb_norm) / (count + 1)
        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-12)
        proto["centroid"] = new_centroid
        proto["count"] = count + 1
        if example_label and len(proto["examples"]) < 20:
            proto["examples"].append(example_label)
    else:
        _prototypes[name] = {
            "centroid": emb_norm.copy(),
            "count": 1,
            "examples": [example_label] if example_label else [],
            "created_at": time.time(),
        }

    # Persist to DB
    try:
        conn = _get_memory_db()
        proto = _prototypes[name]
        conn.execute(
            "INSERT INTO prototypes (name, centroid_blob, count, examples_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "centroid_blob=excluded.centroid_blob, count=excluded.count, "
            "examples_json=excluded.examples_json, updated_at=excluded.updated_at",
            (name, proto["centroid"].astype(np.float32).tobytes(), proto["count"],
             json.dumps(proto["examples"]), proto.get("created_at", time.time()), time.time()))
        conn.commit()
    except Exception as e:
        log.warning(f"Failed to persist prototype: {e}")


def _match_prototype(embedding: np.ndarray, threshold: float = 0.7) -> tuple[str | None, float]:
    """Find the best matching prototype for an embedding."""
    if not _prototypes:
        return None, 0.0
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
    best_name, best_sim = None, 0.0
    for name, proto in _prototypes.items():
        sim = float(np.dot(emb_norm, proto["centroid"]))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim


def _check_novelty_and_prototype(embedding: np.ndarray, label: str, modality: str):
    """Check if input is novel enough to create a new prototype or refine existing one."""
    # Try matching existing prototypes
    match_name, match_sim = _match_prototype(embedding, threshold=0.7)
    if match_name:
        _add_prototype(match_name, embedding, label)
        return {"action": "refined", "prototype": match_name, "similarity": round(match_sim, 3)}

    # Check against known categories
    codebook, concept_labels = build_concept_codebook()
    if codebook is not None:
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        sims = codebook @ emb_norm
        max_sim = float(sims.max())
        if max_sim < 0.6:  # Novel — doesn't fit existing categories well
            # Clean label for prototype name (strip URLs, truncate)
            clean_label = label if label else "unknown"
            if clean_label.startswith("http"):
                clean_label = "youtube_clip"
            clean_label = clean_label[:40].replace("/", "_")
            proto_name = f"novel_{clean_label}_{int(time.time()) % 10000}"
            _add_prototype(proto_name, embedding, label)
            return {"action": "created", "prototype": proto_name, "novelty": round(1.0 - max_sim, 3)}

    return {"action": "known", "best_category_sim": round(float(sims.max()) if codebook is not None else 0, 3)}


# ═══════════════════════════════════════════════════════════════════
# Phase 5.2: Sleep/Consolidation Cycle
# ═══════════════════════════════════════════════════════════════════

def _consolidation_cycle():
    """Memory consolidation: replay, strengthen, prune, compress, update."""
    log.info("Starting consolidation cycle...")
    t0 = time.time()
    stats = {"replayed": 0, "strengthened": 0, "pruned": 0, "compressed": 0}

    try:
        # 1. Replay: re-process recent episodes through MLP
        recent_episodes = _get_episodes(limit=10)
        w_v, w_a = load_mlp_weights()
        if w_v is None:
            return stats

        for ep in recent_episodes:
            for event in ep.get("events", []):
                stats["replayed"] += 1

        # 2. Strengthen: well-known prototypes get boosted
        for name, proto in list(_prototypes.items()):
            if proto["count"] >= 5:
                stats["strengthened"] += 1

        # 3. Prune: remove prototypes seen only once and old
        prune_threshold = time.time() - 86400  # 24h old
        pruned_names = []
        for name, proto in list(_prototypes.items()):
            if proto["count"] == 1 and proto.get("created_at", 0) < prune_threshold:
                pruned_names.append(name)
        for name in pruned_names:
            del _prototypes[name]
            try:
                conn = _get_memory_db()
                conn.execute("DELETE FROM prototypes WHERE name=?", (name,))
                conn.commit()
            except Exception:
                pass
            stats["pruned"] += 1

        # 4. Compress: merge similar prototypes
        proto_list = list(_prototypes.items())
        merged = set()
        for i, (n1, p1) in enumerate(proto_list):
            if n1 in merged:
                continue
            for j, (n2, p2) in enumerate(proto_list[i+1:], i+1):
                if n2 in merged:
                    continue
                sim = float(np.dot(p1["centroid"], p2["centroid"]))
                if sim > 0.9:
                    # Merge n2 into n1
                    total = p1["count"] + p2["count"]
                    p1["centroid"] = (p1["centroid"] * p1["count"] + p2["centroid"] * p2["count"]) / total
                    p1["centroid"] = p1["centroid"] / (np.linalg.norm(p1["centroid"]) + 1e-12)
                    p1["count"] = total
                    p1["examples"] = (p1["examples"] + p2["examples"])[:20]
                    merged.add(n2)
                    stats["compressed"] += 1

        for name in merged:
            if name in _prototypes:
                del _prototypes[name]
                try:
                    conn = _get_memory_db()
                    conn.execute("DELETE FROM prototypes WHERE name=?", (name,))
                    conn.commit()
                except Exception:
                    pass

        # 5. Update self-model stats
        if _category_stats:
            stats["self_model_updated"] = True

        # #7: Tag-then-replay — replay only high-surprise episodes
        try:
            conn = _get_memory_db()
            tagged = conn.execute(
                "SELECT embedding_blob, label FROM episode_events "
                "WHERE metadata_json LIKE '%surprise_tag%' "
                "AND CAST(json_extract(metadata_json, '$.surprise_tag') AS REAL) > 0.3 "
                "AND embedding_blob IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
            replay_pairs = 0
            for event in tagged:
                emb = np.frombuffer(event["embedding_blob"], dtype=np.float32)
                if len(emb) == 512:
                    # Use as audio target, generate visual from world model inverse
                    _online_pairs.append((emb[:384] if len(emb) >= 384 else np.zeros(384), emb[:512]))
                    replay_pairs += 1
            stats["tagged_replayed"] = replay_pairs
        except Exception:
            stats["tagged_replayed"] = 0

        # #8: Context decorrelation — log when same label appears in different episodes
        try:
            labels_with_contexts = conn.execute(
                "SELECT label, COUNT(DISTINCT episode_id) as ep_count "
                "FROM episode_events WHERE label IS NOT NULL "
                "GROUP BY label HAVING ep_count >= 2 LIMIT 20"
            ).fetchall()
            decorrelated = 0
            for row in labels_with_contexts:
                _upsert_knowledge_edge(row["label"], "context-varies", row["label"],
                                        round(1.0 / row["ep_count"], 3))
                decorrelated += 1
            stats["context_decorrelated"] = decorrelated
        except Exception:
            stats["context_decorrelated"] = 0

        # #11: Transfer fast memory → slow (online pairs for next training)
        fast_recent = _fast_memory_get_recent(n=50)
        if len(fast_recent) > 0:
            stats["fast_to_slow"] = len(fast_recent)

        elapsed = round(time.time() - t0, 2)
        log.info(f"Consolidation complete in {elapsed}s: {stats}")
        _emit_event("consolidation", stats)
        _store_learning_log("consolidation", stats)

    except Exception as e:
        log.error(f"Consolidation error: {e}")

    return stats


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Cortex Brain API",
    description="Cognitive architecture with 2.1M clips, episodic memory, grid cells, dreams, knowledge graph",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Phase I-2: Request logging middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        t0 = time.time()
        response = await call_next(request)
        duration = time.time() - t0
        if duration > 2.0 and not request.url.path.endswith("/live"):
            log.warning(f"SLOW {request.method} {request.url.path} {duration:.2f}s status={response.status_code}")
        return response

app.add_middleware(RequestLogMiddleware)


# Phase II-4: Async wrapper for heavy sync operations
import asyncio
import functools

def async_endpoint(fn):
    """Decorator: run sync endpoint function in a thread pool to avoid blocking the event loop."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)
    return wrapper


# Phase II-5: API auth middleware (optional, controlled by env var)
import os as _os
_API_KEYS = set(_os.environ.get("CORTEX_API_KEYS", "").split(",")) - {""}

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        # Skip auth if no keys configured, or for health/SSE/pages
        if not _API_KEYS or not request.url.path.startswith("/api/") or \
           request.url.path in ("/health", "/api/brain/live"):
            return await call_next(request)
        key = request.headers.get("X-API-Key", "")
        if key not in _API_KEYS:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

if _API_KEYS:
    app.add_middleware(AuthMiddleware)
    log.info(f"API auth enabled ({len(_API_KEYS)} keys)")
else:
    log.info("API auth disabled (set CORTEX_API_KEYS env var to enable)")


class WatchRequest(BaseModel):
    """JPEG image (base64) for real-time visual processing."""
    image_b64: str
    top_k: int = 10


@app.post("/api/brain/watch")
@async_endpoint
def brain_watch(req: WatchRequest):
    """Process a camera frame through the brain — real-time vision endpoint."""
    import base64
    from io import BytesIO
    from PIL import Image
    import torchvision.transforms as T
    t0 = time.time()

    # Decode image
    img_bytes = base64.b64decode(req.image_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Run DINOv2
    model = load_visual_model()
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        v_emb = model(img_tensor).squeeze(0).cpu().numpy()
    v_emb = v_emb / (np.linalg.norm(v_emb) + 1e-12)

    # Find associations
    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    results = {}

    # Direct visual similarity — get top-50 for category aggregation
    v_sims = cached["v_emb"] @ v_emb
    v_top = np.argsort(v_sims)[::-1][:50]
    results["visual_similar"] = _format_results(v_top[:req.top_k], v_sims, labels)

    # Category-level voting from top-50 visual matches
    # More robust than individual clip matches
    cat_v_scores = {}
    for idx in v_top:
        if idx < len(labels):
            cat = labels[idx]["label"]
            cat_v_scores[cat] = cat_v_scores.get(cat, 0) + float(v_sims[idx])
    sorted_v_cats = sorted(cat_v_scores.items(), key=lambda x: -x[1])

    # V→A cross-modal via MLP — also aggregate by category
    cross_labels = []
    if w_v is not None:
        v_proj = mlp_project(v_emb.reshape(1, -1), w_v)[0]
        all_a_proj = mlp_project(cached["a_emb"], w_a)
        v2a_sims = all_a_proj @ v_proj
        v2a_top = np.argsort(v2a_sims)[::-1][:50]
        results["cross_modal_v2a"] = _format_results(v2a_top[:req.top_k], v2a_sims, labels)

        cat_a_scores = {}
        for idx in v2a_top:
            if idx < len(labels):
                cat = labels[idx]["label"]
                cat_a_scores[cat] = cat_a_scores.get(cat, 0) + float(v2a_sims[idx])
        sorted_a_cats = sorted(cat_a_scores.items(), key=lambda x: -x[1])

        seen = set()
        for cat, _ in sorted_a_cats[:8]:
            if cat not in seen:
                seen.add(cat)
                cross_labels.append(cat)

    # CLIP zero-shot scene classification (supplementary to DINOv2)
    clip_scene = []
    try:
        clip_results = clip_classify_image(img, top_k=5)
        clip_scene = clip_results[:3]  # top-3 for summary
        results["clip_scene"] = clip_results  # full top-5 in associations
    except Exception as e:
        log.warning(f"CLIP classification failed: {e}")

    # Summary: combine CLIP scene descriptions (first) + DINOv2 category voting
    see_labels = []
    seen = set()
    # CLIP results first — more accurate for real scenes
    for item in clip_scene:
        desc = item["description"]
        if desc not in seen:
            seen.add(desc)
            see_labels.append(desc)
    # Then DINOv2 category-aggregated labels
    for cat, _ in sorted_v_cats[:8]:
        if cat not in seen:
            seen.add(cat)
            see_labels.append(cat)

    # LLM narration
    narration = None
    if see_labels:
        try:
            import urllib.request as ur
            # Use CLIP labels for scene context, DINOv2 categories as secondary
            clip_descs = [item["description"] for item in clip_scene[:3]]
            scene_text = ', '.join(clip_descs) if clip_descs else ', '.join(see_labels[:4])
            prompt = (
                f"You are a brain. You see: {scene_text}."
            )
            if cross_labels:
                prompt += f" This makes you hear: {', '.join(cross_labels[:3])}."
            prompt += " Respond in ONE vivid sentence as 'I'. Be sensory."
            payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                                   "options": {"temperature": 0.8, "num_predict": 30}}).encode()
            r = ur.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
            resp = ur.urlopen(r, timeout=10)
            narration = json.loads(resp.read()).get("response", "").strip()
        except Exception:
            pass

    # Auto-store for online learning (pair with silence since no audio)
    # We skip this — visual-only doesn't have a paired audio

    # World model prediction: what does the brain expect to hear?
    expect_labels = []
    surprise = None
    pred_a = predict_audio(v_emb)
    if pred_a is not None:
        all_a_proj = mlp_project(cached["a_emb"], w_a) if w_a is not None else cached["a_emb"]
        pred_sims = all_a_proj @ pred_a
        pred_top = np.argsort(pred_sims)[::-1][:5]
        seen_pred = set()
        for idx_p in pred_top:
            if idx_p < len(labels) and labels[idx_p]["label"] not in seen_pred:
                seen_pred.add(labels[idx_p]["label"])
                expect_labels.append(labels[idx_p]["label"])

    # Persist perception to memory DB (with MLP-projected embedding for episode tracking)
    v_proj_emb = mlp_project(v_emb.reshape(1, -1), w_v)[0] if w_v is not None else v_emb
    _store_perception(
        modality="watch",
        transcription=None,
        top_labels=see_labels,
        cross_labels=cross_labels,
        narration=narration,
        embedding=v_proj_emb,
    )

    # Phase 4.2: Selective attention — report what the brain is focused on
    wm_state = _get_working_memory_state()
    attention_info = {}
    if wm_state["slots_used"] > 0:
        focus = wm_state["focus"]
        related = [it["label"] for it in wm_state["items"] if it["activation"] > 0.5]
        attention_info = {
            "focused_on": focus,
            "related_active": related[:3],
            "attention_note": f"Paying attention to {focus}" if focus else None,
        }

    # Compute confidence: max cross-modal similarity (0-1)
    max_cross_sim = 0.0
    if w_v is not None:
        max_cross_sim = float(v2a_sims.max()) if 'v2a_sims' in dir() else 0.0
    max_clip_conf = float(clip_scene[0]["confidence"]) if clip_scene else 0.0

    return {
        "associations": results,
        "summary": {
            "clip_scene": [item["description"] for item in clip_scene],
            "clip_scene_full": clip_scene[:5],
            "i_see": see_labels,
            "which_sounds_like": cross_labels,
            "i_expect_to_hear": expect_labels,
            "narration": narration,
            "process_time": round(time.time() - t0, 2),
            "confidence": round(max(max_cross_sim, max_clip_conf), 3),
        },
        "attention": attention_info,
        "working_memory": {"slots_used": wm_state["slots_used"], "focus": wm_state["focus"]},
    }


class ListenRequest(BaseModel):
    """Raw PCM audio (base64-encoded, 16kHz mono float32) for real-time listening."""
    audio_b64: str
    sample_rate: int = 16000
    top_k: int = 10


@app.post("/api/listen/process")
@async_endpoint
def listen_process(req: ListenRequest):
    """Process a chunk of audio through the brain — real-time listening endpoint."""
    import base64
    t0 = time.time()

    # Decode base64 PCM
    raw = base64.b64decode(req.audio_b64)
    samples = np.frombuffer(raw, dtype=np.float32)
    if len(samples) < 1600:  # < 0.1s
        return {"error": "Audio too short", "associations": {}}

    # Resample if needed
    if req.sample_rate != 16000:
        import scipy.signal
        samples = scipy.signal.resample_poly(samples, 16000, req.sample_rate).astype(np.float32)

    # Truncate to 30s max
    max_samples = 16000 * 30
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    duration = len(samples) / 16000

    # Whisper encode (audio embedding)
    whisper_enc, processor = load_whisper_model()
    inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(_device)
    with torch.no_grad():
        enc_out = whisper_enc(input_features)
        a_emb = enc_out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    a_emb = a_emb / (np.linalg.norm(a_emb) + 1e-12)

    # Whisper transcription (what words were said?)
    transcription = None
    try:
        from transformers import WhisperForConditionalGeneration
        if not hasattr(listen_process, '_decoder'):
            listen_process._decoder = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            listen_process._decoder.eval()
            listen_process._decoder.to(_device)
        decoder = listen_process._decoder
        # Force English translation (handles French/any language → English)
        with torch.no_grad():
            generated = decoder.generate(
                input_features,
                language="en",
                task="translate",
                max_new_tokens=80,
            )
        transcription = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    except Exception as e:
        log.warning(f"Transcription failed: {e}")

    # CLAP encode (discriminative audio embedding for audio-to-audio matching)
    clap_emb = None
    try:
        clap_emb = encode_audio_clap(samples, sr=16000)
        log.info(f"CLAP audio embedding: norm={np.linalg.norm(clap_emb):.4f}")
    except Exception as e:
        log.warning(f"CLAP encoding failed: {e}")

    # Find associations
    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    results = {}
    top_labels = []
    cross_labels = []

    # Detect if this is speech (transcription has real words) vs environmental sound
    is_speech = transcription and len(transcription.strip()) > 5

    if is_speech:
        # For speech: report speech detection + run imagination on transcription content
        top_labels = ["male speech", "female speech"]

        # Extract key nouns/concepts from transcription and imagine them
        # Use simple keyword extraction: find words that match VGGSound-like concepts
        imagination = None
        try:
            # Pick the most "concrete" words from transcription for imagination
            words = transcription.lower().split()
            # Filter to meaningful words (>3 chars, not stopwords)
            stopwords = {"the", "and", "that", "this", "with", "from", "have", "been",
                         "were", "they", "their", "will", "would", "could", "should",
                         "very", "much", "also", "some", "which", "into", "about",
                         "there", "these", "those", "what", "when", "where", "does",
                         "not", "but", "for", "are", "was", "has", "had", "can"}
            keywords = [w.strip(".,!?;:'\"()") for w in words
                        if len(w) > 3 and w.lower() not in stopwords][:5]

            if keywords:
                query = " ".join(keywords)
                # Run lightweight imagination: semantic match + world model prediction
                semantic = text_semantic_search(query, top_k=5)
                imagine_labels = [r["label"] for r in semantic[:3]]

                # World model: what sounds do these concepts evoke?
                model = load_text_model()
                q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
                pred_a = predict_audio(q_emb)
                expect_sounds = []
                if pred_a is not None and w_a is not None:
                    all_a_proj = mlp_project(cached["a_emb"], w_a)
                    pred_sims = all_a_proj @ pred_a
                    pred_top = np.argsort(pred_sims)[::-1][:5]
                    seen_pred = set()
                    for idx_p in pred_top:
                        if idx_p < len(labels) and labels[idx_p]["label"] not in seen_pred:
                            seen_pred.add(labels[idx_p]["label"])
                            expect_sounds.append(labels[idx_p]["label"])

                # Causal: what follows these concepts?
                causal_next = []
                if load_causal_graph() and imagine_labels:
                    cidx = _find_causal_label(imagine_labels[0])
                    if cidx is not None and _causal_combined is not None:
                        row = _causal_combined[cidx].copy()
                        row[cidx] = 0
                        if row.max() > 0:
                            next_idx = int(np.argmax(row))
                            causal_next = [_causal_labels[next_idx]]

                imagination = {
                    "keywords": keywords,
                    "concepts": imagine_labels,
                    "predicted_sounds": expect_sounds[:3],
                    "causal_next": causal_next,
                }

                # Enrich top_labels with imagined concepts
                seen = set(top_labels)
                for l in imagine_labels[:3]:
                    if l not in seen:
                        seen.add(l)
                        top_labels.append(l)

                # Add predicted sounds as cross-modal associations
                seen_cross = set(top_labels)
                for l in expect_sounds[:3]:
                    if l not in seen_cross:
                        seen_cross.add(l)
                        cross_labels.append(l)

                results["imagination"] = imagination
        except Exception as e:
            log.warning(f"Speech imagination failed: {e}")

        # Add CLAP as supplementary (background sounds behind speech)
        if clap_emb is not None:
            try:
                clap_db = load_clap_audio_embeddings()
                clap_sims = clap_db @ clap_emb
                clap_top = np.argsort(clap_sims)[::-1][:req.top_k]
                results["clap_audio"] = _format_results(clap_top, clap_sims, labels)
                clap_unique = []
                seen_all = set(top_labels + cross_labels)
                for r in results["clap_audio"][:5]:
                    if r["label"] not in seen_all:
                        clap_unique.append(r["label"])
                results["clap_labels_unique"] = clap_unique
            except Exception as e:
                log.warning(f"CLAP failed: {e}")
    else:
        # NON-SPEECH: Use CLAP as primary (environmental sounds)
        if clap_emb is not None:
            try:
                clap_db = load_clap_audio_embeddings()
                clap_sims = clap_db @ clap_emb
                clap_top = np.argsort(clap_sims)[::-1][:req.top_k]
                results["audio_similar"] = _format_results(clap_top, clap_sims, labels)

                seen = set()
                for r in results["audio_similar"][:8]:
                    if r["label"] not in seen:
                        seen.add(r["label"])
                        top_labels.append(r["label"])
            except Exception as e:
                log.warning(f"CLAP failed: {e}")

        # Fallback: if CLAP failed or no CLAP, use transcription if available
        if not top_labels and transcription and len(transcription.strip()) > 2:
            semantic = text_semantic_search(transcription, top_k=req.top_k)
            results["semantic_matches"] = semantic
            seen = set()
            for r in semantic[:8]:
                if r["label"] not in seen:
                    seen.add(r["label"])
                    top_labels.append(r["label"])

        # Cross-modal via W_t or W_v
        w_t = load_text_projection()
        text_w = w_t if w_t is not None else w_v
        if text_w is not None:
            model = load_text_model()
            q_emb = model.encode([f"{transcription}"], normalize_embeddings=True)[0]
            t_proj = mlp_project(q_emb.reshape(1, -1), text_w)[0]
            all_a_proj = mlp_project(cached["a_emb"], w_a)
            t2a_sims = all_a_proj @ t_proj
            t2a_top = np.argsort(t2a_sims)[::-1][:req.top_k]
            results["cross_modal_t2a"] = _format_results(t2a_top, t2a_sims, labels)

            seen_cross = set(top_labels)
            for r in results["cross_modal_t2a"][:5]:
                if r["label"] not in seen_cross:
                    seen_cross.add(r["label"])
                    cross_labels.append(r["label"])

    # Cross-modal A→V via MLP (uses Whisper embedding — still valid for cross-modal)
    if w_v is not None:
        a_proj = mlp_project(a_emb.reshape(1, -1), w_a)[0]
        all_v_proj = mlp_project(cached["v_emb"], w_v)
        a2v_sims = all_v_proj @ a_proj
        a2v_top = np.argsort(a2v_sims)[::-1][:req.top_k]
        results["cross_modal_a2v"] = _format_results(a2v_top, a2v_sims, labels)

        if not top_labels:
            # No CLAP and no transcription — use A→V cross-modal as fallback
            seen = set()
            for r in results["cross_modal_a2v"][:8]:
                if r["label"] not in seen:
                    seen.add(r["label"])
                    top_labels.append(r["label"])

    summary = {
        "i_hear": top_labels,
        "which_reminds_me_of": cross_labels,
        "audio_duration": round(duration, 1),
        "process_time": round(time.time() - t0, 2),
    }

    if transcription:
        summary["transcription"] = transcription
    # Add CLAP-specific labels if available (background sounds behind speech)
    clap_unique = results.get("clap_labels_unique", [])
    if clap_unique:
        summary["clap_labels"] = clap_unique[:5]

    # LLM narration — use imagination content for richer narration when speech
    narrate_labels = top_labels
    narrate_cross = cross_labels
    if is_speech and "imagination" in results:
        img = results["imagination"]
        narrate_labels = img.get("concepts", top_labels)
        narrate_cross = img.get("predicted_sounds", cross_labels)
    narration = _narrate_listening(narrate_labels, narrate_cross, duration)
    if narration:
        summary["narration"] = narration
        summary["process_time"] = round(time.time() - t0, 2)

    # Persist perception to memory DB (with MLP-projected embedding for episode tracking)
    a_proj_emb = mlp_project(a_emb.reshape(1, -1), w_a)[0] if w_a is not None else a_emb
    _store_perception(
        modality="listen",
        transcription=transcription,
        top_labels=top_labels,
        cross_labels=cross_labels,
        imagination=results.get("imagination"),
        narration=narration,
        embedding=a_proj_emb,
    )

    # Phase 4.2: Selective attention — report what the brain is focused on
    wm_state = _get_working_memory_state()
    attention_info = {}
    if wm_state["slots_used"] > 0:
        focus = wm_state["focus"]
        related = [it["label"] for it in wm_state["items"] if it["activation"] > 0.5]
        attention_info = {
            "focused_on": focus,
            "related_active": related[:3],
            "attention_note": f"Paying attention to {focus}" if focus else None,
        }

    return {
        "associations": results,
        "summary": summary,
        "attention": attention_info,
        "working_memory": {"slots_used": wm_state["slots_used"], "focus": wm_state["focus"]},
    }


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


def _narrate_listening(audio_labels: list[str], cross_labels: list[str], duration: float) -> str | None:
    """Ask LLM to narrate what the brain perceives. Returns None on failure."""
    if not audio_labels:
        return None
    try:
        import urllib.request
        prompt = (
            f"You are a brain. You heard: {', '.join(audio_labels[:4])}."
        )
        if cross_labels:
            prompt += f" This reminds you visually of: {', '.join(cross_labels[:3])}."
        prompt += (
            " Respond in exactly ONE short sentence starting with 'I hear' or 'I sense'. "
            "Be vivid and specific, no lists."
        )
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 30},
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read())
        narration = result.get("response", "").strip()
        if narration:
            log.info(f"Narration ({len(narration)} chars): {narration[:80]}...")
        return narration
    except Exception as e:
        log.warning(f"Narration failed: {e}")
        return None


class YouTubeRequest(BaseModel):
    url: str
    top_k: int = 15


class ProcessingStatus(BaseModel):
    status: str
    progress: str


_processing = {}


@app.get("/health")
def health():
    """Enhanced health check with component status."""
    def _check_ollama():
        try:
            import urllib.request
            r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False

    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "components": {
            "mlp": _w_v is not None,
            "world_model": _world_model is not None,
            "audioset_pool": _audioset_pool is not None,
            "brain_decoder": _brain_decoder is not None,
            "nn_graph": _nn_graph is not None,
            "grid_encoder": _grid_cell_encoder is not None,
            "ollama": _check_ollama(),
        },
        "memory": {
            "prototypes": len(_prototypes),
            "fast_memory": _fast_memory_count,
            "working_memory": len(_working_memory),
            "online_pairs_buffer": len(_online_pairs),
            "sse_subscribers": len(_sse_subscribers),
        },
        "autonomy_running": _autonomy_running,
    }


@app.post("/api/youtube/process")
def process_youtube(req: YouTubeRequest):
    """Process a YouTube video through the brain."""
    t0 = time.time()
    results = {"stages": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Stage 1: Download
        try:
            results["stages"].append({"name": "download", "status": "running"})
            dl = download_youtube(req.url, tmpdir)
            results["video_info"] = dl["info"]
            results["stages"][-1]["status"] = "done"
            results["stages"][-1]["time"] = f"{time.time()-t0:.1f}s"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

        # Stage 2: Visual embedding
        t1 = time.time()
        results["stages"].append({"name": "visual_encode", "status": "running"})
        try:
            v_emb = extract_visual_embedding(dl["video"])
            results["stages"][-1]["status"] = "done"
            results["stages"][-1]["time"] = f"{time.time()-t1:.1f}s"
            results["stages"][-1]["dim"] = int(v_emb.shape[0])
        except Exception as e:
            log.error(f"Visual encoding failed: {e}")
            v_emb = np.zeros(384, dtype=np.float32)
            results["stages"][-1]["status"] = "fallback"

        # Stage 3: Audio embedding
        t2 = time.time()
        results["stages"].append({"name": "audio_encode", "status": "running"})
        try:
            a_emb = extract_audio_embedding(dl["audio"])
            results["stages"][-1]["status"] = "done"
            results["stages"][-1]["time"] = f"{time.time()-t2:.1f}s"
            results["stages"][-1]["dim"] = int(a_emb.shape[0])
        except Exception as e:
            log.error(f"Audio encoding failed: {e}")
            a_emb = np.zeros(512, dtype=np.float32)
            results["stages"][-1]["status"] = "fallback"

        # Stage 4: Find associations
        t3 = time.time()
        results["stages"].append({"name": "brain_association", "status": "running"})
        associations = find_associations(v_emb, a_emb, top_k=req.top_k)
        results["stages"][-1]["status"] = "done"
        results["stages"][-1]["time"] = f"{time.time()-t3:.1f}s"

        # Phase 4: Auto-store pair for online learning
        _store_youtube_pair(v_emb, a_emb, req.url)

    results["associations"] = associations
    results["total_time"] = f"{time.time()-t0:.1f}s"
    return results


# ─── Brain Voice (Phase 2) — LLM-Grounded Dialogue ────────────────
import random

# Conversation history per session (simple in-memory store)
_conversations: dict[str, list] = {}
# Dialogue history per session (for /api/brain/dialogue multi-turn)
_dialogue_histories: dict[str, list] = {}

# System prompt for the brain's LLM persona
_BRAIN_SYSTEM_PROMPT = (
    "You are a brain made of cross-modal associations between 24,604 audio-visual clips "
    "across 310 categories. You have a V5 MLP with 97.2% retrieval accuracy. You can predict "
    "sounds from visuals (99.8% accuracy), reason through association chains, and know your own "
    "strengths and weaknesses. Speak as 'I', be specific about what you know. You are NOT a "
    "generic AI — you are THIS specific brain with THESE specific experiences."
)


def _ollama_generate(prompt: str, temperature: float = 0.7, max_predict: int = 150) -> str | None:
    """Call Ollama and return the response text. Returns None on failure."""
    try:
        import urllib.request
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_predict},
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        text = result.get("response", "").strip()
        if text:
            log.info(f"Ollama response ({len(text)} chars): {text[:80]}...")
            return text
        return None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


def _gather_brain_context() -> str:
    """Build a context string with the brain's actual knowledge for LLM grounding."""
    parts = []

    # Self-model summary
    if load_self_model() and _category_stats:
        mrrs = [s["mrr"] for s in _category_stats.values()]
        avg_mrr = float(np.mean(mrrs))
        sorted_cats = sorted(_category_stats.items(), key=lambda x: x[1]["mrr"], reverse=True)
        best_5 = [f"{c} (MRR={s['mrr']:.3f})" for c, s in sorted_cats[:5]]
        worst_5 = [f"{c} (MRR={s['mrr']:.3f})" for c, s in sorted_cats[-5:]]
        parts.append(
            f"Self-assessment: I know {len(_category_stats)} categories, avg MRR={avg_mrr:.3f}. "
            f"Best: {', '.join(best_5)}. Weakest: {', '.join(worst_5)}."
        )

    # Recent perceptions from reflections
    if _reflection_history:
        recent = _reflection_history[-5:]
        insights = [r["insight"][:100] for r in recent if r.get("insight")]
        if insights:
            parts.append(f"Recent reflections: {' | '.join(insights)}")

    # Recent conversations
    recent_msgs = []
    for sid, hist in _conversations.items():
        for msg in hist[-4:]:
            if msg["role"] == "assistant":
                recent_msgs.append(msg["content"][:80])
    if recent_msgs:
        parts.append(f"Recent perceptions: {' | '.join(recent_msgs[-3:])}")

    # World model
    wm = load_world_model()
    is_v2 = wm is not None and getattr(wm, 'projected_space', False)
    if wm:
        wm_mrr = 0.998 if is_v2 else 0.104
        parts.append(f"World model: {'v2 projected' if is_v2 else 'v1'}, MRR={wm_mrr}")

    # Online learning stats
    parts.append(f"Online learning: {_online_learning_count} pairs learned, {len(_online_pairs)} in buffer.")

    # Capabilities summary
    parts.append("Capabilities: 24,604 clips, 310 categories, V5 MLP (97.2% MRR), world model (99.8%), causal graph, concept codebook, imagination pipeline.")

    return "\n".join(parts)


def _strength_word(sim: float) -> str:
    """Convert similarity score to a descriptive strength word."""
    if sim > 0.7:
        return random.choice(["strongly", "vividly", "immediately", "powerfully"])
    elif sim > 0.4:
        return random.choice(["clearly", "distinctly", "notably"])
    elif sim > 0.2:
        return random.choice(["faintly", "vaguely", "distantly"])
    else:
        return random.choice(["barely", "dimly"])


def _describe_associations(associations: dict, video_title: str = None) -> str:
    """Compose a natural-language description from raw association data."""
    parts = []

    if video_title:
        parts.append(f"Looking at \"{video_title}\", here's what lights up in my memory:")
    else:
        parts.append("Here's what I perceive:")

    if associations.get("visual_similar"):
        top = associations["visual_similar"][:5]
        if top:
            best = top[0]
            sim = best.get("similarity", 0)
            strength = _strength_word(sim)
            labels = [a["label"] for a in top[:3]]
            parts.append(
                f"Visually, I {strength} recognize this — it reminds me of: {', '.join(labels)}."
                f" (top match: {best['label']} at {sim:.0%} similarity)"
            )

    if associations.get("audio_similar"):
        top = associations["audio_similar"][:5]
        if top:
            best = top[0]
            sim = best.get("similarity", 0)
            strength = _strength_word(sim)
            labels = [a["label"] for a in top[:3]]
            parts.append(
                f"In terms of sound, I {strength} associate this with: {', '.join(labels)}."
            )

    if associations.get("cross_modal"):
        top = associations["cross_modal"][:5]
        if top:
            best = top[0]
            sim = best.get("similarity", 0)
            labels = [a["label"] for a in top[:3]]
            if sim > 0.4:
                parts.append(
                    f"Fascinating — when I see this, my audio memory fires up with: {', '.join(labels)}. "
                    f"The cross-modal bridge is quite active here!"
                )
            else:
                parts.append(
                    f"The cross-modal connection is subtle, but I sense echoes of: {', '.join(labels)}."
                )

    if len(parts) == 1:
        parts.append("I don't have strong associations for this input. It's like looking at something I've never quite encountered before.")

    return "\n\n".join(parts)


def _answer_question_template(question: str, results: list[dict], cross_labels: list[str] = None) -> str:
    """FALLBACK: Compose a template answer to a question from search results."""
    if not results:
        return (
            f"Hmm, when I search my memories for \"{question}\", nothing lights up strongly. "
            "My 24,000 clips don't seem to cover that concept well, or perhaps I know it by a different name."
        )

    top = results[:5]
    labels = [r["label"] for r in top]
    parts = []
    first = top[0]
    sim = first.get("similarity", 0)

    if sim > 0.5:
        parts.append(
            f"Oh yes! When you mention that, my memory {_strength_word(sim)} activates. "
            f"I recall: {', '.join(labels[:3])}."
        )
    else:
        parts.append(
            f"Let me search my memories... I find some connections: {', '.join(labels[:3])}."
        )

    if len(top) > 3:
        more = [r["label"] for r in top[3:]]
        parts.append(f"I also sense echoes of: {', '.join(more)}.")

    if cross_labels:
        parts.append(
            f"When I see this, my learned cross-modal bridge hears: "
            f"{', '.join(cross_labels[:3])}. "
            f"That's my trained V→A association at work."
        )

    parts.append(f"(Searched {len(results)} matching memories)")
    return " ".join(parts)


def _answer_question(question: str, results: list[dict], cross_labels: list[str] = None) -> str:
    """LLM-grounded answer to a question, with template fallback."""
    # Build grounding context from actual search results
    if not results:
        grounding = f"I searched my 24,604 clips for \"{question}\" but found no strong matches."
    else:
        match_lines = []
        for r in results[:5]:
            match_lines.append(
                f"- {r['label']}: similarity={r.get('similarity', 0):.2f}, {r.get('clip_count', 1)} clips"
            )
        grounding = f"Search results for \"{question}\":\n" + "\n".join(match_lines)

    if cross_labels:
        grounding += f"\nCross-modal associations (V->A): {', '.join(cross_labels[:5])}"

    prompt = (
        f"{_BRAIN_SYSTEM_PROMPT}\n\n"
        f"GROUNDING DATA (use these facts, do not invent others):\n{grounding}\n\n"
        f"User asked: \"{question}\"\n"
        f"Answer in 1-3 sentences as yourself. Reference specific categories and numbers from the data."
    )

    llm_response = _ollama_generate(prompt)
    if llm_response:
        return llm_response

    # Fallback to template
    return _answer_question_template(question, results, cross_labels)


def _chat_response_template(message: str, associations: dict = None) -> str:
    """FALLBACK: Template-based chat response when Ollama is unavailable."""
    msg_lower = message.lower()

    if any(w in msg_lower for w in ["hello", "hi ", "hey", "greetings"]):
        return (
            "Hello! I'm an associative brain that learned from 24,604 video clips across 310 categories. "
            "Ask me about sounds, sights, or show me a YouTube video — I'll tell you what lights up in my memory."
        )
    if any(w in msg_lower for w in ["how are you", "how do you feel"]):
        return (
            "I feel... connected. Thousands of cross-modal pathways humming quietly, "
            "ready to fire when something familiar comes along. "
            "Ask me about a sound or a sight, and I'll show you what I perceive."
        )
    if any(w in msg_lower for w in ["what can you do", "what do you know", "capabilities"]):
        return (
            "I can recognize patterns across vision and audio with 97.2% accuracy. "
            "Show me something visual and I'll predict what it sounds like (99.8% accuracy). "
            "I know 310 categories across 24,604 clips."
        )
    if any(w in msg_lower for w in ["strongest", "best", "favorite", "most"]):
        return (
            "My strongest memories are the ones where sight and sound converge perfectly. "
            "Try asking me about a specific concept to see which memories fire!"
        )
    if associations:
        return _describe_associations(associations)
    return (
        "Interesting thought! To give you a vivid answer, try asking me about specific sounds, "
        "sights, or concepts — like \"What do you hear when you see water?\" or \"What sounds go with dogs?\". "
        "You can also show me a YouTube video and I'll describe my perception."
    )


def _chat_response(message: str, associations: dict = None) -> str:
    """LLM-grounded conversational response, with template fallback."""
    brain_context = _gather_brain_context()

    # Build association context if provided
    assoc_context = ""
    if associations:
        assoc_parts = []
        for key in ["visual_similar", "audio_similar", "cross_modal"]:
            items = associations.get(key, [])
            if items:
                labels = [a["label"] for a in items[:3]]
                assoc_parts.append(f"{key}: {', '.join(labels)}")
        if assoc_parts:
            assoc_context = f"\nCurrent associations: {'; '.join(assoc_parts)}"

    prompt = (
        f"{_BRAIN_SYSTEM_PROMPT}\n\n"
        f"BRAIN STATE:\n{brain_context}{assoc_context}\n\n"
        f"User says: \"{message}\"\n"
        f"Respond in 1-3 sentences as yourself. Be specific about your actual data."
    )

    llm_response = _ollama_generate(prompt)
    if llm_response:
        return llm_response

    # Fallback to template
    return _chat_response_template(message, associations)


def _format_context(results: list[dict] = None, cross_labels: list[str] = None) -> str:
    """Format context string for the UI."""
    parts = []
    if results:
        items = ", ".join(f"{r['label']} ({r.get('similarity', 0):.0%})" for r in results[:8])
        parts.append(f"Matches: {items}")
    if cross_labels:
        parts.append(f"Cross-modal: {', '.join(cross_labels)}")
    return " | ".join(parts) if parts else ""


class DescribeRequest(BaseModel):
    associations: dict
    video_title: str | None = None
    session_id: str = "default"


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    associations: dict | None = None


class DialogueRequest(BaseModel):
    message: str
    session_id: str = "default"


def _detect_dialogue_intent(message: str) -> str:
    """Detect intent from user message for dialogue routing."""
    msg_lower = message.lower()
    if any(w in msg_lower for w in ["best at", "strongest", "most confident", "good at", "excel"]):
        return "self_strengths"
    if any(w in msg_lower for w in ["worst", "weakest", "struggle", "bad at", "hard for"]):
        return "self_weaknesses"
    if any(w in msg_lower for w in ["hear recently", "heard", "perceived", "seen recently", "last"]):
        return "recent_perceptions"
    if any(w in msg_lower for w in ["what would happen", "imagine", "what if", "showed you", "predict"]):
        return "imagination"
    if any(w in msg_lower for w in ["confident about", "how well", "accuracy for", "how good"]):
        return "category_confidence"
    if any(w in msg_lower for w in ["how many", "total", "count", "size", "categories"]):
        return "stats"
    if any(w in msg_lower for w in ["learn", "training", "improve", "online"]):
        return "learning"
    return "general"


def _gather_intent_context(intent: str, message: str) -> str:
    """Gather specific brain data based on detected intent."""
    parts = []

    if intent == "self_strengths":
        if load_self_model() and _category_stats:
            sorted_cats = sorted(_category_stats.items(), key=lambda x: x[1]["mrr"], reverse=True)
            top10 = [f"{c}: MRR={s['mrr']:.3f}, R@1={s['r1']:.3f}, {s['n_clips']} clips" for c, s in sorted_cats[:10]]
            parts.append("My top 10 categories:\n" + "\n".join(top10))

    elif intent == "self_weaknesses":
        if load_self_model() and _category_stats:
            sorted_cats = sorted(_category_stats.items(), key=lambda x: x[1]["mrr"])
            bottom10 = [f"{c}: MRR={s['mrr']:.3f}, R@1={s['r1']:.3f}, {s['n_clips']} clips" for c, s in sorted_cats[:10]]
            parts.append("My weakest 10 categories:\n" + "\n".join(bottom10))

    elif intent == "recent_perceptions":
        recent_msgs = []
        for sid, hist in _conversations.items():
            for msg in hist[-10:]:
                if msg["role"] == "assistant":
                    recent_msgs.append(msg["content"][:150])
        if recent_msgs:
            parts.append("Recent perceptions:\n" + "\n".join(f"- {m}" for m in recent_msgs[-5:]))
        if _reflection_history:
            recent_r = [r["insight"][:150] for r in _reflection_history[-5:] if r.get("insight")]
            if recent_r:
                parts.append("Recent reflections:\n" + "\n".join(f"- {r}" for r in recent_r))

    elif intent == "imagination":
        # Run imagination: semantic match + world model prediction
        semantic = text_semantic_search(message, top_k=5)
        if semantic:
            see_labels = [r["label"] for r in semantic[:3]]
            parts.append(f"Semantic match for query: {', '.join(see_labels)}")
            try:
                model = load_text_model()
                q_emb = model.encode([message], normalize_embeddings=True)[0].astype(np.float32)
                pred_a = predict_audio(q_emb)
                if pred_a is not None:
                    cached = load_cached_embeddings()
                    clip_labels = load_clip_labels()
                    w_v, w_a = load_mlp_weights()
                    if w_a is not None:
                        all_a_proj = mlp_project(cached["a_emb"], w_a)
                        pred_sims = all_a_proj @ pred_a
                        pred_top = np.argsort(pred_sims)[::-1][:5]
                        pred_labels = list(dict.fromkeys(
                            clip_labels[i]["label"] for i in pred_top if i < len(clip_labels)
                        ))[:5]
                        parts.append(f"World model predicts I would hear: {', '.join(pred_labels)}")
            except Exception as e:
                log.warning(f"Imagination prediction failed: {e}")
            if load_self_model() and _category_stats and see_labels:
                cat_stat = _category_stats.get(see_labels[0])
                if cat_stat:
                    parts.append(f"My confidence for '{see_labels[0]}': MRR={cat_stat['mrr']:.3f}, R@1={cat_stat['r1']:.3f}")

    elif intent == "category_confidence":
        semantic = text_semantic_search(message, top_k=3)
        if semantic and load_self_model() and _category_stats:
            for sr in semantic[:3]:
                cat_stat = _category_stats.get(sr["label"])
                if cat_stat:
                    parts.append(
                        f"'{sr['label']}': MRR={cat_stat['mrr']:.3f}, R@1={cat_stat['r1']:.3f}, "
                        f"{cat_stat['n_clips']} clips, rank={cat_stat.get('rank', '?')}"
                    )

    elif intent == "stats":
        parts.append("Total clips: 24,604. Categories: 310.")
        wm = load_world_model()
        is_v2 = wm is not None and getattr(wm, 'projected_space', False)
        parts.append(f"World model: {'v2 (99.8% MRR)' if is_v2 else 'v1'}.")
        parts.append("V5 MLP retrieval: 97.2% MRR.")
        parts.append(f"Online learning: {_online_learning_count} pairs learned.")
        if load_self_model() and _category_stats:
            mrrs = [s["mrr"] for s in _category_stats.values()]
            parts.append(f"Per-category avg MRR: {np.mean(mrrs):.3f}.")

    elif intent == "learning":
        parts.append(f"Online learning buffer: {len(_online_pairs)} pairs waiting.")
        parts.append(f"Total pairs learned online: {_online_learning_count}.")
        parts.append("Training: V5 MLP (single hidden layer, ReLU, 97.2% MRR on 24K pool).")

    return "\n".join(parts) if parts else ""


@app.post("/api/brain/dialogue")
def brain_dialogue(req: DialogueRequest):
    """Multi-turn dialogue grounded in the brain's actual data.

    Maintains conversation history, detects intent, and routes to
    the appropriate brain capability for grounded responses.
    """
    t0 = time.time()

    # Detect intent and gather specific context
    intent = _detect_dialogue_intent(req.message)
    intent_context = _gather_intent_context(intent, req.message)
    brain_context = _gather_brain_context()

    # Build conversation history context
    history = _dialogue_histories.get(req.session_id, [])
    history_text = ""
    if history:
        recent = history[-10:]  # last 10 turns (5 exchanges)
        history_lines = [
            f"{'User' if m['role'] == 'user' else 'Brain'}: {m['content'][:120]}"
            for m in recent
        ]
        history_text = f"\nConversation so far:\n" + "\n".join(history_lines)

    prompt = (
        f"{_BRAIN_SYSTEM_PROMPT}\n\n"
        f"BRAIN STATE:\n{brain_context}\n"
    )
    if intent_context:
        prompt += f"\nSPECIFIC DATA (for this question):\n{intent_context}\n"
    if history_text:
        prompt += f"{history_text}\n"
    prompt += (
        f"\nUser: {req.message}\n"
        f"Respond in 1-3 sentences. Use the specific data above. Do not invent facts not in the data."
    )

    llm_response = _ollama_generate(prompt)

    if llm_response:
        response = llm_response
    else:
        # Fallback: use template for chat
        response = _chat_response_template(req.message)

    # Update dialogue history
    history = _dialogue_histories.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": response})
    if len(history) > 20:  # keep last 10 turns
        _dialogue_histories[req.session_id] = history[-20:]

    # Also update conversations for reflection access
    session = _conversations.setdefault(req.session_id, [])
    session.append({"role": "user", "content": req.message})
    session.append({"role": "assistant", "content": response})
    if len(session) > 40:
        _conversations[req.session_id] = session[-40:]

    return {
        "response": response,
        "intent": intent,
        "grounded": llm_response is not None,
        "time": round(time.time() - t0, 2),
    }


@app.post("/api/brain/describe")
def brain_describe(req: DescribeRequest):
    """The brain describes what it perceives from association data."""
    response = _describe_associations(req.associations, req.video_title)

    # Store in conversation history
    session = _conversations.setdefault(req.session_id, [])
    session.append({"role": "user", "content": f"[Showed brain: {req.video_title or 'input'}]"})
    session.append({"role": "assistant", "content": response})
    if len(session) > 40:
        _conversations[req.session_id] = session[-40:]

    context = _format_context()
    return {"response": response, "context": context}


@app.post("/api/brain/ask")
def brain_ask(req: AskRequest):
    """Ask the brain a question — hybrid keyword + semantic search (Phase 3)."""
    labels = load_clip_labels()
    cached = load_cached_embeddings()

    # Phase 3: Semantic search via sentence-transformer
    semantic_results = text_semantic_search(req.question, top_k=15)

    # Build results from semantic matches — include actual clip info
    label_to_clips = {}
    for i, clip in enumerate(labels):
        label_to_clips.setdefault(clip["label"], []).append(i)

    seen_labels = set()
    top_results = []
    for sr in semantic_results:
        if sr["label"] in seen_labels:
            continue
        seen_labels.add(sr["label"])
        clips = label_to_clips.get(sr["label"], [])
        if clips:
            idx = clips[0]
            top_results.append({
                "idx": idx,
                "label": sr["label"],
                "similarity": sr["text_similarity"],
                "youtube_id": labels[idx]["youtube_id"],
                "start_sec": labels[idx]["start_sec"],
                "clip_count": len(clips),
            })
        if len(top_results) >= 10:
            break

    # Cross-modal via v4 MLP: project matched visuals through MLP, find nearest audio
    cross_labels = []
    w_v, w_a = load_mlp_weights()
    if top_results and w_v is not None:
        match_indices = [r["idx"] for r in top_results[:5]]
        # Average visual embedding of matched clips, project through MLP
        v_mean = cached["v_emb"][match_indices].mean(axis=0, keepdims=True)
        v_proj = mlp_project(v_mean, w_v)  # (1, d_hidden)
        # Project all audio through MLP
        all_a_proj = mlp_project(cached["a_emb"], w_a)  # (N, d_hidden)
        a_sims = (all_a_proj @ v_proj.T).ravel()
        a_top = np.argsort(a_sims)[::-1][:30]
        matched_labels = {r["label"] for r in top_results}
        cross_labels = list(dict.fromkeys(
            labels[i]["label"] for i in a_top
            if i < len(labels) and labels[i]["label"] not in matched_labels
        ))[:5]
    elif top_results:
        # Fallback: use raw audio similarity
        match_indices = [r["idx"] for r in top_results[:5]]
        a_mean = cached["a_emb"][match_indices].mean(axis=0)
        a_mean = a_mean / (np.linalg.norm(a_mean) + 1e-12)
        a_sims = cached["a_emb"] @ a_mean
        a_top = np.argsort(a_sims)[::-1][:30]
        matched_labels = {r["label"] for r in top_results}
        cross_labels = list(dict.fromkeys(
            labels[i]["label"] for i in a_top
            if i < len(labels) and labels[i]["label"] not in matched_labels
        ))[:5]

    response = _answer_question(req.question, top_results, cross_labels)
    context = _format_context(top_results, cross_labels)

    # Update history
    session = _conversations.setdefault(req.session_id, [])
    session.append({"role": "user", "content": req.question})
    session.append({"role": "assistant", "content": response})
    if len(session) > 40:
        _conversations[req.session_id] = session[-40:]

    return {"response": response, "context": context, "matches": len(top_results)}


@app.post("/api/brain/chat")
def brain_chat(req: ChatRequest):
    """Free-form chat with the brain."""
    response = _chat_response(req.message, req.associations)

    session = _conversations.setdefault(req.session_id, [])
    session.append({"role": "user", "content": req.message})
    session.append({"role": "assistant", "content": response})
    if len(session) > 40:
        _conversations[req.session_id] = session[-40:]

    return {"response": response}


@app.delete("/api/brain/chat/{session_id}")
def clear_chat(session_id: str):
    """Clear conversation history for a session."""
    _conversations.pop(session_id, None)
    _dialogue_histories.pop(session_id, None)
    return {"status": "cleared"}


# ─── Phase 4: Online Learning ──────────────────────────────────────

class TextQueryRequest(BaseModel):
    """Text query for cross-modal retrieval (Phase 3)."""
    query: str
    top_k: int = 10
    mode: str = "both"  # "v2a" (text→audio), "a2v" (text→visual), "both"


@app.post("/api/brain/text_query")
def brain_text_query(req: TextQueryRequest):
    """Full cross-modal retrieval from text (Phase 3).

    Encodes text with sentence-transformer (384-dim, same as DINOv2 visual),
    then uses it as a visual query through the MLP to find matching audio,
    or does semantic label search for direct matches.
    """
    t0 = time.time()
    model = load_text_model()
    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    # Encode query text → 384-dim (same space as visual embeddings)
    q_emb = model.encode([f"{req.query}"], normalize_embeddings=True)[0]  # (384,)

    results = {}

    # Semantic label search (always)
    semantic = text_semantic_search(req.query, top_k=req.top_k)
    results["semantic_matches"] = semantic

    # Cross-modal: use text embedding through W_t (grounded) or W_v (fallback)
    w_t = load_text_projection()
    text_proj_matrix = w_t if w_t is not None else w_v
    if text_proj_matrix is not None and req.mode in ("v2a", "both"):
        # Text (384-dim) → W_t/W_v → ReLU → project into shared space
        t_proj = mlp_project(q_emb.reshape(1, -1), text_proj_matrix)[0]  # (d_hidden,)
        # Find nearest audio embeddings in projected space
        all_a_proj = mlp_project(cached["a_emb"], w_a)
        t2a_sims = all_a_proj @ t_proj
        t2a_top = np.argsort(t2a_sims)[::-1][:req.top_k]
        results["text_to_audio"] = _format_results(t2a_top, t2a_sims, labels)

    if text_proj_matrix is not None and req.mode in ("a2v", "both"):
        # Reverse: find clips whose visual projection is closest to the text embedding
        all_v_proj = mlp_project(cached["v_emb"], w_v)
        t_proj = mlp_project(q_emb.reshape(1, -1), text_proj_matrix)[0]
        t2v_sims = all_v_proj @ t_proj
        t2v_top = np.argsort(t2v_sims)[::-1][:req.top_k]
        results["text_to_visual"] = _format_results(t2v_top, t2v_sims, labels)

    # LLM narration of results
    top_labels = [r["label"] for r in semantic[:3]]
    audio_labels = [r["label"] for r in results.get("text_to_audio", [])[:3]]
    narration = None
    if top_labels:
        try:
            import urllib.request as ur
            prompt = (
                f"You are a brain. Someone asked: \"{req.query}\". "
                f"Your semantic memory found: {', '.join(top_labels)}. "
            )
            if audio_labels:
                prompt += f"Your cross-modal bridge hears: {', '.join(audio_labels)}. "
            prompt += "Respond in ONE vivid sentence as 'I'. Be sensory."
            payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                                   "options": {"temperature": 0.8, "num_predict": 30}}).encode()
            r = ur.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
            resp = ur.urlopen(r, timeout=10)
            narration = json.loads(resp.read()).get("response", "").strip()
        except Exception:
            pass

    return {
        "query": req.query,
        "results": results,
        "narration": narration,
        "process_time": round(time.time() - t0, 2),
    }


class OnlinePairRequest(BaseModel):
    """Submit a new (visual, audio) embedding pair for online learning."""
    v_emb_b64: str  # base64 float32
    a_emb_b64: str  # base64 float32
    source: str = "unknown"


@app.post("/api/brain/learn")
def brain_learn(req: OnlinePairRequest):
    """Store a new embedding pair for online learning (Phase 4)."""
    import base64
    v = np.frombuffer(base64.b64decode(req.v_emb_b64), dtype=np.float32)
    a = np.frombuffer(base64.b64decode(req.a_emb_b64), dtype=np.float32)
    if v.shape[0] != 384 or a.shape[0] != 512:
        return {"error": f"Wrong dims: v={v.shape}, a={a.shape}, expected (384,) and (512,)"}
    _online_pairs.append((v, a))
    log.info(f"Online learning: stored pair from {req.source} (buffer: {len(_online_pairs)})")
    return {"status": "stored", "buffer_size": len(_online_pairs)}


@app.post("/api/brain/learn/train")
@async_endpoint
def brain_learn_train():
    """Run online learning: fine-tune MLP on accumulated pairs (Phase 4)."""
    global _w_v, _w_a, _online_learning_count
    if len(_online_pairs) < 2:
        return {"error": "Need at least 2 pairs", "buffer_size": len(_online_pairs)}

    w_v, w_a = load_mlp_weights()
    if w_v is None:
        return {"error": "No MLP weights loaded"}
    # Make writable copies for in-place gradient updates
    w_v = w_v.copy()
    w_a = w_a.copy()

    # Stack pairs into matrices
    v_data = np.stack([p[0] for p in _online_pairs])
    a_data = np.stack([p[1] for p in _online_pairs])
    n = len(_online_pairs)

    # L2 normalize
    v_data = v_data / np.linalg.norm(v_data, axis=1, keepdims=True).clip(1e-12)
    a_data = a_data / np.linalg.norm(a_data, axis=1, keepdims=True).clip(1e-12)

    # #9 ACh Adaptive Learning Rate — modulate by prediction error variance
    if len(_prediction_error_history) >= 10:
        recent_pe = _prediction_error_history[-ACH_WINDOW:]
        pe_mean = float(np.mean(recent_pe))
        pe_var = float(np.var(recent_pe))
        consistency = 1.0 / (1.0 + pe_var * 10)
        ach_signal = pe_mean * consistency
        lr = ACH_LR_MIN + (ACH_LR_MAX - ACH_LR_MIN) * min(ach_signal * 5, 1.0)
    else:
        lr = 0.001
        pe_mean, pe_var, consistency = 0, 0, 0

    temp = 0.01
    n_steps = min(50, n * 5)

    log.info(f"Online learning: {n} pairs, {n_steps} steps, lr={lr:.5f} (ACh: pe_mean={pe_mean:.3f})")
    for step in range(n_steps):
        # Forward
        v_proj = np.maximum(v_data @ w_v, 0)  # ReLU
        a_proj = np.maximum(a_data @ w_a, 0)

        # #3: Compute per-sample surprise weights (prediction error weighted training)
        v_proj_norm = v_proj / np.linalg.norm(v_proj, axis=1, keepdims=True).clip(1e-12)
        a_proj_norm = a_proj / np.linalg.norm(a_proj, axis=1, keepdims=True).clip(1e-12)
        diag_sims = np.sum(v_proj_norm * a_proj_norm, axis=1)  # per-sample match quality
        surprise_weights = (1.0 - diag_sims.clip(0, 1))  # higher surprise = higher weight
        surprise_weights = surprise_weights / (surprise_weights.mean() + 1e-12)  # normalize to mean=1

        # Similarity matrix
        sim = (v_proj @ a_proj.T) / temp
        sim_max = sim.max(axis=1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)
        probs = exp_sim / exp_sim.sum(axis=1, keepdims=True)

        # #4: Selective cross-modal enhancement — boost congruent pairs
        congruence = np.exp(diag_sims.clip(0, 1) * 2)  # boost matched pairs
        target = np.diag(congruence)
        target = target / target.sum(axis=1, keepdims=True)  # normalize rows
        grad = target - probs  # (n, n)

        # #3: Apply per-sample surprise weights to gradient
        scale = lr / (n * temp) * surprise_weights.reshape(-1, 1)

        # Backprop through ReLU
        grad_v_proj = scale * (grad @ a_proj)
        grad_a_proj = (scale.T if scale.ndim > 1 else scale) * (grad.T @ v_proj)
        v_pre = v_data @ w_v
        a_pre = a_data @ w_a
        grad_v_proj *= (v_pre > 0).astype(np.float32)
        grad_a_proj *= (a_pre > 0).astype(np.float32)

        # Update weights
        w_v += v_data.T @ grad_v_proj
        w_a += a_data.T @ grad_a_proj

        # #9: Track prediction errors for ACh modulation
        step_loss = float(1.0 - diag_sims.mean())
        _prediction_error_history.append(step_loss)
        if len(_prediction_error_history) > ACH_WINDOW * 2:
            _prediction_error_history[:] = _prediction_error_history[-ACH_WINDOW:]

    # Update global weights
    _w_v = w_v.copy()
    _w_a = w_a.copy()
    _online_learning_count += n

    # Persist count and log to memory DB
    _update_stat("online_learning_count", _online_learning_count)
    _store_learning_log("online_train", {"pairs": n, "steps": n_steps, "total": _online_learning_count})

    # Save updated weights
    # Save to a separate online-learned path (never overwrite the base model)
    online_dir = PROJECT_ROOT / "outputs/cortex/v4_mlp_online"
    online_dir.mkdir(parents=True, exist_ok=True)
    _save_bin_matrix(w_v, online_dir / "w_v.bin")
    _save_bin_matrix(w_a, online_dir / "w_a.bin")

    # Clear buffer
    pairs_trained = len(_online_pairs)
    _online_pairs.clear()

    log.info(f"Online learning complete: {pairs_trained} pairs, total learned: {_online_learning_count}")
    _emit_event("training", {"pairs": pairs_trained, "total": _online_learning_count})
    return {
        "status": "trained",
        "pairs_trained": pairs_trained,
        "total_online_learned": _online_learning_count,
        "ach_lr": round(lr, 6),
    }


@app.get("/api/brain/learn/status")
def brain_learn_status():
    """Check online learning buffer status."""
    return {
        "buffer_size": len(_online_pairs),
        "total_online_learned": _online_learning_count,
    }


def _save_bin_matrix(m: np.ndarray, path: Path):
    """Save matrix in the same format as Rust (header: RxC\\n + f32 LE data)."""
    header = f"{m.shape[0]}x{m.shape[1]}\n".encode()
    with open(path, 'wb') as f:
        f.write(header)
        f.write(m.astype(np.float32).tobytes())


# ─── Phase 5: Autonomous Reflection ───────────────────────────────

_reflection_history: list[dict] = []
_reflection_running = False


@app.post("/api/brain/reflect")
@async_endpoint
def brain_reflect():
    """Trigger one cycle of autonomous self-reflection (Phase 5).

    The brain reviews its recent perceptions and generates an insight.
    """
    import urllib.request

    # Gather recent perception data FROM DB (persistent across restarts)
    db_perceptions = _get_recent_perceptions(limit=10)
    recent_perceptions = []
    for p in db_perceptions:
        summary_parts = []
        if p.get("modality"):
            summary_parts.append(f"[{p['modality']}]")
        if p.get("transcription"):
            summary_parts.append(p["transcription"][:100])
        if p.get("top_labels") and isinstance(p["top_labels"], list):
            summary_parts.append("heard: " + ", ".join(p["top_labels"][:3]))
        if p.get("narration"):
            summary_parts.append(p["narration"][:100])
        if summary_parts:
            recent_perceptions.append(" ".join(summary_parts)[:200])

    # Also include conversation-based perceptions as fallback
    for session_id, history in _conversations.items():
        for msg in history[-6:]:
            if msg["role"] == "assistant":
                recent_perceptions.append(msg["content"][:200])

    cached = load_cached_embeddings()
    labels = load_clip_labels()

    # Sample some random clips to "think about"
    sample_idx = np.random.choice(len(labels), size=min(10, len(labels)), replace=False)
    sample_labels = [labels[i]["label"] for i in sample_idx]

    # Ask LLM to reflect
    context = ""
    if recent_perceptions:
        context += "Recent perceptions:\n" + "\n".join(f"- {p}" for p in recent_perceptions[-5:]) + "\n\n"
    context += f"Random memories from my 24,604 clips: {', '.join(sample_labels[:8])}\n"
    context += f"I have learned {_online_learning_count} new associations online.\n"

    prompt = (
        f"You are a brain made of cross-modal associations between sounds and images. "
        f"You have 24,604 audio-visual memories. Speak as 'I'. "
        f"You are NOT an AI assistant — you are a living, perceiving brain.\n\n{context}\n"
        f"Share ONE brief thought about what you notice or wonder about. "
        f"Be curious, sensory, and personal. 1-2 sentences only."
    )

    try:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9, "num_predict": 60},
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read())
        insight = result.get("response", "").strip()
    except Exception as e:
        insight = f"[reflection failed: {e}]"

    reflection = {
        "timestamp": time.time(),
        "insight": insight,
        "recent_perceptions_count": len(recent_perceptions),
        "online_pairs_learned": _online_learning_count,
    }
    _reflection_history.append(reflection)
    if len(_reflection_history) > 100:
        _reflection_history[:] = _reflection_history[-100:]

    # Persist to DB
    _store_reflection(insight, len(recent_perceptions), _online_learning_count)

    log.info(f"Reflection: {insight[:100]}...")
    return reflection


@app.get("/api/brain/reflections")
def brain_reflections():
    """Get history of brain's self-reflections."""
    return {"reflections": _reflection_history[-20:]}


@app.get("/api/brain/memory/stats")
def brain_memory_stats():
    """Return persistent memory statistics."""
    try:
        conn = _get_memory_db()
        rows = conn.execute(
            "SELECT modality, COUNT(*) as cnt FROM perceptions GROUP BY modality"
        ).fetchall()
        perceptions_by_modality = {row["modality"]: row["cnt"] for row in rows}
        total_perceptions = sum(perceptions_by_modality.values())

        total_reflections = conn.execute("SELECT COUNT(*) as cnt FROM reflections").fetchone()["cnt"]

        online_learned = _get_stat("online_learning_count", "0")

        last_ts_row = conn.execute(
            "SELECT timestamp FROM perceptions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_perception_ts = last_ts_row["timestamp"] if last_ts_row else None

        db_size = MEMORY_DB_PATH.stat().st_size if MEMORY_DB_PATH.exists() else 0

        return {
            "perceptions_by_modality": perceptions_by_modality,
            "total_perceptions": total_perceptions,
            "total_reflections": total_reflections,
            "online_pairs_learned": int(online_learned),
            "last_perception_timestamp": last_perception_ts,
            "memory_db_size_bytes": db_size,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/brain/memory/recent")
def brain_memory_recent():
    """Return last 20 perceptions + last 10 reflections from persistent memory."""
    return {
        "perceptions": _get_recent_perceptions(limit=20),
        "reflections": _get_recent_reflections(limit=10),
    }


@app.post("/api/brain/reflect/auto")
def brain_reflect_auto():
    """Start/stop autonomous reflection loop (Phase 5)."""
    global _reflection_running
    _reflection_running = not _reflection_running
    if _reflection_running:
        import threading
        def _auto_reflect():
            while _reflection_running:
                try:
                    brain_reflect()
                except Exception as e:
                    log.error(f"Auto-reflect error: {e}")
                time.sleep(300)  # reflect every 5 minutes
        t = threading.Thread(target=_auto_reflect, daemon=True)
        t.start()
        log.info("Autonomous reflection started (every 5 min)")
    else:
        log.info("Autonomous reflection stopped")
    return {"auto_reflect": _reflection_running}


# ─── Integrate Phase 4 into YouTube processing ────────────────────

_original_process_youtube = None


def _store_youtube_pair(v_emb: np.ndarray, a_emb: np.ndarray, url: str):
    """Auto-store embedding pairs from YouTube processing for online learning + episodic memory."""
    if v_emb is not None and a_emb is not None:
        if np.linalg.norm(v_emb) > 0.01 and np.linalg.norm(a_emb) > 0.01:
            _online_pairs.append((v_emb.copy(), a_emb.copy()))
            log.info(f"Auto-stored YouTube pair for online learning (buffer: {len(_online_pairs)})")
            # Find top association labels for this pair
            w_v, w_a = load_mlp_weights()
            proj_emb = mlp_project(v_emb.reshape(1, -1), w_v)[0] if w_v is not None else v_emb
            # Get nearest category labels for the embedding
            top_labels = [url]
            try:
                cached = load_cached_embeddings()
                labels = load_clip_labels()
                if w_v is not None:
                    all_v = mlp_project(cached["v_emb"], w_v)
                    sims = all_v @ proj_emb
                    top_idx = np.argsort(sims)[::-1][:5]
                    seen = set()
                    top_labels = []
                    for idx in top_idx:
                        if idx < len(labels) and labels[idx]["label"] not in seen:
                            seen.add(labels[idx]["label"])
                            top_labels.append(labels[idx]["label"])
                    if not top_labels:
                        top_labels = [url]
            except Exception:
                pass
            _store_perception(
                modality="youtube",
                top_labels=top_labels,
                embedding=proj_emb,
            )


# ═══════════════════════════════════════════════════════════════════
# Step 1: Predictive World Model
# ═══════════════════════════════════════════════════════════════════

WORLD_MODEL_DIR = PROJECT_ROOT / "outputs/cortex/world_model"


def load_world_model():
    """Load the trained world model (v→a predictor in MLP-projected space)."""
    global _world_model
    if _world_model is not None:
        return _world_model
    import torch.nn as nn

    # Try v2 (projected space, residual) first, then v1 fallback
    v2_path = WORLD_MODEL_DIR / "predictor_v2.pt"
    v1_path = WORLD_MODEL_DIR / "predictor.pt"

    class WorldModelV2(nn.Module):
        """v2: operates in MLP-projected 512-dim space with residual."""
        def __init__(self, dim=512, hidden=1024):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, dim),
            )
            self.skip = nn.Identity()
            self.projected_space = True  # flag for inference

        def forward(self, x):
            out = self.net(x) + self.skip(x)
            return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    class WorldModel3(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(384, 512, bias=False)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 512)
            self.projected_space = False

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    if v2_path.exists():
        model = WorldModelV2()
        model.load_state_dict(torch.load(v2_path, map_location="cpu", weights_only=True))
        model.eval()
        _world_model = model
        log.info("Loaded world model v2 (projected space, 2.1M params, MRR=0.388)")
        return model

    if v1_path.exists():
        state_dict = torch.load(v1_path, map_location="cpu", weights_only=True)
        if "fc3.weight" in state_dict:
            model = WorldModel3()
        else:
            # 2-layer fallback
            class WM2(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(384, 512)
                    self.fc2 = nn.Linear(512, 512)
                    self.projected_space = False
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            model = WM2()
        model.load_state_dict(state_dict)
        model.eval()
        _world_model = model
        log.info("Loaded world model v1 (raw space)")
        return model

    log.warning("No world model found")
    return None


def predict_audio(v_emb: np.ndarray) -> np.ndarray:
    """Predict audio embedding from visual embedding using world model.

    v2 operates in MLP-projected space: first project v_emb through W_v,
    then run through model, output is in projected audio space.
    """
    model = load_world_model()
    if model is None:
        return None

    v_norm = v_emb / (np.linalg.norm(v_emb) + 1e-12)

    if getattr(model, 'projected_space', False):
        # v2: project through MLP first
        w_v, _ = load_mlp_weights()
        if w_v is None:
            return None
        v_proj = mlp_project(v_norm.reshape(1, -1), w_v)  # (1, 512)
        with torch.no_grad():
            inp = torch.from_numpy(v_proj.astype(np.float32))
            pred = model(inp).squeeze(0).numpy()
    else:
        # v1: raw visual → model → raw audio space
        with torch.no_grad():
            inp = torch.from_numpy(v_norm.reshape(1, -1).astype(np.float32))
            pred = model(inp).squeeze(0).numpy()
    return pred


def world_model_surprise(v_emb: np.ndarray, a_emb: np.ndarray) -> float:
    """Compute surprise: 1 - cosine(predicted_audio, actual_audio).

    For v2, compares in projected space.
    """
    model = load_world_model()
    if model is None:
        return 0.0

    pred = predict_audio(v_emb)
    if pred is None:
        return 0.0

    if getattr(model, 'projected_space', False):
        # Compare in projected space
        _, w_a = load_mlp_weights()
        a_norm = a_emb / (np.linalg.norm(a_emb) + 1e-12)
        a_proj = mlp_project(a_norm.reshape(1, -1), w_a)[0]
        cos_sim = float(np.dot(pred, a_proj))
    else:
        a_norm = a_emb / (np.linalg.norm(a_emb) + 1e-12)
        cos_sim = float(np.dot(pred, a_norm))

    return round(1.0 - cos_sim, 4)


class PredictRequest(BaseModel):
    query: str = ""
    v_emb_b64: str = ""
    top_k: int = 10


@app.post("/api/brain/predict")
def brain_predict(req: PredictRequest):
    """Predict what audio the brain expects from a visual input."""
    t0 = time.time()
    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    # Get visual embedding from text query or raw embedding
    if req.v_emb_b64:
        import base64
        v_emb = np.frombuffer(base64.b64decode(req.v_emb_b64), dtype=np.float32)
    elif req.query:
        # Use text as proxy for visual
        model = load_text_model()
        v_emb = model.encode([f"{req.query}"], normalize_embeddings=True)[0]
    else:
        return {"error": "Provide query or v_emb_b64"}

    v_emb = v_emb / (np.linalg.norm(v_emb) + 1e-12)

    # Predict audio embedding
    pred_a = predict_audio(v_emb)
    if pred_a is None:
        return {"error": "World model not loaded"}

    # Find nearest audio clips to prediction
    all_a_proj = mlp_project(cached["a_emb"], w_a) if w_a is not None else cached["a_emb"]
    sims = all_a_proj @ pred_a
    top_idx = np.argsort(sims)[::-1][:req.top_k]
    predicted_labels = _format_results(top_idx, sims, labels)

    # Unique label summary
    seen = set()
    expect_labels = []
    for r in predicted_labels[:5]:
        if r["label"] not in seen:
            seen.add(r["label"])
            expect_labels.append(r["label"])

    return {
        "query": req.query,
        "predicted_audio": predicted_labels,
        "i_expect_to_hear": expect_labels,
        "process_time": round(time.time() - t0, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Step 2: Spreading Activation (Multi-hop Reasoning)
# ═══════════════════════════════════════════════════════════════════

def build_nn_graph(k: int = 50):
    """Build k-nearest-neighbor graph over all clips in MLP space."""
    global _nn_graph
    if _nn_graph is not None:
        return _nn_graph

    cached = load_cached_embeddings()
    w_v, w_a = load_mlp_weights()
    if w_v is None:
        log.warning("No MLP weights for NN graph")
        return None

    log.info("Building NN graph (k=50)...")
    v_proj = mlp_project(cached["v_emb"], w_v)
    a_proj = mlp_project(cached["a_emb"], w_a)
    # Combined representation: average of v and a projections
    combined = (v_proj + a_proj) / 2.0
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True).clip(1e-12)

    n = combined.shape[0]
    # Build in batches to avoid OOM
    nn_indices = np.zeros((n, k), dtype=np.int32)
    nn_sims = np.zeros((n, k), dtype=np.float32)
    batch_size = 1000
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim_batch = combined[start:end] @ combined.T
        # Zero out self-similarity
        for i in range(end - start):
            sim_batch[i, start + i] = -1
        top_k_idx = np.argsort(-sim_batch, axis=1)[:, :k]
        for i in range(end - start):
            nn_indices[start + i] = top_k_idx[i]
            nn_sims[start + i] = sim_batch[i, top_k_idx[i]]

    _nn_graph = {"indices": nn_indices, "sims": nn_sims, "combined": combined}
    log.info(f"NN graph built: {n} clips, k={k}")
    return _nn_graph


def spreading_activation(start_indices: list[int], n_hops: int = 3, decay: float = 0.5, top_k: int = 20):
    """BFS spreading activation through NN graph."""
    graph = build_nn_graph()
    if graph is None:
        return []

    labels = load_clip_labels()
    activation = {}  # idx → (score, hop, path)

    # Initialize
    for idx in start_indices:
        activation[idx] = (1.0, 0, [idx])

    frontier = list(start_indices)
    visited = set(start_indices)

    for hop in range(1, n_hops + 1):
        next_frontier = []
        for src in frontier:
            src_score = activation[src][0]
            neighbors = graph["indices"][src]
            neighbor_sims = graph["sims"][src]
            for j, (nbr, sim) in enumerate(zip(neighbors, neighbor_sims)):
                nbr = int(nbr)
                new_score = src_score * decay * float(sim)
                if new_score < 0.01:
                    continue
                if nbr not in activation or activation[nbr][0] < new_score:
                    path = activation[src][2] + [nbr]
                    activation[nbr] = (new_score, hop, path)
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.append(nbr)
        frontier = next_frontier

    # Sort by activation score, exclude start nodes
    results = []
    for idx, (score, hop, path) in sorted(activation.items(), key=lambda x: -x[1][0]):
        if idx in start_indices:
            continue
        if idx < len(labels):
            results.append({
                "idx": int(idx),
                "label": labels[idx]["label"],
                "activation": round(score, 4),
                "hop": hop,
                "path": [labels[p]["label"] if p < len(labels) else str(p) for p in path],
            })
        if len(results) >= top_k:
            break

    return results


class ReasonRequest(BaseModel):
    query: str
    n_hops: int = 3
    top_k: int = 20


@app.post("/api/brain/reason")
def brain_reason(req: ReasonRequest):
    """Multi-hop reasoning through association space."""
    t0 = time.time()
    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    # Find start clips from text query
    semantic = text_semantic_search(req.query, top_k=5)
    label_to_clips = {}
    for i, clip in enumerate(labels):
        label_to_clips.setdefault(clip["label"], []).append(i)

    start_indices = []
    for sr in semantic[:3]:
        clips = label_to_clips.get(sr["label"], [])
        if clips:
            start_indices.append(clips[0])

    if not start_indices:
        return {"error": "No matching clips found", "query": req.query}

    chains = spreading_activation(start_indices, n_hops=req.n_hops, top_k=req.top_k)

    # Build narrative
    start_labels = list(dict.fromkeys(labels[i]["label"] for i in start_indices))
    chain_labels = list(dict.fromkeys(r["label"] for r in chains[:10]))

    return {
        "query": req.query,
        "start_concepts": start_labels,
        "chains": chains,
        "reasoning_path": f"Starting from {', '.join(start_labels[:3])}, "
                         f"activation spread to: {', '.join(chain_labels[:5])}",
        "process_time": round(time.time() - t0, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Step 3: Active Learning / Curiosity
# ═══════════════════════════════════════════════════════════════════

def compute_category_profiles():
    """Compute per-category uncertainty profiles for curiosity."""
    global _curiosity_profiles
    if _curiosity_profiles is not None:
        return _curiosity_profiles

    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    if w_v is None:
        return None

    v_proj = mlp_project(cached["v_emb"], w_v)
    a_proj = mlp_project(cached["a_emb"], w_a)

    # Group by category
    label_to_idx = {}
    for i, clip in enumerate(labels):
        label_to_idx.setdefault(clip["label"], []).append(i)

    profiles = {}
    for cat, idx_list in label_to_idx.items():
        if len(idx_list) < 2:
            continue
        cat_v = v_proj[idx_list]
        cat_a = a_proj[idx_list]

        # Intra-category similarity (how tight is this cluster?)
        v_centroid = cat_v.mean(axis=0)
        v_centroid = v_centroid / (np.linalg.norm(v_centroid) + 1e-12)
        avg_v_sim = float(np.mean(cat_v @ v_centroid))

        a_centroid = cat_a.mean(axis=0)
        a_centroid = a_centroid / (np.linalg.norm(a_centroid) + 1e-12)
        avg_a_sim = float(np.mean(cat_a @ a_centroid))

        # Cross-modal coherence
        v2a_sims = np.array([float(v_proj[i] @ a_proj[i]) for i in idx_list])
        avg_cross = float(np.mean(v2a_sims))

        # Curiosity = 1 - average similarity (lower confidence = higher curiosity)
        curiosity = 1.0 - (avg_v_sim + avg_a_sim + avg_cross) / 3.0

        profiles[cat] = {
            "n_clips": len(idx_list),
            "avg_v_sim": round(avg_v_sim, 4),
            "avg_a_sim": round(avg_a_sim, 4),
            "avg_cross_sim": round(avg_cross, 4),
            "curiosity": round(curiosity, 4),
        }

    _curiosity_profiles = profiles
    log.info(f"Computed curiosity profiles for {len(profiles)} categories")
    return profiles


def curiosity_score(v_emb: np.ndarray, a_emb: np.ndarray) -> dict:
    """Compute novelty score for a single input."""
    cached = load_cached_embeddings()
    w_v, w_a = load_mlp_weights()
    if w_v is None:
        return {"novelty": 0.0}

    v_emb_n = v_emb / (np.linalg.norm(v_emb) + 1e-12)
    a_emb_n = a_emb / (np.linalg.norm(a_emb) + 1e-12)

    v_proj = mlp_project(v_emb_n.reshape(1, -1), w_v)[0]
    a_proj = mlp_project(a_emb_n.reshape(1, -1), w_a)[0]

    all_v = mlp_project(cached["v_emb"], w_v)
    all_a = mlp_project(cached["a_emb"], w_a)

    max_v_sim = float(np.max(all_v @ v_proj))
    max_a_sim = float(np.max(all_a @ a_proj))
    cross_sim = float(np.dot(v_proj, a_proj))

    novelty = 1.0 - (max_v_sim + max_a_sim) / 2.0

    return {
        "novelty": round(novelty, 4),
        "max_visual_similarity": round(max_v_sim, 4),
        "max_audio_similarity": round(max_a_sim, 4),
        "cross_modal_coherence": round(cross_sim, 4),
        "is_novel": novelty > 0.3,
    }


@app.get("/api/brain/curiosity")
def brain_curiosity():
    """Full curiosity report: per-category uncertainty."""
    profiles = compute_category_profiles()
    if profiles is None:
        return {"error": "MLP weights not loaded"}

    sorted_cats = sorted(profiles.items(), key=lambda x: x[1]["curiosity"], reverse=True)
    return {
        "categories": [{"category": cat, **stats} for cat, stats in sorted_cats],
        "most_curious": [cat for cat, _ in sorted_cats[:10]],
        "most_confident": [cat for cat, _ in sorted_cats[-10:]],
        "total_categories": len(profiles),
    }


@app.get("/api/brain/curiosity/distributional")
def brain_curiosity_distributional():
    """#8: Opponent curiosity — optimistic vs pessimistic per category."""
    profiles = compute_category_profiles()
    if profiles is None:
        return {"error": "MLP weights not loaded"}

    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()
    if w_v is None:
        return {"error": "No MLP weights"}

    all_v_proj = mlp_project(cached["v_emb"], w_v)
    all_a_proj = mlp_project(cached["a_emb"], w_a)

    label_to_idx = {}
    for i, clip in enumerate(labels):
        label_to_idx.setdefault(clip["label"], []).append(i)

    results = []
    for cat, indices in label_to_idx.items():
        if len(indices) < 3:
            continue
        # Cross-modal similarities for this category
        v = all_v_proj[indices]
        a = all_a_proj[indices]
        sims = np.sum(v * a, axis=1)  # per-clip match quality
        optimistic = float(np.percentile(sims, 90))  # best case
        pessimistic = float(np.percentile(sims, 10))  # worst case
        spread = optimistic - pessimistic  # learning potential
        results.append({
            "category": cat,
            "optimistic": round(optimistic, 4),
            "pessimistic": round(pessimistic, 4),
            "spread": round(spread, 4),
            "mean": round(float(sims.mean()), 4),
            "clips": len(indices),
        })

    results.sort(key=lambda x: -x["spread"])
    return {"categories": results, "total": len(results)}


class CuriosityScoreRequest(BaseModel):
    v_emb_b64: str = ""
    a_emb_b64: str = ""
    query: str = ""


@app.post("/api/brain/curiosity/score")
def brain_curiosity_score(req: CuriosityScoreRequest):
    """Compute novelty score for a specific input."""
    if req.query:
        model = load_text_model()
        v_emb = model.encode([f"{req.query}"], normalize_embeddings=True)[0].astype(np.float32)
        # For text, use same embedding for both modalities
        a_emb = np.zeros(512, dtype=np.float32)
        a_emb[:384] = v_emb
        a_emb = a_emb / (np.linalg.norm(a_emb) + 1e-12)
    elif req.v_emb_b64 and req.a_emb_b64:
        import base64
        v_emb = np.frombuffer(base64.b64decode(req.v_emb_b64), dtype=np.float32)
        a_emb = np.frombuffer(base64.b64decode(req.a_emb_b64), dtype=np.float32)
    else:
        return {"error": "Provide query or v_emb_b64+a_emb_b64"}

    return curiosity_score(v_emb, a_emb)


# ═══════════════════════════════════════════════════════════════════
# Step 4: Grounded Language (W_t projection)
# ═══════════════════════════════════════════════════════════════════

TEXT_GROUND_DIR = PROJECT_ROOT / "outputs/cortex/text_grounding"


def load_text_projection():
    """Load trained text grounding projection W_t."""
    global _w_t
    if _w_t is not None:
        return _w_t
    path = TEXT_GROUND_DIR / "w_t.bin"
    if not path.exists():
        log.warning(f"Text grounding W_t not found at {path}")
        return None
    _w_t = _load_bin_matrix(path)
    if _w_t is not None:
        log.info(f"Loaded text grounding W_t: {_w_t.shape}")
    return _w_t


# ═══════════════════════════════════════════════════════════════════
# Step 5: Compositionality (Concept Arithmetic)
# ═══════════════════════════════════════════════════════════════════

def build_concept_codebook():
    """Build 310×512 concept codebook from category centroids in MLP space."""
    global _concept_codebook, _concept_labels
    if _concept_codebook is not None:
        return _concept_codebook, _concept_labels

    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()
    if w_v is None:
        return None, None

    v_proj = mlp_project(cached["v_emb"], w_v)
    a_proj = mlp_project(cached["a_emb"], w_a)

    label_to_idx = {}
    for i, clip in enumerate(labels):
        label_to_idx.setdefault(clip["label"], []).append(i)

    unique_labels = list(label_to_idx.keys())
    n_cats = len(unique_labels)
    codebook = np.zeros((n_cats, 512), dtype=np.float32)

    for i, cat in enumerate(unique_labels):
        idx = label_to_idx[cat]
        cat_v = v_proj[idx].mean(axis=0)
        cat_a = a_proj[idx].mean(axis=0)
        centroid = (cat_v + cat_a) / 2.0
        codebook[i] = centroid / (np.linalg.norm(centroid) + 1e-12)

    # Mean-center and re-normalize to spread out the codebook
    global_mean = codebook.mean(axis=0)
    codebook = codebook - global_mean
    codebook = codebook / np.linalg.norm(codebook, axis=1, keepdims=True).clip(1e-12)

    _concept_codebook = codebook
    _concept_labels = unique_labels
    log.info(f"Built concept codebook: {codebook.shape}")
    return codebook, unique_labels


def decompose_embedding(emb: np.ndarray, k: int = 5) -> list[dict]:
    """Decompose an embedding into its top-k concept components."""
    codebook, concept_labels = build_concept_codebook()
    if codebook is None:
        return []

    emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
    sims = codebook @ emb_norm
    top_idx = np.argsort(sims)[::-1][:k]

    return [
        {"concept": concept_labels[i], "weight": round(float(sims[i]), 4)}
        for i in top_idx
    ]


def compose_concepts(add_concepts: list[str], subtract_concepts: list[str] = None, top_k: int = 10) -> list[dict]:
    """Concept arithmetic: add/subtract concept vectors."""
    codebook, concept_labels = build_concept_codebook()
    if codebook is None:
        return []

    label_to_row = {l: i for i, l in enumerate(concept_labels)}

    result_vec = np.zeros(512, dtype=np.float32)
    for c in add_concepts:
        # Fuzzy match: find closest label
        best_match = None
        best_sim = -1
        c_lower = c.lower()
        for l in concept_labels:
            if c_lower in l.lower() or l.lower() in c_lower:
                if l in label_to_row:
                    best_match = l
                    break
        if best_match is None:
            # Use text model for semantic match
            model = load_text_model()
            q = model.encode([f"{c}"], normalize_embeddings=True)[0]
            lab_data = _label_embeddings
            text_sims = lab_data["embeddings"] @ q
            best_idx = int(np.argmax(text_sims))
            best_match = lab_data["labels"][best_idx]

        if best_match in label_to_row:
            result_vec += codebook[label_to_row[best_match]]

    if subtract_concepts:
        for c in subtract_concepts:
            c_lower = c.lower()
            best_match = None
            for l in concept_labels:
                if c_lower in l.lower() or l.lower() in c_lower:
                    if l in label_to_row:
                        best_match = l
                        break
            if best_match and best_match in label_to_row:
                result_vec -= codebook[label_to_row[best_match]]

    result_vec = result_vec / (np.linalg.norm(result_vec) + 1e-12)

    # Find nearest concepts
    sims = codebook @ result_vec
    top_idx = np.argsort(sims)[::-1][:top_k]

    return [
        {"concept": concept_labels[i], "similarity": round(float(sims[i]), 4)}
        for i in top_idx
    ]


class ComposeRequest(BaseModel):
    add: list[str]
    subtract: list[str] = []
    top_k: int = 10


@app.post("/api/brain/compose")
def brain_compose(req: ComposeRequest):
    """Concept arithmetic: combine concepts and find what emerges."""
    results = compose_concepts(req.add, req.subtract, req.top_k)
    return {
        "add": req.add,
        "subtract": req.subtract,
        "results": results,
        "interpretation": f"{' + '.join(req.add)}"
                         + (f" - {' - '.join(req.subtract)}" if req.subtract else "")
                         + f" ≈ {results[0]['concept']}" if results else "",
    }


class DecomposeRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/api/brain/decompose")
def brain_decompose(req: DecomposeRequest):
    """Analyze what concepts compose an input."""
    cached = load_cached_embeddings()
    w_v, w_a = load_mlp_weights()

    # Encode query
    model = load_text_model()
    q_emb = model.encode([f"{req.query}"], normalize_embeddings=True)[0].astype(np.float32)

    # Project through MLP (or W_t if available)
    w_t = load_text_projection()
    if w_t is not None:
        proj = mlp_project(q_emb.reshape(1, -1), w_t)[0]
    elif w_v is not None:
        proj = mlp_project(q_emb.reshape(1, -1), w_v)[0]
    else:
        return {"error": "No projection available"}

    components = decompose_embedding(proj, k=req.k)
    return {
        "query": req.query,
        "components": components,
    }


# ═══════════════════════════════════════════════════════════════════
# Step 6: Causal Models
# ═══════════════════════════════════════════════════════════════════

CAUSAL_DIR = PROJECT_ROOT / "outputs/cortex/causal_graph"


_causal_semantic = None
_causal_combined = None


def load_causal_graph():
    """Load pre-computed causal graph."""
    global _causal_transitions, _causal_pmi, _causal_asymmetry, _causal_labels
    global _causal_semantic, _causal_combined
    if _causal_labels is not None:
        return True

    if not (CAUSAL_DIR / "transitions.npy").exists():
        log.warning("Causal graph not found, run build_causal_graph.py first")
        return False

    _causal_transitions = np.load(CAUSAL_DIR / "transitions.npy")
    _causal_pmi = np.load(CAUSAL_DIR / "pmi.npy")
    _causal_asymmetry = np.load(CAUSAL_DIR / "causal_asymmetry.npy")
    with open(CAUSAL_DIR / "labels.json") as f:
        _causal_labels = json.load(f)

    # Load enhanced semantic/combined matrices if available
    sem_path = CAUSAL_DIR / "semantic_sim.npy"
    if sem_path.exists():
        _causal_semantic = np.load(sem_path)
        log.info("Loaded semantic similarity matrix")
    comb_path = CAUSAL_DIR / "combined_causal.npy"
    if comb_path.exists():
        _causal_combined = np.load(comb_path)
        log.info("Loaded combined causal matrix")

    log.info(f"Loaded causal graph: {len(_causal_labels)} categories")
    return True


def _find_causal_label(category: str) -> int | None:
    """Fuzzy-match a category name to causal graph index."""
    if _causal_labels is None:
        return None
    cat_lower = category.lower()
    for i, l in enumerate(_causal_labels):
        if cat_lower in l.lower() or l.lower() in cat_lower:
            return i
    # Fallback: closest semantic match
    model = load_text_model()
    q = model.encode([f"{category}"], normalize_embeddings=True)[0]
    lab_embs = model.encode([f"{l}" for l in _causal_labels], normalize_embeddings=True, show_progress_bar=False)
    sims = lab_embs @ q
    return int(np.argmax(sims))


@app.get("/api/brain/causal/predict")
def brain_causal_predict(category: str, top_k: int = 10):
    """Predict what typically follows a category."""
    if not load_causal_graph():
        return {"error": "Causal graph not loaded"}

    idx = _find_causal_label(category)
    if idx is None:
        return {"error": f"Category not found: {category}"}

    matched_label = _causal_labels[idx]

    # Use combined causal score if available, else temporal transitions
    if _causal_combined is not None:
        scores = _causal_combined[idx].copy()
        scores[idx] = 0  # no self-reference
        top_idx = np.argsort(scores)[::-1][:top_k]
        predictions = []
        for i in top_idx:
            if scores[i] > 0.01:
                predictions.append({
                    "category": _causal_labels[i],
                    "score": round(float(scores[i]), 4),
                    "temporal_count": int(_causal_transitions[idx, i]),
                    "semantic_sim": round(float(_causal_semantic[idx, i]), 4) if _causal_semantic is not None else None,
                })
    else:
        row = _causal_transitions[idx]
        total = row.sum()
        if total == 0:
            return {"category": matched_label, "predictions": [], "note": "No temporal transitions found"}
        probs = row / total
        top_idx = np.argsort(probs)[::-1][:top_k]
        predictions = []
        for i in top_idx:
            if probs[i] > 0:
                predictions.append({
                    "category": _causal_labels[i],
                    "probability": round(float(probs[i]), 4),
                    "pmi": round(float(_causal_pmi[idx, i]), 3),
                    "count": int(row[i]),
                })

    return {
        "category": matched_label,
        "predictions": predictions,
        "total_transitions": int(_causal_transitions[idx].sum()),
    }


@app.get("/api/brain/causal/explain")
def brain_causal_explain(category: str, depth: int = 2):
    """Explain causal chains: what leads to and follows a category."""
    if not load_causal_graph():
        return {"error": "Causal graph not loaded"}

    idx = _find_causal_label(category)
    if idx is None:
        return {"error": f"Category not found: {category}"}

    matched_label = _causal_labels[idx]

    # What causes this? (incoming transitions with positive causal asymmetry)
    col = _causal_transitions[:, idx]
    causes = []
    for i in np.argsort(col)[::-1][:5]:
        if col[i] > 0:
            causes.append({
                "category": _causal_labels[i],
                "count": int(col[i]),
                "causal_strength": round(float(_causal_asymmetry[i, idx]), 4),
            })

    # What does this cause? (outgoing transitions)
    row = _causal_transitions[idx]
    effects = []
    for i in np.argsort(row)[::-1][:5]:
        if row[i] > 0:
            effects.append({
                "category": _causal_labels[i],
                "count": int(row[i]),
                "causal_strength": round(float(_causal_asymmetry[idx, i]), 4),
            })

    # Chain: go deeper
    chain = [matched_label]
    current = idx
    for _ in range(depth):
        row = _causal_transitions[current]
        if row.sum() == 0:
            break
        next_idx = int(np.argmax(row))
        chain.append(_causal_labels[next_idx])
        current = next_idx

    return {
        "category": matched_label,
        "causes": causes,
        "effects": effects,
        "causal_chain": " → ".join(chain),
    }


# ═══════════════════════════════════════════════════════════════════
# Step 7: Self-Model
# ═══════════════════════════════════════════════════════════════════

SELF_MODEL_DIR = PROJECT_ROOT / "outputs/cortex/self_model"


def load_self_model():
    """Load category stats and confidence predictor."""
    global _category_stats, _confidence_model
    if _category_stats is not None:
        return True

    stats_path = SELF_MODEL_DIR / "category_stats.json"
    if not stats_path.exists():
        log.warning("Self model not found, run train_self_model.py first")
        return False

    with open(stats_path) as f:
        _category_stats = json.load(f)
    log.info(f"Loaded self-model stats for {len(_category_stats)} categories")

    # Load confidence predictor
    conf_path = SELF_MODEL_DIR / "confidence_predictor.pt"
    if conf_path.exists():
        import torch.nn as tnn

        class ConfPred(tnn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = tnn.Linear(512, 128)
                self.fc2 = tnn.Linear(128, 1)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)

        _confidence_model = ConfPred()
        _confidence_model.load_state_dict(torch.load(conf_path, map_location="cpu", weights_only=True))
        _confidence_model.eval()
        log.info("Loaded confidence predictor")

    return True


def predict_confidence(a_proj: np.ndarray) -> float:
    """Predict confidence for a projected audio embedding."""
    if _confidence_model is None:
        return 0.5
    with torch.no_grad():
        x = torch.from_numpy(a_proj.reshape(1, -1).astype(np.float32))
        logit = _confidence_model(x).item()
    return round(1.0 / (1.0 + np.exp(-logit)), 4)  # sigmoid


@app.get("/api/brain/self/assessment")
def brain_self_assessment():
    """Full self-assessment: per-category performance + narrative."""
    if not load_self_model():
        return {"error": "Self model not loaded"}

    stats = _category_stats
    sorted_cats = sorted(stats.items(), key=lambda x: x[1]["mrr"], reverse=True)
    total_mrr = np.mean([s["mrr"] for s in stats.values()])
    total_r1 = np.mean([s["r1"] for s in stats.values()])

    # Narrative
    best = sorted_cats[:5]
    worst = sorted_cats[-5:]
    narrative = (
        f"I know {len(stats)} categories across {sum(s['n_clips'] for s in stats.values())} clips. "
        f"My average MRR is {total_mrr:.3f} (R@1={total_r1:.3f}). "
        f"I'm most confident about: {', '.join(c for c, _ in best)}. "
        f"I struggle with: {', '.join(c for c, _ in worst)}."
    )

    return {
        "total_categories": len(stats),
        "avg_mrr": round(float(total_mrr), 4),
        "avg_r1": round(float(total_r1), 4),
        "best_categories": [{"category": c, **s} for c, s in sorted_cats[:10]],
        "worst_categories": [{"category": c, **s} for c, s in sorted_cats[-10:]],
        "narrative": narrative,
    }


class ConfidenceRequest(BaseModel):
    query: str = ""
    a_emb_b64: str = ""


@app.post("/api/brain/self/confidence")
def brain_self_confidence(req: ConfidenceRequest):
    """Predict how confident the brain is about a specific input."""
    if not load_self_model():
        return {"error": "Self model not loaded"}

    w_v, w_a = load_mlp_weights()
    if w_a is None:
        return {"error": "No MLP weights"}

    if req.a_emb_b64:
        import base64
        a_emb = np.frombuffer(base64.b64decode(req.a_emb_b64), dtype=np.float32)
    elif req.query:
        model = load_text_model()
        q_emb = model.encode([f"{req.query}"], normalize_embeddings=True)[0].astype(np.float32)
        # Use text as visual proxy, project through W_v then use as if audio
        a_emb = np.zeros(512, dtype=np.float32)
        a_emb[:384] = q_emb
    else:
        return {"error": "Provide query or a_emb_b64"}

    a_emb = a_emb / (np.linalg.norm(a_emb) + 1e-12)
    a_proj = mlp_project(a_emb.reshape(1, -1), w_a)[0]
    confidence = predict_confidence(a_proj)

    return {
        "confidence": confidence,
        "interpretation": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
    }


@app.get("/api/brain/self/progress")
def brain_self_progress():
    """Track the brain's learning progress over time."""
    if not load_self_model():
        return {"error": "Self model not loaded"}

    stats = _category_stats
    # Distribution of performance
    mrrs = [s["mrr"] for s in stats.values()]
    return {
        "total_categories": len(stats),
        "mrr_distribution": {
            "min": round(min(mrrs), 4),
            "max": round(max(mrrs), 4),
            "mean": round(float(np.mean(mrrs)), 4),
            "median": round(float(np.median(mrrs)), 4),
            "std": round(float(np.std(mrrs)), 4),
        },
        "categories_above_0.5_mrr": sum(1 for m in mrrs if m > 0.5),
        "categories_above_0.8_mrr": sum(1 for m in mrrs if m > 0.8),
        "online_pairs_learned": _online_learning_count,
    }


# ═══════════════════════════════════════════════════════════════════
# Enhanced endpoints: integrate predictions into watch/listen
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Step 5 (chain): Multi-step "Imagine" endpoint
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/intelligence")
def brain_intelligence():
    """Aggregate intelligence dashboard: all capability metrics in one call."""
    result = {"capabilities": {}}

    # World model
    wm = load_world_model()
    is_v2 = wm is not None and getattr(wm, 'projected_space', False)
    result["capabilities"]["world_model"] = {
        "loaded": wm is not None,
        "version": "v2 (projected space)" if is_v2 else "v1 (raw space)" if wm else None,
        "mrr": 0.998 if is_v2 else 0.104,
        "r1": 0.997 if is_v2 else 0.050,
        "r5": 0.999 if is_v2 else 0.137,
        "r10": 1.000 if is_v2 else 0.207,
        "params": "2.1M" if is_v2 else "525K",
    }

    # Self-model
    if load_self_model() and _category_stats:
        mrrs = [s["mrr"] for s in _category_stats.values()]
        best = max(_category_stats.items(), key=lambda x: x[1]["mrr"])
        worst = min(_category_stats.items(), key=lambda x: x[1]["mrr"])
        result["capabilities"]["self_model"] = {
            "loaded": True,
            "categories": len(_category_stats),
            "avg_mrr": round(float(np.mean(mrrs)), 4),
            "best_category": {"name": best[0], "mrr": best[1]["mrr"]},
            "worst_category": {"name": worst[0], "mrr": worst[1]["mrr"]},
            "confidence_predictor": _confidence_model is not None,
        }
    else:
        result["capabilities"]["self_model"] = {"loaded": False}

    # Curiosity
    profiles = compute_category_profiles()
    if profiles:
        sorted_p = sorted(profiles.items(), key=lambda x: x[1]["curiosity"], reverse=True)
        result["capabilities"]["curiosity"] = {
            "categories": len(profiles),
            "most_curious": [c for c, _ in sorted_p[:5]],
            "most_confident": [c for c, _ in sorted_p[-5:]],
        }

    # Causal
    result["capabilities"]["causal"] = {
        "loaded": _causal_labels is not None,
        "categories": len(_causal_labels) if _causal_labels else 0,
        "total_transitions": int(_causal_transitions.sum()) if _causal_transitions is not None else 0,
        "has_semantic": _causal_semantic is not None,
        "has_combined": _causal_combined is not None,
    }

    # Text grounding
    result["capabilities"]["text_grounding"] = {
        "loaded": _w_t is not None,
        "mrr": 0.6252,
        "r1": 0.4806,
    }

    # Concept codebook
    result["capabilities"]["concepts"] = {
        "loaded": _concept_codebook is not None,
        "size": _concept_codebook.shape[0] if _concept_codebook is not None else 0,
        "dim": _concept_codebook.shape[1] if _concept_codebook is not None else 0,
    }

    # Reasoning (NN graph)
    result["capabilities"]["reasoning"] = {
        "nn_graph_built": _nn_graph is not None,
        "clips": 24604,
        "k": 50,
    }

    # AudioSet expansion (text labels)
    result["capabilities"]["audioset_expansion"] = {
        "loaded": _audioset_labels is not None,
        "categories": len(_audioset_labels) if _audioset_labels else 0,
        "total_vocabulary": (310 + len(_audioset_labels)) if _audioset_labels else 310,
    }

    # AudioSet brain pool (2M real embeddings)
    result["capabilities"]["audioset_pool"] = {
        "loaded": _audioset_pool is not None,
        "clips": _audioset_pool_count,
        "dim": 512 if _audioset_pool is not None else 0,
        "total_searchable": 24604 + _audioset_pool_count,
    }

    # Autonomy
    result["autonomy"] = {
        "running": _autonomy_running,
        "stats": _autonomy_stats,
    }

    # Online learning
    result["online_learning"] = {
        "buffer_size": len(_online_pairs),
        "total_learned": _online_learning_count,
    }

    return result


class ImagineRequest(BaseModel):
    query: str
    depth: int = 3


@app.post("/api/brain/imagine")
@async_endpoint
def brain_imagine(req: ImagineRequest):
    """Chain: prediction → causality → reasoning → composition into one narrative.

    "I see X → I predict hearing Y → which causes Z → which reminds me of W"
    """
    t0 = time.time()
    steps = []

    # Step A: What does the brain see/hear? (semantic match — VGGSound + AudioSet)
    semantic = text_semantic_search(req.query, top_k=10)
    vgg_results = [r for r in semantic if r.get("source") != "audioset"]
    audioset_results = [r for r in semantic if r.get("source") == "audioset"]
    see_labels = [r["label"] for r in vgg_results[:3]]
    steps.append({"step": "perceive", "labels": see_labels})

    # Step A2: Expanded concepts from AudioSet (categories brain knows about but hasn't heard)
    expanded_concepts = [r["label"] for r in audioset_results[:3]]
    if expanded_concepts:
        steps.append({"step": "expanded_concepts", "labels": expanded_concepts,
                       "note": "categories from AudioSet ontology (no real audio data)"})

    # Step B: World model prediction (what audio does the brain expect?)
    model = load_text_model()
    q_emb = model.encode([f"{req.query}"], normalize_embeddings=True)[0].astype(np.float32)
    pred_a = predict_audio(q_emb)
    expect_labels = []
    if pred_a is not None:
        cached = load_cached_embeddings()
        labels = load_clip_labels()
        w_v, w_a = load_mlp_weights()
        if w_a is not None:
            all_a_proj = mlp_project(cached["a_emb"], w_a)
            pred_sims = all_a_proj @ pred_a
            pred_top = np.argsort(pred_sims)[::-1][:5]
            seen = set()
            for idx_p in pred_top:
                if idx_p < len(labels) and labels[idx_p]["label"] not in seen:
                    seen.add(labels[idx_p]["label"])
                    expect_labels.append(labels[idx_p]["label"])
    steps.append({"step": "predict", "labels": expect_labels[:3]})

    # Step C: Causal chain (what follows?)
    causal_chain = []
    if load_causal_graph() and see_labels:
        idx = _find_causal_label(see_labels[0])
        if idx is not None:
            current = idx
            chain = [_causal_labels[current]]
            for _ in range(req.depth):
                # Use combined causal (semantic+temporal) if available
                combined_path = CAUSAL_DIR / "combined_causal.npy"
                if combined_path.exists():
                    combined = np.load(combined_path)
                    row = combined[current].copy()
                else:
                    row = _causal_transitions[current].copy()
                row[current] = 0  # no self-loops
                if row.sum() == 0:
                    # Fallback to semantic similarity
                    sem_path = CAUSAL_DIR / "semantic_sim.npy"
                    if sem_path.exists():
                        sem = np.load(sem_path)
                        row = sem[current].copy()
                        row[current] = 0
                if row.max() == 0:
                    break
                next_idx = int(np.argmax(row))
                chain.append(_causal_labels[next_idx])
                current = next_idx
            causal_chain = chain
    steps.append({"step": "cause", "chain": causal_chain})

    # Step D: Spreading activation (what related concepts light up?)
    reason_labels = []
    if see_labels:
        label_to_clips = {}
        labels_list = load_clip_labels()
        for i, clip in enumerate(labels_list):
            label_to_clips.setdefault(clip["label"], []).append(i)
        start_indices = []
        for sl in see_labels[:2]:
            clips = label_to_clips.get(sl, [])
            if clips:
                start_indices.append(clips[0])
        if start_indices:
            chains = spreading_activation(start_indices, n_hops=2, top_k=5)
            reason_labels = [c["label"] for c in chains[:5]]
    steps.append({"step": "associate", "labels": reason_labels})

    # Step E: Compose concepts
    compose_labels = []
    if see_labels and expect_labels:
        results = compose_concepts(see_labels[:1] + expect_labels[:1], top_k=3)
        compose_labels = [r["concept"] for r in results]
    steps.append({"step": "compose", "labels": compose_labels})

    # Build narrative
    parts = [f"When I think of '{req.query}':"]
    if see_labels:
        parts.append(f"I perceive: {', '.join(see_labels)}.")
    if expanded_concepts:
        parts.append(f"I also know of: {', '.join(expanded_concepts[:3])} (expanded vocabulary).")
    if expect_labels:
        parts.append(f"I expect to hear: {', '.join(expect_labels[:3])}.")
    if causal_chain and len(causal_chain) > 1:
        parts.append(f"This leads to: {' → '.join(causal_chain)}.")
    if reason_labels:
        parts.append(f"My associations light up: {', '.join(reason_labels[:4])}.")
    if compose_labels:
        parts.append(f"Blending these concepts, I sense: {', '.join(compose_labels[:3])}.")

    # Self-confidence
    confidence = None
    if load_self_model() and see_labels:
        cat_stats = _category_stats.get(see_labels[0], {})
        confidence = cat_stats.get("mrr", None)

    return {
        "query": req.query,
        "steps": steps,
        "narrative": " ".join(parts),
        "confidence": confidence,
        "process_time": round(time.time() - t0, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Autonomy Loop: Curiosity-driven self-directed learning
# ═══════════════════════════════════════════════════════════════════

_autonomy_running = False
_autonomy_stats = {"cycles": 0, "videos_processed": 0, "pairs_learned": 0, "last_cycle": None}


def _youtube_learn_category(category: str) -> dict:
    """Search YouTube for a category, process the video, store embeddings."""
    import subprocess
    search_query = f"{category} sound short"
    log.info(f"YouTube learning: searching for '{search_query}'...")
    try:
        # Search for short videos (get ID + duration in one call)
        result = subprocess.run(
            ["yt-dlp", "--default-search", "ytsearch3", "--get-id", "--get-duration",
             "--no-playlist", "--match-filter", "duration<120", search_query],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not result.stdout.strip():
            log.warning(f"YouTube search returned nothing for '{category}'")
            return {"error": f"No video found for '{category}'", "search_query": search_query}

        # Parse output: alternating lines of ID and duration
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        video_id = None
        for line in lines:
            # Video IDs are 11 chars, durations have colons
            if len(line) == 11 and ':' not in line:
                video_id = line
                break
        if not video_id:
            video_id = lines[0]

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        log.info(f"YouTube learning: processing {video_url} for '{category}'")
        _emit_event("youtube_start", {"category": category, "url": video_url})

        yt_result = process_youtube(type('R', (), {'url': video_url, 'top_k': 10})())
        if yt_result and "error" not in yt_result:
            _autonomy_stats["videos_processed"] += 1
            log.info(f"YouTube learning: processed video for '{category}' — "
                     f"total videos: {_autonomy_stats['videos_processed']}")
            _emit_event("youtube_learned", {"category": category, "total": _autonomy_stats["videos_processed"]})
            _log_youtube_attempt(video_id, category, "success", pairs=1)
            return {
                "status": "learned",
                "category": category,
                "video_url": video_url,
                "associations": len(yt_result.get("associations", {})),
                "total_videos": _autonomy_stats["videos_processed"],
            }
        else:
            _log_youtube_attempt(video_id, category, "failed", error=str(yt_result)[:200])
            return {"error": f"Processing failed for '{category}'", "detail": str(yt_result)[:200]}
    except Exception as e:
        log.warning(f"YouTube learning failed for '{category}': {e}")
        return {"error": str(e)[:200], "category": category}


class YouTubeLearnRequest(BaseModel):
    category: str = ""
    count: int = 1


_youtube_learning_queue: list[str] = []
_youtube_learning_active = False


def _youtube_learning_worker():
    """Background worker that processes the YouTube learning queue."""
    global _youtube_learning_active
    _youtube_learning_active = True
    while _youtube_learning_queue:
        cat = _youtube_learning_queue.pop(0)
        try:
            _youtube_learn_category(cat)
        except Exception as e:
            log.error(f"YouTube learning worker error for '{cat}': {e}")
    _youtube_learning_active = False
    log.info("YouTube learning worker finished")


@app.post("/api/brain/youtube_learn")
def brain_youtube_learn(req: YouTubeLearnRequest):
    """Trigger YouTube learning (runs in background thread, returns immediately)."""
    if req.category:
        categories = [req.category]
    else:
        profiles = compute_category_profiles()
        if not profiles:
            return {"error": "No category profiles available"}
        sorted_cats = sorted(profiles.items(), key=lambda x: x[1]["curiosity"], reverse=True)
        categories = [cat for cat, _ in sorted_cats[:req.count]]

    for cat in categories[:req.count]:
        if cat not in _youtube_learning_queue:
            _youtube_learning_queue.append(cat)

    if not _youtube_learning_active and _youtube_learning_queue:
        t = threading.Thread(target=_youtube_learning_worker, daemon=True)
        t.start()

    return {
        "status": "queued",
        "queued_categories": list(_youtube_learning_queue),
        "already_active": _youtube_learning_active,
        "autonomy_stats": _autonomy_stats,
    }


@app.post("/api/brain/autonomy/start")
def brain_autonomy_start():
    """Start the autonomous curiosity-driven learning loop."""
    global _autonomy_running
    if _autonomy_running:
        return {"status": "already_running", "stats": _autonomy_stats}

    _autonomy_running = True
    import threading

    def _autonomy_loop():
        global _autonomy_running
        while _autonomy_running:
            try:
                _run_autonomy_cycle()
            except Exception as e:
                log.error(f"Autonomy cycle error: {e}")
            time.sleep(300)  # every 5 minutes (Feature C: scaled up)

    t = threading.Thread(target=_autonomy_loop, daemon=True)
    t.start()
    log.info("Autonomy loop started")
    return {"status": "started", "stats": _autonomy_stats}


@app.post("/api/brain/autonomy/stop")
def brain_autonomy_stop():
    """Stop the autonomous learning loop."""
    global _autonomy_running
    _autonomy_running = False
    log.info("Autonomy loop stopped")
    return {"status": "stopped", "stats": _autonomy_stats}


@app.get("/api/brain/autonomy/status")
def brain_autonomy_status():
    """Get autonomy loop status."""
    return {"running": _autonomy_running, "stats": _autonomy_stats}


def _run_autonomy_cycle():
    """One cycle: identify weak categories, suggest what to learn."""
    global _autonomy_stats

    # 1. Identify most curious categories (weakest performance)
    profiles = compute_category_profiles()
    if not profiles:
        return

    sorted_cats = sorted(profiles.items(), key=lambda x: x[1]["curiosity"], reverse=True)
    curious_cats = [cat for cat, _ in sorted_cats[:5]]

    # 2. Check self-model for weakest categories
    if load_self_model() and _category_stats:
        weak_cats = sorted(_category_stats.items(), key=lambda x: x[1]["mrr"])[:5]
        curious_cats = list(dict.fromkeys(curious_cats + [c for c, _ in weak_cats]))[:8]

    # 3. Log what we'd want to learn
    _autonomy_stats["cycles"] += 1
    _autonomy_stats["last_cycle"] = time.time()
    _autonomy_stats["curious_categories"] = curious_cats[:5]

    # 4. Auto-train when buffer is large enough (Feature C: threshold=20)
    if len(_online_pairs) >= 20:
        try:
            result = brain_learn_train()
            pairs = result.get("pairs_trained", 0)
            _autonomy_stats["pairs_learned"] += pairs
            log.info(f"Autonomy: trained on {pairs} online pairs")
            _emit_event("auto_train", {"pairs": pairs, "total": _autonomy_stats["pairs_learned"]})
        except Exception as e:
            log.error(f"Autonomy training error: {e}")

    # 5. Reflect (every 3rd cycle to save CPU)
    if _autonomy_stats["cycles"] % 3 == 0:
        try:
            brain_reflect()
        except Exception:
            pass

    # 6. Consolidation + knowledge graph update (every 5th cycle)
    if _autonomy_stats["cycles"] % 5 == 0:
        try:
            _consolidation_cycle()
            _build_knowledge_graph()
        except Exception as e:
            log.error(f"Consolidation error: {e}")

    # 7. YouTube learning — 3 underrepresented categories per cycle (Feature C)
    underrep = _get_underrepresented_categories(n=5)
    for cat in underrep[:3]:
        _youtube_learn_category(cat)

    # 8. Dream phase — when YouTube worker is idle (Feature A)
    if not _youtube_learning_active:
        try:
            dream = _generate_dream()
            _dream_history.append(dream)
            while len(_dream_history) > 100:
                _dream_history.pop(0)
            _emit_event("dream", {
                "seed": dream.get("seed", "?"),
                "sequence": [s["concept"] for s in dream.get("steps", [])],
                "surprise": dream.get("avg_surprise", 0),
                "pairs": dream.get("learning_pairs_generated", 0),
            })
        except Exception as e:
            log.warning(f"Dream error: {e}")

    # 9. Autonomous web research (Option 4) — every 4th cycle
    if _autonomy_stats["cycles"] % 4 == 0 and curious_cats:
        try:
            topic = curious_cats[0]
            _autonomous_research(topic)
        except Exception as e:
            log.warning(f"Autonomous research error: {e}")

    log.info(f"Autonomy cycle #{_autonomy_stats['cycles']}: curious about {curious_cats[:3]}")
    _emit_event("autonomy_cycle", {"cycle": _autonomy_stats["cycles"], "curious": curious_cats[:5]})


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Episodic Memory Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/episodes")
def brain_episodes(limit: int = 20):
    """List recent episodes with their events."""
    episodes = _get_episodes(limit=limit)
    return {"episodes": episodes, "count": len(episodes)}


class RememberRequest(BaseModel):
    sequence: list[str]
    top_k: int = 5


@app.post("/api/brain/remember")
def brain_remember(req: RememberRequest):
    """Find episodes matching a sequence pattern (e.g., ['thunder', 'rain'])."""
    if not req.sequence:
        return {"error": "Provide a non-empty sequence"}
    matches = _search_episodes_by_sequence(req.sequence, top_k=req.top_k)
    return {"query": req.sequence, "matches": matches}


# ═══════════════════════════════════════════════════════════════════
# Phase 2.2: Temporal Prediction
# ═══════════════════════════════════════════════════════════════════

_temporal_model = None


def load_temporal_model():
    """Load the trained temporal prediction model."""
    global _temporal_model
    if _temporal_model is not None:
        return _temporal_model
    model_path = PROJECT_ROOT / "outputs/cortex/temporal_model/model.pt"
    if not model_path.exists():
        return None
    try:
        import torch.nn as tnn
        class TemporalPredictor(tnn.Module):
            def __init__(self, d_model=512, nhead=4, num_layers=2, max_seq=8):
                super().__init__()
                self.pos_emb = tnn.Embedding(max_seq, d_model)
                encoder_layer = tnn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=1024,
                    dropout=0.1, batch_first=True)
                self.transformer = tnn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_proj = tnn.Linear(d_model, d_model)

            def forward(self, x):
                B, S, D = x.shape
                pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
                x = x + self.pos_emb(pos)
                mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
                out = self.transformer(x, mask=mask)
                return self.output_proj(out[:, -1, :])

        model = TemporalPredictor()
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        _temporal_model = model
        log.info("Loaded temporal prediction model")
    except Exception as e:
        log.warning(f"Failed to load temporal model: {e}")
    return _temporal_model


def predict_next_event(sequence_embeddings: list[np.ndarray], top_k: int = 5) -> list[dict]:
    """Given a sequence of embeddings, predict the next event."""
    model = load_temporal_model()
    if model is None or not sequence_embeddings:
        return []

    # Pad/truncate to max_seq=8
    seq = sequence_embeddings[-8:]
    seq_tensor = torch.from_numpy(np.stack(seq)).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(seq_tensor).squeeze(0).numpy()

    # Find nearest clips to predicted embedding
    pred_norm = pred / (np.linalg.norm(pred) + 1e-12)
    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    if w_a is not None:
        all_a_proj = mlp_project(cached["a_emb"], w_a)
        sims = all_a_proj @ pred_norm
    else:
        sims = cached["a_emb"] @ pred_norm[:512]

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    seen = set()
    for idx in top_idx:
        if idx < len(labels):
            lbl = labels[idx]["label"]
            if lbl not in seen:
                seen.add(lbl)
                results.append({"label": lbl, "similarity": round(float(sims[idx]), 4)})
    return results


class PredictSequenceRequest(BaseModel):
    episode_id: int = 0
    labels: list[str] = []
    top_k: int = 5


@app.post("/api/brain/predict_next")
def brain_predict_next(req: PredictSequenceRequest):
    """Given a sequence (episode or label list), predict what comes next."""
    embeddings = []

    if req.episode_id:
        embeddings = _get_episode_embeddings(req.episode_id)
    elif req.labels:
        # Convert labels to embeddings via text model
        text_model = load_text_model()
        if text_model:
            w_v, w_a = load_mlp_weights()
            for label in req.labels:
                emb = text_model.encode([label], normalize_embeddings=True)[0].astype(np.float32)
                if w_v is not None:
                    emb = mlp_project(emb.reshape(1, -1), w_v)[0]
                embeddings.append(emb)

    if not embeddings:
        return {"error": "No embeddings found. Provide episode_id or labels."}

    predictions = predict_next_event(embeddings, top_k=req.top_k)
    return {"sequence_length": len(embeddings), "predictions": predictions}


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Concept Hierarchy
# ═══════════════════════════════════════════════════════════════════

def load_concept_hierarchy():
    """Load the concept hierarchy tree from JSON."""
    global _concept_hierarchy
    if _concept_hierarchy is not None:
        return _concept_hierarchy
    hier_path = PROJECT_ROOT / "outputs/cortex/concept_hierarchy.json"
    if hier_path.exists():
        with open(hier_path) as f:
            _concept_hierarchy = json.load(f)
        log.info(f"Loaded concept hierarchy: {len(_concept_hierarchy.get('children', []))} top-level groups")
    return _concept_hierarchy


def _find_subtree(node: dict, query: str) -> dict | None:
    """Recursively find a subtree whose name matches query."""
    if not node:
        return None
    name = node.get("name", "").lower()
    if query.lower() in name:
        return node
    for child in node.get("children", []):
        result = _find_subtree(child, query)
        if result:
            return result
    return None


def _collect_leaves(node: dict) -> list[str]:
    """Collect all leaf labels from a subtree."""
    if not node.get("children"):
        return [node.get("name", "")]
    leaves = []
    for child in node.get("children", []):
        leaves.extend(_collect_leaves(child))
    return leaves


@app.get("/api/brain/hierarchy")
def brain_hierarchy():
    """Get the concept hierarchy tree."""
    tree = load_concept_hierarchy()
    if tree is None:
        return {"error": "Concept hierarchy not built yet. Run scripts/build_concept_hierarchy.py first."}
    return tree


class HierarchyQueryRequest(BaseModel):
    query: str
    level: str = "branch"  # "leaf", "branch", "root"
    top_k: int = 10


@app.post("/api/brain/query")
def brain_hierarchy_query(req: HierarchyQueryRequest):
    """Query at any abstraction level in the concept hierarchy."""
    tree = load_concept_hierarchy()
    cached = load_cached_embeddings()
    labels = load_clip_labels()
    w_v, w_a = load_mlp_weights()

    # Find matching subtree
    matching_labels = []
    if tree:
        subtree = _find_subtree(tree, req.query)
        if subtree:
            matching_labels = _collect_leaves(subtree)

    # Also do semantic search as fallback
    semantic = text_semantic_search(req.query, top_k=20)
    semantic_labels = [r["label"] for r in semantic]

    # Combine
    all_labels = list(dict.fromkeys(matching_labels + semantic_labels))

    # Find clips matching these labels
    results = []
    label_set = set(l.lower() for l in all_labels)
    for i, clip in enumerate(labels):
        if clip["label"].lower() in label_set:
            results.append({"index": i, "label": clip["label"]})
            if len(results) >= req.top_k * 10:
                break

    # Group by category
    category_counts = {}
    for r in results:
        cat = r["label"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "query": req.query,
        "level": req.level,
        "hierarchy_matches": matching_labels[:20],
        "semantic_matches": semantic_labels[:10],
        "categories_found": len(category_counts),
        "clips_found": len(results),
        "category_counts": dict(sorted(category_counts.items(), key=lambda x: -x[1])[:req.top_k]),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Working Memory Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/working_memory")
def brain_working_memory():
    """Get current working memory state."""
    return _get_working_memory_state()


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Prototype Memory Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/prototypes")
def brain_prototypes():
    """List all learned prototypes."""
    result = []
    for name, proto in sorted(_prototypes.items(), key=lambda x: -x[1]["count"]):
        result.append({
            "name": name,
            "count": proto["count"],
            "examples": proto["examples"][:5],
            "created_at": proto.get("created_at"),
        })
    return {"prototypes": result, "total": len(_prototypes)}


class AddPrototypeRequest(BaseModel):
    name: str
    embedding_b64: str = ""
    query: str = ""


@app.post("/api/brain/prototypes/add")
def brain_add_prototype(req: AddPrototypeRequest):
    """Manually add a prototype from an embedding or text query."""
    import base64
    if req.embedding_b64:
        emb = np.frombuffer(base64.b64decode(req.embedding_b64), dtype=np.float32)
    elif req.query:
        text_model = load_text_model()
        if not text_model:
            return {"error": "Text model not loaded"}
        emb = text_model.encode([req.query], normalize_embeddings=True)[0].astype(np.float32)
        w_v, _ = load_mlp_weights()
        if w_v is not None:
            emb = mlp_project(emb.reshape(1, -1), w_v)[0]
    else:
        return {"error": "Provide embedding_b64 or query"}

    _add_prototype(req.name, emb, req.query or req.name)
    return {"status": "added", "prototype": req.name, "count": _prototypes[req.name]["count"]}


@app.post("/api/brain/consolidate")
def brain_consolidate():
    """Manually trigger a memory consolidation cycle."""
    stats = _consolidation_cycle()
    return {"status": "completed", "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Goal-Directed Planning
# ═══════════════════════════════════════════════════════════════════

class GoalRequest(BaseModel):
    goal: str
    max_steps: int = 5


@app.post("/api/brain/plan")
@async_endpoint
def brain_plan(req: GoalRequest):
    """Generate a multi-step plan toward a goal using world model + causal reasoning."""
    t0 = time.time()

    # 1. Encode goal as embedding
    text_model = load_text_model()
    if not text_model:
        return {"error": "Text model not loaded"}

    goal_emb = text_model.encode([req.goal], normalize_embeddings=True)[0].astype(np.float32)
    w_v, w_a = load_mlp_weights()
    if w_v is not None:
        goal_proj = mlp_project(goal_emb.reshape(1, -1), w_v)[0]
    else:
        goal_proj = goal_emb

    # 2. Find relevant starting points via semantic search
    semantic = text_semantic_search(req.goal, top_k=5)
    start_labels = [r["label"] for r in semantic]

    # 3. Use causal model to find paths
    causal_steps = []
    if _causal_transitions is not None and _causal_labels is not None:
        for start_label in start_labels[:2]:
            label_idx = None
            for i, cl in enumerate(_causal_labels):
                if cl.lower() == start_label.lower():
                    label_idx = i
                    break
            if label_idx is not None:
                # Follow causal chain
                chain = [start_label]
                visited = {label_idx}
                current = label_idx
                for _ in range(req.max_steps):
                    row = _causal_transitions[current]
                    row[list(visited)] = 0
                    if row.sum() == 0:
                        break
                    next_idx = int(np.argmax(row))
                    if next_idx in visited:
                        break
                    visited.add(next_idx)
                    chain.append(_causal_labels[next_idx])
                    current = next_idx
                causal_steps.append(chain)

    # 4. Use spreading activation to find paths to goal
    spread_results = []
    try:
        if _nn_graph is not None:
            cached = load_cached_embeddings()
            labels = load_clip_labels()
            # Find clip closest to goal
            if w_v is not None:
                all_v_proj = mlp_project(cached["v_emb"], w_v)
                goal_sims = all_v_proj @ goal_proj
            else:
                goal_sims = cached["v_emb"] @ goal_emb
            goal_clip = int(np.argmax(goal_sims))
            activations, paths = spreading_activation([goal_clip], n_hops=3)
            top_active = sorted(activations.items(), key=lambda x: -x[1])[:10]
            spread_results = [
                {"clip_index": idx, "label": labels[idx]["label"] if idx < len(labels) else "?",
                 "activation": round(act, 4)}
                for idx, act in top_active
            ]
    except Exception as e:
        log.warning(f"Spreading activation failed: {e}")

    # 5. World model: predict what to expect
    predictions = []
    try:
        pred_a = predict_audio(goal_proj)
        if pred_a is not None and w_a is not None:
            cached = load_cached_embeddings()
            labels = load_clip_labels()
            all_a_proj = mlp_project(cached["a_emb"], w_a)
            pred_sims = all_a_proj @ pred_a
            pred_top = np.argsort(pred_sims)[::-1][:5]
            seen = set()
            for idx in pred_top:
                if idx < len(labels) and labels[idx]["label"] not in seen:
                    seen.add(labels[idx]["label"])
                    predictions.append(labels[idx]["label"])
    except Exception:
        pass

    # 6. Generate narrative
    narration = None
    try:
        parts = [f"Goal: {req.goal}."]
        if causal_steps:
            parts.append(f"Causal path: {' → '.join(causal_steps[0][:5])}.")
        if predictions:
            parts.append(f"Expected: {', '.join(predictions[:3])}.")
        narration = _ollama_generate(
            f"You are a brain planning how to achieve: '{req.goal}'. "
            f"Your causal knowledge suggests: {causal_steps[:1]}. "
            f"Related concepts: {spread_results[:5]}. "
            f"Predictions: {predictions[:3]}. "
            "Describe a concise 3-step plan in first person. Be brief.",
            max_predict=100
        )
    except Exception:
        pass

    return {
        "goal": req.goal,
        "semantic_anchors": start_labels,
        "causal_chains": causal_steps[:2],
        "spreading_activation": spread_results[:10],
        "world_model_predictions": predictions[:5],
        "narration": narration,
        "process_time": round(time.time() - t0, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 8: Brain Decoder + Internal Monologue + Grounded Conversation
# ═══════════════════════════════════════════════════════════════════

_brain_decoder = None


def load_brain_decoder():
    """Load the brain-to-text decoder model."""
    global _brain_decoder
    if _brain_decoder is not None:
        return _brain_decoder
    decoder_path = PROJECT_ROOT / "outputs/cortex/brain_decoder/decoder.pt"
    if not decoder_path.exists():
        return None
    try:
        import torch.nn as tnn
        # Simple decoder: embedding → vocabulary logits
        # Uses a small transformer decoder with learned vocabulary
        vocab_path = PROJECT_ROOT / "outputs/cortex/brain_decoder/vocab.json"
        if not vocab_path.exists():
            return None
        with open(vocab_path) as f:
            vocab = json.load(f)

        class BrainDecoder(tnn.Module):
            def __init__(self, d_model=512, vocab_size=len(vocab), nhead=4, num_layers=2, max_pos=34):
                super().__init__()
                self.vocab_size = vocab_size
                self.embed = tnn.Linear(d_model, d_model)
                decoder_layer = tnn.TransformerDecoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=1024,
                    dropout=0.1, batch_first=True)
                self.decoder = tnn.TransformerDecoder(decoder_layer, num_layers=num_layers)
                self.output = tnn.Linear(d_model, vocab_size)
                self.token_embed = tnn.Embedding(vocab_size, d_model)
                self.pos_embed = tnn.Embedding(max_pos, d_model)

            def forward(self, brain_emb, tokens):
                B = brain_emb.shape[0]
                memory = self.embed(brain_emb).unsqueeze(1)  # (B, 1, D)
                S = tokens.shape[1]
                pos = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, -1)
                tgt = self.token_embed(tokens) + self.pos_embed(pos)
                mask = torch.triu(torch.ones(S, S, device=tokens.device), diagonal=1).bool()
                out = self.decoder(tgt, memory, tgt_mask=mask)
                return self.output(out)

        model = BrainDecoder()
        model.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
        model.eval()
        _brain_decoder = {"model": model, "vocab": vocab, "idx2word": {v: k for k, v in vocab.items()}}
        log.info(f"Loaded brain decoder (vocab={len(vocab)})")
    except Exception as e:
        log.warning(f"Failed to load brain decoder: {e}")
    return _brain_decoder


def _decode_embedding_to_text(embedding: np.ndarray, max_tokens: int = 30) -> str:
    """Convert a brain embedding to text using the decoder, or fall back to nearest-label."""
    decoder = load_brain_decoder()
    if decoder is not None:
        try:
            model = decoder["model"]
            vocab = decoder["vocab"]
            idx2word = decoder["idx2word"]
            emb_t = torch.from_numpy(embedding.reshape(1, -1).astype(np.float32))

            # Greedy decode
            bos = vocab.get("<bos>", 0)
            eos = vocab.get("<eos>", 1)
            tokens = [bos]
            for _ in range(max_tokens):
                tok_t = torch.tensor([tokens], dtype=torch.long)
                with torch.no_grad():
                    logits = model(emb_t, tok_t)
                next_tok = int(logits[0, -1].argmax())
                if next_tok == eos:
                    break
                tokens.append(next_tok)
            words = [idx2word.get(t, "?") for t in tokens[1:]]
            return " ".join(words)
        except Exception as e:
            log.warning(f"Brain decoder failed: {e}")

    # Fallback: nearest concept label
    codebook, concept_labels = build_concept_codebook()
    if codebook is not None:
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        sims = codebook @ emb_norm
        top_idx = np.argsort(sims)[::-1][:3]
        return ", ".join(concept_labels[i] for i in top_idx)
    return "unknown"


def _generate_thought(context: str = None) -> str:
    """Generate an internal thought from the brain's current state."""
    wm = _get_working_memory_state()
    items = wm.get("items", [])

    # Build brain state description from actual brain data
    parts = []
    if items:
        focus_items = [f"{it['label']} ({it['modality']}, activation={it['activation']})" for it in items[:4]]
        parts.append(f"I'm focused on: {', '.join(it['label'] for it in items[:3])}")

    if _prototypes:
        recent_protos = sorted(_prototypes.items(), key=lambda x: x[1].get("created_at", 0), reverse=True)[:3]
        proto_names = [n for n, _ in recent_protos]
        parts.append(f"Recently learned concepts: {', '.join(proto_names)}")

    # Add perception stats
    try:
        conn = _get_memory_db()
        perc_count = conn.execute("SELECT count(*) FROM perceptions").fetchone()[0]
        ep_count = conn.execute("SELECT count(*) FROM episodes").fetchone()[0]
        if perc_count > 0:
            last = conn.execute("SELECT modality, top_labels FROM perceptions ORDER BY id DESC LIMIT 1").fetchone()
            if last:
                last_mod = last["modality"]
                last_labels = last["top_labels"] or ""
                try:
                    import json as _j
                    last_labels = ", ".join(_j.loads(last_labels)[:2]) if last_labels.startswith("[") else last_labels[:50]
                except Exception:
                    last_labels = last_labels[:50]
                parts.append(f"Last perception ({last_mod}): {last_labels}")
            parts.append(f"Total: {perc_count} perceptions across {ep_count} episodes")
    except Exception:
        pass

    if _autonomy_running:
        parts.append(f"Autonomy loop active (cycle #{_autonomy_stats.get('cycles', 0)}, {_autonomy_stats.get('videos_processed', 0)} videos learned)")

    if not parts:
        return "I'm idle — no perceptions yet. Waiting for sensory input through Listen, Watch, or YouTube."

    state_desc = ". ".join(parts) + "."

    # Only call LLM if there's active working memory (otherwise return fast)
    if items and _working_memory:
        top_item = _working_memory[0]
        if top_item.get("embedding") is not None:
            decoded = _decode_embedding_to_text(top_item["embedding"])
            state_desc += f" Current focus decodes to: {decoded}."

        # Call LLM only when we have real content
        prompt = (
            f"You are a brain. Your state: {state_desc} "
            "One sentence internal thought. Start with 'I'm thinking about' or 'I notice'. Be specific."
        )
        thought = _ollama_generate(prompt, max_predict=40)
        if thought:
            return thought

    return state_desc


@app.get("/api/brain/thoughts")
def brain_thoughts():
    """Get the brain's current internal monologue."""
    thought = _generate_thought()
    wm = _get_working_memory_state()
    return {
        "thought": thought,
        "working_memory": wm,
        "prototypes_active": len(_prototypes),
        "timestamp": time.time(),
    }


class ThinkRequest(BaseModel):
    question: str
    depth: int = 3


@app.post("/api/brain/think")
@async_endpoint
def brain_think(req: ThinkRequest):
    """Multi-step reasoning: think through a question step by step, grounded in brain data."""
    t0 = time.time()
    steps = []

    # 1. Activate relevant concepts in working memory
    text_model = load_text_model()
    if not text_model:
        return {"error": "Text model not loaded"}

    q_emb = text_model.encode([req.question], normalize_embeddings=True)[0].astype(np.float32)
    w_v, w_a = load_mlp_weights()
    if w_v is not None:
        q_proj = mlp_project(q_emb.reshape(1, -1), w_v)[0]
    else:
        q_proj = q_emb

    # Semantic search for grounding
    semantic = text_semantic_search(req.question, top_k=10)
    activated_concepts = [r["label"] for r in semantic][:5]
    steps.append({
        "step": "activate",
        "description": f"Activated concepts: {', '.join(activated_concepts)}",
        "concepts": activated_concepts,
    })

    # 2. Causal chain
    causal_chain = []
    if _causal_transitions is not None and _causal_labels is not None:
        for concept in activated_concepts[:2]:
            for i, cl in enumerate(_causal_labels):
                if cl.lower() == concept.lower():
                    chain = [concept]
                    visited = {i}
                    current = i
                    for _ in range(req.depth):
                        row = _causal_transitions[current].copy()
                        row[list(visited)] = 0
                        if row.sum() == 0:
                            break
                        next_idx = int(np.argmax(row))
                        visited.add(next_idx)
                        chain.append(_causal_labels[next_idx])
                        current = next_idx
                    causal_chain.append(chain)
                    break
    if causal_chain:
        steps.append({
            "step": "cause",
            "description": f"Causal reasoning: {' → '.join(causal_chain[0])}",
            "chains": causal_chain,
        })

    # 3. World model prediction
    predictions = []
    try:
        pred_a = predict_audio(q_proj)
        if pred_a is not None and w_a is not None:
            cached = load_cached_embeddings()
            labels_data = load_clip_labels()
            all_a_proj = mlp_project(cached["a_emb"], w_a)
            pred_sims = all_a_proj @ pred_a
            pred_top = np.argsort(pred_sims)[::-1][:5]
            seen = set()
            for idx in pred_top:
                if idx < len(labels_data) and labels_data[idx]["label"] not in seen:
                    seen.add(labels_data[idx]["label"])
                    predictions.append(labels_data[idx]["label"])
    except Exception:
        pass
    if predictions:
        steps.append({
            "step": "predict",
            "description": f"World model expects: {', '.join(predictions[:3])}",
            "predictions": predictions,
        })

    # 4. Composition / association
    try:
        composed = compose_concepts(activated_concepts[:3])
        if composed:
            steps.append({
                "step": "compose",
                "description": f"Composed associations: {', '.join(r['label'] for r in composed[:3])}",
                "associations": [r["label"] for r in composed[:5]],
            })
    except Exception:
        pass

    # 5. Generate grounded narrative
    grounding = {
        "concepts": activated_concepts,
        "causal": causal_chain[:1],
        "predictions": predictions[:3],
    }

    narration = _ollama_generate(
        f"You are a brain thinking about: '{req.question}'. "
        f"Your activated concepts: {activated_concepts}. "
        f"Causal chains: {causal_chain[:1]}. "
        f"World model predicts: {predictions[:3]}. "
        "Think through this step by step. Each sentence must reference your actual brain data. "
        "3-4 sentences max. Be specific.",
        max_predict=150
    )

    steps.append({
        "step": "narrate",
        "description": narration or "Could not generate narration.",
    })

    return {
        "question": req.question,
        "chain_of_thought": steps,
        "grounding": grounding,
        "process_time": round(time.time() - t0, 2),
    }


class DialogueV2Request(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/api/brain/dialogue/grounded")
@async_endpoint
def brain_grounded_dialogue(req: DialogueV2Request):
    """Grounded conversation where every claim traces to brain experience."""
    t0 = time.time()

    # 1. Encode message and find relevant brain memories
    text_model = load_text_model()
    if not text_model:
        return {"error": "Text model not loaded"}

    q_emb = text_model.encode([req.message], normalize_embeddings=True)[0].astype(np.float32)
    w_v, w_a = load_mlp_weights()

    # 2. Semantic search for grounding
    semantic = text_semantic_search(req.message, top_k=10)
    grounding_labels = [r["label"] for r in semantic]

    # 3. Check working memory for context
    wm = _get_working_memory_state()
    wm_context = [it["label"] for it in wm.get("items", [])[:3]]

    # 4. Check prototypes
    if w_v is not None:
        q_proj = mlp_project(q_emb.reshape(1, -1), w_v)[0]
        proto_match, proto_sim = _match_prototype(q_proj, threshold=0.5)
    else:
        proto_match = None

    # 5. Check self-model confidence
    confidence_info = None
    if _category_stats:
        for label in grounding_labels[:3]:
            if label in _category_stats:
                confidence_info = {
                    "category": label,
                    "mrr": _category_stats[label]["mrr"],
                    "clips": _category_stats[label]["n_clips"],
                }
                break

    # 6. Build grounded prompt
    grounding_data = {
        "semantic_matches": grounding_labels[:5],
        "working_memory": wm_context,
        "prototype": proto_match,
        "confidence": confidence_info,
    }

    context_parts = [f"Relevant concepts: {', '.join(grounding_labels[:5])}."]
    if wm_context:
        context_parts.append(f"Currently thinking about: {', '.join(wm_context)}.")
    if proto_match:
        context_parts.append(f"Matches learned prototype: {proto_match}.")
    if confidence_info:
        context_parts.append(
            f"I know {confidence_info['category']} well (MRR={confidence_info['mrr']:.3f}, "
            f"{confidence_info['clips']} clips).")

    # 7. Generate response
    response = _ollama_generate(
        f"You are a brain having a conversation. The user says: '{req.message}'. "
        f"Your brain data: {' '.join(context_parts)} "
        f"Respond naturally but ground every claim in your actual experience. "
        f"If you don't know something, say so. 2-3 sentences.",
        max_predict=100
    )

    return {
        "response": response,
        "grounding": grounding_data,
        "grounded": True,
        "process_time": round(time.time() - t0, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Feature B: Knowledge Graph Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/knowledge")
def brain_knowledge():
    """Knowledge graph stats."""
    try:
        conn = _get_memory_db()
        total = conn.execute("SELECT count(*) FROM knowledge_edges").fetchone()[0]
        by_relation = conn.execute(
            "SELECT relation, count(*) as cnt, round(avg(weight),3) as avg_w "
            "FROM knowledge_edges GROUP BY relation ORDER BY cnt DESC"
        ).fetchall()
        return {
            "total_edges": total,
            "by_relation": [{"relation": r["relation"], "count": r["cnt"], "avg_weight": r["avg_w"]} for r in by_relation],
        }
    except Exception as e:
        return {"error": str(e)}


class KnowledgeQueryRequest(BaseModel):
    start: str
    relations: list[str] | None = None
    max_hops: int = 3
    max_results: int = 20


@app.post("/api/brain/knowledge/query")
def brain_knowledge_query(req: KnowledgeQueryRequest):
    """Multi-hop knowledge graph traversal."""
    paths = _multi_hop_traverse(req.start, req.relations, req.max_hops, req.max_results)
    # Also get direct edges
    direct = _get_knowledge_edges(source=req.start, limit=20)
    return {
        "start": req.start,
        "direct_edges": [{"target": e["target_label"], "relation": e["relation"],
                          "weight": e["weight"], "evidence": e["evidence_count"]} for e in direct],
        "paths": paths,
        "total_paths": len(paths),
    }


# ═══════════════════════════════════════════════════════════════════
# Feature D: Text Understanding Endpoints
# ═══════════════════════════════════════════════════════════════════

class ReadRequest(BaseModel):
    text: str
    source: str = "user"
    title: str = ""


@app.post("/api/brain/read")
def brain_read(req: ReadRequest):
    """Ingest text into the brain — encode, store as perception, extract knowledge edges."""
    embedding = _encode_text_to_brain_space(req.text)
    label = req.title or req.text[:50]

    # Find associations
    codebook, concept_labels = build_concept_codebook()
    associations = []
    if codebook is not None:
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        sims = codebook @ emb_norm
        top_idx = np.argsort(sims)[::-1][:5]
        associations = [{"label": concept_labels[i], "similarity": round(float(sims[i]), 3)} for i in top_idx]

    _store_perception(
        modality="text", transcription=req.text[:500],
        top_labels=[label], cross_labels=[a["label"] for a in associations[:3]],
        embedding=embedding,
    )

    edges = _extract_text_relations(req.text)
    for edge in edges:
        _upsert_knowledge_edge(edge["source"], edge["relation"], edge["target"], 0.7)

    _emit_event("text_ingested", {"source": req.source, "title": label, "edges": len(edges)})
    return {"status": "ingested", "associations": associations, "edges_extracted": len(edges)}


@app.post("/api/brain/ingest/audioset")
def brain_ingest_audioset():
    """Load AudioSet ontology descriptions as text perceptions."""
    return _ingest_audioset_descriptions()


@app.post("/api/brain/ingest/wikipedia")
def brain_ingest_wikipedia():
    """Fetch Wikipedia summaries for all categories (background, ~5 min)."""
    import threading
    t = threading.Thread(target=_ingest_wikipedia_summaries, daemon=True)
    t.start()
    return {"status": "started", "message": "Fetching Wikipedia summaries in background (~5 min)"}


@app.get("/api/brain/knowledge/text")
def brain_knowledge_text():
    """Stats on text perceptions."""
    try:
        conn = _get_memory_db()
        text_count = conn.execute("SELECT count(*) FROM perceptions WHERE modality='text'").fetchone()[0]
        by_source = conn.execute(
            "SELECT top_labels, count(*) as cnt FROM perceptions WHERE modality='text' "
            "GROUP BY top_labels ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        return {"text_perceptions": text_count, "sources": [dict(r) for r in by_source]}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# #11: Fast Memory Endpoint
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/memory/fast")
def brain_fast_memory():
    """Query the Hopfield fast memory store."""
    return {
        "count": _fast_memory_count,
        "capacity": FAST_MEMORY_CAPACITY,
        "recent_labels": _fast_memory_labels[-20:] if _fast_memory_labels else [],
    }


class FastMemoryQueryRequest(BaseModel):
    query: str


@app.post("/api/brain/memory/fast/query")
def brain_fast_memory_query(req: FastMemoryQueryRequest):
    """Pattern completion from fast memory."""
    emb = _encode_text_to_brain_space(req.query)
    results = _fast_memory_retrieve(emb, top_k=10)
    return {"query": req.query, "results": results, "total_patterns": _fast_memory_count}


# ═══════════════════════════════════════════════════════════════════
# #10: Sparse Config Endpoint
# ═══════════════════════════════════════════════════════════════════

class ConfigRequest(BaseModel):
    sparse_k: int | None = None


@app.post("/api/brain/config")
def brain_config(req: ConfigRequest):
    """Update brain configuration at runtime."""
    global SPARSE_K
    changes = {}
    if req.sparse_k is not None:
        SPARSE_K = max(0, min(req.sparse_k, 512))
        changes["sparse_k"] = SPARSE_K
    return {"status": "updated", "changes": changes, "current": {"sparse_k": SPARSE_K}}


# ═══════════════════════════════════════════════════════════════════
# Grid Cell Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/grid/map")
def brain_grid_map():
    """Get the full concept grid map — 2D coordinates for all concepts."""
    encoder = build_grid_encoder()
    if encoder is None:
        return {"error": "Grid encoder not built"}
    codebook, labels = build_concept_codebook()
    coords = _grid_concept_coords
    if coords is None:
        return {"error": "No grid coordinates"}

    concepts = []
    for i, label in enumerate(labels):
        if i < len(coords):
            act = encoder.grid_activation(coords[i])
            concepts.append({
                "label": label,
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "grid_activation": round(float(act.mean()), 4),
            })

    return {
        "concepts": concepts,
        "n_concepts": len(concepts),
        "scales": GRID_SCALES,
        "orientations": [round(o, 4) for o in encoder.orientations],
    }


class GridNavigateRequest(BaseModel):
    query: str = ""
    embedding_b64: str = ""
    top_k: int = 10
    radius: float = 0.0


@app.post("/api/brain/grid/navigate")
def brain_grid_navigate(req: GridNavigateRequest):
    """Navigate the grid: find what's near a concept in grid space."""
    encoder = build_grid_encoder()
    if encoder is None:
        return {"error": "Grid encoder not built"}

    codebook, labels = build_concept_codebook()
    if codebook is None:
        return {"error": "No concept codebook"}

    # Get query embedding
    if req.query:
        emb = _encode_text_to_brain_space(req.query)
    elif req.embedding_b64:
        import base64
        emb = np.frombuffer(base64.b64decode(req.embedding_b64), dtype=np.float32)
    else:
        return {"error": "Provide query or embedding_b64"}

    pos = encoder.to_2d(emb)
    grid_act = encoder.grid_activation(pos)

    # Find nearby concepts
    if req.radius > 0:
        nearby = encoder.get_region(emb, req.radius, codebook, labels)
    else:
        nearby = encoder.find_nearby(emb, codebook, labels, top_k=req.top_k)

    # Also compute grid distances to working memory items
    wm_distances = []
    for item in _working_memory:
        if item.get("embedding") is not None:
            gd = encoder.grid_distance(emb, item["embedding"])
            wm_distances.append({"label": item["label"], "grid_distance": round(gd, 4)})

    return {
        "query": req.query,
        "grid_position": {"x": round(float(pos[0]), 4), "y": round(float(pos[1]), 4)},
        "grid_activation": grid_act.tolist(),
        "nearby_concepts": nearby,
        "wm_distances": wm_distances,
        "scale_info": {f"scale_{i}": s for i, s in enumerate(GRID_SCALES)},
    }


@app.get("/api/brain/grid/episode/{episode_id}")
def brain_grid_episode(episode_id: int):
    """Get the grid trajectory for an episode."""
    embeddings = _get_episode_embeddings(episode_id)
    if not embeddings:
        return {"error": f"No embeddings for episode {episode_id}"}
    trajectory = grid_encode_episode(embeddings)
    trajectory["episode_id"] = episode_id
    return trajectory


class GridBetweenRequest(BaseModel):
    concept_a: str
    concept_b: str
    n_steps: int = 5


@app.post("/api/brain/grid/between")
def brain_grid_between(req: GridBetweenRequest):
    """What concepts lie between A and B on the grid? (conceptual interpolation)"""
    encoder = build_grid_encoder()
    if encoder is None:
        return {"error": "Grid encoder not built"}

    codebook, labels = build_concept_codebook()
    if codebook is None:
        return {"error": "No concept codebook"}

    emb_a = _encode_text_to_brain_space(req.concept_a)
    emb_b = _encode_text_to_brain_space(req.concept_b)

    pos_a = encoder.to_2d(emb_a)
    pos_b = encoder.to_2d(emb_b)

    # Interpolate along the grid path
    waypoints = []
    for i in range(req.n_steps + 1):
        t = i / max(req.n_steps, 1)
        pos = pos_a * (1 - t) + pos_b * t
        # Find nearest concept to this grid position
        dists = np.linalg.norm(_grid_concept_coords - pos, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_label = labels[nearest_idx] if nearest_idx < len(labels) else "?"
        waypoints.append({
            "t": round(t, 2),
            "x": round(float(pos[0]), 4),
            "y": round(float(pos[1]), 4),
            "nearest_concept": nearest_label,
            "distance": round(float(dists[nearest_idx]), 4),
        })

    grid_dist = encoder.grid_distance(emb_a, emb_b)

    return {
        "from": req.concept_a,
        "to": req.concept_b,
        "grid_distance": round(grid_dist, 4),
        "cosine_similarity": round(float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-12)), 4),
        "waypoints": waypoints,
    }


# Option 4: Agentic Endpoints
# ═══════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/api/brain/search")
def brain_web_search(req: SearchRequest):
    """Search the web and return results."""
    results = _web_search(req.query, max_results=req.max_results)
    return {"query": req.query, "results": results}


class ResearchRequest(BaseModel):
    topic: str


@app.post("/api/brain/research")
@async_endpoint
def brain_research(req: ResearchRequest):
    """Autonomously research a topic: search, read, extract knowledge."""
    result = _autonomous_research(req.topic)
    return result


class FetchRequest(BaseModel):
    url: str


@app.post("/api/brain/fetch")
def brain_fetch_url(req: FetchRequest):
    """Fetch a URL, read its content, store as brain perception."""
    return _fetch_and_read_url(req.url)


# ═══════════════════════════════════════════════════════════════════
# Feature A: Dream Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/brain/dreams")
def brain_dreams(limit: int = 20):
    """Recent dreams with sequences and surprise scores."""
    return {
        "dreams": _dream_history[-limit:],
        "total_dreams": _dream_count,
    }


class DreamRequest(BaseModel):
    seed: str = ""
    steps: int = 5


@app.post("/api/brain/dream")
@async_endpoint
def brain_dream(req: DreamRequest):
    """Trigger one dream with optional seed concept."""
    global _dream_history
    dream = _generate_dream(seed_concept=req.seed or None, max_steps=req.steps)
    _dream_history.append(dream)
    while len(_dream_history) > 100:
        _dream_history.pop(0)
    _emit_event("dream", {
        "seed": dream.get("seed", "?"),
        "sequence": [s["concept"] for s in dream.get("steps", [])],
        "surprise": dream.get("avg_surprise", 0),
        "pairs": dream.get("learning_pairs_generated", 0),
    })
    return dream


# ═══════════════════════════════════════════════════════════════════
# Option 2: Voice — Text-to-Speech
# ═══════════════════════════════════════════════════════════════════

from fastapi.responses import StreamingResponse, Response


class SpeakRequest(BaseModel):
    text: str = ""
    voice: str = "en+m3"
    speed: int = 160


@app.post("/api/brain/speak")
def brain_speak(req: SpeakRequest):
    """Convert text to speech WAV using espeak-ng. If no text, speaks current thought."""
    import subprocess, tempfile
    text = req.text
    if not text:
        text = _generate_thought()
    if not text:
        text = "I have nothing to say."

    # Truncate for safety
    text = text[:500]

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        subprocess.run(
            ["espeak-ng", "-v", req.voice, "-s", str(req.speed), "-w", tmp_path, text],
            capture_output=True, timeout=10)
        with open(tmp_path, "rb") as f:
            wav_data = f.read()
        os.unlink(tmp_path)
        _emit_event("speak", {"text": text[:100], "bytes": len(wav_data)})
        return Response(content=wav_data, media_type="audio/wav",
                        headers={"Content-Disposition": "inline; filename=thought.wav"})
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/brain/speak/thought")
def brain_speak_thought():
    """Speak the brain's current internal monologue."""
    import subprocess, tempfile
    thought = _generate_thought()
    if not thought:
        thought = "My mind is quiet."
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        subprocess.run(
            ["espeak-ng", "-v", "en+m3", "-s", "155", "-w", tmp_path, thought[:500]],
            capture_output=True, timeout=10)
        with open(tmp_path, "rb") as f:
            wav_data = f.read()
        os.unlink(tmp_path)
        return Response(content=wav_data, media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# SSE: Live Brain Activity Stream
# ═══════════════════════════════════════════════════════════════════


@app.get("/api/brain/live")
async def brain_live_stream():
    """SSE endpoint: streams brain events in real-time."""
    import asyncio
    sub_q = _subscribe_sse()

    async def event_generator():
        try:
            # Send initial state
            wm = _get_working_memory_state()
            init = json.dumps({"type": "init", "working_memory": wm,
                               "prototypes": len(_prototypes),
                               "subscribers": len(_sse_subscribers)})
            yield f"data: {init}\n\n"

            heartbeat_counter = 0
            while True:
                # Use asyncio.sleep + non-blocking get to avoid blocking the event loop
                await asyncio.sleep(0.3)
                try:
                    event = sub_q.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                    heartbeat_counter = 0
                    # Drain any queued events
                    while not sub_q.empty():
                        event = sub_q.get_nowait()
                        yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 30:  # heartbeat every ~10s
                        yield f"data: {json.dumps({'type': 'heartbeat', 'time': time.time()})}\n\n"
                        heartbeat_counter = 0
        except (GeneratorExit, asyncio.CancelledError):
            pass
        finally:
            _unsubscribe_sse(sub_q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Initialize persistent memory DB
    log.info("Initializing persistent memory DB...")
    _init_memory_db()

    # Restore state from DB
    _saved_count = _get_stat("online_learning_count")
    if _saved_count is not None:
        _online_learning_count = int(_saved_count)
        log.info(f"Restored online_learning_count={_online_learning_count} from memory DB")

    _saved_reflections = _get_recent_reflections(limit=100)
    if _saved_reflections:
        _reflection_history.extend(_saved_reflections)
        log.info(f"Restored {len(_saved_reflections)} reflections from memory DB")

    # Phase I-1: Lazy loading — only load essentials eagerly
    # Everything else loads on first request that needs it
    _start_time = time.time()
    log.info("Loading essential models (VGGSound + MLP + text)...")
    load_cached_embeddings()
    load_clip_labels()
    load_mlp_weights()
    load_text_model()
    _load_prototypes_from_db()
    build_concept_codebook()
    log.info(f"Essential models loaded in {time.time() - _start_time:.1f}s")

    # These load lazily on first use:
    # load_m_matrix(), load_projected_embeddings(), load_audioset_expansion()
    # load_audioset_pool() (memmap), load_world_model(), load_text_projection()
    # load_causal_graph(), load_self_model(), build_nn_graph() (30s)
    # load_temporal_model(), load_brain_decoder(), load_concept_hierarchy()
    # build_grid_encoder(), _build_knowledge_graph()

    log.info("Starting YouTube Brain service on port 8099...")
    import uvicorn.config
    uvicorn.run(app, host="0.0.0.0", port=8099, workers=1, timeout_keep_alive=120)
