#!/usr/bin/env python3
"""Build AudioSet expansion: encode 527+ AudioSet categories via text encoder + MLP.

Downloads the AudioSet ontology (632 entries, ~527 concrete categories),
filters out abstract parent categories, removes those already in VGGSound,
encodes labels with all-MiniLM-L6-v2 (384-dim), projects through the brain's
MLP W_v (384->512, ReLU, L2 norm), and saves labels + embeddings.

Output:
  /opt/brain/outputs/cortex/audioset_expansion/labels.json
  /opt/brain/outputs/cortex/audioset_expansion/embeddings.npy  (N, 512)
  /opt/brain/outputs/cortex/audioset_expansion/descriptions.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/brain")
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/audioset_expansion"
MODEL_DIR_V4 = PROJECT_ROOT / "outputs/cortex/v4_mlp"
VGGSOUND_CSV = PROJECT_ROOT / "data/vggsound/vggsound.csv"
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"


def load_bin_matrix(path: Path) -> np.ndarray:
    """Load a binary matrix saved by Rust trainer (header: RxC\\n + f32 LE data)."""
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split('x'))
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(rows, cols)


def get_vggsound_labels() -> set[str]:
    """Get the set of unique VGGSound category labels."""
    labels = set()
    with open(VGGSOUND_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                labels.add(parts[2].strip('"'))
    return labels


def fetch_ontology() -> list[dict]:
    """Download AudioSet ontology JSON. Falls back to hardcoded list on failure."""
    try:
        import urllib.request
        log.info(f"Downloading AudioSet ontology from {ONTOLOGY_URL}...")
        with urllib.request.urlopen(ONTOLOGY_URL, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        log.info(f"Downloaded {len(data)} ontology entries")
        return data
    except Exception as e:
        log.warning(f"Download failed ({e}), trying local cache...")

    # Try local cache
    local = Path("/tmp/audioset_ontology.json")
    if local.exists():
        with open(local) as f:
            data = json.load(f)
        log.info(f"Loaded {len(data)} entries from local cache")
        return data

    log.error("No ontology available. Cannot proceed.")
    sys.exit(1)


def mlp_project(emb: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Project embedding through MLP layer: ReLU(emb @ W), L2-normalized."""
    proj = emb @ w
    proj = np.maximum(proj, 0)  # ReLU
    norm = np.linalg.norm(proj, axis=-1, keepdims=True).clip(1e-12)
    return proj / norm


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load AudioSet ontology
    ontology = fetch_ontology()

    # Filter: keep only concrete (non-abstract) categories
    concrete = [e for e in ontology if "abstract" not in e.get("restrictions", [])]
    log.info(f"Concrete (non-abstract) categories: {len(concrete)}")

    # 2. Get VGGSound labels to exclude
    vgg_labels = get_vggsound_labels()
    log.info(f"VGGSound categories: {len(vgg_labels)}")

    # Find new categories not in VGGSound (case-insensitive comparison)
    vgg_lower = {l.lower() for l in vgg_labels}
    new_entries = [e for e in concrete if e["name"].lower() not in vgg_lower]
    log.info(f"New AudioSet categories (not in VGGSound): {len(new_entries)}")

    # Also include VGGSound categories that ARE in AudioSet (with descriptions)
    vgg_in_audioset = [e for e in concrete if e["name"].lower() in vgg_lower]
    log.info(f"VGGSound categories also in AudioSet (for descriptions): {len(vgg_in_audioset)}")

    # Build label list and descriptions for new categories
    new_labels = [e["name"] for e in new_entries]
    new_descriptions = {e["name"]: e.get("description", "") for e in new_entries}

    # 3. Load W_v for MLP projection
    w_v_path = MODEL_DIR_V4 / "w_v.bin"
    if not w_v_path.exists():
        log.error(f"W_v not found at {w_v_path}")
        sys.exit(1)
    w_v = load_bin_matrix(w_v_path)
    log.info(f"Loaded W_v: {w_v.shape}")

    # 4. Load text encoder
    from sentence_transformers import SentenceTransformer
    log.info("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 5. Encode labels (use "label: description" for richer embeddings where available)
    encode_texts = []
    for label in new_labels:
        desc = new_descriptions.get(label, "")
        if desc:
            # Use label + short description for better semantic encoding
            encode_texts.append(f"{label}: {desc[:200]}")
        else:
            encode_texts.append(label)

    log.info(f"Encoding {len(encode_texts)} labels...")
    text_embs = text_model.encode(encode_texts, normalize_embeddings=True,
                                   show_progress_bar=True, batch_size=64)
    log.info(f"Text embeddings shape: {text_embs.shape}")  # (N, 384)

    # 6. Project through MLP: ReLU(text_emb @ W_v), L2 norm -> (N, 512)
    projected = mlp_project(text_embs.astype(np.float32), w_v)
    log.info(f"Projected embeddings shape: {projected.shape}")  # (N, 512)

    # Check for dead neurons (all-zero after ReLU)
    alive = (projected.sum(axis=0) != 0).sum()
    log.info(f"Active dimensions after ReLU: {alive}/{projected.shape[1]}")

    # 7. Save outputs
    labels_path = OUTPUT_DIR / "labels.json"
    emb_path = OUTPUT_DIR / "embeddings.npy"
    desc_path = OUTPUT_DIR / "descriptions.json"

    with open(labels_path, "w") as f:
        json.dump(new_labels, f, indent=2)
    np.save(emb_path, projected)
    with open(desc_path, "w") as f:
        json.dump(new_descriptions, f, indent=2)

    log.info(f"Saved {len(new_labels)} labels to {labels_path}")
    log.info(f"Saved {projected.shape} embeddings to {emb_path}")
    log.info(f"Saved descriptions to {desc_path}")

    # Summary
    log.info("=" * 60)
    log.info(f"AudioSet expansion complete!")
    log.info(f"  VGGSound categories:  {len(vgg_labels)}")
    log.info(f"  New AudioSet categories: {len(new_labels)}")
    log.info(f"  Total vocabulary: {len(vgg_labels) + len(new_labels)}")
    log.info(f"  Embedding dim: {projected.shape[1]}")
    log.info(f"  Sample new categories: {new_labels[:10]}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
