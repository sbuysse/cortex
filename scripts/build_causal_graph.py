#!/usr/bin/env python3
"""Step 6: Build Causal Graph from VGGSound temporal co-occurrences + semantic similarity.

Parses VGGSound CSV, groups clips by youtube_id, extracts temporal transitions,
builds a 310x310 transition matrix, computes PMI and causal asymmetry.
Also builds a semantic similarity matrix from MLP-projected category centroids,
and a combined causal score blending PMI with semantic similarity.
"""

import numpy as np
import json
import struct
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/opt/brain")
VGGSOUND_CSV = PROJECT_ROOT / "data/vggsound/vggsound.csv"
EMBED_CACHE = PROJECT_ROOT / "data/vggsound/.embed_cache/expanded_embeddings.safetensors"
MLP_W_V = PROJECT_ROOT / "outputs/cortex/v4_mlp/w_v.bin"
MLP_W_A = PROJECT_ROOT / "outputs/cortex/v4_mlp/w_a.bin"
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/causal_graph"


def load_mlp_weight(path):
    """Load MLP weight from binary format: first line 'RxC\\n', then float32 LE data."""
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        rows, cols = [int(x) for x in header.split("x")]
        data = f.read()
        arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
    return arr


def mlp_project(emb, w):
    """Project embeddings through MLP: emb @ w, ReLU, L2 normalize."""
    proj = emb @ w
    proj = np.maximum(proj, 0)  # ReLU
    norms = np.linalg.norm(proj, axis=1, keepdims=True).clip(1e-8)
    proj = proj / norms
    return proj


def load_clips():
    """Load VGGSound CSV and get number of clips from embeddings."""
    import safetensors.numpy as sf
    data = sf.load_file(str(EMBED_CACHE))
    n_clips = data["v_emb"].shape[0]

    clips = []
    with open(VGGSOUND_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                clips.append({
                    "youtube_id": parts[0],
                    "start_sec": int(parts[1]),
                    "label": parts[2].strip('"'),
                })
            if len(clips) >= n_clips:
                break
    return clips


def build():
    print("Loading clips...")
    clips = load_clips()
    n_clips = len(clips)

    # Get unique labels
    unique_labels = list(dict.fromkeys(c["label"] for c in clips))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    n_labels = len(unique_labels)
    print(f"{n_clips} clips, {n_labels} unique labels")

    # Group by youtube_id
    by_video = defaultdict(list)
    for c in clips:
        by_video[c["youtube_id"]].append(c)

    # Count videos with multiple clips
    multi_clip_videos = {k: v for k, v in by_video.items() if len(v) > 1}
    print(f"{len(multi_clip_videos)} multi-clip videos (out of {len(by_video)})")

    # Build transition matrix: T[i,j] = count of label_i followed by label_j
    transitions = np.zeros((n_labels, n_labels), dtype=np.float32)
    label_counts = np.zeros(n_labels, dtype=np.float32)

    for vid, vid_clips in multi_clip_videos.items():
        # Sort by start time
        sorted_clips = sorted(vid_clips, key=lambda c: c["start_sec"])
        for c in sorted_clips:
            li = label_to_idx[c["label"]]
            label_counts[li] += 1
        for i in range(len(sorted_clips) - 1):
            src = label_to_idx[sorted_clips[i]["label"]]
            dst = label_to_idx[sorted_clips[i + 1]["label"]]
            transitions[src, dst] += 1

    # Also count labels from single-clip videos
    for vid, vid_clips in by_video.items():
        if len(vid_clips) == 1:
            li = label_to_idx[vid_clips[0]["label"]]
            label_counts[li] += 1

    total_transitions = transitions.sum()
    print(f"Total transitions: {int(total_transitions)}")

    # Compute PMI (Pointwise Mutual Information)
    # PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
    p_joint = transitions / total_transitions.clip(1)
    p_margin_src = transitions.sum(axis=1) / total_transitions.clip(1)
    p_margin_dst = transitions.sum(axis=0) / total_transitions.clip(1)
    expected = np.outer(p_margin_src, p_margin_dst).clip(1e-12)
    pmi = np.log(p_joint.clip(1e-12) / expected)
    # Zero out where no transitions
    pmi[transitions == 0] = 0

    # Causal asymmetry: A(i,j) = P(j|i) - P(i|j)
    # High positive = i predicts j more than j predicts i
    p_cond_fwd = transitions / transitions.sum(axis=1, keepdims=True).clip(1)  # P(j|i)
    p_cond_bwd = transitions.T / transitions.T.sum(axis=1, keepdims=True).clip(1)  # P(i|j)
    causal_asymmetry = p_cond_fwd - p_cond_bwd.T

    # Save temporal matrices
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "transitions.npy", transitions)
    np.save(OUTPUT_DIR / "pmi.npy", pmi)
    np.save(OUTPUT_DIR / "causal_asymmetry.npy", causal_asymmetry)
    np.save(OUTPUT_DIR / "label_counts.npy", label_counts)

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(unique_labels, f)

    print(f"Saved temporal matrices to {OUTPUT_DIR}")

    # --- Semantic similarity from MLP-projected centroids ---
    print("\nBuilding semantic similarity matrix...")
    import safetensors.numpy as sf
    data = sf.load_file(str(EMBED_CACHE))
    v_emb = data["v_emb"]  # (24604, 384)
    a_emb = data["a_emb"]  # (24604, 512)

    # Load clip-level category labels
    label_data = sf.load_file(str(PROJECT_ROOT / "data/vggsound/.embed_cache/labels.safetensors"))
    clip_labels = label_data["labels"]  # (24604,) int32 indices

    # Load MLP weights
    w_v = load_mlp_weight(MLP_W_V)  # (384, 512)
    w_a = load_mlp_weight(MLP_W_A)  # (512, 512)
    print(f"  MLP weights: w_v={w_v.shape}, w_a={w_a.shape}")

    # Project all embeddings through MLP
    v_proj = mlp_project(v_emb, w_v)  # (24604, 512)
    a_proj = mlp_project(a_emb, w_a)  # (24604, 512)
    print(f"  Projected: v={v_proj.shape}, a={a_proj.shape}")

    # Compute per-category centroids (average of visual + audio projections)
    # The clip_labels are integer indices 0..309 from the embedding cache,
    # but our unique_labels list may have a different ordering from the CSV parse.
    # We need to map between the two orderings.

    # Build a mapping: for each clip index, find which unique_label index it belongs to
    # clip_labels[i] is the category index from the embedding cache ordering
    # We need to figure out the label name for each clip_labels value

    # The CSV-derived clips list has the label name for each clip index
    # clip index i -> clips[i]["label"] -> label_to_idx[label] -> our matrix index
    clip_to_matrix_idx = np.array([label_to_idx[clips[i]["label"]] for i in range(n_clips)], dtype=np.int32)

    centroids = np.zeros((n_labels, 512), dtype=np.float32)
    centroid_counts = np.zeros(n_labels, dtype=np.float32)
    for i in range(n_clips):
        cat = clip_to_matrix_idx[i]
        # Average of visual and audio projections for this clip
        centroids[cat] += (v_proj[i] + a_proj[i]) * 0.5
        centroid_counts[cat] += 1

    # Normalize centroids
    for cat in range(n_labels):
        if centroid_counts[cat] > 0:
            centroids[cat] /= centroid_counts[cat]
    # L2 normalize centroids for cosine similarity
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-8)
    centroids = centroids / centroid_norms

    # Cosine similarity matrix
    semantic_sim = centroids @ centroids.T  # (n_labels, n_labels)
    np.fill_diagonal(semantic_sim, 0)  # zero out self-similarity
    np.save(OUTPUT_DIR / "semantic_sim.npy", semantic_sim)
    print(f"  Saved semantic_sim.npy ({semantic_sim.shape})")

    # --- Combined causal score ---
    # Normalize PMI to [0, 1] range for categories that have temporal data
    # PMI can be negative; we normalize: pmi_norm = (pmi - min) / (max - min) where transitions > 0
    pmi_for_norm = pmi.copy()
    has_temporal = transitions > 0
    if has_temporal.any():
        pmi_min = pmi_for_norm[has_temporal].min()
        pmi_max = pmi_for_norm[has_temporal].max()
        pmi_range = pmi_max - pmi_min if pmi_max > pmi_min else 1.0
        pmi_normalized = (pmi_for_norm - pmi_min) / pmi_range
        pmi_normalized[~has_temporal] = 0
    else:
        pmi_normalized = np.zeros_like(pmi)

    # Normalize semantic_sim to [0, 1] (it's already roughly in [-1, 1] from cosine)
    # Shift to [0, 1]
    sem_min = semantic_sim.min()
    sem_max = semantic_sim.max()
    sem_range = sem_max - sem_min if sem_max > sem_min else 1.0
    semantic_sim_norm = (semantic_sim - sem_min) / sem_range

    # Combined: 0.3 * PMI_normalized + 0.7 * semantic_sim (where temporal data exists)
    # Otherwise: just semantic_sim
    combined = np.where(
        has_temporal,
        0.3 * pmi_normalized + 0.7 * semantic_sim_norm,
        semantic_sim_norm,
    )
    np.fill_diagonal(combined, 0)
    np.save(OUTPUT_DIR / "combined_causal.npy", combined)
    print(f"  Saved combined_causal.npy ({combined.shape})")

    # Print top semantic neighbors
    print("\nTop 20 semantic similarity pairs:")
    flat_sem = semantic_sim.copy()
    for _ in range(20):
        idx = np.unravel_index(flat_sem.argmax(), flat_sem.shape)
        src, dst = idx
        print(f"  {unique_labels[src]} <-> {unique_labels[dst]}  sim={semantic_sim[src,dst]:.4f}")
        flat_sem[src, dst] = -999
        flat_sem[dst, src] = -999  # skip reverse pair

    print("\nTop 20 combined causal scores:")
    flat_comb = combined.copy()
    for _ in range(20):
        idx = np.unravel_index(flat_comb.argmax(), flat_comb.shape)
        src, dst = idx
        print(f"  {unique_labels[src]} -> {unique_labels[dst]}  score={combined[src,dst]:.4f}"
              f"  (pmi_n={pmi_normalized[src,dst]:.3f}, sem={semantic_sim_norm[src,dst]:.3f})")
        flat_comb[src, dst] = -999

    # Print top causal links
    print("\nTop 20 causal links (A → B, PMI):")
    flat_pmi = pmi.copy()
    for _ in range(20):
        idx = np.unravel_index(flat_pmi.argmax(), flat_pmi.shape)
        src, dst = idx
        print(f"  {unique_labels[src]} → {unique_labels[dst]}  PMI={pmi[src,dst]:.3f}  count={int(transitions[src,dst])}")
        flat_pmi[src, dst] = -999

    # Print most predictable categories
    print("\nMost predictable categories (highest avg outgoing PMI):")
    avg_out_pmi = np.where(transitions > 0, pmi, 0).sum(axis=1) / (transitions > 0).sum(axis=1).clip(1)
    for idx in np.argsort(avg_out_pmi)[::-1][:10]:
        print(f"  {unique_labels[idx]}: avg_outgoing_PMI={avg_out_pmi[idx]:.3f}")


if __name__ == "__main__":
    build()
