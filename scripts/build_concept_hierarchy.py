#!/usr/bin/env python3
"""Build a hierarchical concept tree from category embeddings.

Uses K-Means at multiple granularities on category centroids (VGGSound + AudioSet)
in the MLP-projected 512-dim space. Names groups by most central member.

Output: outputs/cortex/concept_hierarchy.json
"""

import os
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
import csv

# ── Config ──────────────────────────────────────────────────────────
EMBED_PATH = "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors"
CSV_PATH = "/opt/brain/data/vggsound/vggsound.csv"
MLP_DIR = "/opt/brain/outputs/cortex/v5_mlp"
AUDIOSET_DIR = "/opt/brain/outputs/cortex/audioset_expansion"
OUT_PATH = "/opt/brain/outputs/cortex/concept_hierarchy.json"

# Hierarchy levels: 10 top groups → ~40 mid groups → leaves
N_TOP = 10
N_MID = 40


def load_rust_matrix(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split("x"))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(rows, cols)


def mlp_project(emb, w):
    proj = emb @ w
    proj = np.maximum(proj, 0)
    norm = np.linalg.norm(proj, axis=-1, keepdims=True).clip(1e-12)
    return proj / norm


def kmeans(X, k, max_iter=100):
    """Simple K-Means on L2-normalized vectors (cosine K-Means)."""
    n = len(X)
    rng = np.random.RandomState(42)
    # K-means++ init
    centroids = [X[rng.randint(n)]]
    for _ in range(k - 1):
        C = np.stack(centroids)
        dists = 1 - X @ C.T  # cosine distance
        min_dists = dists.min(axis=1).clip(0)
        if min_dists.sum() == 0:
            min_dists = np.ones(n)
        probs = min_dists / min_dists.sum()
        centroids.append(X[rng.choice(n, p=probs)])
    centroids = np.stack(centroids)

    for _ in range(max_iter):
        sims = X @ centroids.T
        assignments = sims.argmax(axis=1)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = X[mask].mean(axis=0)
                new_centroids[j] /= np.linalg.norm(new_centroids[j]) + 1e-12
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return assignments, centroids


def name_group(labels: list[str], centroid: np.ndarray, all_centroids: np.ndarray, all_labels: list[str]) -> str:
    """Name a group by its most central member."""
    if len(labels) == 1:
        return labels[0]
    if len(labels) <= 3:
        return " / ".join(labels)

    # Find the label closest to cluster centroid
    group_indices = [i for i, l in enumerate(all_labels) if l in set(labels)]
    if not group_indices:
        return labels[0]
    group_embs = all_centroids[group_indices]
    sims = group_embs @ centroid
    best = group_indices[int(np.argmax(sims))]
    # Use most central label as representative + count
    return f"{all_labels[best]} & related"


def build_hierarchy():
    print("Loading embeddings...")
    with safe_open(EMBED_PATH, framework="numpy") as f:
        v_emb = f.get_tensor("v_emb")
        a_emb = f.get_tensor("a_emb")

    w_v = load_rust_matrix(os.path.join(MLP_DIR, "w_v.bin"))
    w_a = load_rust_matrix(os.path.join(MLP_DIR, "w_a.bin"))

    v_proj = mlp_project(v_emb, w_v)
    a_proj = mlp_project(a_emb, w_a)

    # Load labels
    clips = []
    with open(CSV_PATH) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) >= 3:
                clips.append(row[2].strip())

    label_to_idx = {}
    for i, label in enumerate(clips):
        if i < len(v_proj):
            label_to_idx.setdefault(label, []).append(i)

    # Add AudioSet expansion
    audioset_labels_path = Path(AUDIOSET_DIR) / "labels.json"
    audioset_emb_path = Path(AUDIOSET_DIR) / "embeddings.npy"
    as_labels = []
    as_centroids = None
    if audioset_labels_path.exists() and audioset_emb_path.exists():
        as_labels = json.loads(audioset_labels_path.read_text())
        as_centroids = np.load(str(audioset_emb_path))
        print(f"Loaded {len(as_labels)} AudioSet categories")

    # Compute category centroids
    all_labels = []
    all_centroids = []
    for label, indices in label_to_idx.items():
        cat_v = v_proj[indices].mean(axis=0)
        cat_a = a_proj[indices].mean(axis=0)
        centroid = (cat_v + cat_a) / 2.0
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        all_labels.append(label)
        all_centroids.append(centroid)

    if as_centroids is not None:
        for i, label in enumerate(as_labels):
            if label not in label_to_idx and i < len(as_centroids):
                c = as_centroids[i] / (np.linalg.norm(as_centroids[i]) + 1e-12)
                all_labels.append(label)
                all_centroids.append(c)

    centroids = np.stack(all_centroids)
    n = len(all_labels)
    print(f"Total categories: {n}")

    # Level 1: Top clusters
    print(f"\nK-Means: {N_TOP} top-level clusters...")
    top_assign, top_centroids = kmeans(centroids, N_TOP)

    # Level 2: Mid clusters
    print(f"K-Means: {N_MID} mid-level clusters...")
    mid_assign, mid_centroids = kmeans(centroids, N_MID)

    # Build tree
    tree = {"name": "all sounds", "type": "root", "n_leaves": n, "children": []}

    for top_id in range(N_TOP):
        top_mask = top_assign == top_id
        top_labels = [all_labels[i] for i in range(n) if top_mask[i]]
        if not top_labels:
            continue

        top_name = name_group(top_labels, top_centroids[top_id], centroids, all_labels)
        print(f"  [{len(top_labels):3d}] {top_name}")

        top_node = {"name": top_name, "type": "branch", "n_leaves": len(top_labels), "children": []}

        # Find mid clusters within this top cluster
        mid_ids = set(mid_assign[i] for i in range(n) if top_mask[i])

        for mid_id in sorted(mid_ids):
            mid_mask = top_mask & (mid_assign == mid_id)
            mid_labels = [all_labels[i] for i in range(n) if mid_mask[i]]
            if not mid_labels:
                continue

            if len(mid_labels) <= 6:
                # Small enough — add leaves directly
                for label in mid_labels:
                    top_node["children"].append({"name": label, "type": "leaf"})
            else:
                mid_name = name_group(mid_labels, mid_centroids[mid_id], centroids, all_labels)
                mid_node = {
                    "name": mid_name, "type": "branch", "n_leaves": len(mid_labels),
                    "children": [{"name": l, "type": "leaf"} for l in mid_labels]
                }
                top_node["children"].append(mid_node)

        tree["children"].append(top_node)

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(tree, f, indent=2)

    print(f"\nSaved concept hierarchy to {OUT_PATH}")
    print(f"\n=== Summary ===")
    for child in tree["children"]:
        n_leaves = child.get("n_leaves", 0)
        n_sub = len([c for c in child.get("children", []) if c.get("type") == "branch"])
        n_direct = len([c for c in child.get("children", []) if c.get("type") == "leaf"])
        print(f"  {child['name']}: {n_leaves} cats, {n_sub} sub-groups, {n_direct} direct leaves")


if __name__ == "__main__":
    build_hierarchy()
