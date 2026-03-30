#!/usr/bin/env python3
"""Step 7: Train Self-Model.

Phase A: Compute per-category MRR from V4 MLP on all 24K clips.
Phase B: Train confidence predictor MLP (512→128→1, binary: correct in top-5?).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
EMBED_CACHE = PROJECT_ROOT / "data/vggsound/.embed_cache/expanded_embeddings.safetensors"
VGGSOUND_CSV = PROJECT_ROOT / "data/vggsound/vggsound.csv"
MODEL_DIR_V4 = PROJECT_ROOT / "outputs/cortex/v4_mlp"
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/self_model"


def load_bin_matrix(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split('x'))
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(rows, cols)


def mlp_project(emb, w):
    proj = emb @ w
    proj = np.maximum(proj, 0)
    norm = np.linalg.norm(proj, axis=-1, keepdims=True).clip(1e-12)
    return proj / norm


def load_data():
    import safetensors.numpy as sf
    data = sf.load_file(str(EMBED_CACHE))
    v = data["v_emb"].astype(np.float32)
    a = data["a_emb"].astype(np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)

    labels = []
    with open(VGGSOUND_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                labels.append(parts[2].strip('"'))
            if len(labels) >= v.shape[0]:
                break
    return v, a, labels


class ConfidencePredictor(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def train():
    print("Loading data...")
    v, a, labels = load_data()
    n = v.shape[0]

    w_v = load_bin_matrix(MODEL_DIR_V4 / "w_v.bin")
    w_a = load_bin_matrix(MODEL_DIR_V4 / "w_a.bin")
    print(f"Loaded MLP weights: W_v={w_v.shape}, W_a={w_a.shape}")

    # Project all embeddings
    print("Projecting embeddings...")
    v_proj = mlp_project(v, w_v)  # (N, 512)
    a_proj = mlp_project(a, w_a)  # (N, 512)

    # Phase A: Per-category MRR
    print("Computing per-category MRR...")
    unique_labels = list(dict.fromkeys(labels))
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(l, []).append(i)

    # Compute ranks in batches to avoid OOM
    batch_size = 1000
    ranks = np.zeros(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim_batch = v_proj[start:end] @ a_proj.T  # (batch, N)
        sorted_idx = np.argsort(-sim_batch, axis=1)
        for i in range(end - start):
            global_i = start + i
            rank = np.where(sorted_idx[i] == global_i)[0][0] + 1
            ranks[global_i] = rank
        if (start // batch_size) % 5 == 0:
            print(f"  Computed ranks for {end}/{n} clips")

    # Per-category stats
    category_stats = {}
    for cat in unique_labels:
        idx = label_to_idx[cat]
        cat_ranks = ranks[idx]
        mrr = float(np.mean(1.0 / cat_ranks))
        r1 = float(np.mean(cat_ranks <= 1))
        r5 = float(np.mean(cat_ranks <= 5))
        r10 = float(np.mean(cat_ranks <= 10))
        category_stats[cat] = {
            "n_clips": len(idx),
            "mrr": round(mrr, 4),
            "r1": round(r1, 4),
            "r5": round(r5, 4),
            "r10": round(r10, 4),
            "median_rank": int(np.median(cat_ranks)),
            "mean_rank": round(float(np.mean(cat_ranks)), 1),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "category_stats.json", "w") as f:
        json.dump(category_stats, f, indent=2)
    print(f"Saved category stats for {len(category_stats)} categories")

    # Print best/worst
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["mrr"], reverse=True)
    print("\nTop 10 categories:")
    for cat, stats in sorted_cats[:10]:
        print(f"  {cat}: MRR={stats['mrr']:.3f} R@1={stats['r1']:.3f} ({stats['n_clips']} clips)")
    print("\nBottom 10 categories:")
    for cat, stats in sorted_cats[-10:]:
        print(f"  {cat}: MRR={stats['mrr']:.3f} R@1={stats['r1']:.3f} ({stats['n_clips']} clips)")

    # Phase B: Train confidence predictor
    print("\nPhase B: Training confidence predictor...")
    # Features: projected audio embedding (512-dim)
    # Target: 1 if rank <= 5, 0 otherwise
    targets = (ranks <= 5).astype(np.float32)
    print(f"Positive rate: {targets.mean():.3f}")

    x_tensor = torch.from_numpy(a_proj)
    y_tensor = torch.from_numpy(targets)

    model = ConfidencePredictor(in_dim=512)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 100
    batch_size = 2048

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(x_tensor[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(pred, y_tensor[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                all_pred = model(x_tensor)
                acc = ((all_pred > 0).float() == y_tensor).float().mean().item()
            model.train()
            print(f"Epoch {epoch+1}/{n_epochs}  loss={total_loss/n_batches:.4f}  acc={acc:.3f}")

    torch.save(model.state_dict(), OUTPUT_DIR / "confidence_predictor.pt")
    print(f"Saved confidence predictor to {OUTPUT_DIR / 'confidence_predictor.pt'}")

    # Overall stats
    overall_mrr = float(np.mean(1.0 / ranks))
    overall_r1 = float(np.mean(ranks <= 1))
    overall_r5 = float(np.mean(ranks <= 5))
    print(f"\nOverall: MRR={overall_mrr:.4f}  R@1={overall_r1:.4f}  R@5={overall_r5:.4f}")


if __name__ == "__main__":
    train()
