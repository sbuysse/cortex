#!/usr/bin/env python3
"""Step 4: Train Grounded Text Projection W_t.

Trains W_t (384→512) so text embeddings map into the brain's experiential
MLP space, aligned with averaged V+A projections per category.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
EMBED_CACHE = PROJECT_ROOT / "data/vggsound/.embed_cache/expanded_embeddings.safetensors"
VGGSOUND_CSV = PROJECT_ROOT / "data/vggsound/vggsound.csv"
MODEL_DIR_V4 = PROJECT_ROOT / "outputs/cortex/v4_mlp"
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/text_grounding"


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


def train():
    print("Loading data...")
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

    w_v = load_bin_matrix(MODEL_DIR_V4 / "w_v.bin")
    w_a = load_bin_matrix(MODEL_DIR_V4 / "w_a.bin")

    # Project all through MLP
    v_proj = mlp_project(v, w_v)
    a_proj = mlp_project(a, w_a)

    # Compute per-category centroid in MLP space (average of V+A projections)
    unique_labels = list(dict.fromkeys(labels))
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(l, []).append(i)

    n_cats = len(unique_labels)
    centroids = np.zeros((n_cats, 512), dtype=np.float32)
    for i, cat in enumerate(unique_labels):
        idx = label_to_idx[cat]
        # Average of visual + audio projections
        cat_v = v_proj[idx].mean(axis=0)
        cat_a = a_proj[idx].mean(axis=0)
        centroid = (cat_v + cat_a) / 2.0
        centroids[i] = centroid / (np.linalg.norm(centroid) + 1e-12)

    print(f"{n_cats} categories, centroid shape: {centroids.shape}")

    # Encode labels with sentence-transformer
    print("Encoding labels with sentence-transformer...")
    from sentence_transformers import SentenceTransformer
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embs = text_model.encode(unique_labels, normalize_embeddings=True,
                                   show_progress_bar=False)
    text_embs = text_embs.astype(np.float32)
    print(f"Text embeddings: {text_embs.shape}")

    # Train W_t: text_emb (384) @ W_t → 512, aligned with centroids via InfoNCE
    W_t = nn.Linear(384, 512, bias=False)
    # Initialize close to W_v for warm start
    with torch.no_grad():
        W_t.weight.copy_(torch.from_numpy(w_v.T))

    optimizer = optim.Adam(W_t.parameters(), lr=5e-3)
    temp = 0.07

    text_tensor = torch.from_numpy(text_embs)
    target_tensor = torch.from_numpy(centroids)

    n_epochs = 500
    W_t.train()
    for epoch in range(n_epochs):
        # Project text through W_t
        proj = W_t(text_tensor)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # InfoNCE: each text should match its category centroid
        sim = (proj @ target_tensor.T) / temp
        labels_idx = torch.arange(n_cats)
        loss = nn.functional.cross_entropy(sim, labels_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                top1 = (sim.argmax(dim=1) == labels_idx).float().mean().item()
            print(f"Epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}  top1={top1:.3f}")

    # Save W_t as numpy
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    w_t_np = W_t.weight.detach().numpy().T  # (384, 512)
    header = f"{w_t_np.shape[0]}x{w_t_np.shape[1]}\n".encode()
    with open(OUTPUT_DIR / "w_t.bin", "wb") as f:
        f.write(header)
        f.write(w_t_np.astype(np.float32).tobytes())
    print(f"Saved W_t {w_t_np.shape} to {OUTPUT_DIR / 'w_t.bin'}")

    # Final eval
    W_t.eval()
    with torch.no_grad():
        proj = W_t(text_tensor)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = proj @ target_tensor.T
        ranks = (sim.argsort(dim=1, descending=True) == torch.arange(n_cats).unsqueeze(1)).float().argmax(dim=1) + 1
        mrr = (1.0 / ranks.float()).mean().item()
        r1 = (ranks <= 1).float().mean().item()
        r5 = (ranks <= 5).float().mean().item()
    print(f"Final: MRR={mrr:.4f}  R@1={r1:.4f}  R@5={r5:.4f}")


if __name__ == "__main__":
    train()
