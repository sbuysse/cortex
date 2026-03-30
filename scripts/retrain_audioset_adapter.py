#!/usr/bin/env python3
"""Retrain the AudioSet adapter using REAL AudioSet embeddings.

Now that we have the actual 128-dim AudioSet embeddings, we can train a proper
adapter to project them into the brain's 512-dim MLP space. We use overlapping
VGGSound clips as anchor points — finding VGGSound clips that appear in AudioSet
and using their MLP-projected embeddings as targets.

Since VGGSound IS a subset of AudioSet (both use YouTube clips), many clips overlap.
We match by finding AudioSet embeddings whose 128-dim vectors, when projected through
the old PCA, are closest to VGGSound audio embeddings PCA'd the same way.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
from glob import glob
import struct

# ── Config ──────────────────────────────────────────────────────────
AUDIOSET_DIR = "/opt/brain/data/audioset/audioset_v1_embeddings"
MLP_DIR = "/opt/brain/outputs/cortex/v5_mlp"
EMBED_PATH = "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors"
OUT_DIR = "/opt/brain/outputs/cortex/audioset_adapted"
BRAIN_OUT = "/opt/brain/outputs/cortex/audioset_brain"

EPOCHS = 200
BATCH = 1024
LR = 3e-4
TEMP = 0.05


class AudioSetAdapter(nn.Module):
    def __init__(self, dim_in=128, dim_hidden=256, dim_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def load_rust_matrix(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split("x"))
        return np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)


def mlp_project(emb, w):
    proj = np.maximum(emb @ w, 0)
    return proj / np.linalg.norm(proj, axis=-1, keepdims=True).clip(1e-12)


def read_tfrecord(path):
    examples = []
    with open(path, 'rb') as f:
        while True:
            buf = f.read(8)
            if len(buf) < 8: break
            length = struct.unpack('Q', buf)[0]
            f.read(4)
            data = f.read(length)
            f.read(4)
            examples.append(data)
    return examples


def extract_128dim(path):
    """Extract raw 128-dim embeddings from TFRecord."""
    embeddings = []
    for raw in read_tfrecord(path):
        pos = 0
        chunks = []
        while pos < len(raw) - 128:
            if pos + 2 < len(raw) and raw[pos] == 128 and raw[pos+1] == 1:
                chunk = raw[pos+2:pos+2+128]
                if len(chunk) == 128:
                    arr = np.frombuffer(chunk, dtype=np.uint8)
                    if arr.std() > 5:
                        chunks.append(arr.astype(np.float32) / 255.0 * 2.0 - 1.0)
                        pos += 130
                        continue
            pos += 1
        if chunks:
            avg = np.mean(chunks, axis=0)
            embeddings.append(avg / (np.linalg.norm(avg) + 1e-12))
    return embeddings


def train():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load brain target embeddings
    print("Loading VGGSound embeddings + MLP weights...")
    with safe_open(EMBED_PATH, framework="numpy") as f:
        a_emb = f.get_tensor("a_emb")  # (24604, 512)
    w_a = load_rust_matrix(os.path.join(MLP_DIR, "w_a.bin"))
    target_512 = mlp_project(a_emb, w_a)  # (24604, 512) — brain space
    N_vgg = len(target_512)

    # Load a sample of AudioSet 128-dim embeddings
    print("Loading AudioSet 128-dim embeddings (balanced train)...")
    all_128 = []
    files = sorted(glob(os.path.join(AUDIOSET_DIR, "bal_train", "*.tfrecord")))
    for i, f in enumerate(files):
        embs = extract_128dim(f)
        all_128.extend(embs)
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(files)}] {len(all_128)} embeddings")
    print(f"Total AudioSet 128-dim: {len(all_128)}")
    as_128 = np.stack(all_128).astype(np.float32)

    # Create training pairs: for each AudioSet clip, find nearest VGGSound clip
    # Project VGGSound to 128-dim via PCA for matching
    print("Building cross-space alignment pairs...")
    # PCA VGGSound audio to 128-dim
    mean_a = a_emb.mean(axis=0)
    centered = a_emb - mean_a
    cov = (centered.T @ centered) / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:128]
    pca = eigenvectors[:, idx]
    vgg_128 = centered @ pca
    vgg_128 = vgg_128 / np.linalg.norm(vgg_128, axis=1, keepdims=True).clip(1e-12)

    # For each AudioSet embedding, find closest VGGSound embedding in 128-dim
    # Do this in batches to save memory
    print("Finding nearest VGGSound for each AudioSet clip...")
    N_as = len(as_128)
    nearest_vgg_idx = np.zeros(N_as, dtype=np.int64)
    MATCH_BATCH = 5000
    for start in range(0, N_as, MATCH_BATCH):
        end = min(start + MATCH_BATCH, N_as)
        sims = as_128[start:end] @ vgg_128.T  # (batch, N_vgg)
        nearest_vgg_idx[start:end] = sims.argmax(axis=1)
        if (start // MATCH_BATCH) % 2 == 0:
            print(f"  Matched {end}/{N_as}")

    # Training data: input=128-dim AudioSet, target=512-dim VGGSound MLP projection
    input_128 = torch.from_numpy(as_128).float()
    target_512_t = torch.from_numpy(target_512[nearest_vgg_idx]).float()
    print(f"Training pairs: {N_as} (AudioSet→VGGSound alignment)")

    # Train adapter
    model = AudioSetAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Adapter params: {n_params:,}")

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N_as)
        epoch_loss = 0
        n_batches = 0

        progress = epoch / max(1, EPOCHS - 1)
        lr = 1e-5 + 0.5 * (LR - 1e-5) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups: pg["lr"] = lr

        for i in range(0, N_as - BATCH, BATCH):
            idx = perm[i:i+BATCH]
            x = input_128[idx]
            y = target_512_t[idx]

            pred = model(x)
            # Combined loss: InfoNCE + MSE
            sim = pred @ y.T / TEMP
            labels = torch.arange(sim.size(0))
            nce_loss = F.cross_entropy(sim, labels)
            mse_loss = F.mse_loss(pred, y)
            loss = nce_loss + 0.1 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        if (epoch+1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample = torch.randperm(min(N_as, 2000))[:2000]
                pred = model(input_128[sample])
                tgt = target_512_t[sample]
                cos_sim = (pred * tgt).sum(dim=1).mean().item()
                sim = pred @ tgt.T
                ranks = (sim.argsort(dim=1, descending=True) == torch.arange(len(sample)).unsqueeze(1)).float().argmax(dim=1) + 1
                mrr = (1.0 / ranks).mean().item()
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | loss={avg:.4f} | cos={cos_sim:.4f} | MRR={mrr:.4f} | lr={lr:.6f} | {elapsed:.0f}s")
            if avg < best_loss:
                best_loss = avg
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "adapter.pt"))

    # Re-project ALL AudioSet embeddings
    print("\n=== Re-projecting all AudioSet embeddings ===")
    model.eval()
    for split in ["bal_train", "eval", "unbal_train"]:
        split_dir = os.path.join(AUDIOSET_DIR, split)
        if not os.path.exists(split_dir): continue
        files = sorted(glob(os.path.join(split_dir, "*.tfrecord")))
        all_embs = []
        for i, fpath in enumerate(files):
            embs = extract_128dim(fpath)
            if embs:
                batch = torch.from_numpy(np.stack(embs).astype(np.float32))
                with torch.no_grad():
                    proj = model(batch).numpy()
                all_embs.append(proj)
            if (i+1) % 500 == 0:
                total = sum(len(e) for e in all_embs)
                print(f"  [{split}] {i+1}/{len(files)} — {total:,} clips")
        if all_embs:
            final = np.concatenate(all_embs)
            out_path = os.path.join(BRAIN_OUT, f"{split}_embeddings.npy")
            np.save(out_path, final)
            print(f"  Saved {split}: {final.shape} → {out_path}")

    # Verify alignment
    print("\n=== Verification ===")
    bal = np.load(os.path.join(BRAIN_OUT, "bal_train_embeddings.npy"))
    # Check similarity with VGGSound MLP-projected embeddings
    sample_vgg = target_512[:1000]
    sample_as = bal[:1000]
    cross_sims = sample_vgg @ sample_as.T
    print(f"VGGSound↔AudioSet cross-similarity: mean={cross_sims.mean():.4f}, max={cross_sims.max():.4f}")
    print(f"Done!")


if __name__ == "__main__":
    train()
