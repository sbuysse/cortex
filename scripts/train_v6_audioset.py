#!/usr/bin/env python3
"""V6 MLP: Retrain cross-modal model on VGGSound (24K) + AudioSet (2M).

The V5 model was trained only on VGGSound (24K clips). Now we have 2M AudioSet
clips projected into the brain's 512-dim space via the trained adapter.

Strategy: Multi-task training
- Primary batch: VGGSound pairs (v_emb 384-dim, a_emb 512-dim) → project through MLP
- Secondary batch: AudioSet pairs (already in 512-dim) → self-supervised contrastive
- Warm-start from V5 weights

The AudioSet embeddings are already in 512-dim MLP-projected space, so they serve
as targets for the MLP's output. We train the MLP so that its projections are
consistent with the AudioSet embedding space.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ── Config ──────────────────────────────────────────────────────────
EMBED_PATH = "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors"
V5_DIR = "/opt/brain/outputs/cortex/v5_mlp"
AUDIOSET_DIR = "/opt/brain/outputs/cortex/audioset_brain"
OUT_DIR = "/opt/brain/outputs/cortex/v6_mlp"

V_DIM = 384
A_DIM = 512
HIDDEN = 512

EPOCHS = 500
BATCH_VGG = 2048       # VGGSound batch
BATCH_AS = 4096        # AudioSet batch (larger since it's just 512-dim)
LR_MAX = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

TEMP_START = 0.02
TEMP_END = 0.005

HARD_NEG_FRAC = 0.5
HARD_NEG_K = 64
HARD_NEG_REFRESH = 50

EVAL_EVERY = 25
DEVICE = "cpu"


def load_rust_matrix(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split("x"))
        return np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)


def save_rust_matrix(m, path):
    header = f"{m.shape[0]}x{m.shape[1]}\n".encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(m.astype(np.float32).tobytes())


def train():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load VGGSound data
    print("Loading VGGSound embeddings...")
    with safe_open(EMBED_PATH, framework="numpy") as f:
        v_emb = f.get_tensor("v_emb").astype(np.float32)  # (24604, 384)
        a_emb = f.get_tensor("a_emb").astype(np.float32)  # (24604, 512)
    # L2 normalize
    v_emb /= np.linalg.norm(v_emb, axis=1, keepdims=True).clip(1e-12)
    a_emb /= np.linalg.norm(a_emb, axis=1, keepdims=True).clip(1e-12)
    N_vgg = len(v_emb)
    print(f"VGGSound: {N_vgg} clips")

    # Load AudioSet data (balanced train only — fits in memory)
    print("Loading AudioSet embeddings (balanced train)...")
    as_emb = np.load(os.path.join(AUDIOSET_DIR, "bal_train_embeddings.npy")).astype(np.float32)
    as_emb /= np.linalg.norm(as_emb, axis=1, keepdims=True).clip(1e-12)
    N_as = len(as_emb)
    print(f"AudioSet: {N_as} clips (balanced train)")

    # Load V5 weights as warm start
    print("Loading V5 weights for warm start...")
    w_v_init = load_rust_matrix(os.path.join(V5_DIR, "w_v.bin"))
    w_a_init = load_rust_matrix(os.path.join(V5_DIR, "w_a.bin"))
    print(f"W_v: {w_v_init.shape}, W_a: {w_a_init.shape}")

    # PyTorch model
    W_v = nn.Parameter(torch.from_numpy(w_v_init.copy()))
    W_a = nn.Parameter(torch.from_numpy(w_a_init.copy()))

    optimizer = torch.optim.AdamW([W_v, W_a], lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    v_t = torch.from_numpy(v_emb)
    a_t = torch.from_numpy(a_emb)
    as_t = torch.from_numpy(as_emb)

    best_mrr = 0.0
    steps_per_epoch = max(1, N_vgg // BATCH_VGG)
    t0 = time.time()

    print(f"\nTraining V6: {EPOCHS} epochs, VGG batch={BATCH_VGG}, AS batch={BATCH_AS}")

    for epoch in range(EPOCHS):
        # Cosine schedule
        progress = epoch / max(1, EPOCHS - 1)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
        temp = TEMP_START + (TEMP_END - TEMP_START) * progress
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            # ── VGGSound batch: standard InfoNCE ──
            idx = torch.randint(0, N_vgg, (BATCH_VGG,))
            v_batch = v_t[idx]
            a_batch = a_t[idx]

            v_proj = F.normalize(F.relu(v_batch @ W_v), dim=-1)
            a_proj = F.normalize(F.relu(a_batch @ W_a), dim=-1)

            sim_vgg = v_proj @ a_proj.T / temp
            labels_vgg = torch.arange(BATCH_VGG)
            loss_vgg = (F.cross_entropy(sim_vgg, labels_vgg) + F.cross_entropy(sim_vgg.T, labels_vgg)) / 2

            # ── AudioSet batch: consistency loss ──
            # AudioSet embeddings are already in 512-dim MLP space.
            # Train W_a to be consistent: ReLU(as_emb @ W_a) ≈ as_emb
            # This regularizes W_a to preserve the AudioSet embedding structure
            as_idx = torch.randint(0, N_as, (BATCH_AS,))
            as_batch = as_t[as_idx]
            as_proj = F.normalize(F.relu(as_batch @ W_a), dim=-1)

            # Contrastive within AudioSet: each clip should be closest to itself
            sim_as = as_proj @ as_batch.T / temp
            labels_as = torch.arange(BATCH_AS)
            loss_as = F.cross_entropy(sim_as, labels_as)

            # Combined loss
            loss = loss_vgg + 0.3 * loss_as

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W_v, W_a], GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            # Eval MRR on VGGSound
            with torch.no_grad():
                eval_n = min(2000, N_vgg)
                eidx = torch.randperm(N_vgg)[:eval_n]
                ev = F.normalize(F.relu(v_t[eidx] @ W_v), dim=-1)
                ea = F.normalize(F.relu(a_t[eidx] @ W_a), dim=-1)
                sim = ev @ ea.T
                ranks = (sim.argsort(dim=1, descending=True) == torch.arange(eval_n).unsqueeze(1)).float().argmax(dim=1) + 1
                mrr = (1.0 / ranks).mean().item()

            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | loss={avg_loss:.4f} | MRR={mrr:.4f} | "
                  f"temp={temp:.4f} | lr={lr:.6f} | {elapsed:.0f}s")

            if mrr > best_mrr:
                best_mrr = mrr
                # Save weights
                w_v_np = W_v.detach().numpy()
                w_a_np = W_a.detach().numpy()
                save_rust_matrix(w_v_np, os.path.join(OUT_DIR, "w_v.bin"))
                save_rust_matrix(w_a_np, os.path.join(OUT_DIR, "w_a.bin"))
                torch.save({"W_v": W_v.data, "W_a": W_a.data}, os.path.join(OUT_DIR, "model.pt"))
                print(f"  → Saved best model (MRR={mrr:.4f})")

    total_time = time.time() - t0
    print(f"\nDone! Total time: {total_time:.0f}s")
    print(f"Best MRR: {best_mrr:.4f}")
    print(f"Model saved to: {OUT_DIR}/")


if __name__ == "__main__":
    train()
