#!/usr/bin/env python3
"""Train a small transformer to predict the next event in a temporal sequence.

Architecture: 2-layer transformer encoder (dim=512, 4 heads, ~2M params)
Input: sequence of MLP-projected embeddings (max length 8)
Output: predicted next embedding in the same 512-dim space

Training data: consecutive clip pairs from VGGSound grouped by youtube_id.
"""

import os
import time
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────
EMBED_PATH = "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors"
CSV_PATH = "/opt/brain/data/vggsound/vggsound.csv"
MLP_DIR = "/opt/brain/outputs/cortex/v5_mlp"
OUT_DIR = "/opt/brain/outputs/cortex/temporal_model"

D_MODEL = 512
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 1024
MAX_SEQ = 8
DROPOUT = 0.1

EPOCHS = 200
BATCH = 256
LR_MAX = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

EVAL_EVERY = 20
DEVICE = "cpu"


# ── Model ───────────────────────────────────────────────────────────
class TemporalPredictor(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, max_seq=MAX_SEQ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        # Causal mask: attend only to past
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        out = self.transformer(x, mask=mask)
        return self.output_proj(out[:, -1, :])  # predict from last position


# ── Data ────────────────────────────────────────────────────────────
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


def build_sequences():
    """Group clips by youtube_id and build temporal sequences."""
    print("Loading embeddings...")
    with safe_open(EMBED_PATH, framework="numpy") as f:
        v_emb = f.get_tensor("v_emb")
        a_emb = f.get_tensor("a_emb")

    print("Loading MLP weights...")
    w_v = load_rust_matrix(os.path.join(MLP_DIR, "w_v.bin"))
    w_a = load_rust_matrix(os.path.join(MLP_DIR, "w_a.bin"))

    print("Projecting through MLP...")
    v_proj = mlp_project(v_emb, w_v)
    a_proj = mlp_project(a_emb, w_a)

    # Combined embeddings (average of visual and audio projections)
    combined = (v_proj + a_proj) / 2.0
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True).clip(1e-12)

    print("Loading clip metadata...")
    clips = []
    with open(CSV_PATH) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) >= 3:
                clips.append({
                    "youtube_id": row[0].strip(),
                    "start_sec": float(row[1].strip()),
                    "label": row[2].strip() if len(row) > 2 else "",
                })

    # Group by youtube_id
    yt_groups = defaultdict(list)
    for i, clip in enumerate(clips):
        if i < len(combined):
            yt_groups[clip["youtube_id"]].append((clip["start_sec"], i))

    # Sort each group by start time and build sequences
    sequences = []
    for yt_id, items in yt_groups.items():
        items.sort(key=lambda x: x[0])
        indices = [idx for _, idx in items]
        if len(indices) >= 2:
            # Create sequences of length 2..MAX_SEQ
            for seq_len in range(2, min(len(indices) + 1, MAX_SEQ + 1)):
                for start in range(len(indices) - seq_len + 1):
                    seq_indices = indices[start:start + seq_len]
                    sequences.append(seq_indices)

    print(f"Built {len(sequences)} sequences from {len(yt_groups)} videos")
    return combined, sequences


# ── Training ────────────────────────────────────────────────────────
def train():
    os.makedirs(OUT_DIR, exist_ok=True)

    combined, sequences = build_sequences()
    n_total = len(sequences)
    if n_total < 100:
        print(f"Too few sequences ({n_total}). Need at least 100.")
        return

    # Split: 90% train, 10% val
    np.random.seed(42)
    perm = np.random.permutation(n_total)
    n_val = max(100, n_total // 10)
    val_seqs = [sequences[i] for i in perm[:n_val]]
    train_seqs = [sequences[i] for i in perm[n_val:]]
    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")

    model = TemporalPredictor().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    def get_batch(seqs, batch_size):
        """Sample a batch: each sequence → (context, target)."""
        indices = np.random.choice(len(seqs), batch_size, replace=True)
        max_len = 0
        batch_data = []
        for idx in indices:
            seq = seqs[idx]
            context = seq[:-1]
            target = seq[-1]
            batch_data.append((context, target))
            max_len = max(max_len, len(context))

        # Pad sequences
        ctx_batch = np.zeros((batch_size, max_len, D_MODEL), dtype=np.float32)
        tgt_batch = np.zeros((batch_size, D_MODEL), dtype=np.float32)
        lengths = []
        for i, (ctx, tgt) in enumerate(batch_data):
            embs = combined[ctx]
            ctx_batch[i, :len(ctx)] = embs
            tgt_batch[i] = combined[tgt]
            lengths.append(len(ctx))

        return (torch.from_numpy(ctx_batch).to(DEVICE),
                torch.from_numpy(tgt_batch).to(DEVICE),
                lengths)

    best_val_loss = float("inf")
    steps_per_epoch = max(1, len(train_seqs) // BATCH)

    print(f"\nTraining for {EPOCHS} epochs ({steps_per_epoch} steps/epoch)...")
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        # Cosine LR
        progress = epoch / max(1, EPOCHS - 1)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for step in range(steps_per_epoch):
            ctx, tgt, lengths = get_batch(train_seqs, BATCH)

            pred = model(ctx)  # (B, D_MODEL)
            # Cosine similarity loss: maximize similarity between pred and target
            pred_norm = F.normalize(pred, dim=-1)
            tgt_norm = F.normalize(tgt, dim=-1)
            # InfoNCE-style: pred should be close to its target, far from others
            sim = pred_norm @ tgt_norm.T / 0.07  # temperature
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = F.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                ctx, tgt, lengths = get_batch(val_seqs, min(BATCH, len(val_seqs)))
                pred = model(ctx)
                pred_norm = F.normalize(pred, dim=-1)
                tgt_norm = F.normalize(tgt, dim=-1)
                sim = pred_norm @ tgt_norm.T / 0.07
                val_labels = torch.arange(sim.size(0), device=sim.device)
                val_loss = F.cross_entropy(sim, val_labels).item()

                # MRR on validation
                ranks = (sim.argsort(dim=1, descending=True) == val_labels.unsqueeze(1)).float().argmax(dim=1) + 1
                mrr = (1.0 / ranks).mean().item()

            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | train_loss={avg_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_MRR={mrr:.4f} | lr={lr:.6f} | {elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
                print(f"  → Saved best model (val_loss={val_loss:.4f})")

    total_time = time.time() - t0
    print(f"\nDone! Total time: {total_time:.0f}s")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {OUT_DIR}/model.pt")


if __name__ == "__main__":
    train()
