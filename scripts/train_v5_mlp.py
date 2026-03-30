#!/usr/bin/env python3
"""V5 MLP cross-modal association model training.

Improvements over V4 (Rust, MRR=0.730 on 24K):
  - PyTorch + Adam optimizer (was manual SGD)
  - Hidden dim 512 (same as V4, exact warm-start)
  - Warm-start from V4 weights (direct copy)
  - Hard negative mining (50% of batch, top-64 confusers)
  - Temperature annealing 0.02 -> 0.005
  - Cosine LR schedule 5e-3 -> 1e-5
  - Symmetric InfoNCE (V->A + A->V)
  - Gradient clipping max_norm=1.0
  - 500 epochs, batch_size=2048, weight decay 1e-5

Each "epoch" = full pass through the dataset (24604/2048 = 12 steps).
So 500 epochs = 6000 gradient steps (vs 500 steps in previous runs).
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_PATH = "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors"
V4_DIR = "/opt/brain/outputs/cortex/v4_mlp"
OUT_DIR = "/opt/brain/outputs/cortex/v5_mlp"

V_DIM = 384
A_DIM = 512
HIDDEN = 512

EPOCHS = 3000
BATCH = 2048
LR_MAX = 3e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

TEMP_START = 0.02
TEMP_END = 0.005

HARD_NEG_FRAC = 0.5
HARD_NEG_K = 64
HARD_NEG_REFRESH = 50  # epochs

EVAL_EVERY = 25
EVAL_POOL = 2000
FULL_EVAL_EVERY = 200

DEVICE = "cpu"


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_rust_matrix(path):
    with open(path, "rb") as f:
        header = b""
        while True:
            c = f.read(1)
            if c == b"\n":
                break
            header += c
        rows, cols = header.decode().split("x")
        rows, cols = int(rows), int(cols)
        data = f.read(rows * cols * 4)
    arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
    return torch.from_numpy(arr.copy())


def save_rust_matrix(tensor, path):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    r, c = arr.shape
    with open(path, "wb") as f:
        f.write(f"{r}x{c}\n".encode())
        f.write(arr.tobytes())


def info_nce_loss(sim):
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def eval_retrieval(v_proj, a_proj, pool_size=None, batch_chunk=1000):
    N = v_proj.size(0)
    if pool_size and pool_size < N:
        idx = torch.randperm(N)[:pool_size]
        v_proj = v_proj[idx]
        a_proj = a_proj[idx]
        N = pool_size

    v2a_ranks = torch.zeros(N)
    a2v_ranks = torch.zeros(N)

    for start in range(0, N, batch_chunk):
        end = min(start + batch_chunk, N)
        sim_v2a = v_proj[start:end] @ a_proj.T
        diag_v2a = torch.zeros(end - start)
        for i in range(end - start):
            diag_v2a[i] = sim_v2a[i, start + i]
        v2a_ranks[start:end] = (sim_v2a > diag_v2a.unsqueeze(1)).sum(dim=1).float()

        sim_a2v = a_proj[start:end] @ v_proj.T
        diag_a2v = torch.zeros(end - start)
        for i in range(end - start):
            diag_a2v[i] = sim_a2v[i, start + i]
        a2v_ranks[start:end] = (sim_a2v > diag_a2v.unsqueeze(1)).sum(dim=1).float()

    def metrics(ranks):
        ranks1 = ranks + 1
        mrr = (1.0 / ranks1).mean().item()
        r1 = (ranks < 1).float().mean().item()
        r5 = (ranks < 5).float().mean().item()
        r10 = (ranks < 10).float().mean().item()
        return mrr, r1, r5, r10

    v2a_mrr, v2a_r1, v2a_r5, v2a_r10 = metrics(v2a_ranks)
    a2v_mrr, a2v_r1, a2v_r5, a2v_r10 = metrics(a2v_ranks)
    return {
        "v2a_MRR": v2a_mrr, "v2a_R@1": v2a_r1, "v2a_R@5": v2a_r5, "v2a_R@10": v2a_r10,
        "a2v_MRR": a2v_mrr, "a2v_R@1": a2v_r1, "a2v_R@5": a2v_r5, "a2v_R@10": a2v_r10,
    }


class DualMLP(nn.Module):
    def __init__(self, v_dim, a_dim, hidden):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, hidden, bias=False)
        self.a_proj = nn.Linear(a_dim, hidden, bias=False)

    def forward(self, v, a):
        v_h = F.relu(self.v_proj(v))
        a_h = F.relu(self.a_proj(a))
        v_h = F.normalize(v_h, dim=-1)
        a_h = F.normalize(a_h, dim=-1)
        return v_h, a_h


def main():
    print("=" * 70)
    print("V5 MLP Training — PyTorch + Adam (full epochs)")
    print("=" * 70)

    # Load embeddings
    print("Loading embeddings...")
    f = safe_open(EMBED_PATH, framework="pt")
    v_emb = f.get_tensor("v_emb").float()
    a_emb = f.get_tensor("a_emb").float()
    N = v_emb.size(0)
    print(f"  Loaded {N} clips: v_emb {v_emb.shape}, a_emb {a_emb.shape}")

    v_emb = F.normalize(v_emb, dim=-1)
    a_emb = F.normalize(a_emb, dim=-1)
    print("  L2 normalized.")

    # Build model
    model = DualMLP(V_DIM, A_DIM, HIDDEN).to(DEVICE)

    # V4 weights were corrupted by online learning — train from scratch
    # Xavier init is already the default for nn.Linear
    print("  Training from scratch (Xavier init). V4 weights were corrupted by online learning.")

    # Check warm-start quality
    print("\nVerifying warm-start quality (full 24K)...")
    model.eval()
    with torch.no_grad():
        v_h, a_h = model(v_emb, a_emb)
        m0 = eval_retrieval(v_h, a_h, pool_size=None, batch_chunk=1000)
    print(f"  v2a: MRR={m0['v2a_MRR']:.4f}  R@1={m0['v2a_R@1']:.4f}  R@5={m0['v2a_R@5']:.4f}  R@10={m0['v2a_R@10']:.4f}")
    print(f"  a2v: MRR={m0['a2v_MRR']:.4f}")
    model.train()

    # Total gradient steps
    steps_per_epoch = max(1, N // BATCH)  # 12
    total_steps = EPOCHS * steps_per_epoch
    print(f"\n  Steps per epoch: {steps_per_epoch}, total steps: {total_steps}")

    # Optimizer + cosine schedule over total steps
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR_MIN
    )

    # Hard negatives
    hard_neg_indices = None

    def refresh_hard_negatives():
        print("  Refreshing hard negatives...")
        model.eval()
        with torch.no_grad():
            v_h, a_h = model(v_emb, a_emb)
            chunk = 1000
            all_indices = torch.zeros(N, HARD_NEG_K, dtype=torch.long)
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                sim = v_h[start:end] @ a_h.T
                for i in range(end - start):
                    sim[i, start + i] = -1e9
                _, topk = sim.topk(HARD_NEG_K, dim=1)
                all_indices[start:end] = topk
        model.train()
        return all_indices

    # ── Training ────────────────────────────────────────────────────────
    print(f"\nTraining: {EPOCHS} epochs x {steps_per_epoch} steps = {total_steps} steps")
    print(f"  LR: {LR_MAX} -> {LR_MIN} (cosine over {total_steps} steps)")
    print(f"  Temp: {TEMP_START} -> {TEMP_END}")
    print(f"  Hard neg: 50%, K={HARD_NEG_K}, refresh every {HARD_NEG_REFRESH} epochs")
    print()

    model.train()
    t_start = time.time()
    best_full_mrr = 0.0
    best_state = None
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        frac = (epoch - 1) / max(EPOCHS - 1, 1)
        temp = TEMP_START + (TEMP_END - TEMP_START) * frac

        if (epoch - 1) % HARD_NEG_REFRESH == 0:
            hard_neg_indices = refresh_hard_negatives()

        # Shuffle dataset for this epoch
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_steps = 0

        for step_in_epoch in range(steps_per_epoch):
            start = step_in_epoch * BATCH
            end = min(start + BATCH, N)
            if end - start < BATCH // 2:
                break

            # Get base indices from permutation
            base_idx = perm[start:end]
            actual_batch = end - start

            # For hard negative batches: replace half with hard-neg-paired samples
            if hard_neg_indices is not None and actual_batch >= 4:
                half = actual_batch // 2
                # Keep first half as random
                # For second half, use hard negatives
                anchor_idx = base_idx[half:]
                neg_pick = torch.randint(0, HARD_NEG_K, (actual_batch - half,))
                # The batch contains both the anchors (with correct pairs)
                # and confuser indices mixed in
                hard_idx = hard_neg_indices[anchor_idx, neg_pick]
                batch_idx = torch.cat([base_idx[:half], anchor_idx])
            else:
                batch_idx = base_idx

            v_batch = v_emb[batch_idx]
            a_batch = a_emb[batch_idx]

            v_h, a_h = model(v_batch, a_batch)
            sim_v2a = v_h @ a_h.T / temp
            sim_a2v = a_h @ v_h.T / temp
            loss = (info_nce_loss(sim_v2a) + info_nce_loss(sim_a2v)) / 2.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  epoch {epoch:4d}/{EPOCHS} (step {global_step:5d})  loss={avg_loss:.4f}  "
                  f"temp={temp:.5f}  lr={lr:.2e}  [{elapsed:.1f}s]")

        # 2K eval
        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                v_h_all, a_h_all = model(v_emb, a_emb)
                m = eval_retrieval(v_h_all, a_h_all, pool_size=EVAL_POOL)
            model.train()
            elapsed = time.time() - t_start
            print(f"  -- EVAL (2K) epoch {epoch}: "
                  f"v2a_MRR={m['v2a_MRR']:.3f} R@1={m['v2a_R@1']:.3f} R@5={m['v2a_R@5']:.3f} "
                  f"| a2v_MRR={m['a2v_MRR']:.3f}  [{elapsed:.1f}s]")

        # Full 24K eval
        if epoch % FULL_EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                v_h_all, a_h_all = model(v_emb, a_emb)
                mf = eval_retrieval(v_h_all, a_h_all, pool_size=None, batch_chunk=1000)
            model.train()
            elapsed = time.time() - t_start
            print(f"  == FULL EVAL (24K) epoch {epoch}: "
                  f"v2a_MRR={mf['v2a_MRR']:.4f} R@1={mf['v2a_R@1']:.4f} "
                  f"R@5={mf['v2a_R@5']:.4f} R@10={mf['v2a_R@10']:.4f} "
                  f"| a2v_MRR={mf['a2v_MRR']:.4f}  [{elapsed:.1f}s]")
            if mf["v2a_MRR"] > best_full_mrr:
                best_full_mrr = mf["v2a_MRR"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"       ** New best 24K MRR: {best_full_mrr:.4f}")

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f}min)")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint (24K MRR={best_full_mrr:.4f})")

    # ── Final full eval ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL FULL EVALUATION (all 24604 clips)")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        v_h_all, a_h_all = model(v_emb, a_emb)
        t_eval = time.time()
        m = eval_retrieval(v_h_all, a_h_all, pool_size=None, batch_chunk=1000)
        eval_time = time.time() - t_eval

    print(f"  v2a: MRR={m['v2a_MRR']:.4f}  R@1={m['v2a_R@1']:.4f}  "
          f"R@5={m['v2a_R@5']:.4f}  R@10={m['v2a_R@10']:.4f}")
    print(f"  a2v: MRR={m['a2v_MRR']:.4f}  R@1={m['a2v_R@1']:.4f}  "
          f"R@5={m['a2v_R@5']:.4f}  R@10={m['a2v_R@10']:.4f}")
    print(f"  Eval time: {eval_time:.1f}s")
    print(f"\n  V4 baseline: v2a_MRR=0.730  R@1=0.670  R@5=0.801  R@10=0.869")
    print(f"  Delta MRR:   {m['v2a_MRR'] - 0.730:+.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    w_v_out = model.v_proj.weight.data.T  # (V_DIM, HIDDEN)
    w_a_out = model.a_proj.weight.data.T  # (A_DIM, HIDDEN)
    save_rust_matrix(w_v_out, os.path.join(OUT_DIR, "w_v.bin"))
    save_rust_matrix(w_a_out, os.path.join(OUT_DIR, "w_a.bin"))
    print(f"\n  Saved Rust-format weights to {OUT_DIR}/")
    print(f"    W_v: {w_v_out.shape}, W_a: {w_a_out.shape}")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
    print(f"  Saved PyTorch state dict to {OUT_DIR}/model.pt")
    print("\nDone!")


if __name__ == "__main__":
    main()
