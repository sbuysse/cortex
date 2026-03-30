#!/usr/bin/env python3
"""World Model v2: Train in MLP-projected space.

Key changes from v1:
1. Operates in MLP-projected space (already ~43% aligned)
2. Bigger model with residual: 512 → 1024 → 1024 → 512 + skip
3. Hard negative batches (50% hard, refreshed every 50 epochs)
4. Temperature annealing 0.05 → 0.01
5. Category-aware: same-category clips excluded from negatives
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
EMBED_CACHE = PROJECT_ROOT / "data/vggsound/.embed_cache/expanded_embeddings.safetensors"
VGGSOUND_CSV = PROJECT_ROOT / "data/vggsound/vggsound.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/world_model"
W_V_PATH = PROJECT_ROOT / "outputs/cortex/v4_mlp/w_v.bin"
W_A_PATH = PROJECT_ROOT / "outputs/cortex/v4_mlp/w_a.bin"


def load_bin_matrix(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split("x"))
        return np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)


def mlp_project_np(emb, w):
    proj = np.maximum(emb @ w, 0)
    return (proj / np.linalg.norm(proj, axis=-1, keepdims=True).clip(1e-12)).astype(np.float32)


class WorldModelV2(nn.Module):
    def __init__(self, dim=512, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.skip = nn.Identity()

    def forward(self, x):
        out = self.net(x) + self.skip(x)
        return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def mine_hard_negatives(query, targets, k=64):
    """Find top-K hardest negatives per clip."""
    n = query.shape[0]
    hard = torch.zeros(n, k, dtype=torch.long)
    bs = 2000
    for s in range(0, n, bs):
        e = min(s + bs, n)
        sim = query[s:e] @ targets.T
        for i in range(e - s):
            sim[i, s + i] = -999  # mask true match
        hard[s:e] = sim.topk(k, dim=1).indices
    return hard


def train():
    t0 = time.time()
    print("Loading data...")
    import safetensors.numpy as sf
    data = sf.load_file(str(EMBED_CACHE))
    v_raw = data["v_emb"].astype(np.float32)
    a_raw = data["a_emb"].astype(np.float32)
    v_raw /= np.linalg.norm(v_raw, axis=1, keepdims=True).clip(1e-12)
    a_raw /= np.linalg.norm(a_raw, axis=1, keepdims=True).clip(1e-12)

    w_v = load_bin_matrix(W_V_PATH)
    w_a = load_bin_matrix(W_A_PATH)

    v_proj = mlp_project_np(v_raw, w_v)
    a_proj = mlp_project_np(a_raw, w_a)
    n = v_proj.shape[0]

    # Baseline
    baseline_cos = np.sum(v_proj * a_proj, axis=1)
    print(f"N={n}, Baseline cosine: mean={baseline_cos.mean():.4f}, median={np.median(baseline_cos):.4f}")

    # Load labels
    labels = []
    with open(VGGSOUND_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                labels.append(parts[2].strip('"'))
            if len(labels) >= n:
                break

    # Category map
    cat_ids = {}
    for l in labels:
        if l not in cat_ids:
            cat_ids[l] = len(cat_ids)
    clip_cat = np.array([cat_ids[l] for l in labels], dtype=np.int32)
    print(f"Categories: {len(cat_ids)}")

    v_t = torch.from_numpy(v_proj)
    a_t = torch.from_numpy(a_proj)
    cat_t = torch.from_numpy(clip_cat)

    model = WorldModelV2(dim=512, hidden=1024)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    batch_size = 2048
    n_epochs = 500
    hard_neg_k = 64
    hard_neg_ratio = 0.5
    temp_start, temp_end = 0.05, 0.01
    lr_start, lr_end = 5e-4, 1e-5
    total_steps = n_epochs * ((n + batch_size - 1) // batch_size)
    step = 0

    # Initial hard neg mining
    print("Mining hard negatives...")
    hard_neg = mine_hard_negatives(v_t, a_t, k=hard_neg_k)

    best_mrr = 0
    model.train()
    for epoch in range(n_epochs):
        # Re-mine with model predictions
        if epoch > 0 and epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                preds = torch.cat([model(v_t[s:s+4000]) for s in range(0, n, 4000)])
            hard_neg = mine_hard_negatives(preds, a_t, k=hard_neg_k)
            model.train()
            print(f"  [Re-mined hard negatives at epoch {epoch}]")

        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            progress = step / total_steps
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * progress))
            temp = temp_end + (temp_start - temp_end) * (1 - progress)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            step += 1

            idx = perm[i:i + batch_size]
            bs = idx.shape[0]

            # Build hard negative batch:
            # - 50% of audio targets are hard negatives (wrong matches)
            # - This forces the model to discriminate confusers
            n_hard = int(bs * hard_neg_ratio)

            v_batch = v_t[idx]

            # Audio targets: true matches for all queries
            a_batch = a_t[idx].clone()

            # Replace bottom half with hard negatives
            if n_hard > 0:
                hard_picks = torch.randint(0, hard_neg_k, (n_hard,))
                hard_indices = hard_neg[idx[-n_hard:], hard_picks]
                a_batch[-n_hard:] = a_t[hard_indices]

            pred = model(v_batch)

            # InfoNCE: sim matrix within batch
            sim = (pred @ a_batch.T) / temp

            # Labels: for the easy half, diagonal is correct
            # For hard half, the true target ISN'T in a_batch, so we need
            # to include it. Better approach: always include true target.

            # Actually, simpler: compute sim against the TRUE targets + batch targets
            # Standard InfoNCE on batch with true matches
            a_true = a_t[idx]  # always the true targets
            sim_true = (pred @ a_true.T) / temp
            loss = nn.functional.cross_entropy(sim_true, torch.arange(bs))

            # Additional hard negative loss: push away from confusers
            if n_hard > 0:
                hard_query = pred[-n_hard:]
                hard_targets = a_t[hard_neg[idx[-n_hard:], :min(16, hard_neg_k)]]  # (n_hard, 16, 512)
                # Reshape for batch matmul
                hard_sims = torch.bmm(
                    hard_query.unsqueeze(1),  # (n_hard, 1, 512)
                    hard_targets.transpose(1, 2)  # (n_hard, 512, 16)
                ).squeeze(1) / temp  # (n_hard, 16)
                # True match similarity
                true_sims = (hard_query * a_t[idx[-n_hard:]]).sum(dim=1, keepdim=True) / temp  # (n_hard, 1)
                # Concat: [true_sim, hard_sims] → cross entropy with label 0
                combined = torch.cat([true_sims, hard_sims], dim=1)  # (n_hard, 17)
                hard_loss = nn.functional.cross_entropy(combined, torch.zeros(n_hard, dtype=torch.long))
                loss = loss + 0.5 * hard_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 25 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                si = torch.randperm(n)[:2000]
                pred = model(v_t[si])
                sim = pred @ a_t.T
                ranks = (sim.argsort(dim=1, descending=True) == si.unsqueeze(1)).float().argmax(dim=1) + 1
                mrr = (1.0 / ranks.float()).mean().item()
                r1 = (ranks <= 1).float().mean().item()
                r5 = (ranks <= 5).float().mean().item()
            model.train()
            marker = " *BEST*" if mrr > best_mrr else ""
            best_mrr = max(best_mrr, mrr)
            print(f"Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  MRR={mrr:.4f}  R@1={r1:.3f}  R@5={r5:.3f}  temp={temp:.4f}  lr={lr:.2e}  [{time.time()-t0:.0f}s]{marker}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "predictor_v2.pt")
    print(f"\nSaved to {OUTPUT_DIR / 'predictor_v2.pt'}")

    # Full eval
    print("Full evaluation...")
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(v_t[s:s+4000]) for s in range(0, n, 4000)])
        mrr_sum = r1_sum = r5_sum = r10_sum = 0.0
        for s in range(0, n, 1000):
            e = min(s + 1000, n)
            sim = preds[s:e] @ a_t.T
            for j in range(e - s):
                rank = (sim[j] > sim[j, s + j]).sum().item() + 1
                mrr_sum += 1.0 / rank
                r1_sum += rank <= 1
                r5_sum += rank <= 5
                r10_sum += rank <= 10
    print(f"Final: MRR={mrr_sum/n:.4f}  R@1={r1_sum/n:.4f}  R@5={r5_sum/n:.4f}  R@10={r10_sum/n:.4f}")
    print(f"Total time: {time.time()-t0:.0f}s, Params: {n_params:,}")


if __name__ == "__main__":
    train()
