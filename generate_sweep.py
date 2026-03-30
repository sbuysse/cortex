#!/usr/bin/env python3
"""Generate 800 curated mutations (100 per strategy) for sweep screening.

Each mutation is designed with specific parameter values based on analysis:
- Current best: MRR=0.042, R@1=0.012, estimated_rank≈1.3
- 512x512 M matrix, InfoNCE loss (temp=0.006), batch_size=512, 60K steps
- Rank-1 collapse is the #1 bottleneck
"""
import json
import math

mutations = []

# ============================================================
# Strategy 1: SpectralRegularization (target: HebbianUpdate)
# Adds alpha*I to M every 10 steps to fight rank collapse.
# Key insight: estimated_rank≈1.3 means M is nearly rank-1.
# Need enough alpha to spread energy but not so much it dominates.
# M frobenius norm ≈ 0.05, so alpha should be 0.0001-0.01 range.
# ============================================================
for i in range(100):
    # Log-uniform sweep from 1e-5 to 0.02
    alpha = 10 ** (-5 + i * 3.3 / 99)  # 1e-5 to ~0.02
    mutations.append({
        "target": "hebbian_update",
        "hypothesis": f"Spectral regularization alpha={alpha:.6f} to increase effective rank",
        "strategy": {"type": "spectral_regularization", "alpha": round(alpha, 8)}
    })

# ============================================================
# Strategy 2: DiagonalBoost (target: HebbianUpdate)
# Periodic identity injection. Different from spectral_reg:
# - Spectral reg: small alpha every 10 steps (continuous)
# - Diagonal boost: larger boost at wider intervals (periodic)
# Explore boost magnitude × interval combinations.
# ============================================================
boosts = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
intervals = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000]
idx = 0
for boost in boosts:
    for interval in intervals:
        if idx >= 100:
            break
        mutations.append({
            "target": "hebbian_update",
            "hypothesis": f"Diagonal boost={boost:.5f} every {interval} steps for rank recovery",
            "strategy": {"type": "diagonal_boost", "boost": boost, "interval": interval}
        })
        idx += 1

# ============================================================
# Strategy 3: LabelSmoothing (target: HebbianUpdate)
# Soft InfoNCE targets: positive = 1-eps, negative = eps/(B-1).
# With B=512, even small eps redistributes significant mass.
# Too much smoothing → uniform → no signal.
# Sweet spot likely 0.01-0.2.
# ============================================================
for i in range(100):
    # Linear sweep from 0.005 to 0.4
    epsilon = 0.005 + i * 0.395 / 99
    mutations.append({
        "target": "hebbian_update",
        "hypothesis": f"Label smoothing eps={epsilon:.4f} for softer contrastive targets",
        "strategy": {"type": "label_smoothing", "epsilon": round(epsilon, 6)}
    })

# ============================================================
# Strategy 4: WarmupTemperature (target: TrainingLoop)
# Start high temp (broad softmax, explore), decay to low (sharp, exploit).
# Current fixed temp = 0.006 (very sharp).
# Hypothesis: starting sharp causes early commitment to wrong structure.
# ============================================================
start_temps = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
end_temps = [0.001, 0.002, 0.003, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03]
idx = 0
for st in start_temps:
    for et in end_temps:
        if idx >= 100:
            break
        if st <= et:
            continue  # start must be > end
        mutations.append({
            "target": "training_loop",
            "hypothesis": f"Warmup temp {st:.3f}→{et:.4f}: broad exploration then sharp focus",
            "strategy": {"type": "warmup_temperature", "start_temp": st, "end_temp": et}
        })
        idx += 1
# Pad if needed
while idx < 100:
    st = 0.1 + (idx - 90) * 0.05
    et = 0.004 + (idx - 90) * 0.001
    mutations.append({
        "target": "training_loop",
        "hypothesis": f"Warmup temp {st:.3f}→{et:.4f}",
        "strategy": {"type": "warmup_temperature", "start_temp": round(st, 4), "end_temp": round(et, 5)}
    })
    idx += 1

# ============================================================
# Strategy 5: EmaEval (target: TrainingLoop)
# Polyak averaging: shadow_M = beta * shadow_M + (1-beta) * M.
# Higher beta = slower tracking, smoother model.
# Too high → shadow lags too much. Too low → no benefit.
# Sweet spot: 0.99-0.9999.
# ============================================================
for i in range(100):
    # Log-uniform in (1-beta): from 1-0.9999=0.0001 to 1-0.9=0.1
    one_minus = 10 ** (-4 + i * 3 / 99)  # 0.0001 to 0.1
    beta = 1.0 - one_minus
    mutations.append({
        "target": "training_loop",
        "hypothesis": f"EMA beta={beta:.6f} (half-life≈{0.693/one_minus:.0f} steps) for Polyak averaging",
        "strategy": {"type": "ema_eval", "beta": round(beta, 8)}
    })

# ============================================================
# Strategy 6: Mixup (target: TrainingLoop)
# Interpolate pairs: (λ*v_i + (1-λ)*v_j, λ*a_i + (1-λ)*a_j).
# Alpha controls the interpolation range (Beta distribution shape).
# Small alpha → λ near 0.5 (strong mixing).
# Large alpha → λ near 0 or 1 (mild mixing).
# ============================================================
for i in range(100):
    # Linear sweep from 0.02 to 0.8
    alpha = 0.02 + i * 0.78 / 99
    mutations.append({
        "target": "training_loop",
        "hypothesis": f"Mixup alpha={alpha:.3f} for synthetic positive pair augmentation",
        "strategy": {"type": "mixup", "alpha": round(alpha, 5)}
    })

# ============================================================
# Strategy 7: CyclicLr (target: TrainingLoop)
# Oscillating LR instead of monotonic cosine decay.
# Can escape local optima by periodically increasing LR.
# Parameters: min_factor, max_factor, cycle_steps.
# Base LR = 0.00063, so factor 0.1-3.0 → LR 0.00006-0.0019.
# ============================================================
min_factors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.05]
max_factors = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 1.5, 3.0, 2.0, 10.0]
cycle_steps_opts = [2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 5000, 1000]
idx = 0
for mi, ma, cs in zip(min_factors, max_factors, cycle_steps_opts):
    for variant in range(10):
        if idx >= 100:
            break
        # Perturb slightly for diversity
        mi_v = mi * (0.8 + variant * 0.04)
        ma_v = ma * (0.8 + variant * 0.04)
        cs_v = int(cs * (0.7 + variant * 0.06))
        mutations.append({
            "target": "training_loop",
            "hypothesis": f"Cyclic LR [{mi_v:.2f}-{ma_v:.2f}x] period={cs_v} to escape plateau",
            "strategy": {
                "type": "cyclic_lr",
                "min_factor": round(mi_v, 4),
                "max_factor": round(ma_v, 4),
                "cycle_steps": cs_v
            }
        })
        idx += 1

# ============================================================
# Strategy 8: GradientAccumulation (target: TrainingLoop)
# Effectively larger batches. Current batch=512.
# accumulate_steps multiplies batch: 512*2=1024, 512*4=2048, etc.
# More negatives per effective step, smoother gradients.
# Also try smaller effective batches (by reducing batch_size).
# ============================================================
for i in range(100):
    if i < 25:
        acc = 2  # batch 1024
    elif i < 50:
        acc = 3  # batch 1536
    elif i < 75:
        acc = 4  # batch 2048
    else:
        # Also try batch size reductions (encoded as accumulate_steps=1 with note)
        # But grad_accum implements as batch scale, so use small values
        acc = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2][i - 75]

    mutations.append({
        "target": "training_loop",
        "hypothesis": f"Gradient accumulation {acc}x (effective batch={512*acc}) for smoother updates",
        "strategy": {"type": "gradient_accumulation", "accumulate_steps": acc}
    })

# Verify counts
from collections import Counter
strategy_counts = Counter()
for m in mutations:
    stype = m["strategy"]["type"]
    strategy_counts[stype] += 1

print(f"Total mutations: {len(mutations)}")
for s, c in sorted(strategy_counts.items()):
    print(f"  {s}: {c}")

# Write to sweep file
with open("/opt/brain/sweep_queue.json", "w") as f:
    json.dump(mutations, f, indent=2)

print(f"\nWrote {len(mutations)} mutations to /opt/brain/sweep_queue.json")
print(f"At 20K steps screening with 4 concurrent: ~{len(mutations) * 17 / 4 / 60:.1f} hours")
