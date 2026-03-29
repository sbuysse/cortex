# Brain-Grounded Companion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Condition HOPE language generation on a live 512-dim Brain state vector (emotion + working memory + fast memory + concepts) via layer-wise bias injection into ContinuumMemoryBlock FFN outputs.

**Architecture:** A `BrainProjection` linear layer (512 → 384) converts the Brain state to a bias added to every ContinuumMemoryBlock FFN output at every generation timestep. New `generate_grounded()` TorchScript method passes this bias. Rust composes the Brain state from live cognitive signals and calls `generate_grounded()` instead of `generate()`.

**Tech Stack:** Python (PyTorch, TorchScript), Rust (tch-rs, brain-cognition, brain-inference, brain-server), ndarray, no new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `scripts/train_hope_companion.py` | Add `BrainProjection`, modify `ContinuumMemoryBlock.forward()`, add `HOPE.generate_grounded()`, thread brain_bias through `HOPE.forward()` |
| `scripts/generate_emotion_table.py` | **New** — generate 8×512 emotion embedding table as `emotion_table.bin` |
| `scripts/export_companion_grounded.py` | **New** — load `best.pt` + emotion table, export TorchScript with both methods |
| `scripts/train_companion_grounded.py` | **New** — Phase 2 fine-tune on existing triples with synthetic brain states |
| `rust/crates/brain-cognition/src/fast_memory.rs` | Add `pattern_at(idx)` method |
| `rust/crates/brain-cognition/src/concepts.rs` | Add `top1_centroid(emb)` method |
| `rust/crates/brain-cognition/src/brain_state.rs` | **New** — `compose_brain_state()`, `load_emotion_table()`, `emotion_to_idx()` |
| `rust/crates/brain-cognition/src/lib.rs` | Export `brain_state` module |
| `rust/crates/brain-cognition/src/state.rs` | Add `emotion_table: Vec<[f32; 512]>` field, load at startup |
| `rust/crates/brain-inference/src/companion_decoder.rs` | Add `generate_grounded()` method |
| `rust/crates/brain-server/src/routes.rs` | Replace companion call with `generate_grounded()` |

---

## Task 1: Modify Python HOPE Model Architecture

**Files:**
- Modify: `scripts/train_hope_companion.py`

This task adds `BrainProjection`, threads `Optional[Tensor]` brain_bias through the model, and adds the `generate_grounded()` TorchScript export method.

- [ ] **Step 1: Add import for Optional at top of file**

The file already imports from `typing`. Add `Optional` to the import:

```python
from typing import List, Optional, Tuple
```

- [ ] **Step 2: Add `BrainProjection` class after `SelfModifyingLayer`**

Insert this class between `SelfModifyingLayer` and `ContinuumMemoryBlock`:

```python
class BrainProjection(nn.Module):
    """Projects 512-dim Brain state to d_model bias, shared across all layers.

    Initialized near-zero so Phase 1 checkpoints are unaffected — bias ≈ 0
    means generate_grounded() behaves like generate() before fine-tuning.
    """

    def __init__(self, brain_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(brain_dim, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=0.001)

    def forward(self, brain_vec: Tensor) -> Tensor:
        # brain_vec: (1, 512) → proj → (1, d_model) → unsqueeze → (1, 1, d_model)
        # The (1, 1, d_model) broadcasts over (B, S, d_model) in ContinuumMemoryBlock
        return self.proj(brain_vec).unsqueeze(1)
```

- [ ] **Step 3: Modify `ContinuumMemoryBlock.forward()` to accept optional brain_bias**

Replace the existing `forward` method:

```python
def forward(self, x: Tensor, brain_bias: Optional[Tensor] = None) -> Tensor:
    h = self.norm(x)
    h = self.linear1(h)
    h = self.act(h)
    h = self.dropout(h)
    h = self.linear2(h)
    if brain_bias is not None:
        h = h + brain_bias  # (B, S, d_model) + (1, 1, d_model) broadcast
    return x + h
```

- [ ] **Step 4: Add `brain_proj` to `HOPE.__init__()`**

Add `brain_dim: int = 512` parameter and instantiate `BrainProjection`:

```python
def __init__(
    self,
    d_model: int,
    n_layers: int,
    vocab_size: int,
    seq_len: int,
    dropout: float = 0.1,
    brain_dim: int = 512,
) -> None:
    super().__init__()
    self.vocab_size = vocab_size
    self.seq_len = seq_len

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.fast_memory = SelfModifyingLayer(d_model)
    self.norm = nn.LayerNorm(d_model)
    self.cms_layers = nn.ModuleList(
        [ContinuumMemoryBlock(d_model, dropout) for _ in range(n_layers)]
    )
    self.head = nn.Linear(d_model, vocab_size)
    self.brain_proj = BrainProjection(brain_dim, d_model)
```

- [ ] **Step 5: Modify `HOPE.forward()` to thread brain_bias through cms_layers**

Replace the existing `forward` method:

```python
def forward(self, tokens: Tensor, brain_bias: Optional[Tensor] = None) -> Tensor:
    """
    Args:
        tokens:     (B, S) long tensor of byte values 0-255
        brain_bias: (1, 1, d_model) optional bias from BrainProjection, or None

    Returns:
        logits: (B, S, vocab_size)
    """
    mask = (tokens != 0).float()  # (B, S)

    x = self.embedding(tokens)                    # (B, S, d_model)
    x = x + self.fast_memory(x, mask)             # fast memory residual
    x = self.norm(x)

    for layer in self.cms_layers:
        x = layer(x, brain_bias)                  # pass bias to every layer

    return self.head(x)                           # (B, S, vocab_size)
```

- [ ] **Step 6: Add `generate_grounded()` TorchScript export method to `HOPE`**

Add this method to the `HOPE` class, after the existing `generate()` method:

```python
@torch.jit.export
def generate_grounded(
    self,
    brain_vec: List[int],
    prompt_bytes: List[int],
    max_new: int,
) -> List[int]:
    """Greedy decode with Brain state conditioning.

    Args:
        brain_vec:    512 ints — Brain state floats packed as round(x * 1000)
        prompt_bytes: list of byte values (0-255)
        max_new:      maximum new tokens to generate

    Returns:
        List[int] of generated byte values (not including prompt)
    """
    result = torch.jit.annotate(List[int], [])

    # Unpack brain state: int → float (reverse of Rust packing)
    bv = torch.tensor(brain_vec, dtype=torch.float32).unsqueeze(0) / 1000.0  # (1, 512)
    brain_bias = self.brain_proj(bv)  # (1, 1, d_model)

    # Clamp prompt to seq_len
    seq_len = self.seq_len
    if len(prompt_bytes) >= seq_len:
        prompt_bytes = prompt_bytes[len(prompt_bytes) - seq_len + 1:]

    context: List[int] = list(prompt_bytes)

    for _ in range(max_new):
        if len(context) < seq_len:
            pad_len = seq_len - len(context)
            padded = [0] * pad_len + context
        else:
            padded = context[len(context) - seq_len:]

        tokens = torch.tensor(padded, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        logits = self.forward(tokens, brain_bias)                      # (1, seq_len, vocab_size)

        pad_offset = seq_len - len(context)
        if pad_offset < 0:
            pad_offset = 0
        real_last = pad_offset + len(context) - 1
        if real_last >= seq_len:
            real_last = seq_len - 1

        next_token = int(logits[0, real_last, :].argmax().item())
        result.append(next_token)
        context.append(next_token)

    return result
```

- [ ] **Step 7: Verify training still works with modified model**

Run a quick smoke-test (5 epochs, small dataset):

```bash
cd /home/sbuysse/Documents/Coding/Projects/Akretio/Brain
python scripts/train_hope_companion.py \
  --triples data/companion_training/raw/triples.jsonl \
  --out /tmp/hope_smoke_test \
  --epochs 5
```

Expected: Training runs without errors, prints loss values, no shape errors.

- [ ] **Step 8: Commit**

```bash
git add scripts/train_hope_companion.py
git commit -m "feat: add BrainProjection + generate_grounded to HOPE model"
```

---

## Task 2: Emotion Table Generator + Export Script

**Files:**
- Create: `scripts/generate_emotion_table.py`
- Create: `scripts/export_companion_grounded.py`

- [ ] **Step 1: Write `scripts/generate_emotion_table.py`**

```python
"""Generate 8×512 emotion embedding table as raw float32 binary.

Emotions (index order, must match Rust emotion_to_idx):
  0: neutral, 1: sad, 2: pain, 3: happy,
  4: fearful, 5: angry, 6: confused, 7: tired

Output: emotion_table.bin — 8 × 512 × 4 bytes = 16384 bytes, row-major f32 LE.
"""

import argparse
import struct
from pathlib import Path

import numpy as np


EMOTIONS = ["neutral", "sad", "pain", "happy", "fearful", "angry", "confused", "tired"]
DIM = 512
SEED = 42


def generate(out_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    table = rng.standard_normal((len(EMOTIONS), DIM)).astype(np.float32)

    # L2-normalize each row
    norms = np.linalg.norm(table, axis=1, keepdims=True).clip(min=1e-12)
    table /= norms

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.tofile(str(out_path))

    print(f"Saved emotion table {table.shape} → {out_path}")
    print(f"  Row norms: {np.linalg.norm(table, axis=1).round(4).tolist()}")
    for i, name in enumerate(EMOTIONS):
        print(f"  {i}: {name}  mean={table[i].mean():.4f}  std={table[i].std():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/cortex/hope_companion/emotion_table.bin",
        help="Output path for emotion_table.bin",
    )
    args = parser.parse_args()
    generate(Path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test of emotion table generation**

```bash
python scripts/generate_emotion_table.py --out /tmp/emotion_table_test.bin
```

Expected output:
```
Saved emotion table (8, 512) → /tmp/emotion_table_test.bin
  Row norms: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  0: neutral  mean=...  std=...
  ...
```

- [ ] **Step 3: Write `scripts/export_companion_grounded.py`**

```python
"""Export HOPE companion model to TorchScript with generate_grounded() method.

Loads from training checkpoint (best.pt), attaches BrainProjection weights
(loaded from grounded checkpoint if available, else fresh near-zero init),
and exports as TorchScript.

Usage:
  python export_companion_grounded.py \
    --base-checkpoint outputs/cortex/hope_companion/best.pt \
    --grounded-checkpoint outputs/cortex/hope_companion/grounded_best.pt \  # optional
    --out outputs/cortex/hope_companion/hope_companion_ts.pt
"""

import argparse
import sys
from pathlib import Path

import torch

# Import HOPE from training script (same file, same class definition)
sys.path.insert(0, str(Path(__file__).parent))
from train_hope_companion import HOPE, COMPANION_NANO


def export(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model = HOPE(**COMPANION_NANO).to(device)

    # Load base weights (strict=False: brain_proj not in base checkpoint)
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(base_ckpt["model_state_dict"], strict=False)
    print(f"Base checkpoint loaded. Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    # Load grounded weights if available (overwrites BrainProjection + FFN weights)
    if args.grounded_checkpoint and Path(args.grounded_checkpoint).exists():
        grounded_ckpt = torch.load(args.grounded_checkpoint, map_location="cpu")
        missing2, _ = model.load_state_dict(grounded_ckpt["model_state_dict"], strict=False)
        print(f"Grounded checkpoint loaded. Missing: {missing2}")
    else:
        print("No grounded checkpoint — using near-zero BrainProjection (text-only fallback)")

    model.eval()

    # Verify both methods exist before scripting
    assert hasattr(model, "generate"), "generate() missing"
    assert hasattr(model, "generate_grounded"), "generate_grounded() missing"

    # Quick sanity: generate_grounded with zero brain_vec == generate (approx)
    prompt = list(b"[CTX] Hello. [USR] Hi. [CRT] ")
    brain_zeros = [0] * 512
    with torch.no_grad():
        out_base = model.generate(prompt, 10)
        out_grounded = model.generate_grounded(brain_zeros, prompt, 10)
    print(f"generate():          {bytes(out_base[:20])!r}")
    print(f"generate_grounded(): {bytes(out_grounded[:20])!r}")

    # TorchScript export
    scripted = torch.jit.script(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"\nExported TorchScript model → {out_path}")

    # Verify exported model has both methods
    loaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        v1 = loaded.generate(prompt, 5)
        v2 = loaded.generate_grounded(brain_zeros, prompt, 5)
    print(f"Verification generate():          {bytes(v1)!r}")
    print(f"Verification generate_grounded(): {bytes(v2)!r}")
    print("Export OK.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint",
                        default="outputs/cortex/hope_companion/best.pt")
    parser.add_argument("--grounded-checkpoint", default=None)
    parser.add_argument("--out",
                        default="outputs/cortex/hope_companion/hope_companion_ts.pt")
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run export smoke-test (uses existing base checkpoint if available)**

If `outputs/cortex/hope_companion/best.pt` does not exist, create a fresh model for testing:

```bash
python -c "
import torch, sys
sys.path.insert(0, 'scripts')
from train_hope_companion import HOPE, COMPANION_NANO
import os; os.makedirs('outputs/cortex/hope_companion', exist_ok=True)
m = HOPE(**COMPANION_NANO)
torch.save({'model_state_dict': m.state_dict(), 'epoch': 0, 'val_loss': 99.0, 'config': COMPANION_NANO},
           'outputs/cortex/hope_companion/best.pt')
print('Dummy checkpoint created')
"
python scripts/export_companion_grounded.py
```

Expected: `Export OK.` with both methods verified.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_emotion_table.py scripts/export_companion_grounded.py
git commit -m "feat: emotion table generator + grounded export script"
```

---

## Task 3: Rust Helpers — HopfieldMemory and ConceptCodebook

**Files:**
- Modify: `rust/crates/brain-cognition/src/fast_memory.rs`
- Modify: `rust/crates/brain-cognition/src/concepts.rs`

These helpers expose the internal embedding data needed by `compose_brain_state`.

- [ ] **Step 1: Write failing test for `pattern_at` in `fast_memory.rs`**

Add to the `#[cfg(test)]` block at the bottom of `fast_memory.rs`:

```rust
#[test]
fn test_pattern_at() {
    let mut fm = HopfieldMemory::new(512, 10);
    let pattern: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
    fm.store(&pattern, "test");
    // retrieve gives idx=0 for the only stored pattern
    let matches = fm.retrieve(&pattern, 1);
    assert_eq!(matches.len(), 1);
    let idx = matches[0].idx;
    let retrieved = fm.pattern_at(idx);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().len(), 512);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd rust && cargo test -p brain-cognition test_pattern_at 2>&1 | tail -5
```

Expected: FAIL with `no method named 'pattern_at'`.

- [ ] **Step 3: Add `pattern_at()` method to `HopfieldMemory`**

Add after the `get_recent()` method in `fast_memory.rs`:

```rust
/// Get the stored (L2-normalized) pattern at a given index.
/// Returns None if idx is out of range.
pub fn pattern_at(&self, idx: usize) -> Option<&[f32]> {
    self.patterns.get(idx).map(|p| p.as_slice())
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd rust && cargo test -p brain-cognition test_pattern_at 2>&1 | tail -5
```

Expected: `test test_pattern_at ... ok`

- [ ] **Step 5: Write failing test for `top1_centroid` in `concepts.rs`**

Add to `concepts.rs` (add a `#[cfg(test)]` block if not present):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_top1_centroid_returns_512dim() {
        // Build a minimal codebook with 2 centroids directly
        let mut centroids = Array2::<f32>::zeros((2, 512));
        centroids[[0, 0]] = 1.0;
        centroids[[1, 1]] = 1.0;
        let cb = ConceptCodebook { centroids, labels: vec!["a".into(), "b".into()] };

        // Query near centroid 0
        let mut q = vec![0.0f32; 512];
        q[0] = 1.0;
        let result = cb.top1_centroid(&q);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 512);
    }

    #[test]
    fn test_top1_centroid_empty_returns_none() {
        let cb = ConceptCodebook {
            centroids: Array2::zeros((0, 512)),
            labels: vec![],
        };
        let q = vec![0.0f32; 512];
        assert!(cb.top1_centroid(&q).is_none());
    }
}
```

- [ ] **Step 6: Run tests to verify they fail**

```bash
cd rust && cargo test -p brain-cognition test_top1_centroid 2>&1 | tail -5
```

Expected: FAIL with `no method named 'top1_centroid'`.

- [ ] **Step 7: Add `top1_centroid()` to `ConceptCodebook`**

Add after the `decompose()` method in `concepts.rs`. The `l2_norm` helper already exists in the file:

```rust
/// Return the centroid embedding (512-dim) of the concept nearest to `emb`.
/// Returns None if the codebook is empty.
pub fn top1_centroid(&self, emb: &[f32]) -> Option<Vec<f32>> {
    let n = self.centroids.nrows();
    if n == 0 { return None; }
    let emb_norm = l2_norm(emb);
    let best = (0..n).max_by(|&i, &j| {
        let si: f32 = self.centroids.row(i).iter().zip(&emb_norm).map(|(a, b)| a * b).sum();
        let sj: f32 = self.centroids.row(j).iter().zip(&emb_norm).map(|(a, b)| a * b).sum();
        si.partial_cmp(&sj).unwrap_or(std::cmp::Ordering::Equal)
    })?;
    Some(self.centroids.row(best).to_vec())
}
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd rust && cargo test -p brain-cognition test_top1_centroid 2>&1 | tail -5
```

Expected: both tests pass.

- [ ] **Step 9: Run full brain-cognition test suite**

```bash
cd rust && cargo test -p brain-cognition 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add rust/crates/brain-cognition/src/fast_memory.rs rust/crates/brain-cognition/src/concepts.rs
git commit -m "feat: add pattern_at() to HopfieldMemory, top1_centroid() to ConceptCodebook"
```

---

## Task 4: Rust — `brain_state.rs` and `lib.rs` export

**Files:**
- Create: `rust/crates/brain-cognition/src/brain_state.rs`
- Modify: `rust/crates/brain-cognition/src/lib.rs`

- [ ] **Step 1: Write failing test for `compose_brain_state` in a new test file**

This test will be placed inline in `brain_state.rs`. First confirm there's no existing file:

```bash
ls rust/crates/brain-cognition/src/brain_state.rs 2>&1
```

Expected: `No such file or directory`.

- [ ] **Step 2: Create `brain_state.rs` with implementation and tests**

```rust
//! Brain state composition — assembles a 512-dim cognitive vector from live signals.
//!
//! Called before each `generate_grounded()` call to give HOPE a live snapshot
//! of what the Brain is currently experiencing.

use crate::working_memory::WorkingMemory;
use crate::fast_memory::HopfieldMemory;
use crate::concepts::ConceptCodebook;
use std::path::Path;

/// Emotion name → index mapping (must match `generate_emotion_table.py`).
pub fn emotion_to_idx(emotion: &str) -> usize {
    match emotion {
        "neutral"  => 0,
        "sad"      => 1,
        "pain"     => 2,
        "happy"    => 3,
        "fearful"  => 4,
        "angry"    => 5,
        "confused" => 6,
        "tired"    => 7,
        _          => 0, // unknown → neutral
    }
}

/// Compose a 512-dim Brain state vector from live cognitive signals.
///
/// Components and weights:
///   - Working memory centroid (0.35): average of active WM item embeddings
///   - Fast memory retrieval (0.25):   top-1 Hopfield pattern for WM centroid
///   - Concept centroid (0.25):        nearest concept centroid to WM centroid
///   - Emotion embedding (0.15):       row from emotion_table
///
/// Missing components have their weight redistributed proportionally.
/// Output is L2-normalized to unit length.
/// Returns a zero vector if all signals are unavailable.
pub fn compose_brain_state(
    wm: &WorkingMemory,
    fm: &HopfieldMemory,
    codebook: Option<&ConceptCodebook>,
    emotion: &str,
    emotion_table: &[[f32; 512]],
) -> Vec<f32> {
    const DIM: usize = 512;

    // 1. Working memory centroid: average of 512-dim WM embeddings
    let wm_embs: Vec<&[f32]> = wm.get_embeddings()
        .into_iter()
        .filter(|e| e.len() == DIM)
        .collect();

    let wm_centroid: Option<Vec<f32>> = if wm_embs.is_empty() {
        None
    } else {
        let mut sum = vec![0.0f32; DIM];
        for e in &wm_embs {
            for (s, v) in sum.iter_mut().zip(*e) { *s += v; }
        }
        Some(l2_normalize(&sum))
    };

    // 2. Fast memory retrieval: top-1 Hopfield pattern for WM centroid
    let fm_vec: Option<Vec<f32>> = wm_centroid.as_ref().and_then(|wm_c| {
        fm.retrieve(wm_c, 1)
            .first()
            .and_then(|m| fm.pattern_at(m.idx))
            .map(|p| l2_normalize(p))
    });

    // 3. Concept centroid: nearest concept to WM centroid
    let concept_vec: Option<Vec<f32>> = wm_centroid.as_ref().and_then(|wm_c| {
        codebook.and_then(|cb| cb.top1_centroid(wm_c))
    });

    // 4. Emotion embedding: fixed lookup table row
    let emotion_idx = emotion_to_idx(emotion);
    let emotion_vec: Option<&[f32; 512]> = emotion_table.get(emotion_idx);

    // Weighted sum with proportional redistribution for missing components
    let mut result = vec![0.0f32; DIM];
    let mut total_weight = 0.0f32;

    if let Some(ref v) = wm_centroid {
        let w = 0.35;
        for (r, x) in result.iter_mut().zip(v) { *r += w * x; }
        total_weight += w;
    }
    if let Some(ref v) = fm_vec {
        let w = 0.25;
        for (r, x) in result.iter_mut().zip(v) { *r += w * x; }
        total_weight += w;
    }
    if let Some(ref v) = concept_vec {
        let w = 0.25;
        for (r, x) in result.iter_mut().zip(v) { *r += w * x; }
        total_weight += w;
    }
    if let Some(ev) = emotion_vec {
        let w = 0.15;
        for (r, x) in result.iter_mut().zip(ev.iter()) { *r += w * x; }
        total_weight += w;
    }

    if total_weight < 1e-12 {
        return vec![0.0f32; DIM]; // all signals unavailable
    }

    // Normalize by total weight (equivalent to proportional redistribution)
    for r in &mut result { *r /= total_weight; }
    l2_normalize(&result)
}

/// Load emotion table from `emotion_table.bin` (raw f32 LE, 8×512 row-major).
/// Returns 8 zero rows if the file is missing or malformed — HOPE falls back to text-only.
pub fn load_emotion_table(path: &Path) -> Vec<[f32; 512]> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return vec![[0.0f32; 512]; 8],
    };
    let expected = 8 * 512 * 4;
    if bytes.len() < expected {
        tracing::warn!("emotion_table.bin too small ({} bytes, need {})", bytes.len(), expected);
        return vec![[0.0f32; 512]; 8];
    }
    let mut table = Vec::with_capacity(8);
    for i in 0..8 {
        let mut row = [0.0f32; 512];
        for j in 0..512 {
            let off = (i * 512 + j) * 4;
            row[j] = f32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]);
        }
        table.push(row);
    }
    table
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::working_memory::WorkingMemory;
    use crate::fast_memory::HopfieldMemory;

    fn zero_table() -> Vec<[f32; 512]> {
        vec![[0.0f32; 512]; 8]
    }

    fn unit_table() -> Vec<[f32; 512]> {
        let mut table = vec![[0.0f32; 512]; 8];
        for (i, row) in table.iter_mut().enumerate() {
            row[i] = 1.0; // each emotion: unit vector in its own dimension
        }
        table
    }

    #[test]
    fn test_empty_memory_returns_zeros() {
        let wm = WorkingMemory::new(7, 0.85, 0.15);
        let fm = HopfieldMemory::new(512, 100);
        let result = compose_brain_state(&wm, &fm, None, "neutral", &zero_table());
        // All zeros — no signals, zero table
        assert_eq!(result.len(), 512);
        assert!(result.iter().all(|&x| x == 0.0), "Expected zero vector");
    }

    #[test]
    fn test_emotion_only_produces_unit_vector() {
        let wm = WorkingMemory::new(7, 0.85, 0.15);
        let fm = HopfieldMemory::new(512, 100);
        let table = unit_table();
        // "sad" = index 1 → table[1] = unit vector in dim 1
        let result = compose_brain_state(&wm, &fm, None, "sad", &table);
        assert_eq!(result.len(), 512);
        // Output should be L2-normalized (norm ≈ 1.0) if non-zero
        // With zero wm/fm/concept: only emotion contributes (weight 0.15/0.15 = 1.0 after normalize)
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn test_wm_item_contributes() {
        let mut wm = WorkingMemory::new(7, 0.85, 0.15);
        let emb: Vec<f32> = (0..512).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        wm.update(emb, "test".into(), "audio".into());
        let fm = HopfieldMemory::new(512, 100);
        let result = compose_brain_state(&wm, &fm, None, "neutral", &zero_table());
        assert_eq!(result.len(), 512);
        // WM item at dim 0 should push result toward dim 0
        assert!(result[0] > 0.0, "Expected WM contribution in dim 0");
    }

    #[test]
    fn test_output_is_unit_norm_when_nonempty() {
        let mut wm = WorkingMemory::new(7, 0.85, 0.15);
        let emb: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
        wm.update(emb, "test".into(), "audio".into());
        let fm = HopfieldMemory::new(512, 100);
        let result = compose_brain_state(&wm, &fm, None, "neutral", &zero_table());
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn test_emotion_to_idx_unknown_maps_to_neutral() {
        assert_eq!(emotion_to_idx("blorp"), 0);
        assert_eq!(emotion_to_idx("sad"), 1);
        assert_eq!(emotion_to_idx("tired"), 7);
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cd rust && cargo test -p brain-cognition brain_state 2>&1 | tail -15
```

Expected: 5 tests pass.

- [ ] **Step 4: Add `brain_state` to `lib.rs`**

In `rust/crates/brain-cognition/src/lib.rs`, add:

```rust
pub mod brain_state;
```

And add the re-export line after the existing `pub use state::BrainState;`:

```rust
pub use brain_state::{compose_brain_state, load_emotion_table};
```

- [ ] **Step 5: Verify everything compiles**

```bash
cd rust && cargo build -p brain-cognition 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add rust/crates/brain-cognition/src/brain_state.rs rust/crates/brain-cognition/src/lib.rs
git commit -m "feat: compose_brain_state() — assembles 512-dim cognitive vector from live signals"
```

---

## Task 5: Rust — `CompanionDecoder.generate_grounded()`

**Files:**
- Modify: `rust/crates/brain-inference/src/companion_decoder.rs`

- [ ] **Step 1: Write failing test**

Add to `companion_decoder.rs` (it has no test block yet — add one at the bottom):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_vec_packing_roundtrip() {
        // Floats in [-1, 1] packed as i64 (×1000) and unpacked (/1000) lose < 0.001
        let values: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.333, 0.999, 1.0];
        for &v in &values {
            let packed = (v * 1000.0).round() as i64;
            let unpacked = packed as f32 / 1000.0;
            assert!((unpacked - v).abs() < 0.001, "v={v}, unpacked={unpacked}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails** (it should pass since it's pure logic — verify)

```bash
cd rust && cargo test -p brain-inference test_brain_vec_packing 2>&1 | tail -5
```

Expected: PASS (logic test, no model needed).

- [ ] **Step 3: Add `generate_grounded()` to `CompanionDecoder`**

In `companion_decoder.rs`, add after the `generate()` method:

```rust
/// Generate a response conditioned on the live Brain state vector.
///
/// `brain_vec`: 512-dim unit vector from `compose_brain_state()`.
/// Floats are packed as `round(x * 1000) as i64` for TorchScript compatibility.
///
/// Falls back to `generate()` if the TorchScript model does not export
/// `generate_grounded` (e.g., old model without grounded training).
pub fn generate_grounded(
    &self,
    brain_vec: &[f32],
    context_text: &str,
    user_message: &str,
    max_tokens: usize,
) -> String {
    let brain_ints: Vec<i64> = brain_vec.iter()
        .map(|&x| (x * 1000.0).round() as i64)
        .collect();

    let prompt = format!("[CTX] {} [USR] {} [CRT] ", context_text, user_message);
    let prompt_bytes: Vec<i64> = prompt.bytes().map(|b| b as i64).collect();

    let result = match self.model.method_is("generate_grounded", &[
        IValue::IntList(brain_ints),
        IValue::IntList(prompt_bytes),
        IValue::Int(max_tokens as i64),
    ]) {
        Ok(out) => out,
        Err(e) => {
            eprintln!("[companion_decoder] generate_grounded failed ({e}) — fallback to generate()");
            return self.generate(context_text, user_message, max_tokens);
        }
    };

    let byte_ids: Vec<i64> = match result {
        IValue::IntList(v) => v,
        other => {
            eprintln!("[companion_decoder] generate_grounded unexpected output: {other:?}");
            return self.generate(context_text, user_message, max_tokens);
        }
    };

    let bytes: Vec<u8> = byte_ids.iter()
        .filter_map(|&b| if b >= 0 && b < 256 { Some(b as u8) } else { None })
        .collect();

    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    let mut text = text.trim().to_string();
    if let Some(c) = text.get_mut(0..1) { c.make_ascii_uppercase(); }
    if !text.is_empty()
        && !text.ends_with('.')
        && !text.ends_with('!')
        && !text.ends_with('?')
    {
        text.push('.');
    }
    text
}
```

- [ ] **Step 4: Verify compilation**

```bash
cd rust && cargo build -p brain-inference 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add rust/crates/brain-inference/src/companion_decoder.rs
git commit -m "feat: add generate_grounded() to CompanionDecoder with graceful fallback"
```

---

## Task 6: Rust — `BrainState.emotion_table` + Routes Wiring

**Files:**
- Modify: `rust/crates/brain-cognition/src/state.rs`
- Modify: `rust/crates/brain-server/src/routes.rs`

- [ ] **Step 1: Add `emotion_table` field to `BrainState` struct in `state.rs`**

In the `BrainState` struct definition (around line 17), add after `companion_decoder`:

```rust
pub emotion_table: Vec<[f32; 512]>,
```

- [ ] **Step 2: Load `emotion_table` in `BrainState::new()` in `state.rs`**

After the `companion_decoder` loading block (after line ~137), add:

```rust
let emotion_table_path = config.project_root
    .join("outputs/cortex/hope_companion/emotion_table.bin");
let emotion_table = crate::brain_state::load_emotion_table(&emotion_table_path);
let all_zero = emotion_table.iter().all(|row| row.iter().all(|&x| x == 0.0));
if all_zero {
    tracing::info!("Emotion table not found — grounded mode uses zero emotion bias");
} else {
    tracing::info!("Emotion table loaded ({} emotions)", emotion_table.len());
}
```

- [ ] **Step 3: Add `emotion_table` to the `Ok(Self { ... })` construction block**

In the `Ok(Self { ... })` block at the end of `BrainState::new()`, add:

```rust
emotion_table,
```

- [ ] **Step 4: Verify `state.rs` compiles**

```bash
cd rust && cargo build -p brain-cognition 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 5: Update the companion call in `routes.rs`**

Find the section starting at `// ── 3a. Try native HOPE decoder first` (around line 2537) and replace the entire block:

```rust
// ── 3a. Try native HOPE decoder first (Brain-grounded) ─────────────
let native_response: Option<String> = if let Some(dec) = &brain.companion_decoder {
    let (brain_vec, hope_ctx) = {
        let pm  = brain.personal_memory.lock().unwrap();
        let wm  = brain.working_memory.lock().unwrap();
        let fm  = brain.fast_memory.lock().unwrap();
        let cb  = brain.codebook.lock().unwrap();
        let bv  = brain_cognition::brain_state::compose_brain_state(
            &wm,
            &fm,
            cb.as_ref(),
            detected_emotion,
            &brain.emotion_table,
        );
        let ctx = brain_cognition::personal::build_hope_context(&pm);
        (bv, ctx)
        // all guards drop here — no Mutex held across the generate call
    };
    let response = dec.generate_grounded(&brain_vec, &hope_ctx, &message, 130);
    if response.len() > 5 { Some(response) } else { None }
} else {
    None
};
```

- [ ] **Step 6: Build the full server**

```bash
cd rust && cargo build -p brain-server 2>&1 | tail -15
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add rust/crates/brain-cognition/src/state.rs rust/crates/brain-server/src/routes.rs
git commit -m "feat: wire compose_brain_state into companion route — Brain-grounded generation"
```

---

## Task 7: Phase 2 Fine-tune Script

**Files:**
- Create: `scripts/train_companion_grounded.py`

- [ ] **Step 1: Write `scripts/train_companion_grounded.py`**

```python
"""Phase 2: Fine-tune HOPE with Brain state conditioning.

Loads Phase 1 checkpoint, freezes all weights except BrainProjection
and ContinuumMemoryBlock FFN layers, trains with synthetic brain states
(emotion-only: WM/FM/concept all zero) derived from training triples.

Usage:
  python train_companion_grounded.py \\
    --base-checkpoint outputs/cortex/hope_companion/best.pt \\
    --emotion-table   outputs/cortex/hope_companion/emotion_table.bin \\
    --triples         data/companion_training/raw/triples.jsonl \\
    --out             outputs/cortex/hope_companion/ \\
    --epochs 5
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent))
from train_hope_companion import HOPE, COMPANION_NANO, CompanionDataset, CRT_MARKER

# Emotion label → index (must match generate_emotion_table.py and Rust emotion_to_idx)
EMOTION_TO_IDX = {
    "neutral": 0, "sad": 1, "pain": 2, "happy": 3,
    "fearful": 4, "angry": 5, "confused": 6, "tired": 7,
}


def detect_emotion(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["miss", "sad", "lonely", "cry", "passed away", "died", "alone"]):
        return "sad"
    if any(w in t for w in ["hurt", "pain", "ache", "sick"]):
        return "pain"
    if any(w in t for w in ["happy", "wonderful", "great", "love", "laugh", "smile", "joy"]):
        return "happy"
    if any(w in t for w in ["afraid", "scared", "worried", "anxious"]):
        return "fearful"
    if any(w in t for w in ["angry", "frustrated", "annoyed", "furious"]):
        return "angry"
    if any(w in t for w in ["confused", "don't understand", "what day"]):
        return "confused"
    if any(w in t for w in ["tired", "exhausted", "sleepy"]):
        return "tired"
    return "neutral"


def make_brain_vec(user_message: str, emotion_table: np.ndarray) -> List[int]:
    """Create synthetic brain state from user message emotion.

    WM/FM/concept are zero (not available at training time).
    Emotion embedding is the dominant signal during Phase 2 training.
    """
    emotion = detect_emotion(user_message)
    idx = EMOTION_TO_IDX.get(emotion, 0)
    vec = emotion_table[idx]  # (512,) float32
    return [round(float(v) * 1000) for v in vec]


class GroundedDataset(Dataset):
    """Extends CompanionDataset with synthetic brain state vectors."""

    def __init__(self, triples_path: str, seq_len: int, emotion_table: np.ndarray) -> None:
        self.seq_len = seq_len
        self.samples: List[Tuple[List[int], List[int], str]] = []  # input, labels, user_msg

        with open(triples_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    triple = json.loads(line)
                except json.JSONDecodeError:
                    continue

                context = triple.get("context", "")
                user_message = triple.get("user_message", "")
                response = triple.get("response", "")

                doc = (
                    b"[CTX] "
                    + context.encode("utf-8", errors="replace")
                    + b" [USR] "
                    + user_message.encode("utf-8", errors="replace")
                    + CRT_MARKER
                    + response.encode("utf-8", errors="replace")
                )
                doc = doc[:seq_len]
                doc_bytes = list(doc)

                marker = list(CRT_MARKER)
                marker_len = len(marker)
                crt_pos = -1
                for i in range(len(doc_bytes) - marker_len + 1):
                    if doc_bytes[i: i + marker_len] == marker:
                        crt_pos = i + marker_len
                        break

                if crt_pos == -1 or crt_pos >= len(doc_bytes):
                    continue

                input_ids = doc_bytes[:seq_len]
                pad_len = seq_len - len(input_ids)
                input_ids = input_ids + [0] * pad_len

                labels = [-100] * seq_len
                for i in range(crt_pos - 1, len(doc_bytes) - 1):
                    if i < seq_len:
                        labels[i] = doc_bytes[i + 1]

                self.samples.append((input_ids, labels, user_message))

        self.emotion_table = emotion_table

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        input_ids, labels, user_message = self.samples[idx]
        brain_vec = make_brain_vec(user_message, self.emotion_table)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(brain_vec, dtype=torch.float32),
        )


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load emotion table
    emotion_table = np.fromfile(args.emotion_table, dtype=np.float32).reshape(8, 512)
    print(f"Emotion table loaded: {emotion_table.shape}")

    # Load dataset
    print(f"Loading triples from {args.triples} ...")
    full_dataset = GroundedDataset(args.triples, COMPANION_NANO["seq_len"], emotion_table)
    n_total = len(full_dataset)
    print(f"Loaded {n_total} samples.")
    if n_total == 0:
        print("ERROR: No valid samples.")
        sys.exit(1)

    n_val = max(1, int(n_total * 0.05))
    n_train = n_total - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model, load Phase 1 weights
    model = HOPE(**COMPANION_NANO).to(device)
    ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Phase 1 checkpoint loaded. New params (need training): {missing}")

    # Freeze all except BrainProjection and CMS FFN layers
    for name, param in model.named_parameters():
        is_brain_proj = "brain_proj" in name
        is_ffn = any(x in name for x in ["linear1", "linear2"])
        param.requires_grad = is_brain_proj or is_ffn

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params")

    # Separate LRs: BrainProjection at base LR, FFN at LR/10
    brain_params = [p for n, p in model.named_parameters() if "brain_proj" in n and p.requires_grad]
    ffn_params = [p for n, p in model.named_parameters() if
                  any(x in n for x in ["linear1", "linear2"]) and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": brain_params, "lr": args.lr},
        {"params": ffn_params, "lr": args.lr / 10},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for input_ids, labels, brain_vecs in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            brain_vecs = brain_vecs.to(device)  # (B, 512)

            optimizer.zero_grad()

            # Compute brain_bias: (B, 1, d_model)
            brain_bias = model.brain_proj(brain_vecs)  # (B, 1, d_model)

            logits = model(input_ids, brain_bias)  # (B, S, vocab_size)
            B, S, V = logits.shape
            loss = criterion(logits.reshape(B * S, V), labels.reshape(B * S))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for input_ids, labels, brain_vecs in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                brain_vecs = brain_vecs.to(device)
                brain_bias = model.brain_proj(brain_vecs)
                logits = model(input_ids, brain_bias)
                B, S, V = logits.shape
                loss = criterion(logits.reshape(B * S, V), labels.reshape(B * S))
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.out, "grounded_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": COMPANION_NANO,
            }, ckpt_path)
            print(f"  Saved grounded checkpoint → {ckpt_path}")

    print(f"\nFine-tune complete. Best val loss: {best_val_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint",
                        default="outputs/cortex/hope_companion/best.pt")
    parser.add_argument("--emotion-table",
                        default="outputs/cortex/hope_companion/emotion_table.bin")
    parser.add_argument("--triples",
                        default="data/companion_training/raw/triples.jsonl")
    parser.add_argument("--out",
                        default="outputs/cortex/hope_companion/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=3e-4)
    return train(parser.parse_args())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the fine-tune script (1 epoch)**

```bash
python scripts/train_companion_grounded.py \
  --base-checkpoint outputs/cortex/hope_companion/best.pt \
  --emotion-table outputs/cortex/hope_companion/emotion_table.bin \
  --epochs 1
```

Expected: runs without errors, prints `Epoch 1/1 train=X val=X`, saves `grounded_best.pt`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_companion_grounded.py
git commit -m "feat: Phase 2 fine-tune script — Brain-grounded HOPE with synthetic emotion states"
```

---

## Task 8: Generate Emotion Table, Export, and Deploy

This task runs the operational pipeline: generate emotion table → (optionally fine-tune) → export → deploy to Pi 5.

- [ ] **Step 1: Generate emotion table**

```bash
python scripts/generate_emotion_table.py \
  --out outputs/cortex/hope_companion/emotion_table.bin
```

Expected:
```
Saved emotion table (8, 512) → outputs/cortex/hope_companion/emotion_table.bin
  Row norms: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

- [ ] **Step 2: (Optional) Run full fine-tune if training data is available**

```bash
python scripts/train_companion_grounded.py \
  --base-checkpoint outputs/cortex/hope_companion/best.pt \
  --emotion-table outputs/cortex/hope_companion/emotion_table.bin \
  --epochs 5
```

Skip if no triples data — the export in Step 3 works without grounded fine-tune (uses base checkpoint with near-zero BrainProjection, functionally equivalent to old `generate()`).

- [ ] **Step 3: Export to TorchScript**

```bash
python scripts/export_companion_grounded.py \
  --base-checkpoint outputs/cortex/hope_companion/best.pt \
  --grounded-checkpoint outputs/cortex/hope_companion/grounded_best.pt \
  --out outputs/cortex/hope_companion/hope_companion_ts.pt
```

Expected: `Export OK.` with both `generate()` and `generate_grounded()` verified.

- [ ] **Step 4: Deploy emotion table and model to Pi 5**

```bash
rsync -av outputs/cortex/hope_companion/hope_companion_ts.pt \
  sbuysse@192.168.202.9:~/Brain/outputs/cortex/hope_companion/

rsync -av outputs/cortex/hope_companion/emotion_table.bin \
  sbuysse@192.168.202.9:~/Brain/outputs/cortex/hope_companion/
```

- [ ] **Step 5: Rebuild and restart on Pi 5**

```bash
ssh sbuysse@192.168.202.9 "cd ~/Brain/rust && cargo build --release -p brain-server 2>&1 | tail -5"
ssh sbuysse@192.168.202.9 "sudo systemctl restart brain-server"
```

- [ ] **Step 6: Verify server log shows grounded loading**

```bash
ssh sbuysse@192.168.202.9 "sudo journalctl -u brain-server -n 30 --no-pager" | grep -E "Emotion|HOPE|Companion"
```

Expected log lines:
```
Loaded HOPE CompanionDecoder
Emotion table loaded (8 emotions)
```

- [ ] **Step 7: End-to-end test via curl**

```bash
curl -s -X POST http://192.168.202.9:8000/companion/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "My knee hurts today"}' | python3 -m json.tool | grep response
```

Expected: a non-empty response string. Server log should show `[HOPE]` not `[Ollama fallback]`.

- [ ] **Step 8: Commit operational scripts**

```bash
git add outputs/cortex/hope_companion/emotion_table.bin
git commit -m "feat: deploy Brain-grounded HOPE companion — emotion table + grounded model"
```
