# TurboQuant Embedding Quantization — Design Spec

**Date:** 2026-03-29
**Status:** Approved
**Goal:** Replace brute-force f32 dot-product search in all Brain embedding indices with 4-bit quantized integer search for 8× memory reduction and 4–8× speedup on ARM NEON (Pi 5).

---

## Background

All embedding search in Brain is currently brute-force f32 cosine similarity:

| Index | Struct | Vectors | Dims |
|-------|--------|---------|------|
| Label semantic search | `TextEncoder::label_embeddings` | 865 | 384 |
| Concept nearest | `ConceptCodebook::centroids` | 865+ | 512 |
| Fast associative recall | `HopfieldMemory::patterns` | ≤2000 | 512 |

At f32 (4 bytes/dim), a 2000×512 Hopfield search costs ~4MB of data movement per query. At 4 bits/dim (0.5 bytes/dim), this drops to ~0.5MB and integer SIMD handles it natively on Cortex-A76.

**Reference:** TurboQuant paper (https://arxiv.org/pdf/2504.19874) — random rotation + scalar quantization for inner product–preserving compression.

---

## Architecture

### `QuantizedIndex` (new, `brain-inference/src/quantized_index.rs`)

A standalone struct with no ML framework dependency.

```rust
pub struct QuantizedIndex {
    rotation: Vec<f32>,   // d×d row-major orthogonal rotation matrix
    scale: Vec<f32>,      // per-dim scale factor (d values)
    zero: Vec<f32>,       // per-dim zero point (d values)
    codes: Vec<Vec<u8>>,  // packed 4-bit codes — each entry is d/2 bytes
    labels: Vec<String>,
    dim: usize,
    bits: u8,             // always 4
}
```

**API:**

| Method | Description |
|--------|-------------|
| `QuantizedIndex::build(labels, vecs, bits)` | Compute rotation, calibrate scale/zero, quantize all vectors |
| `insert(label, vec)` | Rotate + quantize one vector, append (for dynamic indices) |
| `nearest(query, k) -> Vec<(String, f32)>` | Rotate query, integer dot product search, top-k |
| `len() / is_empty()` | Size queries |

**Rotation:** Random orthogonal matrix via Gram-Schmidt on a random Gaussian matrix. Computed once at build time from the input vectors. Stored as `Vec<f32>` (d² floats).

**Calibration:** Per-dim, use 1st–99th percentile of rotated values across all vectors to define `[lo, hi]` range. `scale[i] = (hi - lo) / 15.0`, `zero[i] = lo`. Clips outliers before quantization.

**Quantization:** Each rotated coordinate mapped to `clamp((x - zero[i]) / scale[i], 0, 15)` → 4-bit value. Two values packed per byte: `(a & 0xF) | ((b & 0xF) << 4)`.

**Search:** Query is rotated by R, then for each stored vector: dequantize codes to `i16` (`code * scale + zero`), compute dot product as `Σ q_i * v_i` in `i32` accumulator. Return top-k by score.

**Approximate recall:** At 4 bits, >95% top-1 recall expected on L2-normalized vectors (validated by unit test).

---

## Integration Strategy: Internal Delegation (Option B)

Existing public APIs are unchanged. Each struct holds an `Option<QuantizedIndex>` internally and delegates search to it when available.

### `ConceptCodebook` (`brain-cognition/src/concepts.rs`)

```rust
pub struct ConceptCodebook {
    pub centroids: Array2<f32>,  // kept — needed for compose()/add_concept()
    pub labels: Vec<String>,
    qi: Option<QuantizedIndex>,  // built at construction, rebuilt on add_concept()
}
```

- `build()` constructs `qi` from the projected centroids
- `nearest()` / `open_nearest()` delegate to `qi.nearest()` when `qi.is_some()`
- `add_concept()` appends to `qi` via `qi.insert()`
- `grow_from_prototypes()` does bulk insert then rebuilds `qi`

### `HopfieldMemory` (`brain-cognition/src/fast_memory.rs`)

```rust
pub struct HopfieldMemory {
    patterns: Vec<Vec<f32>>,   // kept — needed for get_recent() consolidation
    labels: Vec<String>,
    count: usize,
    capacity: usize,
    dim: usize,
    qi: Option<QuantizedIndex>,
    dirty: bool,
}
```

- `store()` sets `dirty = true`
- `retrieve()` checks: if `dirty && count % 50 == 0`, call `rebuild_qi()` from current patterns; delegate to `qi.nearest()`
- `rebuild_qi()` cost: 2000×512×4 ops ≈ <1ms on Pi 5, acceptable inline

### `TextEncoder` (`brain-inference/src/text.rs`)

```rust
pub struct TextEncoder {
    // existing fields unchanged
    label_qi: Option<QuantizedIndex>,  // built on load from label_embeddings
}
```

- `load()` builds `label_qi` after loading `label_embeddings`
- `semantic_search()` delegates to `label_qi.nearest()` when available

---

## Data Flow

```
build(labels, vecs)
  → rotate all vecs by R
  → calibrate scale/zero per dim (1st–99th percentile)
  → pack to 4-bit codes

nearest(query, k)
  → rotate query by R
  → for each stored vec: dequant codes → i16 dot product in i32 accum
  → partial sort top-k
  → return (label, score) pairs
```

---

## Error Handling

- `build()` with 0 vectors or mismatched dims → returns empty index, search returns empty
- `insert()` with wrong dim → silently ignored (same contract as `HopfieldMemory::store()`)
- `nearest()` on empty index → returns empty vec (no panic)

---

## Testing

All tests in `brain-inference/src/quantized_index.rs` (inline `#[cfg(test)]`):

1. **Exact recall:** Build 100-vector 512-dim index, query with one stored vector, assert it ranks first
2. **Recall@1 rate:** Build 500-vector index, query each vector, assert >95% top-1 match rate
3. **ConceptCodebook regression:** `nearest()` results identical top-1 before and after QuantizedIndex integration (test in `concepts.rs`)
4. **HopfieldMemory regression:** `retrieve()` top-1 unchanged after `rebuild_qi()`

---

## Files Changed

| File | Change |
|------|--------|
| `rust/crates/brain-inference/src/quantized_index.rs` | **New** — QuantizedIndex implementation |
| `rust/crates/brain-inference/src/lib.rs` | Export `QuantizedIndex` |
| `rust/crates/brain-cognition/src/concepts.rs` | Add `qi: Option<QuantizedIndex>`, delegate search |
| `rust/crates/brain-cognition/src/fast_memory.rs` | Add `qi`, `dirty` flag, lazy rebuild |
| `rust/crates/brain-inference/src/text.rs` | Add `label_qi`, delegate `semantic_search()` |

No new crates. No new dependencies. No Python preprocessing. No API changes.

---

## Out of Scope

- SIMD intrinsics (compiler auto-vectorizes i16 dot products on ARM NEON; explicit NEON intrinsics are a future micro-optimization)
- Hadamard transform (requires power-of-2 dim padding; Gram-Schmidt rotation is simpler and correct for 384 and 512)
- Saving/loading `QuantizedIndex` to disk (indices are rebuilt from f32 at startup; cold start cost is negligible)
- TurboQuant 2-stage inner product variant (overkill for current index sizes)
