# Brain-Grounded Companion — Design Spec

**Date:** 2026-03-29
**Status:** Approved
**Goal:** Condition HOPE language generation directly on the Brain's live embedding state (emotion + working memory + fast memory + concepts + personal facts) via layer-wise bias injection into `ContinuumMemoryBlock`, replacing the current text-summary-only context.

---

## Background

The current companion pipeline passes a short text string (`[CTX] Name: Albert, 82. Daughter: Marie.`) as the only Brain-derived signal. HOPE processes this as bytes — ~60 bytes out of a 512-byte context window. The model ignores it in favor of its training priors, generating off-topic names and topics.

The root cause: the Brain's embedding state (512-dim working memory, concept activations, fast memory retrieval, emotion) is never connected to HOPE. The language model generates without any cognitive grounding.

**Fix:** Project the Brain's live 512-dim state into a bias vector injected at every `ContinuumMemoryBlock` FFN output, at every layer, at every generation timestep. The Brain state conditions the entire generation trajectory, not just the first few bytes.

---

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         Rust Brain Server         │
                    │                                   │
  User message ──►  │  compose_brain_state() → Vec<f32> │
                    │  (512-dim weighted sum of signals) │
                    └───────────────┬──────────────────┘
                                    │ brain_vec (512 floats)
                                    ▼
                    ┌──────────────────────────────────┐
                    │       HOPE TorchScript Model      │
                    │                                   │
                    │  BrainProjection: 512 → d_model   │
                    │         = brain_bias              │
                    │                                   │
                    │  Layer 0: ContinuumMemoryBlock    │
                    │    FFN_out += brain_bias  ◄───────┤
                    │  Layer 1: ContinuumMemoryBlock    │
                    │    FFN_out += brain_bias  ◄───────┤
                    │  ...×8 layers                     │
                    └───────────────┬──────────────────┘
                                    │ response bytes
                                    ▼
                             companion reply
```

The existing `generate()` method is unchanged — `brain_bias=None` at all blocks gives identical output to current behavior. The new `generate_grounded()` method passes the Brain bias through all layers.

---

## Brain State Composition (Rust)

New file: `rust/crates/brain-cognition/src/brain_state.rs`

```rust
pub fn compose_brain_state(
    working_memory: &WorkingMemory,
    fast_memory: &HopfieldMemory,
    concepts: &ConceptCodebook,
    emotion: &str,
    emotion_table: &[[f32; 512]],  // 8 rows, one per emotion class
) -> Vec<f32>
```

| Component | Source | Default weight |
|-----------|--------|----------------|
| Working memory centroid | Average of `WorkingMemory` item embeddings, L2-normalized | 0.35 |
| Fast memory retrieval | `HopfieldMemory::retrieve(wm_centroid)` top-1, normalized | 0.25 |
| Concept activation | `ConceptCodebook::nearest(wm_centroid)` centroid, normalized | 0.25 |
| Emotion embedding | Fixed 512-dim lookup table (8 emotions), learned during training | 0.15 |

Output: weighted sum → L2-normalize → 512-dim `Vec<f32>`.

**Fallback:** If a component is unavailable (empty memory, no concepts loaded), its weight redistributes equally to the remaining components. If all components are unavailable, returns a zero vector — HOPE falls back to text-only mode transparently.

**Emotion table:** 8 × 512 floats (16KB) stored as `emotion_table.npy` alongside the HOPE checkpoint. Loaded at `BrainState::load()` startup. Emotion labels (index mapping):

```
0: neutral, 1: sad, 2: pain, 3: happy, 4: fearful, 5: angry, 6: confused, 7: tired
```

These match the labels returned by `brain_cognition::personal::detect_text_emotion()`.

---

## HOPE Model Modification (Python)

**Two files change.** No other model files are touched.

### `hope_model.py`

Add `BrainProjection` module and `generate_grounded()` TorchScript method:

```python
class BrainProjection(nn.Module):
    """Projects 512-dim Brain state to d_model bias, shared across all layers."""
    def __init__(self, brain_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(brain_dim, d_model, bias=False)
        # Initialize to near-zero so Phase 1 checkpoint is unaffected
        nn.init.normal_(self.proj.weight, std=0.001)

    def forward(self, brain_vec: torch.Tensor) -> torch.Tensor:
        # brain_vec: (1, 512) → (1, 1, d_model) for broadcast over (B, S, d_model)
        return self.proj(brain_vec).unsqueeze(1)
```

New TorchScript-exportable method on `HOPEModel`:

```python
@torch.jit.export
def generate_grounded(
    self,
    brain_vec: List[int],    # 512 ints: float * 1000 packed for TorchScript compat
    prompt_bytes: List[int],
    max_new: int,
) -> List[int]:
    bv = torch.tensor(brain_vec, dtype=torch.float32).unsqueeze(0) / 1000.0
    brain_bias = self.brain_proj(bv)  # (1, 1, d_model)
    # generation loop: pass brain_bias to each ContinuumMemoryBlock
    ...
```

### `continuum_block.py`

Add optional `brain_bias` parameter to `forward()`:

```python
def forward(self, x: torch.Tensor, brain_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = self.ffn(x)
    if brain_bias is not None:
        out = out + brain_bias  # (B, S, d_model) + (1, 1, d_model) broadcast
    return out
```

---

## TorchScript Interface

**Float packing:** TorchScript only accepts `List[int]` for cross-language calls. Brain state floats are packed as `round(x * 1000) as i64` in Rust, unpacked as `tensor / 1000.0` in Python. Round-trip error < 0.001 for all x ∈ [-1, 1].

**Export script:** `scripts/export_companion_grounded.py`

```python
model = HOPEModel.load_checkpoint("companion_nano_grounded.pt")
model.eval()
scripted = torch.jit.script(model)
scripted.save("companion_nano_grounded.pt")
# Verify both methods are present
assert hasattr(scripted, 'generate')
assert hasattr(scripted, 'generate_grounded')
```

---

## Rust Integration

### `brain-inference/src/companion_decoder.rs`

Add `generate_grounded()` method:

```rust
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
    ]) { ... };
    // same byte → String conversion as generate()
}
```

### `brain-cognition/src/brain_state.rs` (new file)

Contains `compose_brain_state()` as specified above.

### `brain-server/src/routes.rs`

Replace the companion call with the grounded version:

```rust
let native_response: Option<String> = if let Some(dec) = &brain.companion_decoder {
    let (brain_vec, hope_ctx) = {
        let pm = brain.personal_memory.lock().unwrap();
        let bv = brain_cognition::brain_state::compose_brain_state(
            &brain.working_memory,
            &brain.fast_memory,
            &brain.concepts,
            &detected_emotion,
            &brain.emotion_table,
        );
        let ctx = brain_cognition::personal::build_hope_context(&pm);
        (bv, ctx)
    };
    let response = dec.generate_grounded(&brain_vec, &hope_ctx, &message, 130);
    if response.len() > 5 { Some(response) } else { None }
} else {
    None
};
```

`brain.emotion_table` is a `Vec<[f32; 512]>` loaded at startup from `emotion_table.npy` alongside the `.pt` model path.

---

## Training Curriculum

### Phase 1 — Text-only pretraining (existing, no change needed)

- Input: `[CTX] {personal_context} [USR] {message} [CRT] {response}` bytes
- `brain_bias = None` at all layers
- If `companion_nano.pt` already trained: skip Phase 1, use existing checkpoint as base

### Phase 2 — Brain-state fine-tune

- **Freeze:** all weights except `BrainProjection` and `ContinuumMemoryBlock` FFN layers
- **Unfreeze learning rates:** `BrainProjection` at base LR, FFN layers at LR ÷ 10
- **Synthetic Brain states:** For each training sample, synthesize a brain vector:
  - Emotion from `detect_text_emotion([USR] text)` → emotion table lookup
  - Working memory / concept embeddings: sample from pre-extracted embedding pool (companion training data preprocessed with MiniLM)
  - Or use zero vector for missing components (15–30% of samples, randomly)
- **Loss:** cross-entropy on response bytes only
- **Stop:** when validation perplexity ≤ Phase 1 baseline (Brain state must not degrade base quality)
- **Typical duration:** 3–5 epochs, ~15 minutes on a laptop GPU

**Data for Phase 2:** 500+ (Brain state, conversation) pairs. The brain server logs conversations — these accumulate automatically. Synthetic pairs from the companion training set are sufficient to bootstrap.

---

## Files Changed

| File | Change |
|------|--------|
| `scripts/train_companion_grounded.py` | **New** — Phase 2 fine-tune script |
| `scripts/export_companion_grounded.py` | **New** — TorchScript export with both methods |
| `scripts/generate_emotion_table.py` | **New** — Train/export 8×512 emotion embedding table |
| `hope_model.py` | Add `BrainProjection`, `generate_grounded()` |
| `continuum_block.py` | Add optional `brain_bias` arg to `forward()` |
| `rust/crates/brain-cognition/src/brain_state.rs` | **New** — `compose_brain_state()` |
| `rust/crates/brain-cognition/src/lib.rs` | Export `brain_state` module |
| `rust/crates/brain-inference/src/companion_decoder.rs` | Add `generate_grounded()` |
| `rust/crates/brain-server/src/routes.rs` | Use `generate_grounded()` with Brain state |
| `rust/crates/brain-cognition/src/state.rs` | Add `emotion_table: Vec<[f32; 512]>` field, load at startup |

No new crates. No new Python dependencies. No API changes (existing `/chat` endpoint unchanged, just smarter internally).

---

## Testing

### Unit — `brain-cognition/src/brain_state.rs`

- Empty memory → returns 512 zeros (no panic)
- One working memory item → output L2-norm ≈ 1.0
- All components available → output L2-norm ≈ 1.0
- Emotion index clamps: unknown emotion string → neutral (index 0)

### Unit — `brain-inference/src/companion_decoder.rs`

- Float packing round-trip: `(x * 1000).round() / 1000.0` error < 0.001 for x ∈ [-1, 1]
- `generate_grounded()` with zero brain_vec → no panic, returns non-empty string

### Integration — Python

- `generate_grounded(zeros, prompt, 50)` output matches `generate(prompt, 50)` before fine-tune (projection initialized near-zero)
- After fine-tune: validation perplexity ≤ Phase 1 baseline
- Qualitative: emotion="sad" vs emotion="happy" brain vec produces measurably different output sentiment

### Regression

- All existing `generate()` tests pass unchanged
- `/health` endpoint still returns correct companion_decoder status

---

## Out of Scope

- Per-layer separate Brain projections (one shared `BrainProjection` is sufficient)
- Saving/loading Brain state snapshots to disk (composed fresh per request)
- SIMD optimization of `compose_brain_state()` (simple weighted average, negligible cost)
- Real-time Brain state updates mid-generation (Brain state set once per request)
- Replacing [CTX] text with Brain state entirely (keep both — text for human-readable facts, Brain state for emotional/cognitive tone)
