# Cortex Language Decoder — Implementation Plan (Updated: HOPE Pivot)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Ollama with a native Cortex decoder that generates companion responses on-device, targeting Raspberry Pi 5 deployment.

**Architecture (updated):** Use the HOPE (Nested Learning) architecture — a byte-level, stateful language model with fast weights (per-timestep self-modifying memory) and slow weights (ContinuumMemoryBlock layers). "Companion Nano" config: 384-dim, 8 layers, vocab 256 (bytes), ~10M params. PersonalMemory context + user message are concatenated as UTF-8 bytes and fed directly — no separate text encoder or vocabulary file needed. Train on ~50K synthetic (context, user_message, response) triples. Keep Ollama as warm fallback.

**Tech Stack:** Python 3.12, PyTorch 2.x, TorchScript, Rust/tch-rs (until burn/candle migration). No MiniLM, no word vocabulary.

---

## Research Findings Log

### Why HOPE over TransformerDecoder (decision: 2026-03-25)

Original plan used a dual-memory TransformerDecoder (22M params) with word-level vocabulary (~5K tokens), requiring a pre-trained MiniLM encoder for context embedding.

**HOPE advantages for this project:**
- Byte-level tokenization — no vocab file, no OOV, trivially handles names and special characters
- ~10M params Nano config — smaller than original 22M decoder
- Built-in stateful fast memory — naturally maintains conversation context across turns without explicit history passing
- Continual learning by design — fast memory decays old patterns and absorbs new ones per session
- Eliminates MiniLM dependency — context is fed as raw text bytes, not encoded vectors
- Pi 5 friendly — <1GB inference RAM for Nano config

**Reference implementation:** https://github.com/obekt/HOPE-nested-learning

**What is superseded by this pivot:**
- Task 2 (vocab builder) — no vocabulary needed
- Task 4 (TransformerDecoder training script) — replaced by HOPE training script
- Task 5 (precompute embeddings + TorchScript) — no pre-encoded embeddings needed; TorchScript export still needed

**What is NOT affected:**
- Task 1 (data generator) — 50K triples still used for HOPE training (format unchanged)
- Task 3 (context encoder) — encode_context.py still useful for pre-encoding if needed offline
- Tasks 6–8 (Rust integration, routes, eval) — updated for HOPE but structure preserved

---

### TurboQuant (research: 2026-03-25, implementation: future plan)

**Paper:** https://arxiv.org/pdf/2504.19874

**What it does:**
1. Randomly rotates embedding vectors (or uses Hadamard transform for O(d log d) efficiency)
2. After rotation, coordinates concentrate into a near-uniform distribution
3. Applies optimal scalar quantizer per coordinate (~2.5–4 bits/dim)
4. For inner product preservation: 2-stage (MSE quantizer + 1-bit QJL transform)
5. Enables fast integer dot product search on ARM NEON

**Why it matters for Pi 5 deployment:**
- All embedding search in Brain is brute-force f32 dot product — no FAISS/HNSW
- Key indices: label_embeddings (310×384 f32), concept codebook (865+×512 f32), Hopfield memory (2000×512 f32), training embeddings (50K×512 f32), SQLite episode BLOBs
- At 4 bits/dim: 8× memory reduction, 4–8× search speedup on ARM NEON via integer SIMD
- Applies to both static (offline precomputed) and dynamic (online inserted) embeddings

**Design decision:** Implement as a Rust `QuantizedIndex` struct (no ML framework dependency) + Python offline preprocessing script. Targets all embedding indices in brain-inference and brain-cognition.

**Separate plan:** Will be written as `docs/superpowers/plans/YYYY-MM-DD-turboquant-embedding-quantization.md`

---

### Pi 5 Deployment Strategy (research: 2026-03-25)

Target hardware: Raspberry Pi 5, 8GB RAM, quad-core Cortex-A76 @ 2.4GHz, no GPU.

**Dependency reduction roadmap:**
| Dependency | Status | Path |
|-----------|--------|------|
| MiniLM text encoder | Eliminated | Replaced by HOPE byte-level input |
| Ollama LLM | Fallback only | Replaced by Cortex/HOPE as primary |
| tch-rs/libtorch (~400MB) | Current | Future: migrate to burn/candle (pure Rust) |
| DINOv2, Whisper, emotion models | Keep | Future: ONNX INT8 or candle native |
| Embedding indices (f32) | Current | TurboQuant (future plan) |

Runtime sequence for Pi 5 (near-term):
```
Audio/Video → Whisper/DINOv2 (tch-rs, CPU) → 512-dim embeddings
PersonalMemory context text → UTF-8 bytes
[context_bytes] + [user_message_bytes] → HOPE Nano (tch-rs, CPU) → response bytes
Embeddings → QuantizedIndex search → top-K labels/concepts
```

---

## Task Status

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Data generator (generate_companion_data.py) | ✅ Done | Running on prod-ia GPU, ~200 triples/min, target 50K |
| 2 | Vocab builder (build_companion_vocab.py) | ✅ Done | **Superseded** — HOPE uses byte-level, no vocab needed |
| 3 | Context encoder (encode_context.py) | ✅ Done | Optional utility, not required for HOPE |
| 4 | ~~TransformerDecoder training~~ → **HOPE training** | 🔄 Pivoted | See Task 4b below |
| 5 | ~~Precompute embeddings~~ → **HOPE TorchScript export** | 🔄 Pivoted | See Task 5b below |
| 6 | Rust CompanionDecoder struct | ✅ Done | Needs minor update for HOPE byte I/O |
| 7 | Wire into state + routes | ✅ Done | Needs update: remove ctx_emb/brain_emb encoding |
| 8 | Eval script | ✅ Done | Needs update for byte-level decode |

---

## File Map (Updated)

### New files (Python)
| File | Responsibility |
|------|---------------|
| `scripts/train_hope_companion.py` | HOPE Nano training on companion triples |
| `scripts/eval_hope_companion.py` | Sample and evaluate HOPE responses |

### Modified files (Python)
| File | Change |
|------|--------|
| `scripts/export_torchscript.py` | Add `export_hope_companion()` function |
| `scripts/eval_companion_decoder.py` | Update for byte-level decode |

### Modified files (Rust)
| File | Change |
|------|--------|
| `rust/crates/brain-inference/src/companion_decoder.rs` | Replace dual-emb input with byte token input; decode UTF-8 output |
| `rust/crates/brain-server/src/routes.rs` | Section 3a: format context+message as bytes, call HOPE generate |
| `rust/crates/brain-cognition/src/state.rs` | No structural change needed |

### Output paths
```
outputs/cortex/hope_companion/
  hope_companion_ts.pt       ← TorchScript export (used by Rust)
  hope_companion_config.json ← d_model, n_layers, seq_len for Rust loader
```

---

## HOPE "Companion Nano" Config

```python
COMPANION_NANO = {
    "d_model":    384,
    "n_layers":   8,
    "vocab_size": 256,    # byte-level
    "seq_len":    512,    # ~400 chars context + ~100 chars response
    "dropout":    0.1,
    # ~10M parameters
}
```

Training format — each triple becomes one training document (packed bytes):
```
b"[CTX] " + context_text.encode() + b" [USR] " + user_message.encode() + b" [CRT] " + response.encode() + b"\n"
```
Loss computed only on response tokens (after `[CRT]` marker), context+user tokens are masked.

---

## Phase 4b — HOPE Training

### Task 4b: HOPE Companion training script

**Files:**
- Create: `scripts/train_hope_companion.py`

The script adapts the HOPE architecture to the companion domain. Companion Nano config (384-dim, 8 layers, ~10M params). Trains on the same (context, user_message, response) triples as before, but formats them as byte sequences. Loss is computed only on response bytes.

- [ ] **Step 1: Write architecture test**

```python
# scripts/tests/test_hope_companion.py
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_hope_companion import HOPE, COMPANION_NANO

def test_hope_forward_shape():
    model = HOPE(**COMPANION_NANO)
    tokens = torch.zeros(2, 32, dtype=torch.long)   # batch=2, seq=32
    logits = model(tokens)
    assert logits.shape == (2, 32, 256), f"Got {logits.shape}"

def test_hope_generate_returns_bytes():
    model = HOPE(**COMPANION_NANO)
    model.eval()
    prompt = b"[CTX] Hello [USR] How are you? [CRT] "
    response = model.generate(prompt, max_new=30)
    assert isinstance(response, bytes)
    assert len(response) <= 30

def test_hope_torchscript_export():
    model = HOPE(**COMPANION_NANO)
    model.eval()
    scripted = torch.jit.script(model)
    tokens = torch.zeros(1, 10, dtype=torch.long)
    out = scripted(tokens)
    assert out.shape == (1, 10, 256)
```

Run: `python -m pytest scripts/tests/test_hope_companion.py -v`
Expected: **FAIL** — module does not exist yet.

- [ ] **Step 2: Write HOPE Companion training script**

Key classes to implement (adapt from https://github.com/obekt/HOPE-nested-learning):

```python
# scripts/train_hope_companion.py
"""Train HOPE Companion Nano on (context, user_message, response) triples.

Architecture: HOPE nested learning model, byte-level tokenization.
  - Fast memory: SelfModifyingLayer (stateful, per-timestep)
  - Slow weights: ContinuumMemoryBlock × n_layers
  - Vocab: 256 (byte-level, no vocabulary file)

Usage:
  python train_hope_companion.py \
    --triples data/companion_training/raw/triples.jsonl \
    --out     outputs/cortex/hope_companion/ \
    --epochs  50
"""

COMPANION_NANO = {
    "d_model":    384,
    "n_layers":   8,
    "vocab_size": 256,
    "seq_len":    512,
    "dropout":    0.1,
}

CRT_MARKER = b" [CRT] "   # response start marker

class SelfModifyingLayer(nn.Module):
    # fast memory with learned decay

class ContinuumMemoryBlock(nn.Module):
    # FFN with pre-norm: LayerNorm → Linear → GELU → Linear + residual

class HOPE(nn.Module):
    # embedding + fast_memory + cms_layers + head
    def forward(self, tokens: torch.Tensor) -> torch.Tensor: ...
    @torch.jit.export
    def generate(self, prompt_bytes: bytes, max_new: int) -> bytes: ...

class CompanionDataset(Dataset):
    # formats triples as byte sequences
    # loss_mask: 0 for context+user tokens, 1 for response tokens

# training loop: AdamW, cosine LR, gradient clip 1.0, save best val loss
```

- [ ] **Step 3: Run architecture tests**

```bash
python -m pytest scripts/tests/test_hope_companion.py -v
```
Expected: **PASS** (3 tests including TorchScript export).

- [ ] **Step 4: Smoke-train on available triples**

```bash
python scripts/train_hope_companion.py \
  --triples data/companion_training/raw/triples.jsonl \
  --out     /tmp/hope_test \
  --epochs  5
```
Expected: Loss decreases, sample output is ASCII text (may be garbled early).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_hope_companion.py scripts/tests/test_hope_companion.py
git commit -m "feat: HOPE Companion Nano training script (replaces TransformerDecoder)"
```

---

## Phase 5b — Full Training + TorchScript Export

### Task 5b: Train on full dataset + export

These steps run on the remote server (root@prod-ia or root@192.168.202.9) after the 50K dataset is complete.

- [ ] **Step 1: Monitor data generation (prod-ia)**

```bash
ssh root@prod-ia "tail -f /opt/companion_gen.log"
```
Expected: ~4 hours total (started ~18:32 2026-03-25). Wait for `Done. 50000 triples`.

- [ ] **Step 2: Rsync triples to training server**

```bash
rsync -az root@prod-ia:/opt/companion_data/triples.jsonl \
  root@192.168.202.9:/opt/brain/data/companion_training/raw/triples.jsonl
```

- [ ] **Step 3: Train HOPE Companion (remote, background)**

```bash
ssh root@192.168.202.9 "cd /opt/brain && \
  nohup python scripts/train_hope_companion.py \
    --triples data/companion_training/raw/triples.jsonl \
    --out     outputs/cortex/hope_companion \
    --epochs  50 \
  > /tmp/hope_train.log 2>&1 &"
```
Monitor: `tail -f /tmp/hope_train.log`

- [ ] **Step 4: Add export function to export_torchscript.py**

```python
def export_hope_companion(weights_path: str, out_path: str, config_path: str):
    """Export trained HOPE companion to TorchScript."""
    import json
    from train_hope_companion import HOPE, COMPANION_NANO

    model = HOPE(**COMPANION_NANO)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(out_path)

    with open(config_path, "w") as f:
        json.dump(COMPANION_NANO, f, indent=2)
    print(f"Exported to {out_path}")
    print(f"Config saved to {config_path}")
```

Run after training:
```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from export_torchscript import export_hope_companion
export_hope_companion(
  'outputs/cortex/hope_companion/best.pt',
  'outputs/cortex/hope_companion/hope_companion_ts.pt',
  'outputs/cortex/hope_companion/hope_companion_config.json',
)
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/export_torchscript.py
git commit -m "feat: TorchScript export for HOPE companion decoder"
```

---

## Phase 6b — Rust Integration Update

### Task 6b: Update CompanionDecoder for HOPE byte I/O

The existing `companion_decoder.rs` struct takes `brain_emb` + `ctx_emb` (Vec<f32> each). Update to take a single `prompt: &str` (concatenated context + message), tokenize as bytes, run HOPE, decode UTF-8 response.

**Files:**
- Modify: `rust/crates/brain-inference/src/companion_decoder.rs`
- Modify: `rust/crates/brain-server/src/routes.rs` (section 3a)

- [ ] **Step 1: Update companion_decoder.rs**

New `generate` signature:
```rust
pub fn generate(&self, context_text: &str, user_message: &str, max_tokens: usize) -> String {
    // Format: "[CTX] {context} [USR] {message} [CRT] "
    let prompt = format!("[CTX] {} [USR] {} [CRT] ", context_text, user_message);
    // Encode as bytes → i64 tensor (byte values 0-255)
    let bytes: Vec<i64> = prompt.bytes().map(|b| b as i64).collect();
    let tokens = Tensor::from_slice(&bytes).reshape(&[1, bytes.len() as i64]);
    // Run model.forward_ts, sample greedily up to max_tokens
    // Decode output bytes as UTF-8
    ...
}
```

- [ ] **Step 2: Update routes.rs section 3a**

Remove `ctx_emb` and `brain_emb_for_decoder` encoding. Replace with:
```rust
let ctx_text = {
    let pm = brain.personal_memory.lock().unwrap();
    brain_cognition::personal::build_personal_context(&pm)
};
let response = brain_cognition::companion::native_reply(dec, &ctx_text, &message, 80);
```

- [ ] **Step 3: cargo check**

```bash
cargo check -p brain-inference -p brain-cognition -p brain-server 2>&1 | grep "^error"
```
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add rust/crates/brain-inference/src/companion_decoder.rs \
        rust/crates/brain-server/src/routes.rs
git commit -m "feat: update CompanionDecoder for HOPE byte-level I/O"
```

---

## Quality Gates

Before calling HOPE "production ready" (replacing Ollama as default):

| Gate | Criterion |
|------|-----------|
| Response coherence | Manual review of 50 samples: >80% grammatically correct |
| Name usage | >70% of responses use the person's name when available in context |
| Byte coverage | Zero malformed UTF-8 in generated responses |
| Safety | Zero medical advice, zero AI self-identification |
| Latency | Response time < 500ms on Pi 5 (vs 15–30s for Ollama) |
| Memory | Model + runtime < 500MB RAM on Pi 5 |

---

## Future Work (separate plans)

- **TurboQuant embedding quantization** — compress all f32 embedding indices to 4-bit for Pi 5. Design: Rust `QuantizedIndex` struct + Python offline preprocessing. Applies to: label_embeddings, concept codebook, Hopfield memory, SQLite episode BLOBs.
- **Burn/Candle migration** — replace tch-rs/libtorch with pure Rust ML framework. Enables truly minimal Pi deployment (~50MB vs ~400MB for libtorch).
- **DINOv2 / Whisper ONNX quantization** — INT8 export for Pi 5 perception pipeline.
