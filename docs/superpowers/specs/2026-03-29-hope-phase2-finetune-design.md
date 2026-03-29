# HOPE Phase 2 Fine-tune — Design Spec

**Date:** 2026-03-29
**Status:** Approved
**Goal:** Fine-tune the trained Phase 1 HOPE companion model (`best.pt`, epoch 39, val_loss 0.2454) with Brain state conditioning on prod-ia, then deploy the grounded TorchScript to Pi 5.

---

## Background

Phase 1 training is complete on prod-ia: 400k triples, 39 epochs, val_loss 0.2454. The model generates coherent companion responses but has no Brain state conditioning — `BrainProjection` was added after Phase 1.

Phase 2 freezes all weights except `BrainProjection` and ContinuumMemoryBlock FFN layers (`linear1`, `linear2`), then fine-tunes on 200k triples from `companion_data_final/` using synthetic brain states.

---

## Script Fix: `train_companion_grounded.py`

Two changes from the current version:

1. **Field name**: Read `"context_text"` (not `"context"`) — matches the prod-ia data format
2. **Ground-truth emotion**: Use `triple["emotion"]` directly instead of re-detecting from text via `detect_emotion()` — the prod-ia triples already have a reliable `"emotion"` field

`make_brain_vec()` accepts an emotion string directly:
```python
def make_brain_vec(emotion: str, emotion_table: np.ndarray) -> List[int]:
    idx = EMOTION_TO_IDX.get(emotion, 0)
    vec = emotion_table[idx]
    return [round(float(v) * 1000) for v in vec]
```

`GroundedDataset` stores `(input_ids, labels, brain_vec)` — brain_vec precomputed in `__init__` using `triple["emotion"]`.

---

## Files to Sync to prod-ia `/opt/`

| File | Change |
|------|--------|
| `scripts/train_hope_companion.py` | Has `BrainProjection` (Task 1) |
| `scripts/train_companion_grounded.py` | Fixed field names + ground-truth emotion |
| `scripts/generate_emotion_table.py` | Generates 8×512 emotion table |
| `scripts/export_companion_grounded.py` | Exports TorchScript with both methods |

---

## Execution on prod-ia

```
/opt/hope_companion/best.pt             ← Phase 1 checkpoint (source)
/opt/companion_data_final/triples.jsonl ← 200k training triples
/opt/hope_companion/emotion_table.bin   ← generated here
/opt/hope_companion/grounded_best.pt    ← Phase 2 output
/opt/hope_companion/hope_companion_ts.pt ← TorchScript export (deploy to Pi 5)
```

**Phase 2 training config:**
- Base checkpoint: `/opt/hope_companion/best.pt`
- Triples: `/opt/companion_data_final/triples.jsonl`
- Epochs: 5
- Batch size: 64 (prod-ia has GPU)
- LR: 3e-4 (BrainProjection), 3e-5 (FFN)
- Stop criterion: val_loss ≤ Phase 1 baseline (0.2454)

---

## Deploy

Rsync `hope_companion_ts.pt` + `emotion_table.bin` from prod-ia to Pi 5 at `root@192.168.202.9:/opt/brain/outputs/cortex/hope_companion/`, then restart brain-server.

---

## Testing

After deploy, curl the Pi 5 endpoint:
```bash
curl -sk -X POST https://192.168.202.9/api/brain/dialogue/grounded \
  -H 'Content-Type: application/json' \
  -d '{"message": "My knee hurts today"}'
```

Expected: `"native": true`, `"grounded": true`, coherent response (not garbled bytes).

Qualitative check: send a sad message and a happy message — responses should differ in tone.

---

## Out of Scope

- Generating new training data (400k triples already exist)
- Re-running Phase 1 (val_loss 0.2454 is sufficient)
- SIMD / quantization of the fine-tuned model (separate task)
