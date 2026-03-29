# HOPE Phase 2 Fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the trained Phase 1 HOPE companion model with Brain state conditioning on prod-ia, then deploy the grounded TorchScript to Pi 5.

**Architecture:** Fix `train_companion_grounded.py` to use the prod-ia data format (`"context_text"` field, ground-truth `"emotion"` field), sync all scripts to `root@prod-ia:/opt/`, run Phase 2 fine-tune on the 200k triples already there, export TorchScript, deploy to Pi 5.

**Tech Stack:** Python 3.14, PyTorch, prod-ia GPU, `root@prod-ia:/opt/`, `root@192.168.202.9:/opt/brain/`

---

## File Map

| File | Change |
|------|--------|
| `scripts/train_companion_grounded.py` | Fix `"context"` → `"context_text"`, use ground-truth `emotion` field |
| `scripts/train_hope_companion.py` | Sync to prod-ia (has BrainProjection added in Task 1) |
| `scripts/generate_emotion_table.py` | Sync to prod-ia |
| `scripts/export_companion_grounded.py` | Sync to prod-ia |

---

## Task 1: Fix train_companion_grounded.py for prod-ia data format

**Files:**
- Modify: `scripts/train_companion_grounded.py:57-66` (make_brain_vec), `scripts/train_companion_grounded.py:93` (context field), `scripts/train_companion_grounded.py:129` (brain_vec call)

The prod-ia triples use `"context_text"` (not `"context"`) and already have a ground-truth `"emotion"` field. Use it directly instead of re-detecting from text.

- [ ] **Step 1: Fix `make_brain_vec` to accept an emotion string directly**

Replace the current `make_brain_vec(user_message, emotion_table)` function (lines 57–66) with:

```python
def make_brain_vec(emotion: str, emotion_table: np.ndarray) -> List[int]:
    """Create synthetic brain state from ground-truth emotion label.

    WM/FM/concept are zero (not available at training time).
    Emotion embedding is the dominant signal during Phase 2 training.
    """
    idx = EMOTION_TO_IDX.get(emotion, 0)
    vec = emotion_table[idx]  # (512,) float32
    return [round(float(v) * 1000) for v in vec]
```

Also remove the `detect_emotion()` function entirely (lines 38–54) — it is no longer used.

- [ ] **Step 2: Fix `GroundedDataset.__init__` to use correct field names**

Change line 93 from:
```python
context = triple.get("context", "")
```
to:
```python
context = triple.get("context_text", "")
```

Change line 129 from:
```python
brain_vec = make_brain_vec(user_message, emotion_table)
```
to:
```python
emotion = triple.get("emotion", "neutral")
brain_vec = make_brain_vec(emotion, emotion_table)
```

- [ ] **Step 3: Verify the fix with a quick parse test**

```bash
cd /home/sbuysse/Documents/Coding/Projects/Akretio/Brain
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import numpy as np
from train_companion_grounded import make_brain_vec, EMOTION_TO_IDX

et = np.random.randn(8, 512).astype(np.float32)

# make_brain_vec now takes emotion string, not user_message
vec = make_brain_vec('sad', et)
assert len(vec) == 512
assert all(isinstance(v, int) for v in vec)

# Neutral fallback for unknown emotion
vec2 = make_brain_vec('unknown_emotion', et)
assert vec2 == make_brain_vec('neutral', et)

print('make_brain_vec fix verified.')
"
```
Expected: `make_brain_vec fix verified.`

- [ ] **Step 4: Commit the fix**

```bash
cd /home/sbuysse/Documents/Coding/Projects/Akretio/Brain
git add scripts/train_companion_grounded.py
git commit -m "fix: train_companion_grounded — use context_text field and ground-truth emotion"
```

---

## Task 2: Sync scripts to prod-ia and generate emotion table

**Files:**
- Sync to `root@prod-ia:/opt/`: `train_hope_companion.py`, `train_companion_grounded.py`, `generate_emotion_table.py`, `export_companion_grounded.py`

- [ ] **Step 1: Sync all four scripts to prod-ia**

```bash
cd /home/sbuysse/Documents/Coding/Projects/Akretio/Brain
rsync -av \
  scripts/train_hope_companion.py \
  scripts/train_companion_grounded.py \
  scripts/generate_emotion_table.py \
  scripts/export_companion_grounded.py \
  root@prod-ia:/opt/
```

Expected output: four files listed as transferred.

- [ ] **Step 2: Verify imports work on prod-ia**

```bash
ssh root@prod-ia "cd /opt && python3 -c 'import sys; sys.path.insert(0, \".\"); from train_hope_companion import HOPE, COMPANION_NANO; from train_companion_grounded import make_brain_vec; print(COMPANION_NANO)'"
```

Expected:
```
{'d_model': 384, 'n_layers': 8, 'vocab_size': 256, 'seq_len': 512, 'dropout': 0.1, 'brain_dim': 512}
```

- [ ] **Step 3: Generate emotion table on prod-ia**

```bash
ssh root@prod-ia "cd /opt && python3 generate_emotion_table.py --out /opt/hope_companion/emotion_table.bin"
```

Expected:
```
Saved emotion table (8, 512) → /opt/hope_companion/emotion_table.bin
  Row norms: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

- [ ] **Step 4: Confirm the triples have the right fields**

```bash
ssh root@prod-ia "head -1 /opt/companion_data_final/triples.jsonl | python3 -c 'import sys,json; d=json.load(sys.stdin); print(list(d.keys())); print(\"emotion:\", d[\"emotion\"]); print(\"context_text prefix:\", d[\"context_text\"][:60])'"
```

Expected: keys include `context_text`, `user_message`, `response`, `emotion`. Emotion is one of: neutral, sad, pain, happy, fearful, angry, confused, tired.

---

## Task 3: Run Phase 2 fine-tune on prod-ia

**Files:**
- Runs on prod-ia; output: `/opt/hope_companion/grounded_best.pt`

Phase 2 freezes all weights except `brain_proj.*` and FFN layers (`linear1`, `linear2`), trains for 5 epochs on 200k triples.

- [ ] **Step 1: Smoke-test with 1 epoch on 500 samples**

```bash
ssh root@prod-ia "cd /opt && python3 train_companion_grounded.py \
  --base-checkpoint /opt/hope_companion/best.pt \
  --emotion-table   /opt/hope_companion/emotion_table.bin \
  --triples         /opt/companion_data_final/triples.jsonl \
  --out             /opt/hope_companion/ \
  --epochs 1 \
  --batch-size 64 \
  2>&1 | head -30"
```

Expected output includes:
```
Device: cuda   (or cpu if no GPU available)
Emotion table loaded: (8, 512)
Parsed 200000 lines. Dropped 0 JSON errors, 0 truncated ...
Loaded 200000 samples.
Trainable: X / Y params
Phase 1 checkpoint loaded. New params (will be randomly initialized): ['brain_proj.proj.weight']
Epoch   1/1  train=X.XXXX  val=X.XXXX
  Saved grounded checkpoint → /opt/hope_companion/grounded_best.pt
Fine-tune complete. Best val loss: X.XXXX
```

If `n_dropped_crt` is high (>20%), stop — the `seq_len=512` may be truncating the CRT marker. Check by printing a few triples.

- [ ] **Step 2: Run full 5-epoch fine-tune in the background**

```bash
ssh root@prod-ia "cd /opt && nohup python3 train_companion_grounded.py \
  --base-checkpoint /opt/hope_companion/best.pt \
  --emotion-table   /opt/hope_companion/emotion_table.bin \
  --triples         /opt/companion_data_final/triples.jsonl \
  --out             /opt/hope_companion/ \
  --epochs 5 \
  --batch-size 64 \
  > /opt/hope_companion/phase2_train.log 2>&1 &
echo PID: \$!"
```

Monitor progress:
```bash
ssh root@prod-ia "tail -f /opt/hope_companion/phase2_train.log"
```

Wait for: `Fine-tune complete. Best val loss: X.XXXX`

- [ ] **Step 3: Verify checkpoint was saved**

```bash
ssh root@prod-ia "python3 -c \"
import torch
ckpt = torch.load('/opt/hope_companion/grounded_best.pt', map_location='cpu', weights_only=True)
print('epoch:', ckpt['epoch'])
print('val_loss:', ckpt['val_loss'])
print('has brain_proj:', any('brain_proj' in k for k in ckpt['model_state_dict']))
\""
```

Expected: epoch ≤ 5, val_loss ≤ 0.2454 (Phase 1 baseline), `has brain_proj: True`.

---

## Task 4: Export grounded TorchScript on prod-ia

**Files:**
- Runs on prod-ia; output: `/opt/hope_companion/hope_companion_ts.pt`

- [ ] **Step 1: Export TorchScript**

```bash
ssh root@prod-ia "cd /opt && python3 export_companion_grounded.py \
  --base-checkpoint      /opt/hope_companion/best.pt \
  --grounded-checkpoint  /opt/hope_companion/grounded_best.pt \
  --out                  /opt/hope_companion/hope_companion_ts.pt"
```

Expected:
```
Base checkpoint loaded. Missing keys: ['brain_proj.proj.weight']
Grounded checkpoint loaded. Missing: []
generate():          b'...'
generate_grounded(): b'...'
Exported TorchScript model → /opt/hope_companion/hope_companion_ts.pt
Verification generate():          b'...'
Verification generate_grounded(): b'...'
Export OK.
```

The two `generate()` and `generate_grounded()` outputs should now differ (unlike Phase 1 where BrainProjection was near-zero).

---

## Task 5: Deploy to Pi 5 and verify

**Files:**
- Deploys to `root@192.168.202.9:/opt/brain/outputs/cortex/hope_companion/`

- [ ] **Step 1: Rsync model and emotion table to Pi 5**

```bash
rsync -av \
  root@prod-ia:/opt/hope_companion/hope_companion_ts.pt \
  root@prod-ia:/opt/hope_companion/emotion_table.bin \
  root@192.168.202.9:/opt/brain/outputs/cortex/hope_companion/
```

Expected: both files transferred.

- [ ] **Step 2: Restart brain-server on Pi 5**

```bash
ssh root@192.168.202.9 "systemctl restart brain-server && sleep 5 && journalctl -u brain-server -n 50 --no-pager" | grep -iE "Companion|Emotion|HOPE"
```

Expected:
```
Loaded HOPE CompanionDecoder
Emotion table loaded (8 emotions)
```

- [ ] **Step 3: End-to-end test — pain message**

```bash
curl -sk -X POST https://192.168.202.9/api/brain/dialogue/grounded \
  -H 'Content-Type: application/json' \
  -d '{"message": "My knee hurts today"}' | python3 -m json.tool
```

Expected: `"native": true`, `"grounded": true`, response is coherent English (not garbled bytes).

- [ ] **Step 4: Qualitative tone check — sad vs happy**

```bash
curl -sk -X POST https://192.168.202.9/api/brain/dialogue/grounded \
  -H 'Content-Type: application/json' \
  -d '{"message": "I miss my daughter so much today"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('SAD:', d['response'])"

curl -sk -X POST https://192.168.202.9/api/brain/dialogue/grounded \
  -H 'Content-Type: application/json' \
  -d '{"message": "My grandchildren are visiting today!"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('HAPPY:', d['response'])"
```

Expected: the two responses differ in emotional tone — sad message gets a comforting reply, happy message gets an uplifting reply.

- [ ] **Step 5: Commit completion marker**

```bash
cd /home/sbuysse/Documents/Coding/Projects/Akretio/Brain
git commit --allow-empty -m "feat: HOPE Phase 2 grounded companion deployed to Pi 5"
```
