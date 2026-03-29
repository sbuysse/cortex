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
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent))
from train_hope_companion import HOPE, COMPANION_NANO, CRT_MARKER

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
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
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
