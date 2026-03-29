"""Train HOPE Companion Nano on (context, user_message, response) triples.

Architecture: HOPE nested learning model, byte-level tokenization.
  - Fast memory: SelfModifyingLayer (stateful, per-timestep self-modifying)
  - Slow weights: ContinuumMemoryBlock × n_layers
  - Vocab: 256 (byte-level, no vocabulary file)

Usage:
  python train_hope_companion.py \\
    --triples data/companion_training/raw/triples.jsonl \\
    --out     outputs/cortex/hope_companion/ \\
    --epochs  50
"""

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPANION_NANO = {
    "d_model": 384,
    "n_layers": 8,
    "vocab_size": 256,    # byte-level
    "seq_len": 512,
    "dropout": 0.1,
}

CRT_MARKER = b" [CRT] "   # 7 bytes — response start marker

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class SelfModifyingLayer(nn.Module):
    """Fast memory with learned decay.

    Per-timestep update:
        memory_t = decay * memory_{t-1} + linear_update(x_t)
        out_t    = linear_f(memory_t)

    TorchScript compatible.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear_f = nn.Linear(d_model, d_model)
        self.linear_update = nn.Linear(d_model, d_model)
        self.decay_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    (B, S, d_model) — embedded tokens
            mask: (B, S) float — 1.0 for real tokens, 0.0 for padding

        Returns:
            out: (B, S, d_model)

        Vectorized via causal conv1d — single GPU kernel instead of S sequential steps.
        h_t = sum_{i<=t} decay^(t-i) * u_i   where u_i = mask_i * linear_update(x_i)
        """
        B = x.shape[0]
        S = x.shape[1]
        D = self.d_model
        decay = torch.sigmoid(self.decay_param)  # scalar in (0,1)

        # Compute updates, zeroing out padding positions
        u = self.linear_update(x)                          # (B, S, D)
        u = u * mask.unsqueeze(-1)                         # zero padding

        # Build causal decay kernel: [decay^(S-1), ..., decay^1, decay^0]
        t_idx = torch.arange(S, device=x.device, dtype=x.dtype)
        kernel = (decay ** t_idx).flip(0)                  # (S,)
        kernel = kernel.view(1, 1, S)                      # (1, 1, S)

        # Apply causal conv per channel: (B*D, 1, S) * (1,1,S) -> (B*D, 1, S)
        u_t = u.permute(0, 2, 1).reshape(B * D, 1, S)     # (B*D, 1, S)
        h_t = torch.nn.functional.conv1d(u_t, kernel, padding=S - 1)  # (B*D, 1, 2S-1)
        h = h_t[:, :, :S].squeeze(1).reshape(B, D, S).permute(0, 2, 1)  # (B, S, D)

        return self.linear_f(h.reshape(-1, D)).reshape(B, S, D)  # (B, S, D)


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


class ContinuumMemoryBlock(nn.Module):
    """Slow weights FFN block with pre-norm.

    Pre-norm: LayerNorm → Linear(d → 4d) → GELU → Linear(4d → d) + residual
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor, brain_bias: Optional[Tensor] = None) -> Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        if brain_bias is not None:
            h = h + brain_bias  # (B, S, d_model) + (1, 1, d_model) broadcast
        return x + h


class HOPE(nn.Module):
    """HOPE (Nested Learning) Companion Nano model.

    Byte-level language model with:
      - Fast memory (SelfModifyingLayer)
      - Slow weights (ContinuumMemoryBlock × n_layers)
    """

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

    @torch.jit.export
    def generate(self, prompt_bytes: List[int], max_new: int) -> List[int]:
        """Greedy decode up to max_new bytes.

        Args:
            prompt_bytes: list of byte values (0-255)
            max_new:      maximum new tokens to generate

        Returns:
            List[int] of generated byte values (not including prompt)
        """
        result = torch.jit.annotate(List[int], [])

        # Clamp prompt to seq_len
        seq_len = self.seq_len
        if len(prompt_bytes) >= seq_len:
            prompt_bytes = prompt_bytes[len(prompt_bytes) - seq_len + 1:]

        context: List[int] = list(prompt_bytes)

        for _ in range(max_new):
            # Pad or truncate to seq_len
            if len(context) < seq_len:
                pad_len = seq_len - len(context)
                padded = [0] * pad_len + context
            else:
                padded = context[len(context) - seq_len:]

            tokens = torch.tensor(padded, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            logits = self.forward(tokens)  # (1, seq_len, vocab_size)

            # Get logits at the last real position
            last_pos = len(context) - 1
            if last_pos >= seq_len:
                last_pos = seq_len - 1
            # Account for left-padding offset
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CompanionDataset(Dataset):
    """Loads (context, user_message, response) triples and formats as byte sequences.

    Format:
        b"[CTX] " + context.encode() + b" [USR] " + user_message.encode()
        + b" [CRT] " + response.encode()

    Labels: -100 for context+user tokens (masked), byte value for response tokens.
    """

    def __init__(self, triples_path: str, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.samples: List[Tuple[List[int], List[int]]] = []

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

                # Truncate to seq_len
                doc = doc[:seq_len]
                doc_bytes = list(doc)

                # Find where the response starts (after CRT_MARKER)
                marker = list(CRT_MARKER)
                marker_len = len(marker)
                crt_pos = -1
                for i in range(len(doc_bytes) - marker_len + 1):
                    if doc_bytes[i : i + marker_len] == marker:
                        crt_pos = i + marker_len
                        break

                if crt_pos == -1 or crt_pos >= len(doc_bytes):
                    # No response in truncated doc — skip
                    continue

                # Pad to seq_len
                input_ids = doc_bytes[:seq_len]
                pad_len = seq_len - len(input_ids)
                input_ids = input_ids + [0] * pad_len

                # Labels: -100 for masked positions, byte value for response
                labels = [-100] * seq_len
                for i in range(crt_pos, len(doc_bytes)):
                    if i < seq_len:
                        # Label at position i predicts token at i+1 (shift)
                        if i + 1 < len(doc_bytes) and i < seq_len:
                            labels[i] = doc_bytes[i + 1] if i + 1 < seq_len else -100

                # Shift: input[0:S-1] → labels[1:S] (next-token prediction)
                # Recompute properly with shifted labels
                labels = [-100] * seq_len
                for i in range(crt_pos - 1, len(doc_bytes) - 1):
                    if i < seq_len:
                        labels[i] = doc_bytes[i + 1]

                self.samples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        input_ids, labels = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def print_sample(model: HOPE, device: torch.device) -> None:
    model.eval()
    model.cpu()
    prompt = b"[CTX] You are talking to Alice. [USR] How are you today? [CRT] "
    prompt_bytes = list(prompt)
    with torch.no_grad():
        response_bytes = model.generate(prompt_bytes, max_new=80)
    try:
        text = bytes(response_bytes).decode("utf-8", errors="replace")
    except Exception:
        text = str(response_bytes)
    print(f"  Sample response: {repr(text)}")
    model.to(device)
    model.train()


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    print(f"Loading triples from {args.triples} ...")
    full_dataset = CompanionDataset(args.triples, COMPANION_NANO["seq_len"])
    n_total = len(full_dataset)
    print(f"Loaded {n_total} samples.")

    if n_total == 0:
        print("ERROR: No valid samples found in triples file.")
        sys.exit(1)

    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = HOPE(**COMPANION_NANO).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer + LR schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_val_loss = float("inf")

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Update LR
            lr = cosine_lr(step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            logits = model(input_ids)  # (B, S, vocab_size)

            # Reshape for cross-entropy: (B*S, vocab_size) vs (B*S,)
            B, S, V = logits.shape
            loss = criterion(logits.reshape(B * S, V), labels.reshape(B * S))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                B, S, V = logits.shape
                loss = criterion(logits.reshape(B * S, V), labels.reshape(B * S))
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
            f"lr={lr:.2e}"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.out, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "config": COMPANION_NANO,
                },
                ckpt_path,
            )
            print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f}) → {ckpt_path}")

        # Print sample every 10 epochs
        if epoch % 10 == 0:
            print_sample(model, device)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints in: {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOPE Companion Nano")
    parser.add_argument(
        "--triples",
        default="data/companion_training/raw/triples.jsonl",
        help="Path to triples.jsonl",
    )
    parser.add_argument(
        "--out",
        default="outputs/cortex/hope_companion/",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.05, dest="val_split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
