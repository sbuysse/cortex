"""Train the Cortex companion language decoder.

Architecture: dual-memory TransformerDecoder
  Memory inputs: ctx_emb (512) + brain_emb (512)
  Generates: response tokens (word-level vocab ~5K)

Usage:
  python train_companion_decoder.py \\
    --triples data/companion_training/raw/triples.jsonl \\
    --vocab   data/companion_training/vocab/companion_vocab.json \\
    --embs    data/companion_training/processed/embeddings.safetensors \\
    --out     outputs/cortex/companion_decoder/ \\
    --epochs  100
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Hyper-parameters ──────────────────────────────────────────────────────────
D_MODEL    = 512
N_HEAD     = 8
NUM_LAYERS = 4
FF_DIM     = 2048
DROPOUT    = 0.1
MAX_TOKENS = 40
BATCH_SIZE = 128
LR_START   = 3e-4
LR_END     = 1e-5
WARMUP_STEPS = 500


# ── Model ─────────────────────────────────────────────────────────────────────

class CompanionDecoder(nn.Module):
    """Dual-memory Transformer decoder for companion response generation."""

    def __init__(
        self,
        vocab_size: int,
        max_tokens: int = MAX_TOKENS,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens

        self.ctx_proj   = nn.Linear(512, d_model)
        self.brain_proj = nn.Linear(512, d_model)

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_tokens + 2, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output  = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        brain_emb: torch.Tensor,
        ctx_emb: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            brain_emb : (B, 512)
            ctx_emb   : (B, 512)
            tokens    : (B, S)  token ids
        Returns:
            logits    : (B, S, vocab_size)
        """
        B = tokens.shape[0]
        S = tokens.shape[1]
        device = tokens.device

        mem = torch.stack([
            self.ctx_proj(ctx_emb),
            self.brain_proj(brain_emb),
        ], dim=1)  # (B, 2, d_model)

        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        tgt = self.token_embed(tokens) + self.pos_embed(pos)

        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).to(torch.bool)
        out = self.decoder(tgt, mem, tgt_mask=mask)
        return self.output(out)

    @torch.jit.export
    def generate(
        self,
        brain_emb: torch.Tensor,
        ctx_emb: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new: int = 40,
    ) -> List[int]:
        """Greedy decoding — returns list of token ids (excluding bos)."""
        device = brain_emb.device
        tokens = torch.tensor([[bos_id]], device=device)
        result = torch.jit.annotate(List[int], [])
        for _ in range(max_new):
            logits = self.forward(brain_emb, ctx_emb, tokens)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == eos_id:
                break
            result.append(next_id)
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], device=device)], dim=1
            )
        return result


# ── Dataset ───────────────────────────────────────────────────────────────────

class CompanionDataset(Dataset):
    def __init__(
        self,
        responses: List[str],
        vocab: dict,
        ctx_embs: torch.Tensor,
        msg_embs: torch.Tensor,
        max_tokens: int = MAX_TOKENS,
    ):
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from build_companion_vocab import encode as vocab_encode
        self._encode   = vocab_encode
        self.responses = responses
        self.vocab     = vocab
        self.ctx_embs  = ctx_embs
        self.msg_embs  = msg_embs
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.responses)

    def __getitem__(self, idx: int):
        ctx_emb  = self.ctx_embs[idx]
        msg_emb  = self.msg_embs[idx]
        brain_emb = F.normalize(0.7 * ctx_emb + 0.3 * msg_emb, dim=-1)

        target_ids = self._encode(self.responses[idx], self.vocab, self.max_tokens)
        target = torch.tensor(target_ids, dtype=torch.long)

        return ctx_emb, brain_emb, target[:-1], target[1:]


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, lr_start: float, lr_end: float) -> float:
    if step < warmup:
        return lr_start * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * progress))


# ── Training ──────────────────────────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.vocab) as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    print(f"Vocabulary: {len(vocab)} tokens")

    from safetensors.torch import load_file
    tensors  = load_file(args.embs)
    ctx_embs = tensors["context_embs"]
    msg_embs = tensors["message_embs"]

    responses: List[str] = []
    with open(args.triples) as f:
        for line in f:
            responses.append(json.loads(line)["response"])
    assert len(responses) == ctx_embs.shape[0], "Mismatch: triples vs embeddings"

    n_val = max(500, len(responses) // 20)
    train_resp, val_resp   = responses[:-n_val], responses[-n_val:]
    train_ctx,  val_ctx    = ctx_embs[:-n_val],  ctx_embs[-n_val:]
    train_msg,  val_msg    = msg_embs[:-n_val],  msg_embs[-n_val:]

    train_ds = CompanionDataset(train_resp, vocab, train_ctx, train_msg)
    val_ds   = CompanionDataset(val_resp,   vocab, val_ctx,   val_msg)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=256, num_workers=2)

    model = CompanionDecoder(vocab_size=len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    total_steps  = args.epochs * len(train_dl)
    step         = 0
    best_val_loss = float("inf")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from build_companion_vocab import decode as vocab_decode

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for ctx_emb, brain_emb, inp_tokens, tgt_tokens in train_dl:
            ctx_emb    = ctx_emb.to(device)
            brain_emb  = brain_emb.to(device)
            inp_tokens = inp_tokens.to(device)
            tgt_tokens = tgt_tokens.to(device)

            lr = get_lr(step, WARMUP_STEPS, total_steps, LR_START, LR_END)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            logits = model(brain_emb, ctx_emb, inp_tokens)
            loss   = criterion(logits.reshape(-1, len(vocab)), tgt_tokens.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            step += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ctx_emb, brain_emb, inp_tokens, tgt_tokens in val_dl:
                ctx_emb    = ctx_emb.to(device)
                brain_emb  = brain_emb.to(device)
                inp_tokens = inp_tokens.to(device)
                tgt_tokens = tgt_tokens.to(device)
                logits   = model(brain_emb, ctx_emb, inp_tokens)
                val_loss += criterion(
                    logits.reshape(-1, len(vocab)), tgt_tokens.reshape(-1)
                ).item()

        val_loss    /= len(val_dl)
        train_loss   = epoch_loss / len(train_dl)
        elapsed      = time.time() - t0
        lr = get_lr(step, WARMUP_STEPS, total_steps, LR_START, LR_END)
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr:.2e}  {elapsed:.1f}s"
        )

        if (epoch + 1) % 10 == 0:
            model.eval()
            ctx   = val_ctx[0:1].to(device)
            msg   = val_msg[0:1].to(device)
            brain = F.normalize(0.7 * ctx + 0.3 * msg, dim=-1)
            ids   = model.generate(brain, ctx, bos_id=1, eos_id=2, max_new=30)
            text  = vocab_decode(ids, inv_vocab)
            print(f'  Sample: "{text}"')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "companion_decoder.pt")
            print(f"  Saved best (val={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/companion_training/raw/triples.jsonl")
    parser.add_argument("--vocab",   default="data/companion_training/vocab/companion_vocab.json")
    parser.add_argument("--embs",    default="data/companion_training/processed/embeddings.safetensors")
    parser.add_argument("--out",     default="outputs/cortex/companion_decoder")
    parser.add_argument("--epochs",  type=int, default=100)
    args = parser.parse_args()
    train(args)
