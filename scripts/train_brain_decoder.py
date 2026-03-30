#!/usr/bin/env python3
"""Train a brain-to-text decoder: 512-dim brain embeddings → natural language.

Architecture: embedding → 2-layer transformer decoder (dim=512, 4 heads) → tokens
Training data: (MLP-projected centroid, category label) pairs for 898 categories
Augmented with concept compositions and description expansions.

The brain speaks FROM its embeddings — language grounded in experience.
"""

import os
import json
import time
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open
from collections import Counter
import urllib.request

# ── Config ──────────────────────────────────────────────────────────
EMBED_PATH = "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors"
CSV_PATH = "/opt/brain/data/vggsound/vggsound.csv"
MLP_DIR = "/opt/brain/outputs/cortex/v5_mlp"
AUDIOSET_DIR = "/opt/brain/outputs/cortex/audioset_expansion"
OUT_DIR = "/opt/brain/outputs/cortex/brain_decoder"

D_MODEL = 512
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 1024
MAX_TOKENS = 32
DROPOUT = 0.1

EPOCHS = 300
BATCH = 64
LR_MAX = 5e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

DEVICE = "cpu"


# ── Helpers ─────────────────────────────────────────────────────────
def load_rust_matrix(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        rows, cols = map(int, header.split("x"))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(rows, cols)


def mlp_project(emb, w):
    proj = emb @ w
    proj = np.maximum(proj, 0)
    norm = np.linalg.norm(proj, axis=-1, keepdims=True).clip(1e-12)
    return proj / norm


def ollama_expand(label: str) -> str:
    """Get a short natural language description of a sound category."""
    try:
        prompt = (
            f"Describe the sound of '{label}' in exactly one short sentence (under 10 words). "
            "Start with 'The sound of' or 'A'. Only the description, nothing else."
        )
        payload = json.dumps({
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.5, "num_predict": 20},
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        return result.get("response", label).strip().strip('"\'.')
    except Exception:
        return label


# ── Tokenizer (simple word-level) ──────────────────────────────────
def build_vocab(texts: list[str], min_freq: int = 2) -> dict:
    """Build a simple word-level vocabulary."""
    counter = Counter()
    for text in texts:
        words = text.lower().split()
        words = [re.sub(r'[^\w]', '', w) for w in words]
        words = [w for w in words if w]
        counter.update(words)

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for word, count in counter.most_common():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def tokenize(text: str, vocab: dict) -> list[int]:
    """Convert text to token IDs."""
    words = text.lower().split()
    words = [re.sub(r'[^\w]', '', w) for w in words]
    words = [w for w in words if w]
    tokens = [vocab.get("<bos>", 1)]
    for w in words:
        tokens.append(vocab.get(w, vocab.get("<unk>", 3)))
    tokens.append(vocab.get("<eos>", 2))
    return tokens


# ── Model ───────────────────────────────────────────────────────────
class BrainDecoder(nn.Module):
    def __init__(self, d_model=D_MODEL, vocab_size=100, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Linear(d_model, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(MAX_TOKENS + 2, d_model)

    def forward(self, brain_emb, tokens):
        B = brain_emb.shape[0]
        memory = self.embed(brain_emb).unsqueeze(1)  # (B, 1, D)
        S = tokens.shape[1]
        pos = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, -1)
        tgt = self.token_embed(tokens) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(S, S, device=tokens.device), diagonal=1).bool()
        out = self.decoder(tgt, memory, tgt_mask=mask)
        return self.output(out)


# ── Training ────────────────────────────────────────────────────────
def build_training_data():
    """Build (embedding, text) pairs for training."""
    import csv

    print("Loading embeddings and MLP weights...")
    with safe_open(EMBED_PATH, framework="numpy") as f:
        v_emb = f.get_tensor("v_emb")
        a_emb = f.get_tensor("a_emb")

    w_v = load_rust_matrix(os.path.join(MLP_DIR, "w_v.bin"))
    w_a = load_rust_matrix(os.path.join(MLP_DIR, "w_a.bin"))

    v_proj = mlp_project(v_emb, w_v)
    a_proj = mlp_project(a_emb, w_a)

    # Load labels
    clips = []
    with open(CSV_PATH) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) >= 3:
                clips.append(row[2].strip())

    # Per-category centroids
    label_to_idx = {}
    for i, label in enumerate(clips):
        if i < len(v_proj):
            label_to_idx.setdefault(label, []).append(i)

    pairs = []  # (embedding, text)

    print(f"Building pairs from {len(label_to_idx)} VGGSound categories...")
    for label, indices in label_to_idx.items():
        cat_v = v_proj[indices].mean(axis=0)
        cat_a = a_proj[indices].mean(axis=0)
        centroid = (cat_v + cat_a) / 2.0
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Basic label
        pairs.append((centroid, label))
        # Expanded description
        pairs.append((centroid, f"the sound of {label}"))

    # AudioSet expansion
    audioset_labels_path = Path(AUDIOSET_DIR) / "labels.json"
    audioset_emb_path = Path(AUDIOSET_DIR) / "embeddings.npy"
    if audioset_labels_path.exists() and audioset_emb_path.exists():
        as_labels = json.loads(audioset_labels_path.read_text())
        as_embs = np.load(str(audioset_emb_path))
        print(f"Adding {len(as_labels)} AudioSet categories...")
        for i, label in enumerate(as_labels):
            if i < len(as_embs):
                emb = as_embs[i] / (np.linalg.norm(as_embs[i]) + 1e-12)
                pairs.append((emb, label))
                pairs.append((emb, f"the sound of {label}"))

    # LLM-expanded descriptions (sample 100 categories)
    print("Generating LLM-expanded descriptions for 100 sample categories...")
    sampled_labels = []  # skip LLM expansion for speed
    for label in sampled_labels:
        indices = label_to_idx[label]
        cat_v = v_proj[indices].mean(axis=0)
        cat_a = a_proj[indices].mean(axis=0)
        centroid = (cat_v + cat_a) / 2.0
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        expanded = ollama_expand(label)
        if expanded and expanded != label:
            pairs.append((centroid, expanded))

    # Concept compositions (random pairs of categories)
    rng = np.random.RandomState(42)
    label_list = list(label_to_idx.keys())
    for _ in range(200):
        l1, l2 = rng.choice(label_list, 2, replace=False)
        idx1, idx2 = label_to_idx[l1], label_to_idx[l2]
        c1 = (v_proj[idx1].mean(0) + a_proj[idx1].mean(0)) / 2
        c2 = (v_proj[idx2].mean(0) + a_proj[idx2].mean(0)) / 2
        composed = (c1 + c2) / 2
        composed = composed / (np.linalg.norm(composed) + 1e-12)
        pairs.append((composed, f"{l1} and {l2}"))

    print(f"Total training pairs: {len(pairs)}")
    return pairs


def train():
    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = build_training_data()
    all_texts = [text for _, text in pairs]

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(all_texts, min_freq=1)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Save vocab
    with open(os.path.join(OUT_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # Tokenize all texts
    all_tokens = [tokenize(text, vocab) for text in all_texts]
    all_embs = np.stack([emb for emb, _ in pairs])

    N = len(pairs)
    print(f"Training on {N} pairs with vocab size {vocab_size}")

    model = BrainDecoder(vocab_size=vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    def get_batch(batch_size):
        indices = np.random.choice(N, batch_size, replace=True)
        max_len = max(len(all_tokens[i]) for i in indices)
        embs = np.stack([all_embs[i] for i in indices])
        tokens = np.zeros((batch_size, max_len), dtype=np.int64)
        for j, i in enumerate(indices):
            toks = all_tokens[i]
            tokens[j, :len(toks)] = toks

        embs_t = torch.from_numpy(embs).float()
        tokens_t = torch.from_numpy(tokens).long()
        return embs_t, tokens_t

    best_loss = float("inf")
    t0 = time.time()
    steps_per_epoch = max(1, N // BATCH)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        progress = epoch / max(1, EPOCHS - 1)
        lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for step in range(steps_per_epoch):
            embs, tokens = get_batch(BATCH)
            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            logits = model(embs, input_tokens)  # (B, S-1, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_tokens.reshape(-1),
                ignore_index=0)  # ignore padding

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | loss={avg_loss:.4f} | lr={lr:.6f} | {elapsed:.0f}s")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "decoder.pt"))
                print(f"  → Saved best model (loss={avg_loss:.4f})")

            # Sample decode
            model.eval()
            with torch.no_grad():
                # Pick a random embedding and decode
                idx = np.random.randint(N)
                emb_t = torch.from_numpy(all_embs[idx:idx+1]).float()
                bos = vocab.get("<bos>", 1)
                eos = vocab.get("<eos>", 2)
                idx2word = {v: k for k, v in vocab.items()}

                gen_tokens = [bos]
                for _ in range(MAX_TOKENS):
                    tok_t = torch.tensor([gen_tokens], dtype=torch.long)
                    logits = model(emb_t, tok_t)
                    next_tok = int(logits[0, -1].argmax())
                    if next_tok == eos:
                        break
                    gen_tokens.append(next_tok)

                decoded = " ".join(idx2word.get(t, "?") for t in gen_tokens[1:])
                target_text = all_texts[idx]
                print(f"  Sample: '{decoded}' (target: '{target_text}')")

    total_time = time.time() - t0
    print(f"\nDone! Total time: {total_time:.0f}s")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model: {OUT_DIR}/decoder.pt, Vocab: {OUT_DIR}/vocab.json")


if __name__ == "__main__":
    train()
