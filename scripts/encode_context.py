"""Context encoder: text → 512-dim vector via MiniLM + MLP projection.

Used during training data preprocessing.
In production, Rust uses the equivalent pipeline in brain-inference.
"""
import argparse
import struct
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def _load_matrix(path: str) -> torch.Tensor:
    """Load Rust binary matrix format: u32 rows, u32 cols, f32 LE data."""
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(rows * cols * 4), dtype="<f4").reshape(rows, cols)
    return torch.from_numpy(data.copy())


class ContextEncoder:
    def __init__(self, minilm_path: str, tokenizer_path: str, mlp_v_path: str):
        self.model = torch.jit.load(minilm_path, map_location="cpu")
        self.model.eval()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.W_v = _load_matrix(mlp_v_path)  # (384, 512)

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode context text → 512-dim L2-normalized tensor."""
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, padding="max_length",
        )
        out = self.model(enc["input_ids"], enc["attention_mask"])  # (1, 384)
        emb = F.normalize(out.squeeze(0), dim=-1)    # (384,)
        projected = F.relu(emb @ self.W_v)            # (512,)
        return F.normalize(projected, dim=-1)          # L2-normalize


def precompute_context_embeddings(triples_path: str, out_path: str, encoder: "ContextEncoder"):
    """Precompute and cache context + message embeddings for all training triples."""
    import json
    from safetensors.torch import save_file

    contexts, messages, responses = [], [], []
    with open(triples_path) as f:
        for line in f:
            rec = json.loads(line)
            contexts.append(rec["context_text"])
            messages.append(rec["user_message"])
            responses.append(rec["response"])

    print(f"Encoding {len(contexts)} context strings...")
    ctx_embs = torch.stack([encoder.encode(c) for c in contexts])

    print(f"Encoding {len(messages)} user messages...")
    msg_embs = torch.stack([encoder.encode(m) for m in messages])

    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_file({"context_embs": ctx_embs, "message_embs": msg_embs}, out_path)
    print(f"Saved {ctx_embs.shape} to {out_path}")
    return contexts, messages, responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True, help="Path to triples.jsonl")
    parser.add_argument("--output",  required=True, help="Output .safetensors path")
    parser.add_argument("--minilm",  default="outputs/cortex/text_encoder/minilm_ts.pt")
    parser.add_argument("--tokenizer", default="outputs/cortex/text_encoder/tokenizer")
    parser.add_argument("--mlp-v",   default="outputs/cortex/v5_mlp/w_v.bin")
    args = parser.parse_args()

    encoder = ContextEncoder(
        minilm_path=args.minilm,
        tokenizer_path=args.tokenizer,
        mlp_v_path=args.mlp_v,
    )
    precompute_context_embeddings(args.triples, args.output, encoder)
