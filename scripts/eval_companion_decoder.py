# scripts/eval_companion_decoder.py
"""Sample responses from the trained decoder and compare to Ollama baseline.

Usage:
  python eval_companion_decoder.py \
    --decoder outputs/cortex/companion_decoder/companion_decoder_ts.pt \
    --vocab   outputs/cortex/companion_decoder/companion_vocab.json \
    --triples data/companion_training/raw/triples.jsonl \
    --n 50
"""

import argparse
import json
import random
import torch
import torch.nn.functional as F

# Import decoder from training script
import sys; sys.path.insert(0, "scripts")
from train_companion_decoder import CompanionDecoder
from encode_context import ContextEncoder
from build_companion_vocab import decode as vocab_decode


def load_decoder(decoder_path: str, vocab_path: str):
    with open(vocab_path) as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    model = torch.jit.load(decoder_path, map_location="cpu")
    model.eval()
    return model, vocab, inv_vocab


def sample_response(model, brain_emb, ctx_emb, vocab, inv_vocab, max_new=40):
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    tokens = torch.tensor([[bos_id]])
    result = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(brain_emb, ctx_emb, tokens)
            next_id = int(logits[0, -1].argmax())
            if next_id == eos_id:
                break
            result.append(next_id)
            tokens = torch.cat([tokens, torch.tensor([[next_id]])], dim=1)
    return vocab_decode(result, inv_vocab)


def main(args):
    model, vocab, inv_vocab = load_decoder(args.decoder, args.vocab)
    enc = ContextEncoder(
        minilm_path="outputs/cortex/text_encoder/minilm_ts.pt",
        tokenizer_path="outputs/cortex/text_encoder/tokenizer",
        mlp_v_path="outputs/cortex/v5_mlp/w_v.bin",
    )

    triples = []
    with open(args.triples) as f:
        for line in f:
            triples.append(json.loads(line))
    sample = random.sample(triples, min(args.n, len(triples)))

    print(f"{'USER MESSAGE':<40} {'DECODER RESPONSE':<60} {'REFERENCE'}")
    print("-" * 140)
    for rec in sample:
        ctx_emb   = enc.encode(rec["context_text"]).unsqueeze(0)
        msg_emb   = enc.encode(rec["user_message"]).unsqueeze(0)
        brain_emb = F.normalize(0.7 * ctx_emb + 0.3 * msg_emb, dim=-1)
        response  = sample_response(model, brain_emb, ctx_emb, vocab, inv_vocab)
        print(f"{rec['user_message'][:40]:<40}  {response[:60]:<60}  {rec['response'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--vocab",   required=True)
    parser.add_argument("--triples", required=True)
    parser.add_argument("--n",       type=int, default=50)
    main(parser.parse_args())
