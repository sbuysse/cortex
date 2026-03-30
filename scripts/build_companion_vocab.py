"""Build companion vocabulary from training responses.

Usage:
  python build_companion_vocab.py \
    --input data/companion_training/raw/triples.jsonl \
    --output data/companion_training/vocab/companion_vocab.json \
    --max-vocab 5000
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

SEED_WORDS = [
    "i", "you", "me", "my", "your", "we", "it", "that", "this", "is",
    "am", "are", "was", "were", "be", "been", "have", "has", "had", "do",
    "did", "will", "would", "could", "should", "can", "may", "might",
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "from", "by", "not", "so", "if", "then", "when", "how",
    "what", "who", "why", "where", "here", "there", "yes", "no",
    "good", "morning", "evening", "night", "today", "day", "time",
    "feel", "feeling", "well", "better", "sorry", "thank", "please",
    "know", "think", "like", "love", "miss", "hurt", "pain", "sad",
    "happy", "lonely", "tired", "worried", "confused", "remember",
    "dear", "friend", "name", "family", "daughter", "son", "husband",
    "wife", "grandmother", "grandfather", "grandchildren", "visit",
    "nurse", "doctor", "mention", "help", "care", "listen", "hear",
    "tell", "talk", "say", "think", "know", "understand", "hope",
    "nice", "lovely", "beautiful", "wonderful", "warm", "kind",
]


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def build_vocab(responses: list[str], min_freq: int = 2, max_vocab: int = 5000) -> dict[str, int]:
    vocab = dict(SPECIAL)
    counter = Counter()
    for text in responses:
        counter.update(tokenize(text))

    # Always add seed words
    for word in SEED_WORDS:
        if word not in vocab:
            vocab[word] = len(vocab)

    # Add by frequency
    for word, count in counter.most_common(max_vocab * 2):
        if len(vocab) >= max_vocab + len(SPECIAL):
            break
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = 40) -> list[int]:
    tokens = tokenize(text)[:max_len - 2]
    ids = [vocab["<bos>"]]
    ids += [vocab.get(t, vocab["<unk>"]) for t in tokens]
    ids += [vocab["<eos>"]]
    # Pad to max_len
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]


def decode(ids: list[int], inv_vocab: dict[int, str]) -> str:
    words = []
    for i in ids:
        token = inv_vocab.get(i, "<unk>")
        if token in ("<pad>", "<bos>"):
            continue
        if token == "<eos>":
            break
        words.append(token)
    return " ".join(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-vocab", type=int, default=5000)
    parser.add_argument("--min-freq", type=int, default=2)
    args = parser.parse_args()

    responses = []
    with open(args.input) as f:
        for line in f:
            rec = json.loads(line)
            responses.append(rec["response"])
    print(f"Loaded {len(responses)} responses.")

    vocab = build_vocab(responses, min_freq=args.min_freq, max_vocab=args.max_vocab)
    print(f"Vocabulary size: {len(vocab)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved to {args.output}")
