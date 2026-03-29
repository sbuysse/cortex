"""Generate 8×512 emotion embedding table as raw float32 binary.

Emotions (index order, must match Rust emotion_to_idx):
  0: neutral, 1: sad, 2: pain, 3: happy,
  4: fearful, 5: angry, 6: confused, 7: tired

Output: emotion_table.bin — 8 × 512 × 4 bytes = 16384 bytes, row-major f32 LE.
"""

import argparse
import struct
from pathlib import Path

import numpy as np


EMOTIONS = ["neutral", "sad", "pain", "happy", "fearful", "angry", "confused", "tired"]
DIM = 512
SEED = 42


def generate(out_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    table = rng.standard_normal((len(EMOTIONS), DIM)).astype(np.float32)

    # L2-normalize each row
    norms = np.linalg.norm(table, axis=1, keepdims=True).clip(min=1e-12)
    table /= norms

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.tofile(str(out_path))

    print(f"Saved emotion table {table.shape} → {out_path}")
    print(f"  Row norms: {np.linalg.norm(table, axis=1).round(4).tolist()}")
    for i, name in enumerate(EMOTIONS):
        print(f"  {i}: {name}  mean={table[i].mean():.4f}  std={table[i].std():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="outputs/cortex/hope_companion/emotion_table.bin",
        help="Output path for emotion_table.bin",
    )
    args = parser.parse_args()
    generate(Path(args.out))


if __name__ == "__main__":
    main()
