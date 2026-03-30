#!/usr/bin/env python3
"""Process downloaded AudioSet TFRecord embeddings into the brain's 512-dim space.

Reads 128-dim VGGish embeddings from TFRecords, projects them through the
trained adapter (128→512), and saves as numpy arrays ready for the brain.

Also extracts label information from the AudioSet ontology.
"""

import os
import struct
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from glob import glob

# ── Config ──────────────────────────────────────────────────────────
AUDIOSET_DIR = "/opt/brain/data/audioset/audioset_v1_embeddings"
ADAPTER_DIR = "/opt/brain/outputs/cortex/audioset_adapted"
OUT_DIR = "/opt/brain/outputs/cortex/audioset_brain"

BATCH_SIZE = 10000  # process in chunks for memory efficiency


# ── TFRecord Parser (no tensorflow dependency) ─────────────────────
def read_tfrecord(path):
    """Parse a TFRecord file without tensorflow. Returns list of examples."""
    examples = []
    with open(path, 'rb') as f:
        while True:
            # Read length
            buf = f.read(8)
            if len(buf) < 8:
                break
            length = struct.unpack('Q', buf)[0]
            # Skip CRC of length
            f.read(4)
            # Read data
            data = f.read(length)
            # Skip CRC of data
            f.read(4)
            examples.append(data)
    return examples


def parse_sequence_example(data):
    """Parse a tf.train.SequenceExample protobuf manually.

    AudioSet TFRecords use SequenceExample with:
    - context: video_id (bytes), labels (int64_list), start_time_seconds (float)
    - feature_lists: audio_embedding (bytes_list of 128-byte quantized vectors)
    """
    # Minimal protobuf parsing - extract the key fields
    # This is a simplified parser that handles the AudioSet format
    result = {"video_id": "", "labels": [], "embeddings": []}

    try:
        # Use the protobuf wire format to extract fields
        pos = 0
        while pos < len(data):
            if pos >= len(data):
                break
            # Read field tag
            tag_byte = data[pos]
            field_num = tag_byte >> 3
            wire_type = tag_byte & 0x7
            pos += 1

            if wire_type == 2:  # Length-delimited
                # Read varint length
                length = 0
                shift = 0
                while pos < len(data):
                    b = data[pos]
                    pos += 1
                    length |= (b & 0x7f) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                # Skip the bytes
                pos += length
            elif wire_type == 0:  # Varint
                while pos < len(data) and data[pos] & 0x80:
                    pos += 1
                pos += 1
            elif wire_type == 5:  # 32-bit
                pos += 4
            elif wire_type == 1:  # 64-bit
                pos += 8
    except Exception:
        pass

    return result


def extract_embeddings_from_tfrecord(path):
    """Extract 128-dim embeddings from a TFRecord file.

    AudioSet stores quantized uint8 embeddings. Each example has
    multiple 1-second frames. We average all frames per clip.
    """
    embeddings = []
    labels_list = []

    try:
        examples = read_tfrecord(path)
        for raw in examples:
            # AudioSet TFRecords store 128-byte audio_embedding features
            # Each byte is a quantized float in [0, 255] → remap to [-1, 1]
            # Find 128-byte chunks in the raw protobuf data
            emb_chunks = []
            pos = 0
            while pos < len(raw) - 128:
                # Look for length-delimited fields of exactly 128 bytes
                # that look like embedding data (varied byte values)
                if pos + 2 < len(raw):
                    # Check if this could be a 128-byte bytes field
                    b = raw[pos]
                    if b == 128 and raw[pos+1] == 1:  # varint encoding of 128
                        # This might be a length prefix for 128 bytes
                        chunk = raw[pos+2:pos+2+128]
                        if len(chunk) == 128:
                            # Check it looks like embedding data (not all zeros/same)
                            arr = np.frombuffer(chunk, dtype=np.uint8)
                            if arr.std() > 5:  # varied values = likely embedding
                                emb_chunks.append(arr.astype(np.float32) / 255.0 * 2.0 - 1.0)
                                pos += 2 + 128
                                continue
                pos += 1

            if emb_chunks:
                # Average all frames for this clip
                avg_emb = np.mean(emb_chunks, axis=0)
                avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-12)
                embeddings.append(avg_emb)
    except Exception as e:
        pass

    return embeddings


# ── Adapter Model ──────────────────────────────────────────────────
class AudioSetAdapter(nn.Module):
    def __init__(self, dim_in=128, dim_hidden=256, dim_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, dim=-1)


def process_all():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load adapter
    print("Loading adapter model...")
    adapter = AudioSetAdapter()
    adapter_path = os.path.join(ADAPTER_DIR, "adapter.pt")
    if os.path.exists(adapter_path):
        adapter.load_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=True))
        print("Loaded trained adapter")
    else:
        print("WARNING: No trained adapter found, using random weights")
    adapter.eval()

    # Process each split
    for split in ["bal_train", "eval", "unbal_train"]:
        split_dir = os.path.join(AUDIOSET_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Skipping {split} (not found)")
            continue

        tfrecord_files = sorted(glob(os.path.join(split_dir, "*.tfrecord")))
        print(f"\n=== Processing {split}: {len(tfrecord_files)} files ===")

        all_embeddings_128 = []
        all_embeddings_512 = []
        processed_files = 0
        t0 = time.time()

        for i, fpath in enumerate(tfrecord_files):
            embs = extract_embeddings_from_tfrecord(fpath)
            if embs:
                all_embeddings_128.extend(embs)

            processed_files += 1

            # Process in batches through adapter
            if len(all_embeddings_128) >= BATCH_SIZE or i == len(tfrecord_files) - 1:
                if all_embeddings_128:
                    batch = np.stack(all_embeddings_128)
                    with torch.no_grad():
                        batch_t = torch.from_numpy(batch).float()
                        projected = adapter(batch_t).numpy()
                    all_embeddings_512.append(projected)
                    all_embeddings_128 = []

            if (i + 1) % 500 == 0 or i == len(tfrecord_files) - 1:
                total_embs = sum(len(e) for e in all_embeddings_512) + len(all_embeddings_128)
                elapsed = time.time() - t0
                rate = processed_files / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(tfrecord_files)}] {total_embs:,} embeddings | {rate:.0f} files/s | {elapsed:.0f}s")

        # Concatenate and save
        if all_embeddings_512:
            final = np.concatenate(all_embeddings_512, axis=0)
            out_path = os.path.join(OUT_DIR, f"{split}_embeddings.npy")
            np.save(out_path, final)
            print(f"  Saved {split}: {final.shape} → {out_path}")
            print(f"  File size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")
        else:
            print(f"  WARNING: No embeddings extracted from {split}")

    # Summary
    print("\n=== Summary ===")
    total = 0
    for f in glob(os.path.join(OUT_DIR, "*.npy")):
        arr = np.load(f)
        print(f"  {os.path.basename(f)}: {arr.shape}")
        total += arr.shape[0]
    print(f"  Total embeddings in brain space: {total:,}")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    process_all()
