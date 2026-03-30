#!/usr/bin/env python3
"""Download AudioCaps dataset and encode to embeddings for Rust brain ingestion.

AudioCaps: 50K human-written captions for AudioSet clips.
Outputs: audiocaps/audio_embs.npy (N×512), audiocaps/labels.json (N captions)

Stream-process-delete: downloads 10s audio clips, encodes, deletes immediately.
"""
import os, json, csv, subprocess, struct, sys, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/audiocaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("/tmp/audiocaps_dl")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Load Whisper + MiniLM models
print("Loading models...")
import torch
from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, AutoModel

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base").eval()

text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

def encode_audio(wav_path):
    """Encode WAV file to 512-dim Whisper embedding."""
    import torchaudio
    try:
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(0)[:16000*30]  # mono, max 30s
        inputs = whisper_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            enc = whisper_model.encoder(inputs.input_features)
            emb = enc.last_hidden_state.mean(dim=1).squeeze()  # pool to 512-dim
        emb = emb / emb.norm()
        return emb.numpy().astype(np.float32)
    except Exception as e:
        return None

def encode_text(text):
    """Encode text to 384-dim MiniLM embedding."""
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        output = text_model(**inputs)
        emb = output.last_hidden_state.mean(dim=1).squeeze()
    emb = emb / emb.norm()
    return emb.numpy().astype(np.float32)

def download_audiocaps_csv():
    """Download AudioCaps CSV from GitHub."""
    csv_dir = TMP_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)
    for split in ["train", "val", "test"]:
        url = f"https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/{split}.csv"
        out = csv_dir / f"{split}.csv"
        if not out.exists():
            print(f"Downloading {split}.csv...")
            subprocess.run(["wget", "-q", "-O", str(out), url], check=True)
    return csv_dir

def process_audiocaps():
    csv_dir = download_audiocaps_csv()

    audio_embs = []
    text_embs = []
    captions = []
    failed = 0

    # Process train split (largest)
    csv_path = csv_dir / "train.csv"
    if not csv_path.exists():
        print("ERROR: train.csv not found")
        return

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} AudioCaps entries...")

    for i, row in enumerate(rows):
        ytid = row.get("youtube_id", row.get("audiocap_id", ""))
        start = row.get("start_sec", "0")
        caption = row.get("caption", "")

        if not ytid or not caption:
            continue

        # Download 10s audio clip
        wav_path = TMP_DIR / f"clip_{i}.wav"
        try:
            url = f"https://www.youtube.com/watch?v={ytid}"
            result = subprocess.run([
                "yt-dlp", "-x", "--audio-format", "wav",
                "--postprocessor-args", f"ffmpeg:-ss {start} -t 10 -ar 16000 -ac 1",
                "-o", str(wav_path), "--no-playlist",
                "--match-filter", "duration<600",
                url
            ], capture_output=True, timeout=30)

            if not wav_path.exists():
                # yt-dlp may add extension
                for ext in [".wav", ".wav.wav"]:
                    alt = TMP_DIR / f"clip_{i}{ext}"
                    if alt.exists():
                        wav_path = alt
                        break

            if wav_path.exists():
                a_emb = encode_audio(str(wav_path))
                t_emb = encode_text(caption)

                if a_emb is not None and len(a_emb) == 512 and len(t_emb) == 384:
                    audio_embs.append(a_emb)
                    text_embs.append(t_emb)
                    captions.append(caption)

                wav_path.unlink(missing_ok=True)
            else:
                failed += 1
        except Exception as e:
            failed += 1
            wav_path.unlink(missing_ok=True)

        # Progress + save checkpoints
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] encoded={len(audio_embs)}, failed={failed}")

        if (i + 1) % 1000 == 0 and audio_embs:
            save_checkpoint(audio_embs, text_embs, captions)

    # Final save
    if audio_embs:
        save_checkpoint(audio_embs, text_embs, captions)

    print(f"\nDone: {len(audio_embs)} pairs encoded, {failed} failed")

def save_checkpoint(audio_embs, text_embs, captions):
    a = np.array(audio_embs, dtype=np.float32)
    t = np.array(text_embs, dtype=np.float32)
    np.save(OUTPUT_DIR / "audio_embs.npy", a)
    np.save(OUTPUT_DIR / "text_embs.npy", t)
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(captions, f)
    print(f"  Saved checkpoint: {len(audio_embs)} pairs to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_audiocaps()
