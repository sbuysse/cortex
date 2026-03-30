#!/usr/bin/env python3
"""Download WavCaps metadata and encode available audio to embeddings.

WavCaps: 400K audio-text pairs from FreeSound/BBC/SoundBible.
We focus on the FreeSound subset (most accessible).
Outputs: wavcaps/audio_embs.npy (N×512), wavcaps/labels.json (N captions)

Streams audio, encodes, deletes — never stores raw audio.
"""
import os, json, subprocess, sys, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/wavcaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("/tmp/wavcaps_dl")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# WavCaps metadata URLs (from GitHub)
WAVCAPS_META = {
    "freesound": "https://raw.githubusercontent.com/XinhaoMei/WavCaps/master/data/json_files/FreeSound/fsd_final.json",
    "soundbible": "https://raw.githubusercontent.com/XinhaoMei/WavCaps/master/data/json_files/SoundBible/sb_final.json",
}

def download_metadata():
    """Download WavCaps JSON metadata."""
    meta = {}
    for name, url in WAVCAPS_META.items():
        out = TMP_DIR / f"{name}.json"
        if not out.exists():
            print(f"Downloading {name} metadata...")
            subprocess.run(["wget", "-q", "-O", str(out), url])
        if out.exists():
            try:
                data = json.load(open(out))
                if isinstance(data, dict) and "data" in data:
                    meta[name] = data["data"]
                elif isinstance(data, list):
                    meta[name] = data
                print(f"  {name}: {len(meta.get(name, []))} entries")
            except:
                print(f"  {name}: failed to parse")
    return meta

def encode_audio_url(url, idx):
    """Download audio from URL, encode via Whisper, delete."""
    wav_path = TMP_DIR / f"clip_{idx}.wav"
    try:
        # Download with timeout
        result = subprocess.run(
            ["wget", "-q", "--timeout=10", "-O", str(wav_path), url],
            capture_output=True, timeout=15)
        if not wav_path.exists() or wav_path.stat().st_size < 1000:
            wav_path.unlink(missing_ok=True)
            return None

        # Convert to 16kHz mono
        out_path = TMP_DIR / f"clip_{idx}_16k.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-ar", "16000", "-ac", "1",
                       "-t", "30", "-f", "wav", str(out_path)], capture_output=True, timeout=10)
        wav_path.unlink(missing_ok=True)

        if not out_path.exists():
            return None

        import torchaudio, torch
        waveform, sr = torchaudio.load(str(out_path))
        waveform = waveform.mean(0)[:16000*30]
        out_path.unlink(missing_ok=True)

        global whisper_processor, whisper_model
        inputs = whisper_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            enc = whisper_model.encoder(inputs.input_features)
            emb = enc.last_hidden_state.mean(dim=1).squeeze()
        emb = emb / emb.norm()
        return emb.numpy().astype(np.float32)
    except:
        wav_path.unlink(missing_ok=True)
        return None

def encode_text(text):
    """Encode caption to 384-dim."""
    import torch
    global text_tokenizer, text_model
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        output = text_model(**inputs)
        emb = output.last_hidden_state.mean(dim=1).squeeze()
    emb = emb / emb.norm()
    return emb.numpy().astype(np.float32)

def process_wavcaps():
    print("Loading models...")
    import torch
    from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, AutoModel
    global whisper_processor, whisper_model, text_tokenizer, text_model
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-base").eval()
    text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

    meta = download_metadata()
    if not meta:
        print("No metadata downloaded")
        return

    audio_embs = []
    text_embs = []
    captions = []
    failed = 0
    batch = 0

    for source, entries in meta.items():
        print(f"\nProcessing {source}: {len(entries)} entries")
        for i, entry in enumerate(entries):
            # Extract caption and audio URL
            caption = entry.get("caption", entry.get("text", ""))
            audio_url = entry.get("download_link", entry.get("audio", entry.get("url", "")))

            if not caption or not audio_url:
                continue

            # Encode audio
            a_emb = encode_audio_url(audio_url, i)
            if a_emb is not None and len(a_emb) == 512:
                t_emb = encode_text(caption)
                if len(t_emb) == 384:
                    audio_embs.append(a_emb)
                    text_embs.append(t_emb)
                    captions.append(caption)
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(entries)}] encoded={len(audio_embs)}, failed={failed}")

            # Save batch every 5000 clips
            if len(audio_embs) >= 5000:
                save_batch(audio_embs, text_embs, captions, batch)
                batch += 1
                audio_embs, text_embs, captions = [], [], []

            # Rate limiting
            time.sleep(0.1)

    # Save remaining
    if audio_embs:
        save_batch(audio_embs, text_embs, captions, batch)

    # Merge all batches into single files
    merge_batches()

    print(f"\nDone: {sum(1 for f in OUTPUT_DIR.glob('batch_*_audio.npy'))} batches")

def save_batch(audio_embs, text_embs, captions, batch_num):
    a = np.array(audio_embs, dtype=np.float32)
    t = np.array(text_embs, dtype=np.float32)
    np.save(OUTPUT_DIR / f"batch_{batch_num}_audio.npy", a)
    np.save(OUTPUT_DIR / f"batch_{batch_num}_text.npy", t)
    with open(OUTPUT_DIR / f"batch_{batch_num}_captions.json", "w") as f:
        json.dump(captions, f)
    print(f"  Saved batch {batch_num}: {len(audio_embs)} pairs")

def merge_batches():
    """Merge all batch files into single audio_embs.npy + labels.json."""
    all_audio = []
    all_labels = []
    for f in sorted(OUTPUT_DIR.glob("batch_*_audio.npy")):
        batch_num = f.stem.split("_")[1]
        all_audio.append(np.load(f))
        labels_f = OUTPUT_DIR / f"batch_{batch_num}_captions.json"
        if labels_f.exists():
            all_labels.extend(json.load(open(labels_f)))
    if all_audio:
        merged = np.concatenate(all_audio, axis=0)
        np.save(OUTPUT_DIR / "audio_embs.npy", merged)
        with open(OUTPUT_DIR / "labels.json", "w") as f:
            json.dump(all_labels, f)
        print(f"Merged: {merged.shape[0]} total pairs")

if __name__ == "__main__":
    process_wavcaps()
