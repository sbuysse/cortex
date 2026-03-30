#!/usr/bin/env python3
"""Download FSD50K dataset and encode to embeddings for Rust brain ingestion.

FSD50K: 51K Freesound clips with AudioSet-ontology labels.
Outputs: fsd50k/audio_embs.npy (N×512), fsd50k/labels.json (N labels)

Downloads from Zenodo, processes in batches, deletes raw audio.
"""
import os, json, csv, subprocess, sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/fsd50k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("/tmp/fsd50k_dl")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Zenodo URLs for FSD50K
ZENODO_BASE = "https://zenodo.org/record/4060432/files"
FILES = {
    "dev_audio": f"{ZENODO_BASE}/FSD50K.dev_audio.zip",
    "eval_audio": f"{ZENODO_BASE}/FSD50K.eval_audio.zip",
    "ground_truth": f"{ZENODO_BASE}/FSD50K.ground_truth.zip",
}

def download_and_extract():
    """Download FSD50K files from Zenodo."""
    for name, url in FILES.items():
        zip_path = TMP_DIR / f"{name}.zip"
        if not zip_path.exists():
            print(f"Downloading {name}...")
            subprocess.run(["wget", "-q", "--show-progress", "-O", str(zip_path), url], check=True)
        # Extract
        extract_dir = TMP_DIR / name
        if not extract_dir.exists():
            print(f"Extracting {name}...")
            subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(TMP_DIR)], check=True)

def load_ground_truth():
    """Load FSD50K ground truth labels."""
    gt_dir = TMP_DIR / "FSD50K.ground_truth"
    labels = {}
    for split in ["dev", "eval"]:
        csv_path = gt_dir / f"{split}.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        fname = row[0]
                        tags = row[1].split(",")
                        labels[fname] = tags[0].strip()  # primary label
    return labels

def encode_audio_file(wav_path):
    """Encode audio to 512-dim via Whisper."""
    try:
        # Convert to 16kHz mono WAV via ffmpeg
        tmp_wav = str(wav_path) + ".16k.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-ar", "16000", "-ac", "1",
                       "-f", "wav", tmp_wav], capture_output=True, timeout=10)

        if not os.path.exists(tmp_wav):
            return None

        import torchaudio, torch
        waveform, sr = torchaudio.load(tmp_wav)
        waveform = waveform.mean(0)[:16000*30]  # mono, 30s max
        os.unlink(tmp_wav)

        from transformers import WhisperProcessor, WhisperModel
        global whisper_processor, whisper_model
        inputs = whisper_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            enc = whisper_model.encoder(inputs.input_features)
            emb = enc.last_hidden_state.mean(dim=1).squeeze()
        emb = emb / emb.norm()
        return emb.numpy().astype(np.float32)
    except:
        return None

def encode_text(text):
    """Encode text to 384-dim via MiniLM."""
    import torch
    global text_tokenizer, text_model
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        output = text_model(**inputs)
        emb = output.last_hidden_state.mean(dim=1).squeeze()
    emb = emb / emb.norm()
    return emb.numpy().astype(np.float32)

def process_fsd50k():
    print("Loading models...")
    import torch
    from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, AutoModel
    global whisper_processor, whisper_model, text_tokenizer, text_model
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-base").eval()
    text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

    download_and_extract()
    gt = load_ground_truth()
    print(f"Ground truth: {len(gt)} clips")

    audio_embs = []
    labels = []
    failed = 0

    # Process dev + eval audio directories
    for audio_dir_name in ["FSD50K.dev_audio", "FSD50K.eval_audio"]:
        audio_dir = TMP_DIR / audio_dir_name
        if not audio_dir.exists():
            print(f"  {audio_dir} not found, skipping")
            continue

        files = sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac"))
        print(f"Processing {len(files)} files from {audio_dir_name}...")

        for i, fpath in enumerate(files):
            fname = fpath.stem
            label = gt.get(fname, "unknown")
            if label == "unknown":
                continue

            emb = encode_audio_file(fpath)
            if emb is not None and len(emb) == 512:
                audio_embs.append(emb)
                labels.append(label)
            else:
                failed += 1

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(files)}] encoded={len(audio_embs)}, failed={failed}")

            # Delete processed file to save space
            fpath.unlink(missing_ok=True)

    # Save
    if audio_embs:
        a = np.array(audio_embs, dtype=np.float32)
        np.save(OUTPUT_DIR / "audio_embs.npy", a)
        with open(OUTPUT_DIR / "labels.json", "w") as f:
            json.dump(labels, f)
        print(f"\nDone: {len(audio_embs)} pairs, {failed} failed → {OUTPUT_DIR}")

    # Cleanup
    import shutil
    shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    process_fsd50k()
