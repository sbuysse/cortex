#!/usr/bin/env python3
"""Targeted FreeSound download for brain's weakest categories.

Queries the brain's curiosity endpoint to find weak categories,
then downloads targeted audio clips from FreeSound API.
Outputs: freesound/audio_embs.npy (N×512), freesound/labels.json

Requires: FREESOUND_API_KEY environment variable (free at freesound.org)
Without API key, uses FreeSound search page scraping as fallback.
"""
import os, json, subprocess, sys, time, urllib.request, ssl
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/freesound"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("/tmp/freesound_dl")
TMP_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("FREESOUND_API_KEY", "")
CLIPS_PER_CATEGORY = 20
MAX_CATEGORIES = 50

def get_weak_categories():
    """Query brain for categories with highest curiosity (weakest performance)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.Request("https://localhost/api/brain/curiosity/distributional")
        resp = urllib.request.urlopen(req, timeout=5, context=ctx)
        data = json.loads(resp.read())
        cats = data.get("categories", [])
        # Sort by least covered
        uncovered = [c["category"] for c in cats if not c.get("covered", True)]
        return uncovered[:MAX_CATEGORIES]
    except:
        # Fallback: use a predefined list of diverse categories
        return [
            "baby crying", "birds chirping", "car horn", "church bell",
            "clock ticking", "coughing", "cow mooing", "crowd cheering",
            "dog bark", "door knock", "drilling", "engine", "fire crackling",
            "footsteps", "glass breaking", "helicopter", "keyboard typing",
            "laughing", "ocean waves", "phone ringing", "rain", "rooster",
            "siren", "snoring", "thunder", "train horn", "typing",
            "vacuum cleaner", "water dripping", "wind"
        ]

def search_freesound(query, max_results=20):
    """Search FreeSound for clips matching a query."""
    if API_KEY:
        url = f"https://freesound.org/apiv2/search/text/?query={urllib.parse.quote(query)}&page_size={max_results}&fields=id,name,duration,previews&token={API_KEY}"
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            data = json.loads(resp.read())
            results = []
            for r in data.get("results", []):
                preview_url = r.get("previews", {}).get("preview-lq-mp3", "")
                if preview_url and r.get("duration", 0) < 60:
                    results.append({"id": r["id"], "name": r["name"], "url": preview_url})
            return results
        except:
            return []
    else:
        # Without API key, we can't access FreeSound API
        # Use YouTube search as fallback
        return search_youtube_fallback(query)

def search_youtube_fallback(query):
    """Fallback: use yt-dlp to find short sound clips on YouTube."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--default-search", f"ytsearch{CLIPS_PER_CATEGORY}",
             "--print", "%(webpage_url)s\t%(title)s\t%(duration)s",
             "--match-filter", "duration<30",
             "--no-download", f"{query} sound effect short"],
            capture_output=True, text=True, timeout=20)
        results = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                results.append({"url": parts[0], "name": parts[1], "source": "youtube"})
        return results[:CLIPS_PER_CATEGORY]
    except:
        return []

def download_and_encode(entry, idx):
    """Download audio and encode via Whisper."""
    wav_path = TMP_DIR / f"clip_{idx}.wav"
    try:
        url = entry.get("url", "")
        if not url:
            return None

        if entry.get("source") == "youtube":
            # yt-dlp download
            subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "wav",
                 "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1 -t 30",
                 "-o", str(wav_path), "--no-playlist", url],
                capture_output=True, timeout=30)
        else:
            # Direct download (FreeSound preview)
            subprocess.run(["wget", "-q", "--timeout=10", "-O", str(wav_path), url],
                          capture_output=True, timeout=15)
            # Convert to 16kHz
            out = TMP_DIR / f"clip_{idx}_16k.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-ar", "16000", "-ac", "1",
                           "-t", "30", "-f", "wav", str(out)], capture_output=True, timeout=10)
            wav_path.unlink(missing_ok=True)
            wav_path = out

        if not wav_path.exists() or wav_path.stat().st_size < 1000:
            wav_path.unlink(missing_ok=True)
            return None

        import torchaudio, torch
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(0)[:16000*30]
        wav_path.unlink(missing_ok=True)

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
    import torch
    global text_tokenizer, text_model
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        output = text_model(**inputs)
        emb = output.last_hidden_state.mean(dim=1).squeeze()
    emb = emb / emb.norm()
    return emb.numpy().astype(np.float32)

def process_freesound():
    print("Loading models...")
    import torch
    from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, AutoModel
    global whisper_processor, whisper_model, text_tokenizer, text_model
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-base").eval()
    text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

    categories = get_weak_categories()
    print(f"Targeting {len(categories)} weak categories")

    audio_embs = []
    labels = []
    failed = 0

    for cat_idx, category in enumerate(categories):
        print(f"\n[{cat_idx+1}/{len(categories)}] Searching: {category}")
        results = search_freesound(category)
        if not results:
            print(f"  No results for {category}")
            continue

        for i, entry in enumerate(results):
            emb = download_and_encode(entry, cat_idx * 100 + i)
            if emb is not None and len(emb) == 512:
                audio_embs.append(emb)
                labels.append(category)
            else:
                failed += 1
            time.sleep(0.5)  # rate limit

        print(f"  Encoded {len([l for l in labels if l == category])} clips for '{category}'")

    # Save
    if audio_embs:
        a = np.array(audio_embs, dtype=np.float32)
        np.save(OUTPUT_DIR / "audio_embs.npy", a)
        with open(OUTPUT_DIR / "labels.json", "w") as f:
            json.dump(labels, f)
        print(f"\nDone: {len(audio_embs)} clips, {failed} failed → {OUTPUT_DIR}")

    import shutil
    shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    process_freesound()
