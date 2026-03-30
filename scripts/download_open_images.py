#!/usr/bin/env python3
"""Download Open Images V7 validation images + encode via DINOv2/CLIP.

Stream-process-delete: downloads images, encodes, deletes immediately.
Outputs: open_images/dino_embs.npy (N×384), open_images/labels.json
         open_images/audio_embs.npy (alias for dino_embs for Rust ingest compat)

Targets 600 boxable classes × ~70 images each ≈ 42K images.
"""
import os, csv, json, sys, time, io, urllib.request, urllib.parse
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/opt/brain")
OUTPUT_DIR = PROJECT_ROOT / "outputs/cortex/open_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("/tmp/open_images_dl")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# URLs for Open Images V7 metadata
METADATA = {
    "image_urls": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "image_labels": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
    "class_names": str(OUTPUT_DIR / "class_descriptions.csv"),  # already downloaded
}

def download_metadata():
    """Download image URLs and label CSVs."""
    for name, url in METADATA.items():
        if url.startswith("http"):
            out = TMP_DIR / f"{name}.csv"
            if not out.exists():
                print(f"Downloading {name}...")
                urllib.request.urlretrieve(url, str(out))
                print(f"  → {out} ({out.stat().st_size // 1024}KB)")
            else:
                print(f"  {name}: already exists")

def load_metadata():
    """Build mapping: image_id → (url, [label_names])."""
    # Load class MID → name
    mid_to_name = {}
    with open(OUTPUT_DIR / "class_descriptions.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].startswith("/m/"):
                mid_to_name[row[0]] = row[1]

    # Load image labels (only confidence=1, positive labels)
    image_labels = defaultdict(list)
    labels_path = TMP_DIR / "image_labels.csv"
    if labels_path.exists():
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row.get("Confidence", 0)) >= 1.0:
                    mid = row.get("LabelName", "")
                    if mid in mid_to_name:
                        image_labels[row["ImageID"]].append(mid_to_name[mid])

    # Load image URLs
    image_urls = {}
    urls_path = TMP_DIR / "image_urls.csv"
    if urls_path.exists():
        with open(urls_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = row.get("ImageID", "")
                url = row.get("OriginalURL", "")
                if iid and url and iid in image_labels:
                    image_urls[iid] = url

    print(f"Metadata: {len(mid_to_name)} classes, {len(image_labels)} labeled images, {len(image_urls)} with URLs")
    return image_urls, image_labels, mid_to_name

def load_models():
    """Load DINOv2 + text encoder."""
    import torch
    from torchvision import transforms

    print("Loading DINOv2...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True).eval()

    print("Loading MiniLM text encoder...")
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return dino, tok, text_model, transform

def encode_image_batch(dino, images, transform):
    """Encode a batch of PIL images via DINOv2 → 384-dim."""
    import torch
    tensors = torch.stack([transform(img.convert("RGB")) for img in images])
    with torch.no_grad():
        features = dino(tensors)
    # L2 normalize
    features = features / features.norm(dim=1, keepdim=True)
    return features.numpy().astype(np.float32)

def encode_text_batch(tok, model, texts):
    """Encode batch of texts via MiniLM → 384-dim."""
    import torch
    inputs = tok(texts, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state.mean(1)
    out = out / out.norm(dim=1, keepdim=True)
    return out.numpy().astype(np.float32)

def download_image(image_id, fallback_url=None, timeout=10):
    """Download image from S3 mirror (reliable), fallback to original URL."""
    from PIL import Image
    # Primary: CVDF S3 mirror (works reliably)
    s3_url = f"https://s3.amazonaws.com/open-images-dataset/validation/{image_id}.jpg"
    for url in [s3_url, fallback_url]:
        if not url: continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BrainCortex/1.0"})
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = resp.read()
            img = Image.open(io.BytesIO(data))
            return img
        except:
            continue
    return None

def process_open_images():
    download_metadata()
    image_urls, image_labels, mid_to_name = load_metadata()

    if not image_urls:
        print("ERROR: No image URLs loaded")
        return

    dino, tok, text_model, transform = load_models()

    # Group images by primary label for balanced sampling
    label_to_images = defaultdict(list)
    for iid, labels in image_labels.items():
        if iid in image_urls:
            label_to_images[labels[0]].append(iid)

    print(f"\nLabels with images: {len(label_to_images)}")

    # Sample up to 100 images per class, prioritize classes with fewer images
    MAX_PER_CLASS = 100
    selected = []
    for label, iids in sorted(label_to_images.items()):
        for iid in iids[:MAX_PER_CLASS]:
            selected.append((iid, label, image_urls[iid]))

    print(f"Selected: {len(selected)} images across {len(label_to_images)} classes")

    # Process in batches
    from PIL import Image
    BATCH = 32
    all_dino_embs = []
    all_text_embs = []
    all_labels = []
    failed = 0

    batch_images = []
    batch_labels = []

    for i, (iid, label, url) in enumerate(selected):
        img = download_image(iid, url)
        if img is not None and img.size[0] > 10 and img.size[1] > 10:
            batch_images.append(img)
            batch_labels.append(label)
        else:
            failed += 1

        # Process batch
        if len(batch_images) >= BATCH or (i == len(selected) - 1 and batch_images):
            try:
                dino_embs = encode_image_batch(dino, batch_images, transform)
                text_embs = encode_text_batch(tok, text_model, batch_labels)
                all_dino_embs.append(dino_embs)
                all_text_embs.append(text_embs)
                all_labels.extend(batch_labels)
            except Exception as e:
                failed += len(batch_images)
                print(f"  Batch encode error: {e}")

            batch_images = []
            batch_labels = []

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(selected)}] encoded={len(all_labels)}, failed={failed}")

        # Checkpoint every 5000
        if len(all_labels) > 0 and len(all_labels) % 5000 < BATCH:
            save_checkpoint(all_dino_embs, all_text_embs, all_labels)

    # Final save
    if all_dino_embs:
        save_checkpoint(all_dino_embs, all_text_embs, all_labels)

    print(f"\nDone: {len(all_labels)} images encoded, {failed} failed")

def save_checkpoint(dino_embs_list, text_embs_list, labels):
    dino = np.concatenate(dino_embs_list, axis=0)
    text = np.concatenate(text_embs_list, axis=0)

    # Save DINOv2 embeddings as "audio_embs" for Rust ingest compatibility
    # (The Rust ingest handler expects audio_embs.npy + labels.json)
    # We pad 384→512 to match MLP W_a input dimension
    padded = np.zeros((dino.shape[0], 512), dtype=np.float32)
    padded[:, :384] = dino
    np.save(OUTPUT_DIR / "audio_embs.npy", padded)

    # Also save raw DINOv2 at native 384-dim
    np.save(OUTPUT_DIR / "dino_embs.npy", dino)
    np.save(OUTPUT_DIR / "text_embs.npy", text)
    json.dump(labels, open(OUTPUT_DIR / "labels.json", "w"))

    print(f"  Checkpoint: {len(labels)} pairs → {OUTPUT_DIR}")

if __name__ == "__main__":
    process_open_images()
