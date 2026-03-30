#!/usr/bin/env python3
"""Export DINOv2, Whisper-base encoder, and CLIP ViT-B/32 to TorchScript."""

import os
import torch
import torch.nn as nn

BASE = "/opt/brain"


def fmt_size(path):
    sz = os.path.getsize(path)
    if sz > 1e9:
        return f"{sz/1e9:.2f} GB"
    return f"{sz/1e6:.1f} MB"


# ── 1. DINOv2 vits14 ──────────────────────────────────────────────────────────

print("=" * 60)
print("1. Exporting DINOv2 vits14 ...")
print("=" * 60)


class DINOv2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dinov2 forward returns cls token embedding (batch, 384)
        return self.model(x)


dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino.eval()
wrapper_dino = DINOv2Wrapper(dino)
wrapper_dino.eval()

dummy_img = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    ref_dino = wrapper_dino(dummy_img)
    print(f"  Output shape: {ref_dino.shape}")  # (1, 384)

ts_dino = torch.jit.trace(wrapper_dino, dummy_img, check_trace=False)
dino_path = os.path.join(BASE, "outputs/cortex/visual_encoder/dinov2_ts.pt")
ts_dino.save(dino_path)

# Verify
loaded_dino = torch.jit.load(dino_path)
with torch.no_grad():
    out = loaded_dino(dummy_img)
    diff = (out - ref_dino).abs().max().item()
    print(f"  Verification max diff: {diff:.2e}")
    assert diff < 1e-4, f"DINOv2 verification failed: {diff}"
print(f"  Saved to: {dino_path} ({fmt_size(dino_path)})")

# Free memory
del dino, wrapper_dino, ts_dino, loaded_dino, ref_dino
torch.cuda.empty_cache() if torch.cuda.is_available() else None
import gc; gc.collect()


# ── 2. Whisper-base encoder ───────────────────────────────────────────────────

print()
print("=" * 60)
print("2. Exporting Whisper-base encoder ...")
print("=" * 60)

from transformers import WhisperModel


class WhisperEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features: (batch, 80, 3000)
        out = self.encoder(input_features).last_hidden_state  # (batch, seq, 512)
        return out.mean(dim=1)  # (batch, 512)


whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
whisper_model.eval()
encoder_wrapper = WhisperEncoderWrapper(whisper_model.encoder)
encoder_wrapper.eval()

dummy_mel = torch.randn(1, 80, 3000)

with torch.no_grad():
    ref_whisper = encoder_wrapper(dummy_mel)
    print(f"  Output shape: {ref_whisper.shape}")  # (1, 512)

ts_whisper = torch.jit.trace(encoder_wrapper, dummy_mel, check_trace=False)
whisper_path = os.path.join(BASE, "outputs/cortex/audio_encoder/whisper_encoder_ts.pt")
ts_whisper.save(whisper_path)

# Verify
loaded_whisper = torch.jit.load(whisper_path)
with torch.no_grad():
    out = loaded_whisper(dummy_mel)
    diff = (out - ref_whisper).abs().max().item()
    print(f"  Verification max diff: {diff:.2e}")
    assert diff < 1e-4, f"Whisper verification failed: {diff}"
print(f"  Saved to: {whisper_path} ({fmt_size(whisper_path)})")

del whisper_model, encoder_wrapper, ts_whisper, loaded_whisper, ref_whisper
gc.collect()


# ── 3. CLIP ViT-B/32 image encoder ──────────────────────────────────────────

print()
print("=" * 60)
print("3. Exporting CLIP ViT-B/32 image encoder ...")
print("=" * 60)

from transformers import CLIPModel


class CLIPImageEncoderWrapper(nn.Module):
    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (batch, 3, 224, 224)
        vision_out = self.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output  # (batch, 768)
        projected = self.visual_projection(pooled)  # (batch, 512)
        return projected


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

clip_wrapper = CLIPImageEncoderWrapper(
    clip_model.vision_model, clip_model.visual_projection
)
clip_wrapper.eval()

dummy_pixel = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    ref_clip = clip_wrapper(dummy_pixel)
    print(f"  Output shape: {ref_clip.shape}")  # (1, 512)

ts_clip = torch.jit.trace(clip_wrapper, dummy_pixel, check_trace=False)
clip_path = os.path.join(BASE, "outputs/cortex/visual_encoder/clip_ts.pt")
ts_clip.save(clip_path)

# Verify
loaded_clip = torch.jit.load(clip_path)
with torch.no_grad():
    out = loaded_clip(dummy_pixel)
    diff = (out - ref_clip).abs().max().item()
    print(f"  Verification max diff: {diff:.2e}")
    assert diff < 1e-4, f"CLIP verification failed: {diff}"
print(f"  Saved to: {clip_path} ({fmt_size(clip_path)})")

del clip_model, clip_wrapper, ts_clip, loaded_clip, ref_clip
gc.collect()


print()
print("=" * 60)
print("All 3 models exported successfully!")
print("=" * 60)
