#!/usr/bin/env python3
"""Export all TorchScript models to ONNX format for ARM/ONNX Runtime deployment.

Exports:
  - DINOv2 (88MB FP32 → ~22MB INT8)
  - Whisper encoder (82MB → ~21MB INT8)
  - MiniLM text encoder (87MB → ~22MB INT8)
  - World model (8MB → ~2MB INT8)
  - Confidence predictor (0.5MB → ~0.2MB INT8)
  - Temporal predictor (18MB → ~5MB INT8)

Output: /opt/brain/outputs/cortex/onnx/ directory
"""
import torch
import os
from pathlib import Path

PROJECT_ROOT = Path("/opt/brain")
ONNX_DIR = PROJECT_ROOT / "outputs/cortex/onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

def export_torchscript_to_onnx(ts_path, onnx_path, input_shapes, input_names, output_names):
    """Export a TorchScript model to ONNX."""
    print(f"  Loading {ts_path}...")
    model = torch.jit.load(str(ts_path), map_location="cpu")
    model.eval()

    # Create dummy inputs
    dummy_inputs = []
    for shape in input_shapes:
        if isinstance(shape[0], int):
            dummy_inputs.append(torch.randn(*shape))
        else:
            dummy_inputs.append(torch.zeros(*shape, dtype=torch.long))

    dummy_inputs = tuple(dummy_inputs)

    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        model, dummy_inputs, str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ✓ Exported: {size_mb:.1f} MB")
    return onnx_path

def quantize_onnx(onnx_path, quantized_path):
    """Quantize ONNX model to INT8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
        size_mb = os.path.getsize(quantized_path) / 1024 / 1024
        print(f"  ✓ Quantized: {size_mb:.1f} MB")
    except ImportError:
        print("  ⚠ onnxruntime not installed, skipping quantization")
    except Exception as e:
        print(f"  ⚠ Quantization failed: {e}")

def main():
    print("═══ EXPORT ALL MODELS TO ONNX ═══\n")

    models = [
        {
            "name": "DINOv2",
            "ts_path": PROJECT_ROOT / "outputs/cortex/visual_encoder/dinov2_ts.pt",
            "input_shapes": [(1, 3, 224, 224)],
            "input_names": ["image"],
            "output_names": ["embedding"],
        },
        {
            "name": "Whisper Encoder",
            "ts_path": PROJECT_ROOT / "outputs/cortex/audio_encoder/whisper_encoder_ts.pt",
            "input_shapes": [(1, 80, 3000)],
            "input_names": ["mel"],
            "output_names": ["embedding"],
        },
        {
            "name": "MiniLM Text Encoder",
            "ts_path": PROJECT_ROOT / "outputs/cortex/text_encoder/minilm_ts.pt",
            "input_shapes": [([1, 128],), ([1, 128],)],  # input_ids, attention_mask (long tensors)
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["embedding"],
        },
        {
            "name": "World Model",
            "ts_path": PROJECT_ROOT / "outputs/cortex/world_model/predictor_v2_ts.pt",
            "input_shapes": [(1, 512)],
            "input_names": ["visual_embedding"],
            "output_names": ["predicted_audio"],
        },
        {
            "name": "Confidence Predictor",
            "ts_path": PROJECT_ROOT / "outputs/cortex/self_model/confidence_predictor_ts.pt",
            "input_shapes": [(1, 512)],
            "input_names": ["embedding"],
            "output_names": ["confidence"],
        },
        {
            "name": "Temporal Predictor",
            "ts_path": PROJECT_ROOT / "outputs/cortex/temporal_model/model_ts.pt",
            "input_shapes": [(1, 3, 512)],  # sequence of 3 embeddings
            "input_names": ["sequence"],
            "output_names": ["next_embedding"],
        },
    ]

    for m in models:
        name = m["name"]
        ts_path = m["ts_path"]
        if not ts_path.exists():
            print(f"[{name}] ✗ TorchScript not found: {ts_path}")
            continue

        print(f"[{name}]")
        onnx_path = ONNX_DIR / f"{ts_path.stem}.onnx"
        try:
            # Handle MiniLM special case (long tensor inputs)
            if name == "MiniLM Text Encoder":
                model = torch.jit.load(str(ts_path), map_location="cpu")
                model.eval()
                dummy_ids = torch.zeros(1, 128, dtype=torch.long)
                dummy_mask = torch.ones(1, 128, dtype=torch.long)
                torch.onnx.export(
                    model, (dummy_ids, dummy_mask), str(onnx_path),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["embedding"],
                    opset_version=17, do_constant_folding=True,
                )
                size_mb = os.path.getsize(onnx_path) / 1024 / 1024
                print(f"  ✓ Exported: {size_mb:.1f} MB")
            else:
                export_torchscript_to_onnx(
                    ts_path, onnx_path,
                    m["input_shapes"], m["input_names"], m["output_names"]
                )

            # Quantize
            q_path = ONNX_DIR / f"{ts_path.stem}_int8.onnx"
            quantize_onnx(onnx_path, q_path)

        except Exception as e:
            print(f"  ✗ Failed: {e}")

        print()

    # Summary
    print("═══ SUMMARY ═══")
    total_fp32 = 0
    total_int8 = 0
    for f in sorted(ONNX_DIR.glob("*.onnx")):
        size = f.stat().st_size / 1024 / 1024
        tag = "INT8" if "int8" in f.name else "FP32"
        print(f"  {f.name:40s} {size:6.1f} MB  ({tag})")
        if "int8" in f.name:
            total_int8 += size
        else:
            total_fp32 += size
    print(f"\n  Total FP32: {total_fp32:.0f} MB")
    print(f"  Total INT8: {total_int8:.0f} MB")
    print(f"  Savings: {(1 - total_int8/max(total_fp32,1))*100:.0f}%")

if __name__ == "__main__":
    main()
