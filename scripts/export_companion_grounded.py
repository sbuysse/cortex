"""Export HOPE companion model to TorchScript with generate_grounded() method.

Loads from training checkpoint (best.pt), attaches BrainProjection weights
(loaded from grounded checkpoint if available, else fresh near-zero init),
and exports as TorchScript.

Usage:
  python export_companion_grounded.py \
    --base-checkpoint outputs/cortex/hope_companion/best.pt \
    --grounded-checkpoint outputs/cortex/hope_companion/grounded_best.pt  # optional
    --out outputs/cortex/hope_companion/hope_companion_ts.pt
"""

import argparse
import sys
from pathlib import Path

import torch

# Import HOPE from training script (same file, same class definition)
sys.path.insert(0, str(Path(__file__).parent))
from train_hope_companion import HOPE, COMPANION_NANO


def export(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model = HOPE(**COMPANION_NANO).to(device)

    # Load base weights (strict=False: brain_proj not in base checkpoint)
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(base_ckpt["model_state_dict"], strict=False)
    print(f"Base checkpoint loaded. Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    # Load grounded weights if available (overwrites BrainProjection + FFN weights)
    if args.grounded_checkpoint and Path(args.grounded_checkpoint).exists():
        grounded_ckpt = torch.load(args.grounded_checkpoint, map_location="cpu")
        missing2, _ = model.load_state_dict(grounded_ckpt["model_state_dict"], strict=False)
        print(f"Grounded checkpoint loaded. Missing: {missing2}")
    else:
        print("No grounded checkpoint — using near-zero BrainProjection (text-only fallback)")

    model.eval()

    # Verify both methods exist before scripting
    assert hasattr(model, "generate"), "generate() missing"
    assert hasattr(model, "generate_grounded"), "generate_grounded() missing"

    # Quick sanity: generate_grounded with zero brain_vec == generate (approx)
    prompt = list(b"[CTX] Hello. [USR] Hi. [CRT] ")
    brain_zeros = [0] * 512
    with torch.no_grad():
        out_base = model.generate(prompt, 10)
        out_grounded = model.generate_grounded(brain_zeros, prompt, 10)
    print(f"generate():          {bytes(out_base[:20])!r}")
    print(f"generate_grounded(): {bytes(out_grounded[:20])!r}")

    # TorchScript export
    scripted = torch.jit.script(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"\nExported TorchScript model → {out_path}")

    # Verify exported model has both methods
    loaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        v1 = loaded.generate(prompt, 5)
        v2 = loaded.generate_grounded(brain_zeros, prompt, 5)
    print(f"Verification generate():          {bytes(v1)!r}")
    print(f"Verification generate_grounded(): {bytes(v2)!r}")
    print("Export OK.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint",
                        default="outputs/cortex/hope_companion/best.pt")
    parser.add_argument("--grounded-checkpoint", default=None)
    parser.add_argument("--out",
                        default="outputs/cortex/hope_companion/hope_companion_ts.pt")
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
