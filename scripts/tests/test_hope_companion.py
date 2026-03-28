"""Tests for HOPE Companion Nano architecture.

Tests:
1. test_hope_forward_shape     — output (2, 32, 256) for batch=2, seq=32, vocab=256
2. test_hope_generate_returns_bytes — generate returns List[int], len <= 30
3. test_hope_torchscript_export — torch.jit.script(model) succeeds, forward works
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_hope_companion import HOPE, COMPANION_NANO


def test_hope_forward_shape():
    model = HOPE(**COMPANION_NANO)
    tokens = torch.zeros(2, 32, dtype=torch.long)  # batch=2, seq=32
    logits = model(tokens)
    assert logits.shape == (2, 32, 256), f"Got {logits.shape}"
    print(f"PASS test_hope_forward_shape: shape={logits.shape}")


def test_hope_generate_returns_bytes():
    model = HOPE(**COMPANION_NANO)
    model.eval()
    prompt = list(b"[CTX] Hello [USR] How are you? [CRT] ")
    response = model.generate(prompt, max_new=30)
    assert isinstance(response, list), f"Expected list, got {type(response)}"
    assert all(isinstance(b, int) for b in response), "All elements must be int"
    assert len(response) <= 30, f"Expected len <= 30, got {len(response)}"
    print(f"PASS test_hope_generate_returns_bytes: len={len(response)}, sample={response[:5]}")


def test_hope_torchscript_export():
    model = HOPE(**COMPANION_NANO)
    model.eval()
    scripted = torch.jit.script(model)
    tokens = torch.zeros(1, 10, dtype=torch.long)
    out = scripted(tokens)
    assert out.shape == (1, 10, 256), f"Got {out.shape}"
    print(f"PASS test_hope_torchscript_export: scripted forward shape={out.shape}")


if __name__ == "__main__":
    test_hope_forward_shape()
    test_hope_generate_returns_bytes()
    test_hope_torchscript_export()
    print("\nAll tests passed.")
