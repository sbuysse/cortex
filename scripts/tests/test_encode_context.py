import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

# Tests that require model files (skip if not present)
MINILM_PATH = "outputs/cortex/text_encoder/minilm_ts.pt"
TOKENIZER_PATH = "outputs/cortex/text_encoder/tokenizer"
MLP_V_PATH = "outputs/cortex/v5_mlp/w_v.bin"
models_available = os.path.exists(MINILM_PATH)

from build_companion_vocab import tokenize  # reuse tokenizer logic for text check


def test_load_matrix_shape():
    """Test the binary matrix loader independently (no model needed)."""
    import struct, tempfile, numpy as np
    from encode_context import _load_matrix

    # Write a tiny 2x3 matrix in Rust binary format
    rows, cols = 2, 3
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(struct.pack("<II", rows, cols))
        f.write(struct.pack(f"<{rows*cols}f", *data))
        path = f.name

    mat = _load_matrix(path)
    assert mat.shape == (2, 3)
    assert abs(float(mat[0, 0]) - 1.0) < 1e-5
    os.unlink(path)


@pytest.mark.skipif(not models_available, reason="Model files not available locally")
def test_encoder_output_shape():
    from encode_context import ContextEncoder
    enc = ContextEncoder(MINILM_PATH, TOKENIZER_PATH, MLP_V_PATH)
    vec = enc.encode("Name: Marguerite, Age: 84. Likes: gardening.")
    assert vec.shape == (512,)


@pytest.mark.skipif(not models_available, reason="Model files not available locally")
def test_encoder_output_normalized():
    from encode_context import ContextEncoder
    enc = ContextEncoder(MINILM_PATH, TOKENIZER_PATH, MLP_V_PATH)
    vec = enc.encode("Hello, my name is Louise.")
    norm = float(torch.norm(vec))
    assert 0.95 < norm < 1.05


@pytest.mark.skipif(not models_available, reason="Model files not available locally")
def test_different_contexts_give_different_vectors():
    from encode_context import ContextEncoder
    enc = ContextEncoder(MINILM_PATH, TOKENIZER_PATH, MLP_V_PATH)
    v1 = enc.encode("Name: Marguerite, Age: 84. Likes: gardening.")
    v2 = enc.encode("Name: Georges, Age: 78. Health: back pain.")
    sim = float(torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))
    assert sim < 0.99
