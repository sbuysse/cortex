import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_companion_decoder import CompanionDecoder

VOCAB_SIZE = 5000
MAX_TOKENS = 40

def test_decoder_forward_shape():
    model = CompanionDecoder(vocab_size=VOCAB_SIZE, max_tokens=MAX_TOKENS)
    brain_emb = torch.randn(2, 512)
    ctx_emb   = torch.randn(2, 512)
    tokens    = torch.zeros(2, MAX_TOKENS, dtype=torch.long)
    logits    = model(brain_emb, ctx_emb, tokens)
    assert logits.shape == (2, MAX_TOKENS, VOCAB_SIZE), f"Got {logits.shape}"

def test_decoder_generate_returns_tokens():
    model = CompanionDecoder(vocab_size=VOCAB_SIZE, max_tokens=MAX_TOKENS)
    model.eval()
    brain_emb = torch.randn(1, 512)
    ctx_emb   = torch.randn(1, 512)
    tokens = model.generate(brain_emb, ctx_emb, bos_id=1, eos_id=2, max_new=20)
    assert isinstance(tokens, list)
    assert len(tokens) <= 20

def test_decoder_torchscript_export():
    model = CompanionDecoder(vocab_size=VOCAB_SIZE, max_tokens=MAX_TOKENS)
    model.eval()
    # TorchScript: Tensor args are fine; no Optional, no dict, all primitive types
    scripted = torch.jit.script(model)
    brain_emb = torch.randn(1, 512)
    ctx_emb   = torch.randn(1, 512)
    tokens    = torch.zeros(1, 5, dtype=torch.long)
    out = scripted(brain_emb, ctx_emb, tokens)
    assert out.shape == (1, 5, VOCAB_SIZE)
    # Verify generate() is also scriptable (uses List[int] return, all primitives)
    ids = scripted.generate(brain_emb, ctx_emb, bos_id=1, eos_id=2, max_new=5)
    assert isinstance(ids, list)
