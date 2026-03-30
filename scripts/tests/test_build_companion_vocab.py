import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from build_companion_vocab import build_vocab, encode, decode

def test_special_tokens_always_present():
    vocab = build_vocab([])
    assert vocab["<pad>"] == 0
    assert vocab["<bos>"] == 1
    assert vocab["<eos>"] == 2
    assert vocab["<unk>"] == 3

def test_common_words_in_vocab():
    responses = ["Hello Marguerite!", "I am here for you.", "That sounds lovely."]
    vocab = build_vocab(responses, min_freq=1)
    assert "marguerite" in vocab
    assert "lovely" in vocab

def test_encode_decode_roundtrip():
    vocab = build_vocab(["hello dear friend how are you"], min_freq=1)
    ids = encode("hello dear friend", vocab, max_len=10)
    text = decode(ids, {v: k for k, v in vocab.items()})
    assert "hello" in text
    assert "dear" in text

def test_unknown_word_maps_to_unk():
    vocab = build_vocab(["hello world"], min_freq=1)
    ids = encode("xyzzyx", vocab, max_len=5)
    assert 3 in ids  # <unk>
