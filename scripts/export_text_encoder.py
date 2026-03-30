#!/usr/bin/env python3
"""Export MiniLM text encoder to TorchScript for Rust tch-rs loading."""
import torch
import torch.nn.functional as F
import os

os.makedirs("outputs/cortex/text_encoder", exist_ok=True)

print("Loading MiniLM...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

hf_model = model[0].auto_model
tokenizer = model[0].tokenizer

class MiniLMWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

wrapper = MiniLMWrapper(hf_model)
wrapper.eval()

encoded = tokenizer("test text", return_tensors="pt", padding=True, truncation=True, max_length=128)
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (encoded['input_ids'], encoded['attention_mask']), check_trace=False)

traced.save("outputs/cortex/text_encoder/minilm_ts.pt")
print(f"Model: {os.path.getsize('outputs/cortex/text_encoder/minilm_ts.pt') // 1024}KB")

tokenizer.save_pretrained("outputs/cortex/text_encoder/tokenizer")
print("Tokenizer saved")

# Verify
loaded = torch.jit.load("outputs/cortex/text_encoder/minilm_ts.pt")
with torch.no_grad():
    ref = wrapper(encoded['input_ids'], encoded['attention_mask'])
    test = loaded(encoded['input_ids'], encoded['attention_mask'])
    diff = (ref - test).abs().max().item()
print(f"Verification: max diff = {diff:.8f}")
