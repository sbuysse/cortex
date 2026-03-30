#!/usr/bin/env python3
"""Export all PyTorch models to TorchScript format for Rust tch-rs loading."""
import torch
import torch.nn as nn
import json
import os

print("=== Exporting models to TorchScript ===\n")

# 1. World model v2: 512→1024→1024→512 + residual
class WorldModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512))
    def forward(self, x):
        return self.net(x) + x

state = torch.load('outputs/cortex/world_model/predictor_v2.pt', map_location='cpu', weights_only=True)
wm = WorldModelV2()
wm.load_state_dict(state)
wm.eval()
torch.jit.script(wm).save('outputs/cortex/world_model/predictor_v2_ts.pt')
print("1. World model: OK")

# 2. Confidence predictor: 512→128→1
class ConfPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)

state = torch.load('outputs/cortex/self_model/confidence_predictor.pt', map_location='cpu', weights_only=True)
cp = ConfPred()
cp.load_state_dict(state)
cp.eval()
torch.jit.script(cp).save('outputs/cortex/self_model/confidence_predictor_ts.pt')
print("2. Confidence predictor: OK")

# 3. Temporal model: transformer encoder, 2 layers, 4 heads
class TemporalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Embedding(8, 512)
        layer = nn.TransformerEncoderLayer(512, 4, 1024, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.output_proj = nn.Linear(512, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        out = self.transformer(x, mask=mask)
        return self.output_proj(out[:, -1, :])

state = torch.load('outputs/cortex/temporal_model/model.pt', map_location='cpu', weights_only=True)
tm = TemporalPredictor()
tm.load_state_dict(state)
tm.eval()
with torch.no_grad():
    torch.jit.trace(tm, torch.randn(1, 3, 512), check_trace=False).save('outputs/cortex/temporal_model/model_ts.pt')
print("3. Temporal model: OK")

# 4. Brain decoder: transformer decoder, 2 layers, 957 vocab
with open('outputs/cortex/brain_decoder/vocab.json') as f:
    vocab = json.load(f)

class BrainDecoder(nn.Module):
    def __init__(self, vs: int):
        super().__init__()
        self.embed = nn.Linear(512, 512)
        layer = nn.TransformerDecoderLayer(512, 4, 1024, 0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.output = nn.Linear(512, vs)
        self.token_embed = nn.Embedding(vs, 512)
        self.pos_embed = nn.Embedding(34, 512)

    def forward(self, brain_emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        memory = self.embed(brain_emb).unsqueeze(1)
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, -1)
        tgt = self.token_embed(tokens) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(S, S, device=tokens.device), diagonal=1).bool()
        return self.output(self.decoder(tgt, memory, tgt_mask=mask))

state = torch.load('outputs/cortex/brain_decoder/decoder.pt', map_location='cpu', weights_only=True)
bd = BrainDecoder(len(vocab))
bd.load_state_dict(state)
bd.eval()
with torch.no_grad():
    torch.jit.trace(bd, (torch.randn(1, 512), torch.tensor([[1, 2, 3]])), check_trace=False).save('outputs/cortex/brain_decoder/decoder_ts.pt')
print(f"4. Brain decoder: OK (vocab={len(vocab)})")

# Summary
print("\n=== TorchScript exports ===")
for p in [
    'outputs/cortex/world_model/predictor_v2_ts.pt',
    'outputs/cortex/self_model/confidence_predictor_ts.pt',
    'outputs/cortex/temporal_model/model_ts.pt',
    'outputs/cortex/brain_decoder/decoder_ts.pt',
]:
    if os.path.exists(p):
        print(f"  {p}: {os.path.getsize(p)//1024}KB")
    else:
        print(f"  {p}: MISSING")


def export_hope_companion(weights_path: str, out_path: str, config_path: str):
    """Export trained HOPE companion decoder to TorchScript."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from train_hope_companion import HOPE, COMPANION_NANO

    checkpoint = torch.load(weights_path, map_location="cpu")
    model = HOPE(**COMPANION_NANO)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(out_path)

    with open(config_path, "w") as f:
        json.dump(COMPANION_NANO, f, indent=2)

    size_kb = os.path.getsize(out_path) // 1024
    print(f"HOPE companion exported: {out_path} ({size_kb}KB)")
    print(f"Config: {config_path}")
    print(f"Val loss at export: {checkpoint.get('val_loss', 'n/a')}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "hope":
    export_hope_companion(
        weights_path="outputs/cortex/hope_companion/best.pt",
        out_path="outputs/cortex/hope_companion/hope_companion_ts.pt",
        config_path="outputs/cortex/hope_companion/hope_companion_config.json",
    )
