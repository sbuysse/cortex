"""Cortex Brain configuration — all constants, paths, and env vars."""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(os.environ.get("BRAIN_PROJECT_ROOT", "/opt/brain"))
EMBED_CACHE = PROJECT_ROOT / "data/vggsound/.embed_cache/expanded_embeddings.safetensors"
VGGSOUND_CSV = PROJECT_ROOT / "data/vggsound/vggsound.csv"
MODEL_DIR = PROJECT_ROOT / "outputs/cortex/v6_mlp"
MEMORY_DB_PATH = PROJECT_ROOT / "outputs/cortex/brain_memory.db"
AUDIOSET_DIR = PROJECT_ROOT / "outputs/cortex/audioset_brain"
WORLD_MODEL_DIR = PROJECT_ROOT / "outputs/cortex/world_model"
CAUSAL_DIR = PROJECT_ROOT / "outputs/cortex/causal_graph"
SELF_MODEL_DIR = PROJECT_ROOT / "outputs/cortex/self_model"
HIERARCHY_PATH = PROJECT_ROOT / "outputs/cortex/concept_hierarchy.json"
TEMPORAL_MODEL_PATH = PROJECT_ROOT / "outputs/cortex/temporal_model/model.pt"
BRAIN_DECODER_DIR = PROJECT_ROOT / "outputs/cortex/brain_decoder"
TTS_MODEL_DIR = PROJECT_ROOT / "data/tts"

# LLM
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

# Working Memory
WORKING_MEMORY_SLOTS = int(os.environ.get("WM_SLOTS", "7"))
WORKING_MEMORY_DECAY = float(os.environ.get("WM_DECAY", "0.85"))
WM_THETA_FREQ = float(os.environ.get("WM_THETA_FREQ", "0.15"))

# Neuroscience features
SPARSE_K = int(os.environ.get("SPARSE_K", "0"))
ACH_WINDOW = int(os.environ.get("ACH_WINDOW", "50"))
ACH_LR_MIN = float(os.environ.get("ACH_LR_MIN", "0.0002"))
ACH_LR_MAX = float(os.environ.get("ACH_LR_MAX", "0.005"))
FAST_MEMORY_CAPACITY = int(os.environ.get("FAST_MEMORY_CAPACITY", "2000"))
EPISODE_GAP_SECONDS = float(os.environ.get("EPISODE_GAP_SECONDS", "30.0"))

# Grid cells
GRID_SCALES = [0.05, 0.15, 0.5]
GRID_N_ORIENTATIONS = 3

# Autonomy
AUTONOMY_INTERVAL = int(os.environ.get("AUTONOMY_INTERVAL", "300"))

# API
API_KEYS = set(os.environ.get("CORTEX_API_KEYS", "").split(",")) - {""}
DEVICE = "cpu"
