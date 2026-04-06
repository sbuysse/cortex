# Cortex

A cognitive architecture in Rust combining spiking neural networks with foundation model encoders. 2 million ALIF neurons, 2 billion synaptic connections, 10 brain regions, STDP-based learning.

## Overview

Cortex is a research platform for studying how spiking neural dynamics can form associative knowledge from multimodal input. It encodes perception (DINOv2, CLIP, Whisper, MiniLM) into spike trains, imprints knowledge into synaptic weights, and recalls through spreading activation across 2 billion connections — the same mechanism the brain uses.

The system watches YouTube videos, extracts knowledge triples via LLM, imprints them into actual synaptic connections between neuron assemblies, and discovers emergent cross-domain associations through neural propagation. Knowledge lives in the weights, not in a database.

### Neuroscience Alignment

Cortex is grounded in real neuroscience. See [docs/NEUROSCIENCE.md](docs/NEUROSCIENCE.md) for a point-by-point comparison with brain science. Key alignments:

| Mechanism | Biology | Cortex |
|-----------|---------|--------|
| **Neurons** | Leaky integrate-and-fire with spike-frequency adaptation | ALIF neurons (2M), SoA layout |
| **Learning** | Three-factor STDP (pre/post timing × neuromodulator) | Eligibility traces × dopamine/ACh gating |
| **Stability** | Homeostatic multiplicative synaptic scaling | Multiplicative drive scaling (preserves weight ratios) |
| **Recall** | Spreading activation through synaptic connections | Spike propagation through 2B imprinted synapses |
| **Sequences** | Theta phase precession (temporal offset → STDP) | STDP-timed chain imprinting (5-step offset) |
| **Modulation** | DA (reward), ACh (attention), NE (arousal), 5-HT (mood) | Four scalar modulators controlling learning and recall modes |
| **Concepts** | Cell assemblies (~100 co-firing neurons) | 100-neuron dedicated assemblies per concept |

## Architecture

```
                        ┌─────────────────────┐
                        │    brain-server      │
                        │   (axum, 60+ API)    │
                        └──────────┬──────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
┌─────────┴─────────┐  ┌─────────┴─────────┐  ┌──────────┴─────────┐
│  brain-cognition   │  │  brain-inference   │  │   brain-spiking    │
│                    │  │                    │  │                    │
│ Working memory     │  │ DINOv2  (384d)     │  │ 10 brain regions   │
│ Hopfield memory    │  │ CLIP    (512d)     │  │ 2M ALIF neurons    │
│ Knowledge graph    │  │ Whisper (512d)     │  │ 2B CSR synapses    │
│ Personal memory    │  │ MiniLM  (384d)     │  │ Three-factor STDP  │
│ Companion/emotion  │  │ World model        │  │ 4 neuromodulators  │
│ Autonomy loop      │  │ Mel spectrogram    │  │ Cell assemblies    │
│ Sleep consolidation│  │ VAD, faces         │  │ Triple extraction  │
└────────────────────┘  └────────────────────┘  │ Chain recall       │
                                                 │ Sleep/pruning      │
          ┌─────────────────────┐                └────────────────────┘
          │    brain-core       │
          │ Hebbian networks    │
          │ Sparse projections  │
          └─────────────────────┘
```

### Brain Regions

| Region | Neurons | Role |
|--------|---------|------|
| Visual cortex | 200K | Receives DINOv2/CLIP/MiniLM embeddings via latency coding |
| Auditory cortex | 200K | Receives Whisper audio embeddings |
| Association cortex | 500K | Cross-modal binding, cell assemblies for concepts |
| Predictive cortex | 200K | Top-down prediction, bottom-up error signals |
| Hippocampus | 300K | Fast pattern storage (DG/CA3/CA1 subfields) |
| Prefrontal cortex | 200K | Working memory attractors (NMDA-like slow decay) |
| Amygdala | 100K | Emotional valence assignment |
| Motor cortex | 100K | Action/speech output |
| Brainstem | 50K | Neuromodulator source (DA, ACh, NE, 5-HT) |
| Cerebellum | 150K | Timing and error correction |

### Learning Pipeline

```
YouTube video
  → yt-dlp auto-subtitles
  → Sentence chunking (~200 chars, filler filtering)
  → LLM-powered triple extraction (Ollama, batched):
      "From these sentences about X, extract subject|verb|object triples"
      e.g., TurboQuant|compresses|KV cache
  → Batch learning: all triples encoded in one tick (~0ms)
  → Concept association matrix: S→R, R→O, S→O edges strengthened
  → No text stored — association weights ARE the memory
```

### Recall

```
Query → Fuzzy concept matching → Find matching cell assemblies
  → BFS through concept association graph (up to 6 hops)
  → Follow strongest weighted edges, filter noise concepts
  → Chain of associated concepts returned with strength scores
  → Injected into LLM system prompt as learned knowledge
```

## Experiment: Learning TurboQuant from a YouTube Video

To validate the architecture, we taught Cortex about [TurboQuant](https://arxiv.org/abs/2504.19874) — a quantization method published after the LLM's training cutoff. The LLM alone cannot answer questions about it.

**Step 1: Ask the raw LLM (no Cortex)**
```
Q: "How does TurboQuant reduce memory usage?"
A: "I don't have specific information about TurboQuant..."
```

**Step 2: Teach Cortex by watching a YouTube video**
```bash
curl -X POST /api/brain/learn/academic \
  -d '{"query": "https://www.youtube.com/watch?v=7YVrb3-ABYE", "topic": "TurboQuant"}'
# → 50 sentences processed, 12 triples extracted via LLM, learned in 0ms
```

The LLM extracts precise triples like `TurboQuant|compresses|KV cache` and `TurboQuant|reduces|short-term memory of models` from the transcript. All 12 triples are batch-learned in a single tick (~0ms). Total learning time: ~10 seconds.

**Step 3: Ask Cortex**
```
Q: "How does TurboQuant reduce memory usage?"

Brain associations (from chain recall, strength scores):
  - kv cache (80)
  - short-term memory of models (80)
  - short-term memory of an ai assistant (80)
  - stock price (80)
  - moves (80)

A: "TurboQuant optimizes memory usage by employing techniques like
    kv cache management and efficient short-term memory handling of
    AI models, which allows it to operate more effectively without
    overloading its resources. This means it can process information
    and make predictions about stock prices or other data-intensive
    tasks with reduced memory overhead."
```

The LLM now answers using knowledge from the video — "kv cache", "short-term memory of models", and "stock price" come from the brain's learned associations, not from the LLM's training data. The spiking brain decides what to recall; the LLM turns it into language.

## Experiment 2: Cross-Domain Emergent Discovery

After teaching Cortex 24 topics (TurboQuant, FlashAttention, transformers, LoRA, GGUF, spiking networks, diffusion models, tokenization, and more), we asked a question that spans multiple domains.

**Query:** "How does TurboQuant work?"

**Brain associations (dual-pathway recall with confidence tags):**
```
[explicit] kv cache (strength: 200)           — from BFS
[explicit] short-term memory of models (160)   — from BFS
[explicit] formal mathematical proof (160)     — from BFS
[emergent] sparse (strength: 70)               — discovered by spiking propagation
[emergent] word into vector (strength: 70)     — discovered by spiking propagation
[emergent] similar words close to each other (70) — discovered by spiking propagation
```

The `[emergent]` associations were NOT learned from TurboQuant's video — they were discovered by the 2B-synapse spiking network finding lateral pathways to concepts from other topics (tokenization, embeddings). The spiking brain connected "quantization" to "sparsity" and "vector representations" through neural propagation, not text matching.

### How It Works

1. **BFS recall** (0ms): follows explicit learned edges in the HashMap association graph
2. **Spiking recall** (0.1s): fires seed concepts into 500K association cortex neurons, propagates through imprinted + random synapses for 30 steps
3. **Merge**: concepts found by both = `[confirmed]`, BFS only = `[explicit]`, spiking only = `[emergent]`
4. **Neuromodulator control**: single-topic queries use focused mode (high acetylcholine), multi-topic queries use broad mode (high norepinephrine)

### Performance

| Metric | Value |
|--------|-------|
| Topics learned | 24 (from YouTube videos) |
| Concepts | 816 |
| Associations | 1,195 |
| Persisted triples | 423 (survives restarts) |
| Triple extraction | LLM-powered (Ollama), ~12 triples per video in 9.4s |
| Learning | Batch: 12 triples in 0.000s + 803 synapses imprinted |
| BFS recall | 0.000s (instant) |
| Spiking recall | 0.1s (30 steps through association cortex) |
| Brain scale | 2M neurons, 2B synapses, 10 regions |

## UI — Immersive 3D Brain Explorer

Open `https://your-server:8443/` to access the brain explorer.

- **Full-screen 3D brain** with 10 anatomically positioned regions that glow based on spike activity (Three.js)
- **Knowledge graph** visible when zoomed in — 1000+ concept nodes colored by topic, connected by learned associations
- **Ask questions** via the unified input bar — the brain animates during recall, response appears as a floating card with confidence tags
- **Learn from YouTube** — paste a URL, the brain learns in real-time with progress animation
- **Browse knowledge** — slide-out panels for topics, brain regions, and system stats
- **Confidence visualization** — `[confirmed]` green, `[explicit]` blue, `[emergent]` purple, `[predicted]` orange

Built with Three.js, vanilla JS, and Tailwind CSS. Single HTML page, no framework.

## Installation

### Prerequisites

- Rust (edition 2024)
- [libtorch](https://pytorch.org/get-started/locally/) (PyTorch C++ library)
- Ollama (for LLM dialogue — optional)
- yt-dlp + ffmpeg (for video learning — optional)

### Build

```bash
git clone https://github.com/sbuysse/cortex.git
cd cortex/rust

# Point to your libtorch installation
export LIBTORCH=/path/to/libtorch        # e.g., /usr/local/lib64/python3.14/site-packages/torch
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

cargo build --release -p brain-server
```

### Run

```bash
# Minimal (no spiking brain, no models)
BRAIN_PROJECT_ROOT=/path/to/cortex ./target/release/brain-server

# With spiking brain (scale: 0.01=tiny test, 0.1=development, 1.0=full 2M neurons)
BRAIN_PROJECT_ROOT=/path/to/cortex SPIKING_SCALE=0.1 ./target/release/brain-server

# Disable cortex experiment runner (saves CPU for spiking brain + Ollama)
BRAIN_CORTEX_DISABLE=1 SPIKING_SCALE=0.1 BRAIN_PROJECT_ROOT=/path/to/cortex ./target/release/brain-server
```

The server starts on `https://localhost:443` (TLS with self-signed cert).

### Optional: Ollama for dialogue

```bash
# Install Ollama (https://ollama.ai)
ollama pull qwen2.5:1.5b

# Keep model loaded permanently (avoids 25s cold-start)
export OLLAMA_KEEP_ALIVE=-1
```

### Optional: Video learning

```bash
# Install yt-dlp and ffmpeg
pip install yt-dlp
# ffmpeg via your package manager

# Teach Cortex from a YouTube video
curl -sk -X POST https://localhost/api/brain/learn/academic \
  -H 'Content-Type: application/json' \
  -d '{"query": "https://www.youtube.com/watch?v=VIDEO_ID", "topic": "topic name"}'
```

## API Reference

See [docs/API.md](docs/API.md) for the full 60+ endpoint reference.

Key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/brain/learn/academic` | POST | Learn from YouTube video `{query, topic}` |
| `/api/brain/dialogue/grounded` | POST | Conversation with brain associations `{message}` |
| `/api/brain/spiking/status` | GET | Neuron counts, spike rates, neuromodulator levels |
| `/api/brain/watch` | POST | Process image through visual cortex |
| `/api/listen/process` | POST | Process audio through auditory cortex |
| `/api/brain/dream` | POST | Generate imagination chain |
| `/api/companion/greeting` | GET | Time-of-day greeting with personal context |
| `/api/companion/safety` | GET | Caregiver safety alerts |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BRAIN_PROJECT_ROOT` | current dir | Path to project root (templates, outputs) |
| `SPIKING_SCALE` | 0 (disabled) | Neuron count multiplier (0.1 = 200K, 1.0 = 2M) |
| `BRAIN_CORTEX_DISABLE` | not set | Set to `1` to disable experiment runner |
| `COMPANION_MODEL` | qwen2.5:1.5b | Ollama model for dialogue |
| `OLLAMA_MODEL` | qwen2.5:1.5b | Ollama model for triple extraction |
| `OLLAMA_URL` | http://localhost:11434 | Ollama API endpoint |
| `BRAIN_BIND_ADDR` | 0.0.0.0:443 | Server bind address |

## Project Structure

```
rust/
  crates/
    brain-spiking/     # Spiking neural network engine
      src/
        neuron.rs       # ALIF neurons, SoA layout
        synapse.rs      # COO builder → CSR storage, synaptic scaling
        region.rs       # Brain region (neurons + synapses + STDP)
        network.rs      # Multi-region orchestrator
        concepts.rs     # Cell assemblies, triple extraction
        knowledge.rs    # Concept association matrix, BFS chain recall
        plasticity.rs   # Three-factor STDP, TACOS dual-weight
        neuromodulation.rs  # DA, ACh, NE, 5-HT
        sleep.rs        # NREM replay + REM noise + structural pruning
        spike_encoder.rs    # Latency coding (embedding → spikes)
        spike_decoder.rs    # Rate decoding (spikes → embedding)
    brain-server/      # HTTP server (axum)
    brain-cognition/   # Cognitive systems
    brain-inference/   # TorchScript model loading
    brain-core/        # Hebbian networks
    brain-db/          # SQLite persistence
    brain-experiment/  # Self-improving mutation loop
scripts/               # Python training, data download, model export
templates/             # Web UI (8 HTML pages)
docs/                  # API reference, engineering lessons
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal use, research, education, and non-profit organizations. Commercial use requires a separate license from [Akretio](https://akretio.com).
