# Cortex

A cognitive architecture in Rust combining spiking neural networks with foundation model encoders. 2 million ALIF neurons, 2 billion synaptic connections, 10 brain regions, STDP-based learning.

## Overview

Cortex is a research platform for studying how spiking neural dynamics can form associative knowledge from multimodal input. It encodes perception (DINOv2, CLIP, Whisper, MiniLM) into spike trains, learns associations through spike-timing-dependent plasticity, and recalls knowledge through chain propagation across brain regions.

The system can watch YouTube videos, extract subject-verb-object triples from transcripts, encode them as sequential STDP patterns in cell assemblies, and recall associated concepts through learned synaptic pathways — without storing any text.

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
  → Sentence chunking (~200 chars)
  → SVO triple extraction (rule-based, with pronoun resolution)
  → Sequential STDP encoding:
      Subject neurons (20 steps) → Relation neurons (20 steps) → Object neurons (20 steps)
      Repeated 3x per triple
  → Synaptic weights encode directional associations
  → No text stored — weights ARE the memory
```

### Recall

```
Query → Fuzzy concept matching → Activate matching cell assemblies
  → Free propagation through learned connections (up to 30 steps, early stopping)
  → Read activated populations in PFC + Hippocampus
  → Chain of associated concepts returned
```

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
        knowledge.rs    # Sequential STDP learning, chain recall
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
