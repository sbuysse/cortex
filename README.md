# Cortex

An open cognitive architecture in Rust. Single binary, zero Python runtime dependency, runs on a Raspberry Pi 5.

Cortex sees (DINOv2, CLIP), hears (Whisper), remembers (Hopfield networks, knowledge graph), reasons (spreading activation, world model), dreams (imagination chains), learns continuously (gradient InfoNCE), and talks (emotion-aware companion dialogue).

## Why This Exists

Most AI systems are stateless inference endpoints. Cortex is a persistent, self-improving cognitive system that maintains its own memory, knowledge, and internal state across time. It perceives multimodal input, forms associations, builds a knowledge graph, tracks emotions, and generates behavior — all on-device, all private.

This isn't a chatbot wrapper. It's a reference implementation for what a complete cognitive architecture looks like when you build it from scratch.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     brain-server (axum)                           │
│  HTTPS :443 — 7 HTML pages + 60+ JSON endpoints + SSE           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                   brain-cognition                            │  │
│  │                                                              │  │
│  │  Working Memory    Fast Memory     Grid Cells    Codebook    │  │
│  │  (7-slot theta-    (Hopfield       (hexagonal    (865+       │  │
│  │   gamma buffer)     2000 patterns)  spatial 2D)   categories) │  │
│  │                                                              │  │
│  │  Personal Memory   Emotion         Companion     Dreams      │  │
│  │  (facts, family,   (7 classes,     (daily rhythm, (world     │  │
│  │   health, KG)       mood tracking)  safety alerts)  model)    │  │
│  │                                                              │  │
│  │  Knowledge Graph   Autonomy Loop   SSE Bus                   │  │
│  │  (6K+ edges,       (5-min self-    (real-time                │  │
│  │   5 relation types) improvement)    broadcast)               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                   brain-inference (tch-rs)                    │  │
│  │                                                              │  │
│  │  DINOv2   CLIP   Whisper   MiniLM   WorldModel   MLP        │  │
│  │  (384d)   (512d)  (512d)   (384d)   (512→512)   (V+A→512)  │  │
│  │                                                              │  │
│  │  Mel Spectrogram   VAD   Face Database   Emotion Classifier  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  brain-core          brain-db           brain-experiment          │
│  (Hebbian networks)  (SQLite KG)        (cortex mutation loop)   │
└──────────────────────────────────────────────────────────────────┘
```

## What It Can Do

**Perceive** — Process images (DINOv2/CLIP) and audio (Whisper) into a shared embedding space. Real-time voice activity detection, face recognition, mel spectrogram computation.

**Remember** — 7-slot working memory with theta-gamma oscillation. Hopfield associative memory (2000 patterns). Episodic memory with temporal clustering. Personal knowledge graph (family, health, preferences).

**Reason** — World model predicts cross-modal associations. Knowledge graph traversal with spreading activation. Concept arithmetic (add/subtract embeddings). Confidence estimation and novelty detection.

**Dream** — Imagination chains: start from a concept, predict what follows, explore surprising paths. Each dream generates new training pairs for self-improvement.

**Learn** — Gradient InfoNCE training runs in Rust. Online learning from new perceptions. Autonomous 5-minute improvement cycles. Cortex daemon tests mutations against a fitness function.

**Talk** — Emotion detection from text (7 classes). Personal memory informs every response. Mood tracking over time. Safety alerts for caregivers. Brain state (working memory + knowledge + emotion) grounds the dialogue.

## Quick Start

```bash
# Clone
git clone https://github.com/sbuysse/cortex.git
cd cortex

# Build (requires libtorch)
cd rust
export LIBTORCH=/path/to/libtorch
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build --release -p brain-server

# Run (minimal — no models needed for core cognition)
BRAIN_PROJECT_ROOT=$(pwd)/.. ./target/release/brain-server
```

Visit `https://localhost` for the dashboard. See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup.

## API Overview

60+ endpoints organized by cognitive function:

| Domain | Endpoints | Examples |
|--------|-----------|---------|
| **Perception** | `/api/brain/watch`, `/api/listen/process` | Process image or audio → embeddings + associations |
| **Cognition** | `/api/brain/predict`, `/api/brain/reason`, `/api/brain/compose` | World model prediction, KG reasoning, concept arithmetic |
| **Memory** | `/api/brain/working_memory`, `/api/brain/episodes`, `/api/brain/prototypes` | Working memory state, episodic timeline, learned concepts |
| **Dreams** | `/api/brain/dream`, `/api/brain/think` | Imagination chains, multi-step reasoning |
| **Learning** | `/api/brain/learn`, `/api/brain/learn/train` | Buffer pairs, train online |
| **Autonomy** | `/api/brain/autonomy/start` | Self-directed learning loop |
| **Companion** | `/api/brain/dialogue/grounded`, `/api/companion/safety` | Emotion-aware dialogue, caregiver alerts |
| **Spatial** | `/api/brain/grid/map`, `/api/brain/grid/navigate` | Hexagonal grid, concept navigation |

Full reference: [docs/API.md](docs/API.md)

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `brain-server` | Axum web server, 60+ routes, 7 HTML pages, TLS |
| `brain-cognition` | Working memory, fast memory, knowledge graph, companion, dreams, autonomy |
| `brain-inference` | TorchScript model loading (DINOv2, CLIP, Whisper, MiniLM, world model, MLP) |
| `brain-core` | Low-level matrix ops, Hebbian association networks |
| `brain-db` | SQLite persistence (13 tables) |
| `brain-experiment` | Cortex daemon — self-improving mutation loop |
| `brain-traits` | Shared trait definitions |
| `mutation-template` | Template for experimental variants |

## Design Principles

**Single binary** — No Python runtime, no Docker, no microservices. One `cargo build` produces one 28MB binary that loads TorchScript models at startup.

**On-device** — All processing is local. No cloud APIs, no data leaves the device. Runs on a Raspberry Pi 5 (8GB).

**Neuroscience-informed** — Working memory modeled on theta-gamma phase coupling. Hopfield networks for associative memory. Grid cells for spatial representation. Dream replay for memory consolidation.

**Self-improving** — The cortex daemon runs experiments autonomously, testing mutations to its own training loop and architecture against a fitness function. Over 23,000 experiments completed.

**Privacy-first** — No recordings stored, only extracted facts and embeddings. SQLite database stays on-device. No telemetry, no analytics.

## Current Scale

| Metric | Value |
|--------|-------|
| Training pairs learned | 2,086,394 |
| Concept vocabulary | 865+ categories |
| Knowledge graph edges | 6,347 |
| TorchScript models | 8 (1.15 GB total) |
| API endpoints | 60+ |
| Cortex experiments | 23,000+ |

## License

Apache 2.0 — see [LICENSE](LICENSE).
