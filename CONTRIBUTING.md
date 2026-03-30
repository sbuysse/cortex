# Contributing to Cortex

Thank you for your interest in Cortex. This document will help you get set up and oriented.

## Prerequisites

- **Rust** (edition 2024, nightly recommended)
- **Python 3.12+** (for training scripts and data pipelines)
- **PyTorch 2.x** with libtorch (required by `tch-rs`)
- **SQLite 3** (bundled via `rusqlite`, no separate install)
- **espeak-ng** (optional, for TTS)
- **Ollama** (optional, for LLM-powered dialogue)

## Building

```bash
cd rust

# Set libtorch path (adjust for your system)
export LIBTORCH=/path/to/libtorch
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

cargo build --release -p brain-server
```

The binary lands at `rust/target/release/brain-server` (~28MB).

## Running

```bash
# Minimal startup (no models, limited features)
BRAIN_PROJECT_ROOT=$(pwd) ./rust/target/release/brain-server

# Full startup (with TorchScript models in outputs/cortex/)
BRAIN_PROJECT_ROOT=$(pwd) \
  BRAIN_DB_PATH=outputs/cortex/knowledge.db \
  BRAIN_OUTPUT_DIR=outputs/cortex \
  ./rust/target/release/brain-server
```

Server binds to `https://localhost:443` (self-signed TLS). Visit `https://localhost` for the dashboard.

## Project Layout

```
rust/crates/
  brain-server/      Axum web server, routes, templates
  brain-cognition/   Cognitive systems (memory, emotions, companion, dreams)
  brain-inference/   ML model loading and inference (tch-rs)
  brain-core/        Low-level matrix ops, Hebbian networks
  brain-db/          SQLite persistence layer
  brain-experiment/  Cortex daemon (self-improving mutation loop)
  brain-traits/      Shared trait definitions
  mutation-template/ Template for experimental variants

scripts/             Python training, data download, model export
templates/           Jinja2 HTML pages for the web UI
docs/                Architecture, API reference, roadmap
```

## Architecture Orientation

Cortex is a single Rust binary with 8 crates. The key flow:

1. **Perception** (`brain-inference`): Raw sensor input (image, audio, text) is encoded into 384-512 dim embeddings via TorchScript models (DINOv2, Whisper, CLIP, MiniLM).

2. **Association** (`brain-core`, `brain-inference`): A dual-encoder MLP projects visual and audio embeddings into a shared 512-dim space, trained with gradient InfoNCE.

3. **Cognition** (`brain-cognition`): Embeddings flow into working memory (7-slot theta-gamma buffer), fast memory (Hopfield associative store, 2000 patterns), concept codebook (865+ categories), knowledge graph (6K+ edges), and personal memory.

4. **Action** (`brain-server`): 60+ REST endpoints expose cognition to the outside world. The companion module builds system prompts from brain state and calls an LLM for natural dialogue.

5. **Self-improvement** (`brain-experiment`): A cortex daemon runs mutation experiments every 5 minutes, testing architectural variations against a fitness function.

## Development Workflow

1. Pick an issue or open one describing what you want to change.
2. Fork and create a feature branch.
3. Make your changes. Run `cargo build --release` to verify.
4. If you modified Python scripts, run `ruff check` and `pytest`.
5. Open a PR with a clear description of what changed and why.

## Code Style

- **Rust**: Follow standard `rustfmt` conventions. Use `tracing::info!`/`warn!` for logging (not `println!` or `eprintln!`).
- **Python**: Follow `ruff` defaults. Type hints encouraged but not required.
- Keep functions focused. If a function is doing 3 things, split it.
- Comments should explain *why*, not *what*.

## Training Scripts

The `scripts/` directory contains Python pipelines for training models and downloading datasets. These produce TorchScript files that the Rust binary loads at startup.

```bash
# Example: train the MLP dual encoder
python scripts/train_v6_audioset.py --data-dir data/audioset --output outputs/cortex

# Example: export models to TorchScript
python scripts/export_torchscript.py --output outputs/cortex
```

## Questions?

Open an issue on GitHub. We're happy to help you get oriented.
