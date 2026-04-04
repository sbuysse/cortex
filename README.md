# Cortex

A cognitive architecture with a spiking neural brain. Built in Rust. Learns from YouTube videos. Recalls through 2 billion synaptic connections. No text stored — STDP weights are the memory.

## What Makes This Different

Most AI systems retrieve information from databases. Cortex **learns** it — through spike-timing-dependent plasticity across 2 billion connections in 10 brain regions. When you ask a question, the answer comes from neural activation patterns propagating through learned synaptic pathways, not from a lookup table.

**Watch a video → Extract knowledge triples → Learn via sequential STDP → Recall through spike propagation → Answer from neural associations**

No other system combines:
- Spiking neural network (2M neurons, 2B connections, STDP learning)
- 10 biologically-inspired brain regions with inter-region connectivity
- Foundation model encoders (DINOv2, CLIP, Whisper, MiniLM)
- Knowledge triple extraction and sequential STDP encoding
- Associative chain recall through learned synaptic pathways
- Natural language dialogue shaped by neuromodulators
- Learns from YouTube videos with zero LLM dependency for extraction
- Persistent memory across restarts via synaptic weight save/load
- Sleep consolidation (NREM replay + REM noise + structural pruning)
- Runs on commodity hardware (single machine, CPU)

## The Spiking Brain

```
┌─────────────────────────────────────────────────────────────┐
│                    brain-spiking                             │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Visual   │ │ Auditory │ │Association│ │Predictive│       │
│  │ Cortex   │→│ Cortex   │→│ Cortex   │→│ Cortex   │       │
│  │ (200K)   │ │ (200K)   │ │ (500K)   │ │ (200K)   │       │
│  └──────────┘ └──────────┘ └────┬─────┘ └──────────┘       │
│                                  │                           │
│  ┌──────────┐ ┌──────────┐ ┌────┴─────┐ ┌──────────┐       │
│  │Hippocampus│ │Prefrontal│ │ Amygdala │ │  Motor   │       │
│  │ (300K)   │←│ (200K)  │←│ (100K)   │ │ (100K)   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
│  ┌──────────┐ ┌──────────┐                                  │
│  │Brainstem │ │Cerebellum│  4 Neuromodulators:              │
│  │ (50K)    │ │ (150K)   │  DA · ACh · NE · 5-HT           │
│  └──────────┘ └──────────┘                                  │
│                                                              │
│  2,000,000 ALIF neurons · 2,000,000,000 synapses            │
│  Three-factor STDP · TACOS dual-weight · Structural pruning │
│  Cell assemblies · Knowledge triples · Chain recall          │
└─────────────────────────────────────────────────────────────┘
```

### How It Learns

1. **Watch**: YouTube video → auto-subtitles → sentence chunks
2. **Extract**: Rule-based SVO triple extraction with pronoun resolution
   - "This compresses the KV cache" + topic "TurboQuant" → `(turboquant, compresses, kv_cache)`
3. **Encode**: Sequential STDP — subject neurons fire (20 steps) → relation neurons (20 steps) → object neurons (20 steps). Repeated 3x. The directional synaptic chain is the knowledge.
4. **Recall**: Query activates matching concept populations → spikes propagate through STDP-strengthened pathways → downstream populations fire → chain of activated concepts is the answer

No text is stored in the brain. The synaptic weight pattern IS the memory.

### How It Recalls

```
Query: "What compresses the KV cache?"

1. Fuzzy concept matching: "compresses" → concept population
                           "cache" → concept population
2. Activate matching populations (20 timesteps)
3. Free propagation through 2B learned connections (30 steps)
4. Read which populations activated in sequence
5. Chain: turboquant → compresses → kv_cache → is → short_term_memory
6. LLM receives: "TurboQuant compresses the KV cache, which is the short-term memory"
```

## Architecture

9 Rust crates, 60+ API endpoints, 13 SQLite tables, 8 TorchScript models.

| Crate | Purpose |
|-------|---------|
| `brain-spiking` | **NEW** — ALIF neurons, CSR synapses, 10 brain regions, STDP, neuromodulation, knowledge engine, chain recall |
| `brain-server` | Axum web server, 60+ routes, 8 HTML pages, TLS, SSE |
| `brain-cognition` | Working memory, fast memory, knowledge graph, companion, dreams, autonomy |
| `brain-inference` | TorchScript model loading (DINOv2, CLIP, Whisper, MiniLM, world model) |
| `brain-core` | Hebbian association networks, sparse projections |
| `brain-db` | SQLite persistence (13 tables) |
| `brain-experiment` | Cortex daemon — self-improving mutation loop |
| `brain-traits` | Shared trait definitions |
| `mutation-template` | Template for experimental variants |

## Quick Start

```bash
git clone https://github.com/sbuysse/cortex.git
cd cortex/rust

# Build (requires libtorch)
export LIBTORCH=/path/to/libtorch
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build --release -p brain-server

# Run with spiking brain (scale: 0.01=test, 0.1=dev, 1.0=full)
BRAIN_PROJECT_ROOT=$(pwd)/.. SPIKING_SCALE=0.1 ./target/release/brain-server
```

### Teach it something

```bash
# Watch a YouTube video about TurboQuant
curl -sk -X POST https://localhost/api/brain/learn/academic \
  -H 'Content-Type: application/json' \
  -d '{"query": "https://www.youtube.com/watch?v=7YVrb3-ABYE", "topic": "turboquant"}'

# Ask about it (after learning completes)
curl -sk -X POST https://localhost/api/brain/dialogue/grounded \
  -H 'Content-Type: application/json' \
  -d '{"message": "How does TurboQuant reduce memory?"}'
```

## What It Can Do

**Perceive** — DINOv2 (vision), CLIP (scenes), Whisper (audio), MiniLM (text) → shared embedding space

**Remember** — 7-slot working memory, Hopfield associative memory, personal knowledge graph, persistent spiking brain weights

**Reason** — World model predictions, knowledge graph traversal, spiking chain recall through learned associations

**Dream** — Imagination chains with surprise-weighted learning. Sleep consolidation (NREM replay + REM noise + structural pruning)

**Learn** — From YouTube videos (triple extraction → sequential STDP), from conversation (neuromodulator-driven), from perception (online gradient InfoNCE)

**Talk** — Emotion-aware companion dialogue. Neuromodulators (dopamine, acetylcholine, norepinephrine, serotonin) shape personality in real-time. Brain associations guide LLM responses.

## API

60+ endpoints. Key ones:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/brain/learn/academic` | POST | Learn from YouTube video (extract triples, STDP encode) |
| `/api/brain/dialogue/grounded` | POST | Conversation with spiking brain associations |
| `/api/brain/spiking/status` | GET | Per-region spike rates, neuromodulator levels |
| `/api/brain/watch` | POST | Process image → visual cortex |
| `/api/listen/process` | POST | Process audio → auditory cortex |
| `/api/brain/dream` | POST | Generate imagination chain |
| `/api/companion/safety` | GET | Caregiver safety alerts |

Full reference: [docs/API.md](docs/API.md)

## Current Scale

| Metric | Value |
|--------|-------|
| Spiking neurons | 2,000,000 |
| Synaptic connections | 2,000,000,000 |
| Brain regions | 10 |
| Neuromodulators | 4 (DA, ACh, NE, 5-HT) |
| Training pairs learned | 2,086,394 |
| Knowledge graph edges | 6,347+ |
| TorchScript models | 8 |
| API endpoints | 60+ |

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal use, research, education, and non-profit. Commercial use requires a separate license from Akretio.
