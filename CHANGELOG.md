# Changelog

## [0.4.0] - 2026-04-06

### Spiking Recall Resurrection
- Fire BFS-discovered concepts into the 2B-synapse spiking network
- 30-step propagation through association cortex with raised synaptic clamp (1.5)
- Neuromodulator-driven recall: focused mode (high ACh) for single-topic, broad mode (high NE) for cross-domain
- Emergent associations discovered through lateral neural pathways
- Example: querying "TurboQuant" activates "sparse", "word into vector" from different topics
- Confidence tags: [confirmed] (BFS+spiking), [explicit] (BFS only), [emergent] (spiking only)

### Synaptic Knowledge Imprinting
- When triples are learned, strengthen actual CSR weights between concept assembly neurons
- 803 synapses imprinted per 13-triple learning batch (delta 0.8/0.8/0.4)
- Persisted triples re-imprinted on startup
- Imprinted weights cap at 1.0 (2x the normal 0.5 background)

### Persistent Cumulative Knowledge
- Triples persist to `data/triples.log` (pipe-delimited, append-only)
- Topic registry in `data/topics.json` (deduplication, provenance)
- Knowledge survives server restarts: 24 topics, 816 concepts, 1195 associations
- Weight cap raised to 2.0 for multi-source reinforcement

### Cross-Domain Reasoning
- Bidirectional BFS: query matches 2+ concept clusters, BFS from both sides, find bridge nodes
- Bridge concept detection: "kv cache" shared between TurboQuant and FlashAttention
- Cross-domain answers: LLM connects knowledge from different YouTube videos
- Topic provenance tracking on ConceptRegistry

### New API Endpoints
- `POST /api/brain/learn/batch` — learn multiple videos `{"videos": [{url, topic}, ...]}`
- `GET /api/brain/knowledge/stats` — topics, concepts, associations, bridges, top connected

## [0.3.0] - 2026-04-06

### LLM-Powered Triple Extraction
- Replaced rule-based SVO parser with LLM-powered extraction (Ollama)
- Batched sentences (10 per call) for efficient extraction
- Example: "TurboQuant compresses the KV cache" now extracts `TurboQuant|compresses|KV cache`
- Noise filtering: rejects prompt echoes, long rambling objects, meta-text
- Rule-based fallback when Ollama is unavailable

### Batch Learning
- Triple queue now drains all pending triples in a single tick
- 12 triples learned in 0.000s (previously 48s at 1 per 2s tick)

### Improved Recall Quality
- Filtered relation verbs ("is", "are", "relates-to") from recall output
- Only substantive concept names returned as associations

### Stronger LLM Integration
- System prompt now instructs LLM to use brain knowledge as factual learned data
- LLM answers are grounded in brain associations, not hedged guesses

### Triple Extraction Quality (Rule-Based Fallback)
- Sentence-boundary splitting for multi-sentence chunks
- Expanded noise filters: filler subjects, commas, single junk words
- Topic-anchored extraction: catches key phrases near topic even without SVO
- Stop word list expanded (50+ words)

## [0.2.0] - 2026-04-05

### Direct Concept Association Matrix
- Replaced 500M-synapse STDP simulation with HashMap-based associations
- learn_triple: 3 hash map updates (S→R, R→O, S→O edges)
- recall_chain: BFS through association graph (instant)
- Learning went from 90s/triple to 0ms/triple

### GPU Spike Delivery
- Optional CUDA acceleration via tch/libtorch
- COO format on GPU, scatter_add spike delivery
- Feature-gated: `--features gpu`

### Performance Optimizations
- CSR prefetching (x86_64 _mm_prefetch)
- Sorted spike delivery for cache locality
- `target-cpu=native` for SIMD auto-vectorization
- Thread-local reusable buffers for synaptic delivery

## [0.1.0] - 2026-04-04

### Initial Release
- 10 brain regions, 2M ALIF neurons, 2B CSR synapses
- Three-factor STDP with eligibility traces
- TACOS dual-weight synapses for continual learning
- 4 neuromodulators (dopamine, acetylcholine, norepinephrine, serotonin)
- Cell assemblies (~100 neurons per concept)
- Foundation model encoders: DINOv2, CLIP, Whisper, MiniLM
- YouTube video learning pipeline
- 60+ API endpoints via axum
- Sleep consolidation (NREM replay + REM noise + structural pruning)
- PolyForm Noncommercial 1.0.0 license
