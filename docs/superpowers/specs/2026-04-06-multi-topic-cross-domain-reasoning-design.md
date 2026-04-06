# Multi-Topic Cumulative Knowledge + Cross-Domain Reasoning

## Goal

Enable Cortex to learn from dozens of YouTube videos across different domains and answer questions that require connecting knowledge between topics — something no RAG system does well. The spiking brain's association graph naturally bridges domains through shared concepts.

## Architecture

The knowledge engine gains persistence, cross-domain graph traversal, bidirectional recall, and a bulk learning API. No new crates — all changes are in existing files.

```
Learn Phase:
  Video A (TurboQuant)  → LLM extraction → triples → association matrix
  Video B (FlashAttention) → LLM extraction → triples → association matrix
  Video C (GGUF)        → LLM extraction → triples → association matrix
                                                          │
                              Shared concepts ("kv cache") bridge domains
                                                          │
Recall Phase:                                             ▼
  Query "How are quantization and attention related?"
    → Bidirectional BFS from {quantization} and {attention}
    → Meet at bridge node: "kv cache"
    → Return chain: turboquant → compresses → kv cache ← uses ← flash attention
    → LLM gets structured reasoning chain
```

## Section 1: Persistent Knowledge Store

### What

Triples persist to disk so knowledge accumulates across restarts.

### How

- On every `learn_triple`, append to `{BRAIN_PROJECT_ROOT}/data/knowledge.triples`
- Format: one triple per line, pipe-delimited: `subject|relation|object|topic|timestamp`
- On startup, `KnowledgeEngine::load_from_file()` reads the file and replays all triples into the association matrix
- On `learn_triple`, also append to the open file handle (buffered writer, flushed every 10 triples or on shutdown)
- Create `{BRAIN_PROJECT_ROOT}/data/` directory on first write if missing

### Topic Registry

- Sidecar file: `{BRAIN_PROJECT_ROOT}/data/topics.json`
- Schema: `[{topic: string, url: string, triples_count: number, learned_at: string}]`
- Updated after each `youtube_learn_academic` completes
- Used for deduplication (skip already-learned URLs) and stats

### Files Changed

- `brain-spiking/src/knowledge.rs` — add `load_from_file()`, `append_triple()`, file handle field
- `brain-spiking/src/lib.rs` — call `load_from_file()` in SpikingBrain constructor
- `brain-cognition/src/autonomy.rs` — update topics.json after learning
- `brain-cognition/src/state.rs` — pass data directory path to knowledge engine

## Section 2: Cross-Domain Association Graph

### What

When multiple topics share concepts (e.g., both TurboQuant and FlashAttention mention "kv cache"), the association graph naturally forms bridges. We strengthen these bridges and extend BFS to exploit them.

### How

- **Shared nodes are automatic:** ConceptRegistry deduplicates by name. If video A creates concept "kv cache" and video B also references "kv cache", they share the same node ID.
- **Reinforcement on overlap:** When a triple's edge already exists (from a different topic), strengthen by delta (currently capped at 1.0). Change cap to allow accumulation: `min(weight + delta, 2.0)`. Edges learned from multiple independent sources are more trustworthy.
- **Increase max BFS hops:** Change `recall_chain` max_hops from 6 to 10. Cross-domain paths need more hops (topic A → shared concept → topic B = 4+ hops minimum).
- **Topic-weighted recall:** Add optional `topic_hint` parameter to `recall_chain`. When provided, boost weights of edges whose concepts were learned from that topic (requires storing topic provenance per concept in ConceptRegistry).

### Files Changed

- `brain-spiking/src/knowledge.rs` — raise weight cap, increase max_hops, add topic_hint parameter
- `brain-spiking/src/concepts.rs` — add topic provenance to ConceptRegistry (concept_name → set of topics)

## Section 3: Multi-Topic Learning API + Bulk Ingestion

### What

New endpoints to teach Cortex many topics efficiently and inspect the knowledge graph.

### Endpoints

**`POST /api/brain/learn/batch`**
- Body: `{"videos": [{"url": "...", "topic": "..."}, ...]}`
- Processes sequentially (Ollama is single-threaded)
- Skips already-learned URLs (checks topics.json)
- Returns: `{"topics_learned": N, "triples": N, "concepts": N, "skipped": N}`

**`GET /api/brain/knowledge/stats`**
- Returns: total topics, total triples, total unique concepts, top 10 highest-degree concepts (bridge nodes), list of learned topics with triple counts

### Files Changed

- `brain-server/src/routes.rs` — add two new route handlers
- `brain-server/src/app.rs` — register routes
- `brain-cognition/src/autonomy.rs` — extract `learn_single_video()` helper for reuse

## Section 4: Improved Recall for Cross-Domain Queries

### What

Bidirectional BFS that finds bridge paths between concept clusters, with structured chain output for the LLM.

### How

**Bidirectional BFS:**
- Parse query for recognizable concept names (match against all registered concepts, not just words > 3 chars)
- If 2+ concept clusters found, run BFS from both sides simultaneously
- When frontiers overlap, extract the bridge path
- If only 1 cluster found, fall back to current unidirectional BFS

**Path reconstruction:**
- Track parent pointers during BFS: for each visited node, store which node led to it and via which edge
- Reconstruct full chain from start to bridge to end
- Format as: `concept_a --relation--> concept_b --relation--> concept_c`

**System prompt format:**
```
YOU LEARNED THE FOLLOWING CONNECTIONS:
turboquant --compresses--> kv cache --used by--> flash attention
quantization --reduces--> memory usage --enables--> larger models

Use these chains to answer. Follow the arrows to explain relationships.
```

**Multi-seed matching:**
- Current: match query words > 3 chars against concept names
- New: also match against topic names from topics.json
- Also match multi-word phrases: "kv cache" should match concept "kv cache", not just "cache"

### Files Changed

- `brain-spiking/src/knowledge.rs` — bidirectional BFS, path reconstruction, chain formatting
- `brain-server/src/routes.rs` — update system prompt formatting to use chains

## Success Criteria

1. Learn 10+ videos across 3+ domains (e.g., ML optimization, programming languages, neuroscience)
2. Ask a cross-domain question that requires bridging two topics
3. Brain returns a chain that connects concepts from different videos through shared nodes
4. LLM produces an answer that demonstrates cross-domain reasoning
5. Knowledge persists across server restarts
6. `GET /api/brain/knowledge/stats` shows bridge concepts shared by 2+ topics

## Non-Goals

- No temporal/causal ordering (future work — Approach B)
- No spiking propagation recall (future work — Approach C)
- No Pi deployment optimization
- No UI changes (API-only)
