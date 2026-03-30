# Cortex API Reference

Base URL: `https://localhost` (TLS, self-signed cert)

## Companion API

### GET /api/companion/greeting
Time-of-day greeting with personal context.
```json
{"greeting": "Good morning, Marguerite!", "period": "morning", "should_initiate": true}
```

### GET /api/companion/safety
Safety alerts for caregivers.
```json
{"alerts": ["Pain mentioned 4 times"], "alert_count": 1, "personal_summary": "...", "recent_moods": [...]}
```

### GET /api/companion/personal
All personal facts + conversation history.
```json
{"name": "Marguerite", "facts": [...], "context": "...", "recent_conversation": [...]}
```

## Dialogue

### POST /api/brain/dialogue/grounded
Personality-driven conversation with emotion detection and personal memory.
```json
// Request
{"message": "I miss my husband Jean"}

// Response
{
  "response": "I'm here with you. When you say that, I think of Sad music...",
  "facts_extracted": ["Jean is-husband-of user"],
  "grounding": {"semantic_matches": [...], "personal": "Name: Marguerite...", "knowledge": [...]},
  "grounded": true
}
```

### POST /api/brain/dialogue
LLM-based dialogue (requires Ollama or local LLM).

## Perception

### POST /api/brain/watch
Process webcam image → visual understanding.
```json
{"image_b64": "<base64 JPEG>", "top_k": 10}
→ {"summary": {"i_see": [...], "i_expect_to_hear": [...], "confidence": 0.95}}
```

### POST /api/listen/process
Process audio → sound understanding.
```json
{"audio_b64": "<base64 PCM float32 16kHz>", "top_k": 10}
→ {"summary": {"i_hear": [...], "audio_duration": 2.0, "confidence": 0.8}}
```

## Cognition

### POST /api/brain/predict
World model prediction: what audio does this evoke?
```json
{"query": "thunder storm", "top_k": 5}
→ {"predicted_audio": [{"label": "thunder", "similarity": 0.41}, ...]}
```

### POST /api/brain/reason
Spreading activation reasoning through knowledge graph.
```json
{"query": "what follows thunder?", "n_hops": 3, "top_k": 10}
→ {"start_concepts": ["thunder"], "chains": [...]}
```

### POST /api/brain/compose
Concept arithmetic: add/subtract concepts.
```json
{"add": ["dog", "music"], "subtract": ["cat"], "top_k": 5}
→ {"results": [{"concept": "dog growling", "similarity": 0.40}]}
```

### POST /api/brain/decompose
Decompose a phrase into concept components.
```json
{"query": "thunderstorm at night", "k": 5}
→ {"components": [{"concept": "night", "weight": 0.41}, ...]}
```

### POST /api/brain/dream
Generate an imagination chain.
```json
{"seed": "piano", "steps": 5}
→ {"seed": "playing piano", "steps": [...], "avg_surprise": 0.35, "learning_pairs_generated": 5}
```

### POST /api/brain/think
Multi-step grounded reasoning.
```json
{"question": "what is a dog?"}
→ {"chain_of_thought": [{"step": "activate", ...}, {"step": "predict", ...}]}
```

## Memory

### GET /api/brain/working_memory
Current 7-slot attention buffer.

### GET /api/brain/memory/fast
Hopfield associative memory status.

### GET /api/brain/episodes
Episodic memory timeline.

### GET /api/brain/prototypes
Learned concept prototypes.

### GET /api/brain/knowledge
Knowledge graph statistics by relation type.

### POST /api/brain/knowledge/query
BFS traversal through knowledge graph.

## Self-Model

### GET /api/brain/self/assessment
Overall brain state assessment.

### GET /api/brain/self/progress
Learning progress: pairs learned, dreams, autonomy cycles.

### POST /api/brain/self/confidence
Confidence score for a query.

### POST /api/brain/curiosity/score
Novelty detection: is this concept new to the brain?

## Learning

### POST /api/brain/learn
Buffer a (visual, audio) embedding pair for training.

### POST /api/brain/learn/train
Train MLP on buffered pairs (gradient InfoNCE).

### GET /api/brain/learn/status
Training buffer size and total learned count.

### POST /api/brain/ingest/audioset
Bulk ingest pre-computed embeddings from any dataset directory.
```json
{"dataset": "balanced", "batch_size": 22160}
// Also accepts: "eval", "unbalanced", or any directory name under outputs/cortex/
```

## Autonomy

### POST /api/brain/autonomy/start
Start the self-directed learning loop (5-minute cycles).

### POST /api/brain/autonomy/stop
Stop the autonomy loop.

### GET /api/brain/autonomy/status
Autonomy loop status: cycles, videos, pairs learned.

## Grid Cells

### GET /api/brain/grid/map
300 concepts projected to 2D hexagonal grid.

### POST /api/brain/grid/navigate
Waypoints between two concepts in grid space.

### POST /api/brain/grid/between
Grid distance + cosine similarity between concepts.

## Voice

### POST /api/brain/speak
Text-to-speech via espeak-ng. Returns WAV audio.

### GET /api/brain/speak/thought
Speak the brain's current thought.

## Pages

| URL | Page |
|-----|------|
| `/` | Dashboard |
| `/goals` | Goals & Progress |
| `/cognition` | Memory / Learning / Autonomy / Knowledge |
| `/imagine` | Imagination + Concept Arithmetic |
| `/explore` | Clips / Embeddings / Grid Map |
| `/training` | Experiments & Evolution |
| `/face` | Animated Face with mic/camera/chat |
