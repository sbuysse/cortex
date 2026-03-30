# Brain Project — Development Phases

## Phase 1: Perception & Association (COMPLETE)
**Goal**: Learn cross-modal associations between vision and audio

- [x] Hebbian association matrix (512x512 bilinear map)
- [x] DINOv2 visual encoder (384-dim)
- [x] Whisper audio encoder (512-dim)
- [x] Sparse random projection (384/512 → 512 sparse)
- [x] VGGSound dataset (24,604 clips)
- [x] Gradient InfoNCE training (replaced broken Hebbian approximation)
- [x] Symmetric bidirectional loss (V→A + A→V)
- [x] Autonomous self-improvement (Cortex daemon + LLM mutations)
- [x] Web dashboard with goals tracking
- [x] YouTube video interaction

**Results**: v2a_MRR=0.55, a2v_MRR=0.54, 1,240+ experiments

## Phase 2: Give It a Voice (COMPLETE)
**Goal**: The brain can describe what it perceives and answer questions

- [x] `/api/brain/describe` — describe associations for an input
- [x] `/api/brain/ask` — answer questions about learned patterns with cross-modal associations
- [x] `/api/brain/chat` — conversational interface with personality
- [x] Chat UI on the web dashboard (`/chat` page)
- [x] Brain personality (speaks in first person about its associations)
- [x] Conversation context (session-based history)
- [x] Template-based voice engine (instant responses, no LLM latency)
- [x] Stop word filtering + fuzzy stem matching for keyword search
- [x] Cross-modal association retrieval (visual→audio bridge)
- [x] YouTube video processing → brain perception description

**Key insight**: Template-based voice engine composes natural language from raw association
scores instantly. LLM approach was abandoned due to CPU-only inference being too slow
(~5s/token even for 0.5b model). The template engine gives personality and is immediate.

## Phase 3: Real-Time Interaction
**Goal**: Process live audio/video and react instantly

- [ ] WebSocket endpoint for streaming audio
- [ ] Real-time encoding + association (< 2s latency)
- [ ] Live waveform visualization in browser
- [ ] Streaming association updates via SSE
- [ ] Browser microphone capture (Web Audio API)
- [ ] Live camera feed processing

## Phase 4: Continuous Learning & Memory
**Goal**: Learn from new experiences, remember specific interactions

- [ ] Online gradient InfoNCE updates from new inputs
- [ ] Episodic memory store (specific interactions, not just aggregates)
- [ ] "What did you learn today?" summarization
- [ ] Learnable projection (replace random sparse projection)
- [ ] Expand beyond VGGSound — learn from user-provided content
- [ ] Memory consolidation (sleep-like offline replay)

## Phase 5: Compositional Understanding
**Goal**: Reason about relationships, not just similarities

- [ ] Graph of associations (beyond single bilinear matrix)
- [ ] Multi-hop reasoning: A sounds like B, B looks like C → A relates to C
- [ ] Concept formation — cluster similar associations into abstract categories
- [ ] Text embedding bridge (CLIP) for language-grounded queries
- [ ] Causal associations — "rain causes puddles" not just "rain co-occurs with puddles"
- [ ] Attention over association graph for complex queries

## Phase 6: Autonomous Agent
**Goal**: Self-directed exploration and communication

- [ ] Initiate conversation based on interesting patterns discovered
- [ ] Self-reflection on matrix changes ("I learned something new about...")
- [ ] Goal-directed perception — seek out inputs that fill knowledge gaps
- [ ] Multi-agent interaction — share associations with other brain instances
- [ ] Report mutation results and discoveries in natural language
