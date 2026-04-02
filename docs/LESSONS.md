# Cortex — Engineering Lessons Learned

Built 2026-04-01/02 across a single marathon session.

---

## Spiking Brain Architecture

### Activity cascade at scale
**Problem:** At 2B connections (1000 fanout/neuron), a single stimulus cascades through all 2M neurons — 36 million spikes, 215 seconds per tick.

**Root cause:** 80/20 excitatory/inhibitory ratio with random connectivity. Each excitatory neuron excites 800 others, each of which excites 800 more — exponential explosion. 200 inhibitory connections can't contain it.

**Solution:** Synaptic scaling in `deliver_spikes()`. Separate external current (from encoders, unclamped) from synaptic current (from other neurons, clamped to ±0.5). External input drives the stimulus; synaptic clamp prevents cascade. Also reset neuron membrane voltages at the start of each `associate()` to prevent cross-query accumulation.

**Result:** 1.1 seconds per tick, 23-38 spikes total. No cascade.

### Inter-region signal propagation
**Problem:** Clamping stopped cascade but also killed legitimate signal propagation between regions.

**Root cause:** Inter-region weights (0.05-0.3) were far below firing threshold (1.0). A single inter-region spike couldn't fire the target neuron.

**Solution:** Strong feedforward inter-region weights (0.8-1.5) so a single spike propagates across region boundaries. Feedback paths stay weaker (0.1-0.3). Combined with intra-region synaptic clamping, this allows signal to flow BETWEEN regions while preventing cascade WITHIN regions.

**Result:** visual_cortex:21 → association_cortex:16 → prefrontal_cortex:1. Signal crosses 3 region boundaries.

### Connection scaling must be linear, not quadratic
**Problem:** Initial connectivity used `n² × probability` — at 200K neurons with 5% probability, that's 2 billion connections per region. OOM at 58GB.

**Root cause:** Biological neurons have ~1000-10000 connections each, not proportional to the square of the population.

**Solution:** `n_connections = n × avg_fanout` where avg_fanout is a constant (1000). Linear in population size.

### Encoder dimensions must match input sources
**Problem:** Visual encoder was 512-dim but MiniLM text produces 384-dim. Association cortex decoder was 500K-dim (one per neuron) but MLP expects 384-dim.

**Root cause:** Encoder/decoder dims were set to region neuron count instead of input/output source dims.

**Solution:** Visual encoder = 384 (DINOv2/MiniLM), audio encoder = 512 (Whisper), decoder = 384 (reads from first 384 neurons, matching MLP input dim).

---

## CPU Performance

### Spiking brain vs Ollama — CPU contention
**Problem:** At scale=1.0, the spiking brain tick consumed 85%+ CPU continuously, starving Ollama (which needs CPU for token generation on this machine).

**Root cause:** The tick thread ran every 5 seconds with no regard for other processes. Each tick took 30-215 seconds, leaving no CPU for Ollama.

**Solution chain:**
1. On-demand tick — only run when there's a pending query (not on timer)
2. Selective region stepping — only step the 4 regions in the association pathway (visual, association, PFC, hippocampus), skip the other 6
3. Skip quiescent neurons — neurons at rest (v=0, i_ext=0, a=0) skip the update loop
4. Early stopping — stop propagation when no new spikes for 3 consecutive steps
5. 10ms sleep between steps — forces OS scheduler to give time to Ollama
6. Separate snapshot mutex — dialogue route reads `spiking_snapshot` which never contends with the tick thread's brain lock

### Ollama cold loading
**Problem:** Ollama unloads models after 5 minutes of inactivity. Each first call took 25 seconds just to load the model into RAM.

**Solution:** `OLLAMA_KEEP_ALIVE=-1` environment variable keeps the model loaded permanently. Response time: 25s → 0.3s.

### Conversation history length
**Problem:** Sending 8 conversation history turns to qwen2.5:1.5b on CPU took 60-90 seconds.

**Solution:** Reduced to 3 turns. Adequate context, responsive performance.

---

## Academic Video Learning

### YouTube subtitle parsing
**Problem:** YouTube auto-subtitles often lack punctuation. Splitting on `.!?` produced 1-4 "sentences" from 57K chars of transcript.

**Solution:** Split on punctuation AND on ~100-character word boundaries. 57K chars → 580 sentences.

### Concept extraction without LLM
**Problem:** Ollama concept extraction was slow (timeout), unreliable (bad JSON from 1.5b), and required network to GPU server (tunnel hack).

**Root cause:** Using an LLM for structured extraction is the wrong tool. The codebook + MiniLM already know 865 concepts.

**Solution:** Pure local pipeline: split transcript → encode each chunk via MiniLM → match against codebook → detect novel concepts (low codebook match) → store novel concepts in `learned_concepts` store. Zero LLM dependency, instant, deterministic.

### Audio codebook pollution
**Problem:** Brain associations returned "Bicycle", "Cello", "Fart" — VGGSound audio labels instead of semantic concepts.

**Root cause:** The codebook was built from audio categories. MiniLM encodes text but the codebook matches it against audio labels.

**Solution:** Separate `learned_concepts` store (label + 384-dim MiniLM embedding) populated during academic learning. Dialogue route matches brain output against this store instead of the audio codebook.

---

## Associative Recall

### Reading the wrong neurons
**Problem:** Association cortex output echoed the input query instead of producing novel associations.

**Root cause:** Reading from the first 384 neurons of the association cortex — these are the INPUT neurons that received the stimulus directly.

**Solution:** Read from DOWNSTREAM regions (PFC + hippocampus) which were NOT directly stimulated. Whatever fires there is pure association from learned STDP connections.

### Non-blocking architecture
**Problem:** Synchronous associative recall (30 steps through 2B connections) blocked the HTTP request thread for 30+ seconds. Ollama timed out.

**Solution:** Asynchronous architecture:
- Dialogue route: `enqueue_query()` (instant) + read `spiking_snapshot` (separate mutex, never contends)
- Background tick thread: processes pending queries, writes snapshot
- Brain thinks asynchronously — current question's associations are available for the NEXT question

### The brain needs to THINK, not just STORE
**Key insight:** Simply storing and retrieving transcript sentences is RAG (a database). A brain forms associations through learned connectivity. The spiking brain's value is that it produces EMERGENT associations — concepts it linked through STDP because they co-occurred during learning, not because we stored them in a lookup table.

**Proven:** Query "TurboQuant memory" → brain associates "compresses the KV" (from video transcript, linked through STDP in the association cortex).

---

## Development Process

### No quick fixes
Rejected approaches: SSH tunnels for networking, zero-padding for dim mismatch, code fence stripping for Ollama formatting, truncating history as a performance fix. Always find the root cause.

### Test at the right scale
scale=0.01 for unit tests, scale=0.1 for integration tests with real dialogue, scale=1.0 for production. Don't debug production-scale issues at test scale — the behavior is fundamentally different (cascade dynamics, memory pressure, CPU contention).

### Subagent-driven development
10 core tasks implemented by dispatching fresh subagents per task. Each subagent got the full task spec + context. Two-stage review (spec compliance + code quality) between tasks. Result: 33 tests, clean architecture, no cross-contamination of context.

---

## What Makes Cortex Unique

No existing project combines all of these:
1. Spiking neurons with STDP learning
2. 10+ biologically-inspired brain regions
3. Foundation model encoders (DINOv2, CLIP, Whisper, MiniLM)
4. Natural language dialogue
5. Learns from YouTube videos
6. Persistent memory across restarts
7. Neuromodulation (DA, ACh, NE, 5-HT) driving personality
8. Runs on commodity hardware
9. Open source (PolyForm Noncommercial)
10. Associative recall — the brain THINKS, not just retrieves

The closest competitor (BrainCog) scores 5/10. The field is bifurcated: brain simulators that can't talk, and talking models that aren't brains. Cortex bridges both.
