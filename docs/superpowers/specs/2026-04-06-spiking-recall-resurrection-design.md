# Spiking Recall Resurrection

## Goal

Resurrect the 2B-synapse spiking network for knowledge recall alongside the existing HashMap BFS. The spiking network discovers emergent lateral associations that BFS can't find, provides dual-pathway confidence signals, and uses neuromodulator state to control recall breadth.

## Architecture

```
Query arrives
  → BFS (instant, 0ms) → explicit + confirmed associations
  → Store seed concept IDs for spiking propagation
  
Next tick (2s cycle):
  → Set neuromodulator mode (focused/broad/default) based on query
  → Fire seed concepts into association cortex (100 neurons per concept)
  → Propagate 80 steps through 2B synapses
  → Collect activated concept assemblies above threshold
  → Merge with BFS results: confirmed / explicit / emergent
  → Update snapshot with confidence-tagged associations
  
Next dialogue query:
  → LLM gets confidence-tagged associations:
    [confirmed] kv cache — found by both BFS and spiking
    [emergent] matrix multiplication — found by spiking only (lateral)
    [explicit] floating point to integer — found by BFS only
```

## Section 1: Spiking Recall Pipeline

When `recall_chain_bidirectional` runs (BFS), it stores the discovered concept IDs as seeds for spiking propagation on the `SpikingBrain`.

On the next tick cycle, the tick thread:

1. Reads the pending spiking seeds from `SpikingBrain`
2. For each seed concept, looks up its cell assembly in `ConceptRegistry` and injects `stim_current` (5.0) into all 100 neurons of that assembly in the association cortex region
3. Runs `network.step()` for 80 iterations, allowing spikes to propagate through intra-region and inter-region synapses
4. After propagation, scans all concept assemblies in the registry. For each assembly, counts how many neurons fired during the propagation window. If the fire count exceeds a threshold (e.g., 10 out of 100 neurons = 10%), that concept is considered "activated"
5. Collects activated concept names and their activation strength (fire count / assembly size)
6. Resets neuron state after propagation to avoid interference with future ticks

### Key Details

- Spiking propagation runs on the tick thread, NOT on the request thread — no latency impact
- The 80-step window allows signals to traverse 2-3 brain regions (visual → association → prefrontal)
- Seed injection uses the existing `inject_current` method on `BrainRegion`
- Neuron reset after propagation: zero out `i_ext` and optionally reset voltages to resting potential
- The `stim_current` of 5.0 is strong enough to guarantee the seed neurons fire, creating a spike wavefront

### Files Changed

- `brain-spiking/src/lib.rs` — add `pending_spiking_seeds: Option<Vec<usize>>` field, `set_spiking_seeds()`, `run_spiking_recall() -> Vec<(String, usize)>` method
- `brain-spiking/src/knowledge.rs` — after BFS recall, store seed concept IDs on the brain
- `brain-cognition/src/state.rs` — tick thread checks for pending spiking seeds and runs propagation

## Section 2: Neuromodulator-Driven Recall Modes

Before firing seed concepts, set the neuromodulator state to control propagation dynamics:

- **Focused mode (high acetylcholine = 2.0):** Strengthens local connections, suppresses noise. Produces precise, closely-related associations. Triggered when query matches 1 specific concept.
- **Broad mode (high norepinephrine = 2.0):** Lowers effective activation threshold by increasing gain. Allows weaker connections to propagate. Produces more distant, creative associations. Triggered when query matches 2+ concepts from different topics.
- **Default mode (balanced, all 1.0):** Standard propagation. Fallback.

### Mode Selection Logic

In `recall_chain_bidirectional`, after matching concepts to the query:
- Count how many distinct topics the matched concepts span (using topic provenance)
- If 2+ topics → Broad mode
- If 1 topic or 1 concept → Focused mode
- Otherwise → Default mode

Store the selected mode alongside the spiking seeds.

### Implementation

The existing `network.modulators` struct has `acetylcholine`, `norepinephrine`, etc. Before propagation:
1. Save current modulator state
2. Set modulators for the selected mode
3. Run propagation
4. Restore original modulator state

### Files Changed

- `brain-spiking/src/lib.rs` — `run_spiking_recall` sets modulators before propagation
- `brain-spiking/src/knowledge.rs` — `recall_chain_bidirectional` determines mode from topic provenance

## Section 3: Snapshot Format and LLM Integration

### Snapshot Changes

`BrainSnapshot` gains:
- `spiking_associations: Vec<(String, usize)>` — concepts activated by spiking propagation
- `recall_mode: String` — "focused", "broad", or "default"

### Merge Logic

After spiking propagation completes, merge BFS and spiking results:

```
For each association:
  - In BFS AND spiking → tag as [confirmed], weight = max(bfs_weight, spiking_weight) * 1.5
  - In BFS only → tag as [explicit], weight = bfs_weight
  - In spiking only → tag as [emergent], weight = spiking_weight * 0.7
Sort by weight descending, take top 12
```

### System Prompt Format

```
YOU LEARNED THE FOLLOWING (confirmed = high confidence, emergent = discovered by neural propagation):
  [confirmed] kv cache (strength: 160)
  [confirmed] memory usage (strength: 120)
  [emergent] matrix multiplication (strength: 45)
  [explicit] floating point to integer (strength: 80)

Use confirmed facts directly. Emergent associations suggest possible connections worth exploring.
If concepts come from different topics, explain how they connect.
```

### Files Changed

- `brain-spiking/src/lib.rs` — `BrainSnapshot` new fields, merge logic
- `brain-cognition/src/state.rs` — tick thread runs spiking recall, updates snapshot with merged results
- `brain-server/src/routes.rs` — dialogue route formats confidence-tagged associations into system prompt

## Success Criteria

1. After learning 10+ topics, spiking propagation discovers at least 1 emergent association not found by BFS
2. Confirmed associations (found by both pathways) appear in the LLM system prompt with boosted weight
3. Neuromodulator mode is automatically selected based on query (broad for multi-topic, focused for single-topic)
4. Spiking recall adds zero latency to the dialogue response (runs on tick thread)
5. Neuron state is properly reset after propagation (no interference with future queries)

## Non-Goals

- No changes to the learning pipeline (triples still learned via HashMap)
- No real-time spiking propagation during dialogue (all async via tick thread)
- No UI changes
- No background enrichment loop (Approach 3 — future work)
