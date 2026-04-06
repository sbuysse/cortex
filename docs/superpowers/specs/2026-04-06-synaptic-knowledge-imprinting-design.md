# Synaptic Knowledge Imprinting

## Goal

When knowledge triples are learned, strengthen the actual synaptic weights between concept assembly neurons in the CSR matrix. This makes the 2B-synapse spiking network mirror the HashMap knowledge graph, enabling spike propagation to discover emergent associations through lateral connectivity.

## Architecture

```
learn_triple_with_topic("turboquant", "compresses", "kv cache")
  │
  ├── KnowledgeEngine: HashMap update (existing)
  │     turboquant → compresses: 0.8
  │     compresses → kv cache: 0.8
  │     turboquant → kv cache: 0.4
  │
  └── SpikingBrain::imprint_synapses (NEW)
        For each edge (subject → object):
          Find subject assembly neurons (e.g., 300-399)
          Find object assembly neurons (e.g., 700-799)
          Scan CSR rows for subject neurons
          Strengthen any synapse targeting an object neuron by +0.3
          Cap imprinted weights at ±1.0 (above normal ±0.5 clamp)

Later, during spiking recall:
  Fire "turboquant" neurons
  → Strengthened synapses activate "kv cache" assembly (above 5% threshold)
  → "kv cache" synapses activate "flash attention" assembly (second hop)
  → Emergent multi-hop chains discovered through spike propagation
```

## Section 1: Synaptic Imprinting Mechanism

### Method: `SpikingBrain::imprint_synapses(&mut self, triple: &Triple)`

For each directed edge in the triple (S→R, R→O, S→O):

1. Look up source and target concept assemblies via `self.knowledge.registry.get(name)` → `CellAssembly { start, size }`
2. Get the association cortex region's mutable synapses: `self.network.region_mut(assoc_region).synapses_mut()`
3. For each neuron in the source assembly (start..start+size), scan the CSR row:
   - `row_ptr[src]` to `row_ptr[src+1]` gives the synapse index range
   - For each synapse in that range, check if `col_idx[syn_i]` falls within the target assembly range
   - If yes: read `weights[syn_i]` via `weight_from_i16`, add delta (+0.3), cap at 1.0, write back via `weight_to_i16`
4. Track how many synapses were strengthened (for logging)

### Weight Parameters

- Imprint delta: 0.3 per learning event (accumulates with repeated learning)
- Weight cap: 1.0 (normal intra-region cap is 0.5, imprinted connections are 2x stronger)
- Edge deltas mirror HashMap: S→R gets 0.3, R→O gets 0.3, S→O gets 0.15 (proportional to 0.8/0.8/0.4)

### Files Changed

- `brain-spiking/src/lib.rs` — add `imprint_synapses` method to SpikingBrain

## Section 2: Integration with Learning Pipeline

### During Live Learning

In `brain-cognition/src/state.rs`, the tick thread's triple drain loop currently calls:
```rust
sb.knowledge.learn_triple_with_topic(triple, topic);
```

After this line, add:
```rust
sb.imprint_synapses(triple);
```

### During Startup Reload

In `SpikingBrain::new`, after `knowledge.load_from_file()` loads triples into the HashMap, replay them through `imprint_synapses`. This requires storing the loaded triples temporarily.

Change `load_from_file` to return the loaded triples (not just the count), so `SpikingBrain::new` can iterate them:
```rust
let loaded_triples = knowledge.load_from_file_with_triples(&path);
for triple in &loaded_triples {
    // imprint synapses for each loaded triple
}
```

Alternatively, simpler: add a method `knowledge.loaded_triples() -> &[(Triple, String)]` that caches the last-loaded triples, consumed once by the brain constructor.

### Files Changed

- `brain-spiking/src/knowledge.rs` — `load_from_file` also returns or caches loaded triples
- `brain-spiking/src/lib.rs` — `SpikingBrain::new` calls `imprint_synapses` for each loaded triple
- `brain-cognition/src/state.rs` — tick thread calls `imprint_synapses` after `learn_triple_with_topic`

## Section 3: Spiking Recall (No Changes)

The existing `run_spiking_recall` method is unchanged. The only difference is that now the CSR weights between concept assemblies are strengthened, so spike propagation naturally follows learned knowledge paths instead of dispersing uniformly.

### Expected Behavior

- Before imprinting: 0 emergent concepts (random dispersion)
- After imprinting: 1-5 emergent concepts per recall (strengthened pathways activated)
- Multi-hop discovery: A→B and B→C imprinted → firing A activates C through two spiking hops
- `[confirmed]` tags appear when BFS and spiking agree
- `[emergent]` tags appear when spiking finds concepts BFS didn't return (possible through lateral random connections adjacent to imprinted paths)

## Success Criteria

1. After learning 10+ topics, `run_spiking_recall` returns at least 1 activated concept (non-zero emergent/confirmed)
2. The dialogue output shows `[confirmed]` tags for concepts found by both BFS and spiking
3. Imprinting adds zero latency to learning (direct CSR write, no simulation)
4. Persisted triples are re-imprinted on startup (knowledge survives restart in both HashMap and synapses)
5. Log shows "Imprinted N synapses for triple (S, R, O)" with N > 0

## Non-Goals

- No new synapse insertion (only strengthen existing CSR connections)
- No STDP simulation for learning
- No changes to the recall pipeline
- No changes to the system prompt format
