# Temporal Sequence Learning — Brain-Inspired Temporal Coding

## Goal

Encode temporal order and causal direction into the spiking network using neuroscience-grounded mechanisms: ordered triple chains with asymmetric weights, STDP-timed imprinting for forward connections, and predictive recall that surfaces "what comes next" in learned sequences.

## Neuroscience Basis

- **Theta phase precession:** The hippocampus encodes sequences by firing neurons at progressively earlier theta phases. Order is stored through relative spike timing, not explicit labels.
- **Predictive coding:** The prefrontal cortex learns to predict what comes next. Causal understanding = successful prediction.
- **STDP + delays:** Pre-before-post firing strengthens forward connections, post-before-pre weakens them. This creates directional temporal chains naturally.

## Architecture

```
Transcript: "Quantization reduces precision. Lower precision reduces memory. Less memory enables larger batches."

Triple 1 (seq 0): quantization | reduces | precision
Triple 2 (seq 1): lower precision | reduces | memory
Triple 3 (seq 2): less memory | enables | larger batches

Within-triple (direct weight injection, existing):
  quantization → reduces → precision (0.8/0.8/0.4)

Between-triple chain (NEW — STDP-timed):
  precision (triple 1 object) → lower precision (triple 2 subject)
    Fire precision assembly, wait 5 steps, fire lower precision assembly
    STDP strengthens forward: precision → lower precision (potentiation)
    STDP weakens backward: lower precision → precision (depression)

During recall:
  Fire "quantization" →
    Step 1-10: activates precision, memory (direct associations)
    Step 11-30: activates larger batches (chain prediction — traveled forward through temporal links)
    → [predicted] larger batches
```

## Section 1: Ordered Triple Chains

### Persistence Format

Add sequence index to triples.log:
```
subject|relation|object|topic|timestamp|seq_index
```

The seq_index is 0, 1, 2, ... within each learning batch (one video = one sequence). Old triples without seq_index are treated as seq_index=-1 (no chaining).

### Forward Chain Imprinting

When `imprint_synapses` processes a batch of triples, also create chain connections between consecutive triples:
- Triple N's object assembly → Triple N+1's subject assembly: forward weight 0.5
- Triple N+1's subject assembly → Triple N's object assembly: backward weight 0.15

This uses the existing `strengthen_assembly_synapses` method with different deltas for forward vs backward.

### Files Changed

- `brain-spiking/src/knowledge.rs` — append_to_file includes seq_index, load_from_file parses it
- `brain-spiking/src/lib.rs` — new method `imprint_chain` processes consecutive triple pairs
- `brain-cognition/src/state.rs` — tick thread passes ordered triples to imprint_chain

## Section 2: STDP-Timed Imprinting

### Mechanism

For each consecutive triple pair (N, N+1), instead of direct weight injection for the chain connection:

1. Temporarily enable learning on the association cortex region (`learning_enabled = true`)
2. Inject current into Triple N's object assembly neurons — let them spike
3. Run 5 steps (temporal gap — mimics theta phase offset between sequence elements)
4. Inject current into Triple N+1's subject assembly neurons — let them spike
5. Run 5 more steps — STDP rule fires on the pre-before-post timing
6. Disable learning after the batch

The existing three-factor STDP in `region.rs` handles the weight updates automatically. The 5-step gap creates the right timing for potentiation (pre fires ~5 steps before post).

### Cost

Per chain link: 10 steps × ~0.003s (association cortex only) = 0.03s.
Per 12-triple batch: 11 links × 0.03s = ~0.33s. Negligible alongside the 0.003s HashMap learning.

### Reset After STDP

After the STDP imprinting sequence, reset neuron voltages and currents (but NOT synaptic weights — those are the point). Use `neurons_mut().reset()` on the association cortex.

### Files Changed

- `brain-spiking/src/lib.rs` — new method `imprint_chain_stdp` that fires assemblies with temporal offset
- `brain-spiking/src/region.rs` — no changes (STDP already implemented in step())

## Section 3: Predictive Recall

### Two-Window Propagation

During `run_spiking_recall`, split the 30-step propagation into two windows:

- **Window 1 (steps 1-10):** Direct associations. Concepts activated here are close neighbors of the seed concepts.
- **Window 2 (steps 11-30):** Chain predictions. Concepts newly activated in this window (not in Window 1) traveled through forward temporal chains.

### New Tag: `[predicted]`

Concepts activated only in Window 2 get tagged as `[predicted]` instead of `[emergent]`. They represent "what the brain expects to follow" based on learned temporal sequences.

Detection logic:
```
window1_concepts = concepts activated in steps 1-10
window2_concepts = concepts activated in steps 11-30
predicted = window2_concepts - window1_concepts - seed_concepts
emergent = (all spiking concepts) - bfs_concepts - predicted
```

### System Prompt Format

```
[confirmed] kv cache (strength: 160) — known fact
[explicit] memory usage (strength: 80) — learned directly
[predicted] enables larger batch sizes (strength: 45) — your brain predicts this follows
[emergent] sparse representations (strength: 35) — discovered through neural pathways
```

LLM instruction: "Predicted associations represent what you learned typically follows. Use them to explain consequences and next steps."

### Files Changed

- `brain-spiking/src/lib.rs` — `run_spiking_recall` tracks two windows, returns predicted vs emergent
- `brain-cognition/src/state.rs` — merge logic adds `[predicted]` tag
- `brain-server/src/routes.rs` — system prompt includes `[predicted]` tag explanation

## Success Criteria

1. Consecutive triples from the same video have forward chain connections in the CSR
2. STDP-timed imprinting creates asymmetric weights (forward > backward) between chain links
3. Spiking recall produces `[predicted]` concepts that follow the learned temporal order
4. The LLM uses predicted concepts to explain consequences ("this leads to...")
5. Persistence format includes seq_index, backward compatible with old triples

## Non-Goals

- No explicit "causes" or "precedes" edge labels (temporal order is encoded in spike timing, not labels)
- No theta oscillation simulation (we approximate phase offset with 5-step delays)
- No changes to BFS recall (temporal ordering is spiking-only)
- No changes to the LLM triple extraction prompt (ordering comes from transcript sentence order)
