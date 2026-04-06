# Cortex vs. Neuroscience: A Point-by-Point Comparison

This document compares every architectural component of Cortex against current neuroscience research, noting where our implementation aligns with biology, where it simplifies, and where it diverges.

---

## 1. Neuron Model: ALIF (Adaptive Leaky Integrate-and-Fire)

### What the brain does
Biological neurons integrate incoming currents on their membrane, leak charge over time, and fire an action potential when voltage crosses a threshold. After firing, there is a refractory period and spike-frequency adaptation — the neuron becomes temporarily harder to excite. Under sustained input, initial firing at hundreds of Hz drops to tens of Hz over tens of milliseconds.

### What Cortex does
We use ALIF neurons in a Structure-of-Arrays layout. Each neuron has voltage `v`, adaptation variable `a`, and external current `i_ext`. On each step: `v = v * v_decay + i_ext - a`. If `v >= threshold`, the neuron fires, `v` resets, and `a` increases. Adaptation decays each step. We skip quiescent neurons (v=0, i_ext=0, a=0) for performance.

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Membrane integration | Continuous ODE | Discrete step (Euler) | Approximation |
| Leak | Passive ion channels | `v * v_decay` (exponential decay) | Good match |
| Spike-frequency adaptation | Ca²⁺-dependent K⁺ channels | `a` variable increases on spike | Good match |
| Refractory period | Na⁺ channel inactivation (~2ms) | Implicit via voltage reset | Simplified |
| Dendritic computation | Nonlinear integration per dendrite branch | Single compartment | Major simplification |
| Ion channel diversity | Dozens of channel types | None modeled | Major simplification |

**Verdict:** ALIF is a well-established simplification. It captures the two most important dynamic properties (leak and adaptation) while being 1000x cheaper than Hodgkin-Huxley. Recent research confirms ALIF outperforms standard LIF on temporal tasks. Our implementation is **scientifically sound**.

**References:**
- [ALIF Neuron Overview](https://www.emergentmind.com/topics/adaptive-leaky-integrate-and-fire-alif-neuron)
- [Adaptive Generalized LIF for Hippocampal Neurons](https://link.springer.com/article/10.1007/s11538-023-01206-8)

---

## 2. Cell Assemblies: Concept Representation

### What the brain does
Hebb (1949) proposed that concepts are represented by **cell assemblies** — groups of strongly interconnected neurons whose co-activation represents a specific memory or concept. When part of the assembly is activated, recurrent connections complete the pattern (pattern completion). Modern evidence confirms this: cell assemblies with synchronous firing represent specific memories, with typical sizes of 50-300 neurons per concept in hippocampus and cortex.

### What Cortex does
Each concept (e.g., "kv cache", "turboquant") gets a dedicated `CellAssembly` of 100 neurons in the association cortex. Assemblies are allocated sequentially (neurons 0-99, 100-199, etc.) in the `ConceptRegistry`. Activation is checked by counting how many neurons in an assembly fired (threshold: 5%).

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Assembly size | 50-300 neurons | 100 neurons (fixed) | Reasonable |
| Formation | Emerges through co-activation + STDP | Pre-allocated by registry | **Divergence** |
| Overlap | Neurons belong to multiple assemblies | Non-overlapping, sequential blocks | **Divergence** |
| Pattern completion | Recurrent connections auto-complete | No — activation is checked post-hoc | **Divergence** |
| Sparse coding | ~1-5% of neurons active for any concept | Fixed blocks, no sparsity constraint | Simplified |

**Verdict:** Our assemblies are **structurally correct** (groups of neurons = concepts) but **functionally simplified**. The three divergences are significant:

1. **Pre-allocation vs. emergence:** In biology, assemblies form through experience. In Cortex, we allocate them by name. This is pragmatic — emergent assembly formation requires extensive STDP training that we bypass with the HashMap.

2. **No overlap:** Biological assemblies share neurons, enabling generalization (the neuron for "red" might be in both "apple" and "fire truck" assemblies). Our non-overlapping blocks prevent this. This limits emergent generalization.

3. **No pattern completion:** In CA3, activating 30% of an assembly's neurons triggers the full pattern through recurrent connections. Cortex doesn't have this — we check activation after the fact rather than letting the network complete it.

**References:**
- [Cell Assemblies - Scholarpedia](http://www.scholarpedia.org/article/Cell_assemblies)
- [Hebbian Theory - Wikipedia](https://en.wikipedia.org/wiki/Hebbian_theory)
- [2024 Cell Assembly Connectivity Model](https://www.sciencedaily.com/releases/2024/01/240117143741.htm)

---

## 3. Synaptic Connectivity: CSR with 2B Connections

### What the brain does
The human brain has ~86 billion neurons and ~150 trillion synapses (~1,750 synapses per neuron on average, though cortical pyramidal neurons have 5,000-10,000). Connectivity is sparse (~0.01% of all possible pairs) but highly structured — nearby neurons connect more frequently (small-world topology), and specific pathways connect brain regions (white matter tracts).

### What Cortex does
2 million neurons with 2 billion CSR synapses (~1,000 synapses per neuron). Connectivity is random (Erdos-Renyi within regions) with structured inter-region feedforward connections. CSR format stores `row_ptr` (u64), `col_idx` (u32), `weights` (i16), achieving 18x speedup over dense matrices with 95% memory savings.

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Scale | 86B neurons, 150T synapses | 2M neurons, 2B synapses | 1/43,000x scale |
| Fanout per neuron | 5,000-10,000 | ~1,000 | Same order of magnitude |
| Intra-region connectivity | Distance-dependent, small-world | Random (Erdos-Renyi) | **Divergence** |
| Inter-region connectivity | White matter tracts, laminar-specific | Random with controlled probability | Simplified |
| Weight representation | Continuous (analog), stochastic | i16 quantized (32,767 levels) | Good approximation |
| Storage format | Not applicable (biological) | CSR (standard for SNN simulators) | Industry standard |

**Verdict:** The CSR storage and scale are **standard practice** for SNN research. The random intra-region connectivity is the main divergence — biological networks have distance-dependent and experience-shaped topology. Our synaptic imprinting partially compensates by strengthening specific pathways post-hoc.

**References:**
- [Brain Connectivity Matrices](https://www.nature.com/articles/s41597-022-01596-9)
- [CSR for Brain Event Processing](https://brainevent.readthedocs.io/Tutorials/02_sparse_matrices.html)

---

## 4. Spike-Timing-Dependent Plasticity (STDP)

### What the brain does
STDP is the primary mechanism for learning in biological neural networks. If a presynaptic neuron fires 1-20ms **before** the postsynaptic neuron, the synapse is strengthened (LTP). If the order is reversed, the synapse is weakened (LTD). The timing window is asymmetric — potentiation is typically stronger and has a shorter window (~10ms) than depression (~20ms).

Modern research shows STDP is actually more complex: **triplet rules** (pre-post-pre, post-pre-post) explain data better than simple pairs, and the effect depends on **dendritic location** — synapses on distal dendrites show different STDP curves than proximal ones.

### What Cortex does
We implement three-factor STDP: the standard pre/post timing rule modulated by a neuromodulatory "third factor" (eligibility traces). The `update_stdp` function computes weight change from timing difference. Eligibility traces mark synapses for future modification, and the actual weight change happens when a neuromodulator signal (dopamine, acetylcholine) arrives.

We also added **synaptic imprinting** — directly strengthening CSR weights between concept assemblies when triples are learned. This bypasses STDP timing entirely for known associations.

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Pre-before-post = LTP | Yes, ~10ms window | Yes | Good match |
| Post-before-pre = LTD | Yes, ~20ms window | Yes | Good match |
| Asymmetric window | Potentiation stronger, shorter | Configurable in PlasticityParams | Good match |
| Triplet rules | Needed for full accuracy | Not implemented (pairwise only) | Simplified |
| Dendritic location dependence | Different STDP curves per dendrite | Not modeled (single compartment) | Not applicable |
| Third factor (neuromodulation) | Dopamine gates eligibility traces | Yes — eligibility × learning_modulator | **Strong match** |
| Eligibility traces | Decays over ~1s, molecular basis | Decays every 10 steps, i16 representation | Good match |
| Synaptic imprinting | No direct biological equivalent | Direct weight injection for known triples | **Novel addition** |

**Verdict:** Our STDP implementation is **biologically well-grounded**, especially the three-factor learning rule which matches the state-of-the-art in computational neuroscience (Frémaux & Gerstner, 2016). The synaptic imprinting is a pragmatic shortcut without direct biological basis — it's more like "instruction-based learning" than experience-based plasticity.

**References:**
- [STDP: A Comprehensive Overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC3395004/)
- [Three-Factor Learning Rules](https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/)
- [Eligibility Traces: Experimental Support](https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/)

---

## 5. Synaptic Scaling (Homeostatic Plasticity)

### What the brain does
STDP is a positive feedback mechanism — once a synapse is strengthened, it's easier to strengthen further, risking runaway excitation. The brain counteracts this with **homeostatic synaptic scaling**: neurons detect their own firing rate via calcium sensors and multiplicatively scale ALL synaptic weights up or down to maintain a target firing rate. This is a slow process (hours to days) that preserves relative weight ratios while preventing saturation.

### What Cortex does
We use **multiplicative synaptic scaling**: when a neuron's total synaptic drive exceeds a target (0.5 for normal operation, 1.5 during spiking recall), all incoming currents are scaled proportionally by `drive * (target / actual)`. This preserves the relative strength ratios between strong (imprinted) and weak (random) connections while preventing runaway excitation.

During normal operation, the target is 0.5 — strong enough for basic spike propagation. During spiking recall, the target is raised to 1.5 to let imprinted knowledge connections (0.8-1.0 weight) dominate over random background connections (~0.01 weight). The key property: a neuron receiving 0.8 from an imprinted synapse and 0.01 from a random synapse will always fire preferentially toward the imprinted target, regardless of the scaling target.

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Mechanism | Multiplicative scaling of all synapses | Multiplicative scaling per neuron per step | **Good match** |
| Timescale | Hours to days | Every timestep (instantaneous) | Simplified |
| Preserves weight ratios | Yes | Yes — `drive * (target / actual)` | **Good match** |
| Prevents runaway excitation | Yes | Yes | Good match |
| Target firing rate | ~1-5 Hz in cortex | Implicit via target drive (0.5 / 1.5) | Simplified |

**Verdict:** This is now a **strong match** with biology. The multiplicative mechanism is the same principle — scale all inputs proportionally to maintain stability. The main simplification is timescale: biological scaling operates over hours/days via receptor trafficking, while ours is instantaneous. However, the functional effect is equivalent: imprinted connections (0.8-1.0) naturally dominate random connections (~0.01) because their relative strength is preserved through the scaling.

**Result:** With multiplicative scaling, spiking recall discovers 7+ emergent associations (e.g., "self-attention", "ai architecture") compared to 0 with the old hard clamp. The biological approach works better.

**References:**
- [Homeostatic Synaptic Plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC3249629/)
- [Unraveling Mechanisms of Homeostatic Plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC3021747/)
- [Interplay Between Homeostatic Scaling and Structural Plasticity](https://elifesciences.org/articles/88376)

---

## 6. Brain Regions and Hierarchy

### What the brain does
The cortex is organized hierarchically. Sensory cortices (V1, A1) process raw input. Association cortices integrate across modalities. The prefrontal cortex (PFC) sits at the top, generating predictions and maintaining working memory. Information flows bottom-up (feedforward) with increasing abstraction, and top-down (feedback) with predictions and attention. Higher regions use higher-frequency oscillations (~11Hz in V4, ~15Hz in parietal, ~19Hz in PFC).

### What Cortex does
We have 10 brain regions: Visual cortex (200K), Auditory cortex (200K), Association cortex (500K), Predictive cortex (200K), Hippocampus (300K), PFC (200K), Amygdala (100K), Motor cortex (100K), Brainstem (50K), Cerebellum (150K). Feedforward connections go Visual/Auditory → Association → PFC. Feedback connections go PFC → Association → Predictive → Visual/Auditory.

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Hierarchical organization | Yes, well-established | Yes, 10 regions with hierarchy | Good match |
| Feedforward pathway | V1 → V2 → V4 → IT → PFC | Visual → Association → PFC | Simplified but correct |
| Feedback pathway | PFC → lower regions | PFC → Association → Predictive → sensory | Good match |
| Region specialization | Highly specialized subregions | Single population per region | Simplified |
| Hippocampal subfields | DG → CA3 → CA1 (distinct functions) | Single hippocampus region | **Divergence** |
| Laminar structure | 6 cortical layers with distinct roles | Not modeled | Major simplification |
| Oscillation frequencies | Region-specific (theta, beta, gamma) | Not modeled | Not implemented |

**Verdict:** The region hierarchy is **directionally correct** — the right regions exist with the right connectivity pattern. The main simplifications are: no laminar structure (layers within each region), no hippocampal subfields (DG/CA3/CA1 have very different functions), and no oscillations. The hippocampal simplification is notable because CA3's autoassociative recurrent connections are key to pattern completion, which our hippocampus doesn't do.

**References:**
- [Brain Hierarchy and Frequency Waves](https://news.mit.edu/2020/information-flows-through-brains-heirarchy-higher-regions-use-higher-frequency-waves-0910)
- [Higher Cortical Functions](https://nba.uth.tmc.edu/neuroscience/m/s4/chapter09.html)

---

## 7. Neuromodulation: Four Modulators

### What the brain does
Four major neuromodulatory systems regulate brain-wide processing:
- **Dopamine** (VTA/SNc): Reward prediction error, reinforcement learning, motivation
- **Acetylcholine** (basal forebrain): Attention, encoding mode, signal-to-noise ratio
- **Norepinephrine** (locus coeruleus): Arousal, exploration/exploitation tradeoff, gain modulation
- **Serotonin** (raphe nuclei): Mood, behavioral inhibition, patience

These modulators shape learning and recall: high ACh during waking = encoding mode; low ACh during sleep = consolidation mode. Dopamine gates which memories get replayed and consolidated.

### What Cortex does
Four scalar modulators: `dopamine`, `acetylcholine`, `norepinephrine`, `serotonin`. Each decays toward baseline (1.0) per timestep. Dopamine contributes to `learning_modulator()` (gates STDP via three-factor rule). Norepinephrine contributes to `gain_modulator()` (scales neuronal excitability). Serotonin modulates `exploration_noise()`. During spiking recall, we set ACh=2.0 for focused mode and NE=2.0 for broad mode.

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Dopamine = reward/learning | Yes | Yes — gates STDP via learning_modulator | Good match |
| ACh = attention/encoding | Yes | Yes — used for focused recall mode | Good match |
| NE = arousal/exploration | Yes | Yes — gain modulator + broad recall | Good match |
| Serotonin = mood/inhibition | Yes | Partially — exploration noise only | Simplified |
| Separate brain nuclei | Each from different brainstem region | Single brainstem region | Simplified |
| Tonic vs. phasic release | Different timescales, different functions | Single scalar per modulator | Simplified |
| ACh encoding vs. consolidation | High ACh = encode, low ACh = consolidate | Not modeled (ACh only used for recall) | **Gap** |

**Verdict:** The four-modulator system is **well-chosen and correctly mapped** to their biological functions. The main gap is the encoding/consolidation switch — in biology, ACh levels cycle between waking (high = encode) and sleep (low = consolidate). Our sleep consolidation module exists but doesn't use ACh modulation to control the mode switch.

**References:**
- [Memory Trace Replay and Neuromodulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4712256/)
- [Neuromodulators in Learning and Memory](https://www.ijbcp.com/index.php/ijbcp/article/view/3707)
- [ACh and Memory Consolidation](https://pubmed.ncbi.nlm.nih.gov/10461198/)

---

## 8. Knowledge Recall: BFS vs. Spreading Activation

### What the brain does
The brain retrieves memories through **spreading activation** — when a concept is activated, excitation spreads through associated connections, activating related concepts with decreasing strength. This is not BFS (which explores uniformly layer by layer) but rather a **weighted, parallel, continuous** process where strongly connected concepts activate faster than weakly connected ones. The process is modulated by attention (ACh) and arousal (NE).

Collins & Loftus (1975) established the spreading activation theory: "When a concept is processed, activation spreads out along the paths of the network in a decreasing gradient."

### What Cortex does
We have **two parallel recall systems**:
1. **HashMap BFS** (instant, 0ms): Explicit graph traversal through learned association edges. Bidirectional for cross-domain queries.
2. **Spiking propagation** (0.1s): Fire seed concepts into the spiking network, let activation spread through imprinted + random synapses for 30 steps.

### Comparison
| Aspect | Biology | Cortex BFS | Cortex Spiking | 
|--------|---------|------------|----------------|
| Mechanism | Spreading activation via synapses | Graph traversal via HashMap | Spike propagation via CSR | 
| Parallel | Yes, all connections simultaneously | No, sequential BFS layers | Yes, all synapses simultaneously |
| Weighted | Yes, stronger connections activate faster | Yes, sorted by weight | Yes, stronger synapses drive more current |
| Continuous | Yes, graded activation | No, discrete hops | Yes, continuous voltage integration |
| Modulated by attention | Yes (ACh) | No | Yes (neuromodulator modes) |
| Speed | ~200-500ms for recall | 0ms | 100ms |

**Verdict:** Our **spiking recall is a much closer match** to biological spreading activation than the BFS. The BFS is a pragmatic shortcut that guarantees finding known associations. The dual-pathway approach (BFS for reliability + spiking for discovery) is actually a reasonable engineering choice — the BFS handles explicit knowledge while the spiking network handles emergent/implicit knowledge.

> The BFS is NOT biologically plausible. It's a database lookup. But the spiking propagation IS biologically plausible — it's literally spreading activation through synaptic connections.

**References:**
- [Spreading Activation Theory](https://en.wikipedia.org/wiki/Spreading_activation)
- [Collins & Loftus (1975)](https://www.sciencedirect.com/science/article/abs/pii/S0022537183902013)
- [Associative Memory Cells](https://pmc.ncbi.nlm.nih.gov/articles/PMC5806053/)

---

## 9. Temporal Sequence Learning: STDP-Timed Chains

### What the brain does
The hippocampus encodes temporal sequences through **theta phase precession**: neurons representing sequential locations fire at progressively earlier phases of the theta oscillation (~8Hz). This compresses a behavioral sequence (seconds) into a single theta cycle (~125ms), with a compression ratio up to 10:1. The timing differences trigger STDP, strengthening forward connections (A→B→C) and weakening backward connections.

### What Cortex does
When consecutive triples are learned, we fire the object assembly of triple N, wait 5 steps, then fire the subject assembly of triple N+1. The STDP rule (already running in the association cortex) strengthens the forward connections automatically. We then split spiking recall into two windows: direct (steps 1-10) and predicted (steps 11-30) to detect chain-following concepts.

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Temporal compression | 10:1 via theta phase precession | 5-step gap between firings | Functional analog |
| STDP for sequence learning | Pre-before-post strengthens forward | Yes, same mechanism | Good match |
| Theta oscillation | 8Hz rhythm provides temporal framework | Not modeled (fixed step delays) | **Divergence** |
| Replay during sleep | Compressed replay consolidates sequences | Not implemented for chain sequences | Gap |
| Prediction through chains | Predictive coding in PFC | Two-window recall (direct vs. predicted) | Novel approach |
| Bidirectional asymmetry | Forward >> backward connections | Forward 0.5 delta vs. backward through STDP LTD | Good match |

**Verdict:** Our STDP-timed chain imprinting captures the **essential mechanism** — temporal offset driving directional STDP. The 5-step gap is a functional analog of theta phase offset. The main divergence is that we don't model theta oscillations explicitly, which means we lack the natural temporal framework that the brain uses to organize sequences. Our two-window approach for detecting predictions is a novel engineering solution without direct biological analog, but it serves the same functional purpose as predictive coding.

**References:**
- [Theta Phase Precession and Memory](https://www.nature.com/articles/s41562-024-01983-9)
- [Theta-Phase Dependent Neuronal Coding in Humans](https://www.nature.com/articles/s41467-021-25150-0)
- [Sequence Anticipation and STDP](https://www.nature.com/articles/s41467-023-40651-w)

---

## 10. Knowledge Representation: HashMap + Synaptic Weights

### What the brain does
The brain does NOT have a symbolic knowledge graph. All knowledge is encoded in **synaptic weights** — the strength and topology of connections between neurons IS the memory. There is no separate "database" of facts. Recall is reconstruction, not retrieval.

### What Cortex does
We maintain **two parallel representations**, with spiking as the primary source:
1. **CSR synaptic weights** (primary): Actual neural connectivity. Imprinted from learned triples (803 synapses per triple batch). Knowledge lives in the weights — the brain-like approach.
2. **HashMap association matrix** (supplement): Explicit symbolic graph for fast BFS lookup. Provides `[explicit]` facts that the spiking network may miss.

The **spiking network is the primary knowledge source**. During recall:
- Spiking results (emergent, predicted) are reported at full weight
- BFS results supplement as `[explicit]` — facts the spiking network didn't activate
- When both agree → `[confirmed]` (1.5x boosted weight)

### Comparison
| Aspect | Biology | Cortex | Match |
|--------|---------|--------|-------|
| Storage medium | Synaptic weights only | CSR weights (primary) + HashMap (supplement) | **Converging** |
| Symbolic representation | None — all subsymbolic | HashMap exists but is secondary | Improving |
| Recall mechanism | Spreading activation via synapses | Spiking propagation (primary) + BFS (supplement) | **Good match** |
| Persistence | Synaptic consolidation during sleep | triples.log replayed into CSR on startup | Engineering choice |
| Graceful degradation | Partial cues trigger full recall | Spiking: partial activation works | Good match |
| Emergent discovery | Lateral connections find novel associations | Yes — 7 emergent concepts found via neural pathways | **Strong match** |

**Verdict:** This has **improved significantly** from the initial architecture. The spiking network now:
- Carries knowledge in actual synaptic weights (803 imprinted synapses per learning batch)
- Is the primary recall source (emergent/predicted at full weight)
- Discovers cross-domain associations the HashMap can't find (e.g., "self-attention", "ai architecture" discovered when querying "TurboQuant")

The HashMap remains as a reliability scaffold — it guarantees that explicitly learned facts are always available, even when the spiking network's 5% activation threshold filters them out. This dual-pathway approach (fast symbolic + rich neural) may actually be closer to how the brain works than pure spiking: the hippocampus provides fast, explicit recall while the cortex provides slow, associative recall.

---

## Summary: Alignment Scorecard

| Component | Biology Alignment | Notes |
|-----------|:-:|-------|
| ALIF neurons | High | Standard, well-validated model |
| Cell assemblies | Medium | Correct concept, but pre-allocated not emergent |
| CSR connectivity | High | Industry standard for SNN simulators |
| Three-factor STDP | High | Matches Frémaux & Gerstner (2016) |
| Synaptic scaling | **High** | Multiplicative scaling preserves weight ratios |
| Brain region hierarchy | Medium | Right regions, missing laminar structure |
| Neuromodulation | High | Four modulators correctly mapped |
| BFS recall | Low | Symbolic supplement, not biological |
| Spiking recall | **High** | Primary recall via spreading activation |
| STDP chain imprinting | Medium | Right mechanism, missing theta oscillations |
| Knowledge store | **Medium** | Spiking-primary with HashMap supplement |

## Potential Improvements (Ordered by Impact)

1. ~~Replace hard clamp with multiplicative synaptic scaling~~ — **DONE.** Multiplicative scaling preserves weight ratios. Result: 7 emergent associations discovered.

2. ~~Spiking-primary recall~~ — **DONE.** Spiking network is now the primary knowledge source. BFS supplements with explicit facts.

3. **Overlapping cell assemblies** — Allow neurons to participate in multiple concepts. Would enable generalization (shared features). High effort.

4. **Hippocampal subfields (DG/CA3/CA1)** — CA3 autoassociative recall would enable true pattern completion. High effort but high impact.

5. **Theta oscillation framework** — Add a ~8Hz rhythm to provide temporal structure for sequence encoding. Medium effort.
