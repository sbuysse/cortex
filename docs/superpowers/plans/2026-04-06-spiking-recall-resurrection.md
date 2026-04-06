# Spiking Recall Resurrection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fire BFS-discovered concepts into the 2B-synapse spiking network to discover emergent lateral associations, provide dual-pathway confidence signals, and use neuromodulators to control recall breadth.

**Architecture:** BFS runs instantly and stores seed concept IDs. On the next tick, the spiking network fires those seeds, propagates for 80 steps, and collects activated concepts. Results are merged with BFS output using confidence tags (confirmed/explicit/emergent). Neuromodulators are set to focused or broad mode before propagation.

**Tech Stack:** Rust, brain-spiking (lib.rs, knowledge.rs, neuromodulation.rs), brain-cognition (state.rs), brain-server (routes.rs)

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `crates/brain-spiking/src/lib.rs` | SpikingBrain: spiking seeds, run_spiking_recall, BrainSnapshot | Modify |
| `crates/brain-spiking/src/knowledge.rs` | Store seed IDs + recall mode after BFS | Modify |
| `crates/brain-cognition/src/state.rs` | Tick thread: run spiking recall, merge, update snapshot | Modify |
| `crates/brain-server/src/routes.rs` | Confidence-tagged system prompt format | Modify |
| `crates/brain-spiking/tests/knowledge_test.rs` | Test for spiking recall | Modify |

---

### Task 1: Spiking Seeds and Recall Mode on KnowledgeEngine

**Files:**
- Modify: `crates/brain-spiking/src/knowledge.rs`
- Modify: `crates/brain-spiking/tests/knowledge_test.rs`

- [ ] **Step 1: Write test for seed storage after BFS**

Append to `crates/brain-spiking/tests/knowledge_test.rs`:

```rust
#[test]
fn test_spiking_seeds_after_bfs() {
    let mut engine = KnowledgeEngine::new(0, 50000, 100);

    engine.learn_triple_with_topic(
        &Triple::new("turboquant", "compresses", "kv cache"), "TurboQuant");
    engine.learn_triple_with_topic(
        &Triple::new("flash attention", "optimizes", "kv cache"), "FlashAttention");

    let mut net = brain_spiking::network::SpikingNetwork::new();
    let _chain = engine.recall_chain_bidirectional(&mut net, "turboquant flash attention", 10);

    let (seeds, mode) = engine.take_spiking_seeds();
    assert!(!seeds.is_empty(), "Should have spiking seeds after BFS");
    assert_eq!(mode, "broad", "Multi-topic query should trigger broad mode");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p brain-spiking --test knowledge_test test_spiking_seeds_after_bfs`
Expected: FAIL — `take_spiking_seeds` doesn't exist.

- [ ] **Step 3: Add seed storage fields and methods to KnowledgeEngine**

In `crates/brain-spiking/src/knowledge.rs`, add two fields to the struct:

```rust
    /// Concept IDs to fire into spiking network on next tick.
    spiking_seeds: Vec<usize>,
    /// Recall mode: "focused", "broad", or "default".
    spiking_mode: String,
```

Initialize them in `new()`:
```rust
            spiking_seeds: Vec::new(),
            spiking_mode: String::new(),
```

Add methods:

```rust
    /// Take the pending spiking seeds (clears them). Returns (seed_concept_ids, mode).
    pub fn take_spiking_seeds(&mut self) -> (Vec<usize>, String) {
        let seeds = std::mem::take(&mut self.spiking_seeds);
        let mode = std::mem::take(&mut self.spiking_mode);
        (seeds, mode)
    }

    /// Check if spiking seeds are pending.
    pub fn has_spiking_seeds(&self) -> bool {
        !self.spiking_seeds.is_empty()
    }
```

- [ ] **Step 4: Store seeds at the end of `recall_chain_bidirectional`**

In `recall_chain_bidirectional`, just before the noise filtering at the end, add:

```rust
        // Store seeds for spiking propagation
        self.spiking_seeds = all_ids.clone();

        // Determine recall mode from topic provenance
        let mut seed_topics: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (_, name) in &matched_ids {
            for topic in self.registry.get_topics(name) {
                seed_topics.insert(topic);
            }
        }
        self.spiking_mode = if seed_topics.len() >= 2 {
            "broad".to_string()
        } else if !seed_topics.is_empty() {
            "focused".to_string()
        } else {
            "default".to_string()
        };
```

Where `all_ids` is the vec of matched concept IDs already computed in the function.

Note: also store seeds in the unidirectional fallback path. In `recall_chain`, at the start after `start_ids` is built, add:
```rust
        self.spiking_seeds = start_ids.clone();
        self.spiking_mode = "focused".to_string();
```

Make `recall_chain` take `&mut self` instead of `&self` (it already takes `&self` — change to `&mut self`).

- [ ] **Step 5: Run test**

Run: `cargo test -p brain-spiking --test knowledge_test`
Expected: All tests pass including the new one.

- [ ] **Step 6: Commit**

```bash
git add crates/brain-spiking/src/knowledge.rs crates/brain-spiking/tests/knowledge_test.rs
git commit -m "feat: store spiking seeds + recall mode after BFS recall"
```

---

### Task 2: Spiking Recall Method on SpikingBrain

**Files:**
- Modify: `crates/brain-spiking/src/lib.rs`

- [ ] **Step 1: Add spiking_associations and recall_mode to BrainSnapshot**

In `crates/brain-spiking/src/lib.rs`, add to `BrainSnapshot`:

```rust
    /// Concepts activated by spiking propagation (emergent associations).
    pub spiking_associations: Vec<(String, usize)>,
    /// Recall mode used: "focused", "broad", "default", or empty.
    pub recall_mode: String,
```

- [ ] **Step 2: Implement `run_spiking_recall` on SpikingBrain**

Add this method to `impl SpikingBrain`:

```rust
    /// Fire seed concepts into the spiking network and collect activated concepts.
    /// Returns (activated_concepts, mode_used). Resets neurons after propagation.
    pub fn run_spiking_recall(&mut self) -> Option<(Vec<(String, usize)>, String)> {
        let (seeds, mode) = self.knowledge.take_spiking_seeds();
        if seeds.is_empty() {
            return None;
        }

        let assoc_region = self.knowledge.concept_region;

        // Save and set neuromodulators based on mode
        let saved_mods = self.network.modulators.clone();
        match mode.as_str() {
            "focused" => {
                self.network.modulators.acetylcholine = 2.0;
                self.network.modulators.norepinephrine = 0.8;
            }
            "broad" => {
                self.network.modulators.norepinephrine = 2.0;
                self.network.modulators.acetylcholine = 0.8;
            }
            _ => {} // default: leave as-is
        }

        // Inject current into seed concept assemblies
        for &seed_id in &seeds {
            // Convert concept_id back to assembly start: concept_id * assembly_size
            let asm_start = seed_id * 100;
            for neuron in asm_start..(asm_start + 100) {
                if neuron < self.network.region(assoc_region).num_neurons() {
                    self.network.inject_current(assoc_region, neuron, self.knowledge.stim_current());
                }
            }
        }

        // Propagate for 80 steps
        let mut fired_in_assoc: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for _step in 0..80 {
            self.network.step();
            // Count spikes in association cortex
            let spikes = self.network.region(assoc_region).last_spikes();
            for &neuron_idx in spikes {
                *fired_in_assoc.entry(neuron_idx).or_insert(0) += 1;
            }
        }

        // Scan concept assemblies for activation
        let mut activated: Vec<(String, usize)> = Vec::new();
        let seed_set: std::collections::HashSet<usize> = seeds.iter().copied().collect();

        for name in self.knowledge.registry.concept_names() {
            if let Some(asm) = self.knowledge.registry.get(name) {
                let concept_id = asm.start / 100;
                if seed_set.contains(&concept_id) { continue; } // skip seeds themselves

                // Count how many neurons in this assembly fired
                let fire_count: usize = (asm.start..asm.start + asm.size)
                    .filter(|n| fired_in_assoc.contains_key(n))
                    .count();

                // Threshold: 10% of assembly must fire
                if fire_count >= asm.size / 10 {
                    let strength = (fire_count as f32 / asm.size as f32 * 100.0) as usize;
                    activated.push((name.to_string(), strength));
                }
            }
        }

        // Sort by activation strength
        activated.sort_by(|a, b| b.1.cmp(&a.1));
        activated.truncate(10);

        // Reset neurons to avoid interference
        for i in 0..self.network.num_regions() {
            self.network.region_mut(i).neurons_mut().reset();
        }

        // Restore neuromodulators
        self.network.modulators = saved_mods;

        tracing::info!("Spiking recall ({} mode): {} seeds fired, {} concepts activated",
            mode, seeds.len(), activated.len());

        Some((activated, mode))
    }
```

- [ ] **Step 3: Add `stim_current` accessor to KnowledgeEngine**

In `crates/brain-spiking/src/knowledge.rs`, add:

```rust
    /// Stimulation current for spiking seed injection.
    pub fn stim_current(&self) -> f32 {
        self.stim_current
    }
```

- [ ] **Step 4: Add `region_mut` to SpikingNetwork if not present**

Check `crates/brain-spiking/src/network.rs` for a `region_mut` method. If it doesn't exist, add:

```rust
    pub fn region_mut(&mut self, id: BrainRegionId) -> &mut BrainRegion {
        &mut self.regions[id]
    }
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p brain-spiking`
Expected: compiles cleanly.

- [ ] **Step 6: Commit**

```bash
git add crates/brain-spiking/src/lib.rs crates/brain-spiking/src/knowledge.rs crates/brain-spiking/src/network.rs
git commit -m "feat: spiking recall — fire seeds into 2B-synapse network, collect activated concepts"
```

---

### Task 3: Wire Spiking Recall Into Tick Thread

**Files:**
- Modify: `crates/brain-cognition/src/state.rs`

- [ ] **Step 1: Add spiking recall after BFS recall in tick thread**

In `crates/brain-cognition/src/state.rs`, find the recall section (around line 187-202). After the BFS recall completes and the snapshot is written, add spiking recall. Replace the entire recall block:

```rust
                    if let Some(concept) = recall_concept {
                        let t0 = std::time::Instant::now();
                        tracing::info!("Chain recall starting for: {}", concept);

                        // Phase 1: BFS recall (instant)
                        let mut sb = sb_clone.lock().unwrap();
                        let (bfs_chain, knowledge) = sb.recall_knowledge(&concept);
                        let bfs_labels: Vec<String> = bfs_chain.iter().map(|(n, _)| n.clone()).collect();

                        // Phase 2: Spiking recall (fire seeds into network)
                        let spiking_result = sb.run_spiking_recall();

                        // Merge results
                        let mut snap = brain_spiking::BrainSnapshot::default();
                        snap.has_data = true;

                        if let Some((spiking_assoc, mode)) = spiking_result {
                            snap.recall_mode = mode;
                            snap.spiking_associations = spiking_assoc.clone();

                            // Build confidence-tagged labels
                            let spiking_names: std::collections::HashSet<String> =
                                spiking_assoc.iter().map(|(n, _)| n.clone()).collect();
                            let bfs_names: std::collections::HashSet<String> =
                                bfs_chain.iter().map(|(n, _)| n.clone()).collect();

                            let mut merged: Vec<(String, usize, &str)> = Vec::new();

                            // Confirmed: in both BFS and spiking
                            for (name, bfs_w) in &bfs_chain {
                                if spiking_names.contains(name) {
                                    let spike_w = spiking_assoc.iter()
                                        .find(|(n, _)| n == name)
                                        .map(|(_, w)| *w).unwrap_or(0);
                                    let weight = ((*bfs_w).max(spike_w) as f32 * 1.5) as usize;
                                    merged.push((name.clone(), weight, "confirmed"));
                                }
                            }
                            // Explicit: BFS only
                            for (name, w) in &bfs_chain {
                                if !spiking_names.contains(name) {
                                    merged.push((name.clone(), *w, "explicit"));
                                }
                            }
                            // Emergent: spiking only
                            for (name, w) in &spiking_assoc {
                                if !bfs_names.contains(name) {
                                    let weight = (*w as f32 * 0.7) as usize;
                                    merged.push((name.clone(), weight, "emergent"));
                                }
                            }

                            merged.sort_by(|a, b| b.1.cmp(&a.1));
                            merged.truncate(12);

                            // Format knowledge string with tags
                            if !merged.is_empty() {
                                let tagged: Vec<String> = merged.iter()
                                    .map(|(name, strength, tag)| format!("[{tag}] {name} (strength: {strength})"))
                                    .collect();
                                let knowledge_tagged = format!("{} is associated with: {}",
                                    concept, tagged.join(", "));
                                snap.associated_labels.push(format!("KNOWLEDGE: {knowledge_tagged}"));
                            }
                            for (name, _, tag) in &merged {
                                snap.associated_labels.push(format!("[{}] {}", tag, name));
                            }
                        } else {
                            // No spiking results — use BFS only (same as before)
                            if !knowledge.is_empty() {
                                snap.associated_labels.push(format!("KNOWLEDGE: {knowledge}"));
                            }
                            snap.associated_labels.extend(bfs_labels);
                        }

                        drop(sb);
                        *snap_clone.lock().unwrap() = snap;
                        tracing::info!("Recall done in {:.1}s", t0.elapsed().as_secs_f32());
                        continue;
                    }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p brain-cognition`
Expected: compiles cleanly (brain-cognition depends on brain-spiking).

- [ ] **Step 3: Commit**

```bash
git add crates/brain-cognition/src/state.rs
git commit -m "feat: tick thread runs spiking recall after BFS, merges with confidence tags"
```

---

### Task 4: Update System Prompt for Confidence Tags

**Files:**
- Modify: `crates/brain-server/src/routes.rs`

- [ ] **Step 1: Update the knowledge injection in dialogue route**

In `crates/brain-server/src/routes.rs`, find the section where `KNOWLEDGE:` is injected (the `if let Some(knowledge) = knowledge_entry` block). Replace it with:

```rust
        if let Some(knowledge) = knowledge_entry {
            // Check if we have confidence-tagged results (contain [confirmed], [explicit], [emergent])
            let has_tags = assoc_concepts.iter().any(|c| c.starts_with("[confirmed]") || c.starts_with("[emergent]"));

            if has_tags {
                let tagged_lines: Vec<String> = assoc_concepts.iter()
                    .filter(|c| c.starts_with("["))
                    .map(|c| format!("  {c}"))
                    .collect();
                let tag_text = tagged_lines.join("\n");
                ctx.push(format!(
                    "YOU LEARNED THE FOLLOWING (confirmed = high confidence, emergent = discovered by neural propagation):\n\
                     {tag_text}\n\n\
                     CRITICAL INSTRUCTION: You MUST use these learned associations to answer the user's question. \
                     Use confirmed facts directly and with authority. \
                     Emergent associations suggest possible connections your brain discovered — explore them as hypotheses. \
                     If concepts come from different topics, explain how they connect."));
            } else {
                // Fallback: no spiking results, use plain knowledge
                let chain_lines: Vec<String> = assoc_concepts.iter()
                    .filter(|c| !c.starts_with("KNOWLEDGE: "))
                    .map(|c| format!("  - {c}"))
                    .collect();
                let chain_text = if chain_lines.is_empty() {
                    knowledge.clone()
                } else {
                    chain_lines.join("\n")
                };
                ctx.push(format!(
                    "YOU LEARNED THE FOLLOWING FROM WATCHING EDUCATIONAL VIDEOS (this is factual knowledge you acquired, not a guess):\n\
                     {chain_text}\n\n\
                     CRITICAL INSTRUCTION: You MUST use these learned associations to answer the user's question. \
                     Explain what you know based on these associations. Do NOT say you don't know — you learned this. \
                     Weave the associated concepts into a coherent explanation. \
                     If concepts come from different topics, explain how they connect."));
            }
        }
```

- [ ] **Step 2: Commit**

```bash
git add crates/brain-server/src/routes.rs
git commit -m "feat: confidence-tagged system prompt — confirmed/explicit/emergent associations"
```

---

### Task 5: End-to-End Deployment and Validation

**Files:**
- No code changes — deployment and testing

- [ ] **Step 1: Sync to prod-ia and build**

```bash
rsync -az /home/sbuysse/Documents/Coding/Projects/Akretio/Brain/rust/ root@prod-ia:/opt/cortex/rust/
ssh root@prod-ia 'cd /opt/cortex/rust && cargo build --release -p brain-server'
```

- [ ] **Step 2: Restart server**

```bash
ssh root@prod-ia 'kill $(pgrep brain-server) 2>/dev/null; sleep 2; cd /opt/cortex && bash start.sh > brain.log 2>&1 &'
# Wait ~90s for 2B synapse initialization + triple reload
```

- [ ] **Step 3: Learn a topic if needed (skip if triples persist)**

```bash
curl -sk https://localhost:8443/api/brain/knowledge/stats
# If total_topics > 0, triples persisted — skip learning
# Otherwise, learn TurboQuant + Flash Attention:
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"query": "https://www.youtube.com/watch?v=7YVrb3-ABYE", "topic": "TurboQuant"}' \
  https://localhost:8443/api/brain/learn/academic
# Wait 30s for extraction
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"query": "https://www.youtube.com/watch?v=zy8ChVd_oTM", "topic": "Flash Attention"}' \
  https://localhost:8443/api/brain/learn/academic
```

- [ ] **Step 4: Test spiking recall — prime and ask**

```bash
# Prime (triggers BFS + stores spiking seeds)
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"message": "turboquant flash attention", "session_id": "spike1"}' \
  https://localhost:8443/api/brain/dialogue/grounded > /dev/null
# Wait for tick to run spiking propagation
sleep 5
# Ask (should get confidence-tagged associations)
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"message": "How are quantization and attention related?", "session_id": "spike2"}' \
  https://localhost:8443/api/brain/dialogue/grounded
```

Expected: Response includes `[confirmed]` and/or `[emergent]` tagged associations. The LLM answer should distinguish between high-confidence and emergent connections.

- [ ] **Step 5: Check logs for spiking recall**

```bash
ssh root@prod-ia 'grep "Spiking recall\|spiking.*mode\|activated" /opt/cortex/brain.log | tail -5'
```

Expected: log line like `Spiking recall (broad mode): 4 seeds fired, N concepts activated`

- [ ] **Step 6: Commit and push**

```bash
git push origin master
```
