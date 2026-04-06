# Temporal Sequence Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode temporal order into the spiking network using STDP-timed imprinting between consecutive triples, enabling predictive recall that surfaces "what comes next" in learned sequences.

**Architecture:** Triples gain a sequence index. Consecutive triples from the same video are linked by STDP-timed chain imprinting (fire pre-assembly, wait 5 steps, fire post-assembly). Spiking recall splits into two windows: early (direct) and late (predicted). A new `[predicted]` tag surfaces chain-following concepts to the LLM.

**Tech Stack:** Rust, brain-spiking (lib.rs, knowledge.rs), brain-cognition (state.rs), brain-server (routes.rs)

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `crates/brain-spiking/src/knowledge.rs` | Persistence format with seq_index | Modify |
| `crates/brain-spiking/src/lib.rs` | imprint_chain_stdp, two-window recall | Modify |
| `crates/brain-cognition/src/state.rs` | Tick thread passes ordered triples to chain imprinting | Modify |
| `crates/brain-server/src/routes.rs` | [predicted] tag in system prompt | Modify |
| `crates/brain-spiking/tests/knowledge_test.rs` | Test for chain imprinting | Modify |

---

### Task 1: Sequence Index in Persistence and Triple Queue

**Files:**
- Modify: `crates/brain-spiking/src/knowledge.rs`
- Modify: `crates/brain-cognition/src/state.rs`

- [ ] **Step 1: Update persistence format to include seq_index**

In `crates/brain-spiking/src/knowledge.rs`, change `learn_triple_with_topic` to accept a sequence index and write it to the file. Change the signature:

```rust
    pub fn learn_triple_with_topic(&mut self, triple: &Triple, topic: &str, seq_index: i32) {
        self.learn_triple_inner(triple, topic);

        if let Some(w) = self.writer.as_mut() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(
                w,
                "{}|{}|{}|{}|{}|{}",
                triple.subject, triple.relation, triple.object, topic, ts, seq_index
            );
            self.unflushed += 1;
            if self.unflushed >= 100 {
                let _ = w.flush();
                self.unflushed = 0;
            }
        }
    }
```

- [ ] **Step 2: Update load_from_file to parse seq_index**

In `load_from_file`, change the triple collection to include seq_index. Change the return type to `Vec<(Triple, i32)>`:

```rust
    pub fn load_from_file(&mut self, path: &Path) -> Vec<(Triple, i32)> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let mut loaded = Vec::new();
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let parts: Vec<&str> = line.splitn(7, '|').collect();
            if parts.len() < 3 { continue; }
            let triple = Triple::new(parts[0], parts[1], parts[2]);
            let topic = if parts.len() >= 4 { parts[3] } else { "" };
            let seq_index: i32 = if parts.len() >= 6 {
                parts[5].parse().unwrap_or(-1)
            } else {
                -1
            };
            self.learn_triple_inner(&triple, topic);
            loaded.push((triple, seq_index));
        }
        if !loaded.is_empty() {
            tracing::info!("Loaded {} triples from {}", loaded.len(), path.display());
        }
        loaded
    }
```

- [ ] **Step 3: Update triple queue type to include seq_index**

In `crates/brain-cognition/src/state.rs`, change the triple queue type from `Vec<(brain_spiking::Triple, String)>` to `Vec<(brain_spiking::Triple, String, i32)>`:

```rust
    pub triple_queue: std::sync::Arc<std::sync::Mutex<Vec<(brain_spiking::Triple, String, i32)>>>,
```

Update the initialization:
```rust
    let triple_queue = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(brain_spiking::Triple, String, i32)>::new()));
```

Update the tick thread drain loop:
```rust
                    if !triples.is_empty() {
                        let t0 = std::time::Instant::now();
                        let count = triples.len();
                        let mut sb = sb_clone.lock().unwrap();
                        let mut imprinted = 0;
                        for (triple, topic, seq_idx) in &triples {
                            sb.knowledge.learn_triple_with_topic(triple, topic, *seq_idx);
                            imprinted += sb.imprint_synapses(triple);
                        }
                        // Chain imprint consecutive triples (STDP-timed)
                        let chain_count = sb.imprint_chain_stdp(&triples);
                        sb.knowledge.flush();
                        let elapsed = t0.elapsed().as_secs_f32();
                        tracing::info!("Learned {} triples in {:.3}s, imprinted {} synapses, {} chain links",
                            count, elapsed, imprinted, chain_count);
                        drop(sb);
                        continue;
                    }
```

- [ ] **Step 4: Update autonomy.rs to enqueue with seq_index**

In `crates/brain-cognition/src/autonomy.rs`, find the `tokio::spawn` block that pushes triples. Change:
```rust
                    queue.push((triple, topic_key_owned.clone()));
```
to:
```rust
                    queue.push((triple, topic_key_owned.clone(), idx as i32));
```

Where `idx` is the loop index. Change the loop from:
```rust
            for triple in all_triples {
                queue.push(triple);
            }
```
to:
```rust
            for (idx, triple) in all_triples.into_iter().enumerate() {
                queue.push((triple, topic_key_owned.clone(), idx as i32));
            }
```

Also update the rule-based fallback in the `Err` branch similarly.

- [ ] **Step 5: Fix callers of load_from_file**

In `crates/brain-spiking/src/lib.rs`, update `SpikingBrain::new` where `load_from_file` is called. Change from:
```rust
let loaded_triples = knowledge.load_from_file(&dir.join("triples.log"));
```
The loaded triples are now `Vec<(Triple, i32)>`. Update the imprint loop:
```rust
for (triple, _seq) in &loaded_triples {
    total_imprinted += brain.imprint_synapses(triple);
}
```

Also add chain imprinting for loaded triples at startup:
```rust
// Chain imprint loaded triples (convert to tuple format for imprint_chain_stdp)
let as_tuples: Vec<(brain_spiking::Triple, String, i32)> = loaded_triples.iter()
    .map(|(t, seq)| (t.clone(), String::new(), *seq))
    .collect();
let chain_links = brain.imprint_chain_stdp(&as_tuples);
if chain_links > 0 {
    tracing::info!("STDP chain imprinted {} links from persisted triples", chain_links);
}
```

Do the same for `new_association_only`.

- [ ] **Step 6: Fix tests**

In `crates/brain-spiking/tests/knowledge_test.rs`, update all calls to `learn_triple_with_topic` to pass `seq_index`:
- Change `engine.learn_triple_with_topic(&triple, "TurboQuant")` to `engine.learn_triple_with_topic(&triple, "TurboQuant", 0)`
- Change `engine.learn_triple_with_topic(&triple, "FlashAttention")` to `engine.learn_triple_with_topic(&triple, "FlashAttention", 0)`
- etc. for all test calls

Update `test_persist_and_reload_triples` to assert on `loaded.len()` (it returns `Vec<(Triple, i32)>` now).

- [ ] **Step 7: Verify compilation**

Run: `cargo check -p brain-spiking && cargo check -p brain-cognition`
Expected: compiles (brain-server may fail without libtorch, that's OK).

- [ ] **Step 8: Commit**

```bash
git add crates/brain-spiking/src/knowledge.rs crates/brain-spiking/src/lib.rs crates/brain-spiking/tests/knowledge_test.rs crates/brain-cognition/src/state.rs crates/brain-cognition/src/autonomy.rs
git commit -m "feat: sequence index in triple persistence and queue"
```

---

### Task 2: STDP-Timed Chain Imprinting

**Files:**
- Modify: `crates/brain-spiking/src/lib.rs`
- Modify: `crates/brain-spiking/tests/knowledge_test.rs`

- [ ] **Step 1: Write test for chain imprinting**

Append to `crates/brain-spiking/tests/knowledge_test.rs`:

```rust
#[test]
fn test_chain_imprinting() {
    let mut brain = brain_spiking::SpikingBrain::new(0.01, None);

    let t1 = brain_spiking::Triple::new("quantization", "reduces", "precision");
    let t2 = brain_spiking::Triple::new("lower precision", "reduces", "memory");
    let t3 = brain_spiking::Triple::new("less memory", "enables", "larger batches");

    // Learn triples with sequence order
    brain.knowledge.learn_triple_with_topic(&t1, "test", 0);
    brain.knowledge.learn_triple_with_topic(&t2, "test", 1);
    brain.knowledge.learn_triple_with_topic(&t3, "test", 2);

    // Imprint within-triple synapses
    brain.imprint_synapses(&t1);
    brain.imprint_synapses(&t2);
    brain.imprint_synapses(&t3);

    // Chain imprint consecutive pairs via STDP
    let triples = vec![
        (t1, "test".to_string(), 0i32),
        (t2, "test".to_string(), 1i32),
        (t3, "test".to_string(), 2i32),
    ];
    let chain_count = brain.imprint_chain_stdp(&triples);
    println!("Chain imprinted {} links at scale=0.01", chain_count);
    // Should have created 2 chain links (t1→t2, t2→t3)
    assert_eq!(chain_count, 2, "Should have 2 chain links for 3 consecutive triples");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p brain-spiking --test knowledge_test test_chain_imprinting`
Expected: FAIL — `imprint_chain_stdp` doesn't exist.

- [ ] **Step 3: Implement `imprint_chain_stdp`**

Add to `impl SpikingBrain` in `crates/brain-spiking/src/lib.rs`:

```rust
    /// STDP-timed chain imprinting: for consecutive triples (by seq_index),
    /// fire the object assembly of triple N, wait 5 steps, fire the subject assembly
    /// of triple N+1. The STDP rule strengthens forward connections naturally.
    /// Returns number of chain links created.
    pub fn imprint_chain_stdp(&mut self, triples: &[(Triple, String, i32)]) -> usize {
        // Sort by seq_index, filter out unordered triples (seq_index < 0)
        let mut ordered: Vec<&(Triple, String, i32)> = triples.iter()
            .filter(|(_, _, seq)| *seq >= 0)
            .collect();
        ordered.sort_by_key(|(_, _, seq)| *seq);

        if ordered.len() < 2 {
            return 0;
        }

        let assoc_region = self.knowledge.concept_region;
        let stim = self.knowledge.stim_current();
        let mut chain_count = 0;

        // Enable learning on association cortex for STDP
        self.network.region_mut(assoc_region).learning_enabled = true;

        for i in 0..ordered.len() - 1 {
            let (triple_n, _, seq_n) = ordered[i];
            let (triple_n1, _, seq_n1) = ordered[i + 1];

            // Only chain consecutive seq indices
            if *seq_n1 != *seq_n + 1 {
                continue;
            }

            // Get assemblies: object of triple N → subject of triple N+1
            let obj_asm = self.knowledge.registry.get(&triple_n.object).cloned();
            let subj_asm = self.knowledge.registry.get(&triple_n1.subject).cloned();

            let (obj_asm, subj_asm) = match (obj_asm, subj_asm) {
                (Some(o), Some(s)) => (o, s),
                _ => continue,
            };

            // Phase 1: Fire object assembly of triple N (pre-synaptic)
            let region_neurons = self.network.region(assoc_region).num_neurons();
            for neuron in obj_asm.start..(obj_asm.start + obj_asm.size).min(region_neurons) {
                self.network.inject_current(assoc_region, neuron, stim);
            }

            // Run 5 steps — pre-synaptic neurons fire
            for _ in 0..5 {
                self.network.region_mut(assoc_region).step_with_clamp(1.5);
            }

            // Phase 2: Fire subject assembly of triple N+1 (post-synaptic)
            for neuron in subj_asm.start..(subj_asm.start + subj_asm.size).min(region_neurons) {
                self.network.inject_current(assoc_region, neuron, stim);
            }

            // Run 5 more steps — STDP fires on pre-before-post timing
            for _ in 0..5 {
                self.network.region_mut(assoc_region).step_with_clamp(1.5);
            }

            chain_count += 1;
        }

        // Disable learning and reset neurons (but keep weight changes)
        self.network.region_mut(assoc_region).learning_enabled = false;
        self.network.region_mut(assoc_region).neurons_mut().reset();

        if chain_count > 0 {
            tracing::info!("STDP chain imprinted {} temporal links", chain_count);
        }

        chain_count
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p brain-spiking --test knowledge_test`
Expected: All tests pass including the new chain imprinting test.

- [ ] **Step 5: Commit**

```bash
git add crates/brain-spiking/src/lib.rs crates/brain-spiking/tests/knowledge_test.rs
git commit -m "feat: STDP-timed chain imprinting — fire pre-assembly, wait 5 steps, fire post-assembly"
```

---

### Task 3: Two-Window Predictive Recall

**Files:**
- Modify: `crates/brain-spiking/src/lib.rs`

- [ ] **Step 1: Split spiking propagation into two windows**

In `crates/brain-spiking/src/lib.rs`, replace the propagation loop in `run_spiking_recall`. Change the current single-window loop to a two-window approach. Replace:

```rust
        // Propagate for 30 steps through association cortex only (with raised clamp for imprinted weights)
        let mut fired_in_assoc: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for _step in 0..30 {
            let spikes = self.network.region_mut(assoc_region).step_with_clamp(1.5);
            for &neuron_idx in spikes {
                *fired_in_assoc.entry(neuron_idx).or_insert(0) += 1;
            }
        }
```

With:

```rust
        // Window 1 (steps 1-10): Direct associations
        let mut fired_window1: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for _step in 0..10 {
            let spikes = self.network.region_mut(assoc_region).step_with_clamp(1.5);
            for &neuron_idx in spikes {
                *fired_window1.entry(neuron_idx).or_insert(0) += 1;
            }
        }

        // Window 2 (steps 11-30): Chain predictions (traveled through forward temporal links)
        let mut fired_window2: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for _step in 0..20 {
            let spikes = self.network.region_mut(assoc_region).step_with_clamp(1.5);
            for &neuron_idx in spikes {
                *fired_window2.entry(neuron_idx).or_insert(0) += 1;
            }
        }

        // Combine for total activation
        let mut fired_in_assoc: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (k, v) in fired_window1.iter() {
            *fired_in_assoc.entry(*k).or_insert(0) += v;
        }
        for (k, v) in fired_window2.iter() {
            *fired_in_assoc.entry(*k).or_insert(0) += v;
        }
```

- [ ] **Step 2: Classify concepts as direct vs predicted**

After the existing concept activation scanning loop, add classification. Change `run_spiking_recall` return type from `Option<(Vec<(String, usize)>, String)>` to `Option<(Vec<(String, usize)>, Vec<(String, usize)>, String)>` — (direct, predicted, mode).

After the activation scan, split activated concepts:

```rust
        // Classify: concepts activated in window 1 = direct, only in window 2 = predicted
        let mut direct: Vec<(String, usize)> = Vec::new();
        let mut predicted: Vec<(String, usize)> = Vec::new();

        for (name, strength) in &activated {
            // Check if this concept's assembly fired in window 1
            let asm = self.knowledge.registry.get(name);
            let in_window1 = if let Some(asm) = asm {
                (asm.start..asm.start + asm.size).any(|n| fired_window1.contains_key(&n))
            } else {
                false
            };

            if in_window1 {
                direct.push((name.clone(), *strength));
            } else {
                predicted.push((name.clone(), *strength));
            }
        }

        direct.sort_by(|a, b| b.1.cmp(&a.1));
        predicted.sort_by(|a, b| b.1.cmp(&a.1));
        direct.truncate(8);
        predicted.truncate(5);
```

Change the return:
```rust
        Some((direct, predicted, mode))
```

- [ ] **Step 3: Update BrainSnapshot**

Add a new field to `BrainSnapshot`:
```rust
    /// Concepts predicted by forward chain propagation (what comes next).
    pub predicted_associations: Vec<(String, usize)>,
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p brain-spiking`
Expected: FAIL — callers of `run_spiking_recall` need updating (Task 4).

- [ ] **Step 5: Commit**

```bash
git add crates/brain-spiking/src/lib.rs
git commit -m "feat: two-window spiking recall — direct (steps 1-10) vs predicted (steps 11-30)"
```

---

### Task 4: Wire Predicted Tag Into Tick Thread and System Prompt

**Files:**
- Modify: `crates/brain-cognition/src/state.rs`
- Modify: `crates/brain-server/src/routes.rs`

- [ ] **Step 1: Update tick thread merge logic for three-way output**

In `crates/brain-cognition/src/state.rs`, find the spiking recall handling. Change:

```rust
if let Some((spiking_assoc, mode)) = spiking_result {
```

to handle the new three-element tuple:

```rust
if let Some((spiking_direct, spiking_predicted, mode)) = spiking_result {
    snap.recall_mode = mode;
    snap.spiking_associations = spiking_direct.clone();
    snap.predicted_associations = spiking_predicted.clone();

    // Combine direct + predicted for spiking names set
    let all_spiking: Vec<(String, usize)> = spiking_direct.iter()
        .chain(spiking_predicted.iter())
        .cloned().collect();
    let spiking_names: std::collections::HashSet<String> =
        all_spiking.iter().map(|(n, _)| n.clone()).collect();
    let bfs_names: std::collections::HashSet<String> =
        bfs_chain.iter().map(|(n, _)| n.clone()).collect();
    let predicted_names: std::collections::HashSet<String> =
        spiking_predicted.iter().map(|(n, _)| n.clone()).collect();

    let mut merged: Vec<(String, usize, &str)> = Vec::new();

    // Confirmed: in both BFS and spiking direct
    for (name, bfs_w) in &bfs_chain {
        if spiking_names.contains(name) && !predicted_names.contains(name) {
            let spike_w = all_spiking.iter()
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
    // Predicted: spiking window 2 only
    for (name, w) in &spiking_predicted {
        if !bfs_names.contains(name) {
            let weight = (*w as f32 * 0.8) as usize;
            merged.push((name.clone(), weight, "predicted"));
        }
    }
    // Emergent: spiking direct only (not in BFS, not predicted)
    for (name, w) in &spiking_direct {
        if !bfs_names.contains(name) && !predicted_names.contains(name) {
            let weight = (*w as f32 * 0.7) as usize;
            merged.push((name.clone(), weight, "emergent"));
        }
    }

    merged.sort_by(|a, b| b.1.cmp(&a.1));
    merged.truncate(12);

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
}
```

- [ ] **Step 2: Update system prompt for [predicted] tag**

In `crates/brain-server/src/routes.rs`, find the confidence-tagged system prompt section. Update the tag detection and prompt text:

```rust
            let has_tags = assoc_concepts.iter().any(|c|
                c.starts_with("[confirmed]") || c.starts_with("[emergent]") || c.starts_with("[predicted]"));

            if has_tags {
                let tagged_lines: Vec<String> = assoc_concepts.iter()
                    .filter(|c| c.starts_with("["))
                    .map(|c| format!("  {c}"))
                    .collect();
                let tag_text = tagged_lines.join("\n");
                ctx.push(format!(
                    "YOU LEARNED THE FOLLOWING (confirmed = high confidence, predicted = what typically follows, emergent = discovered by neural pathways):\n\
                     {tag_text}\n\n\
                     CRITICAL INSTRUCTION: You MUST use these learned associations to answer the user's question. \
                     Use confirmed facts directly and with authority. \
                     Predicted associations represent what you learned typically follows — use them to explain consequences and next steps. \
                     Emergent associations suggest possible connections your brain discovered — explore them as hypotheses. \
                     If concepts come from different topics, explain how they connect."));
            }
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p brain-cognition`
Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add crates/brain-cognition/src/state.rs crates/brain-server/src/routes.rs
git commit -m "feat: [predicted] tag — forward chain concepts surfaced to LLM"
```

---

### Task 5: End-to-End Deploy and Validate

**Files:**
- No code changes — deployment and testing

- [ ] **Step 1: Sync and build**

```bash
rsync -az /home/sbuysse/Documents/Coding/Projects/Akretio/Brain/rust/ root@prod-ia:/opt/cortex/rust/
ssh root@prod-ia 'cd /opt/cortex/rust && cargo build --release -p brain-server'
```

- [ ] **Step 2: Restart**

```bash
ssh root@prod-ia 'kill $(pgrep brain-server) 2>/dev/null; sleep 2; cd /opt/cortex && bash start.sh > brain.log 2>&1 &'
# Wait ~120s (triple reload + chain imprinting from 400+ triples)
```

- [ ] **Step 3: Check startup logs**

```bash
ssh root@prod-ia 'grep -i "chain\|STDP\|loaded\|imprint" /opt/cortex/brain.log | head -10'
```

Expected: "STDP chain imprinted N links from persisted triples" with N > 0.

- [ ] **Step 4: Learn a new topic to test live chain imprinting**

```bash
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"query": "https://www.youtube.com/watch?v=7YVrb3-ABYE", "topic": "TurboQuant_chain_test"}' \
  https://localhost:8443/api/brain/learn/academic
sleep 30
grep "chain links" /opt/cortex/brain.log | tail -3
```

Expected: "Learned N triples in Xs, imprinted M synapses, K chain links" with K > 0.

- [ ] **Step 5: Test predictive recall**

```bash
# Prime
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"message": "quantization", "session_id": "pred1"}' \
  https://localhost:8443/api/brain/dialogue/grounded > /dev/null
sleep 8
# Ask
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"message": "What happens after quantization is applied?", "session_id": "pred2"}' \
  https://localhost:8443/api/brain/dialogue/grounded
```

Expected: Response includes `[predicted]` tags for concepts that follow quantization in learned sequences (e.g., "memory reduction", "faster inference").

- [ ] **Step 6: Push**

```bash
git push origin master
```
