# Multi-Topic Cross-Domain Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Cortex to persistently learn from many YouTube videos and answer cross-domain questions by finding bridge paths through shared concepts.

**Architecture:** Knowledge triples persist to a flat file, reloaded on startup. The association graph caps weights at 2.0 (reinforcement from multiple sources). Bidirectional BFS connects concept clusters from different topics through shared nodes. New batch-learning and stats endpoints expose the cumulative knowledge.

**Tech Stack:** Rust, brain-spiking crate (knowledge.rs, concepts.rs), brain-cognition (autonomy.rs, state.rs), brain-server (routes.rs, app.rs), serde_json for topics.json

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `crates/brain-spiking/src/knowledge.rs` | Association matrix, persistence, BFS recall | Modify |
| `crates/brain-spiking/src/concepts.rs` | ConceptRegistry, topic provenance | Modify |
| `crates/brain-spiking/src/lib.rs` | SpikingBrain facade, init with data dir | Modify |
| `crates/brain-cognition/src/autonomy.rs` | youtube_learn_academic, topic registry updates | Modify |
| `crates/brain-cognition/src/state.rs` | Pass data_dir to spiking brain, load on startup | Modify |
| `crates/brain-server/src/routes.rs` | New endpoints, updated system prompt format | Modify |
| `crates/brain-server/src/app.rs` | Register new routes | Modify |
| `crates/brain-spiking/tests/knowledge_test.rs` | Tests for persistence, cross-domain BFS, bidirectional recall | Create |

---

### Task 1: Persistent Knowledge Store — File I/O in KnowledgeEngine

**Files:**
- Modify: `crates/brain-spiking/src/knowledge.rs`
- Create: `crates/brain-spiking/tests/knowledge_test.rs`

- [ ] **Step 1: Write test for triple persistence round-trip**

```rust
// crates/brain-spiking/tests/knowledge_test.rs
use brain_spiking::concepts::Triple;
use brain_spiking::knowledge::KnowledgeEngine;
use std::path::PathBuf;

#[test]
fn test_persist_and_reload_triples() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();

    // Create engine, learn triples, persist
    {
        let mut engine = KnowledgeEngine::new(0, 50000, 100);
        engine.set_data_dir(data_dir.clone());
        let triple = Triple::new("turboquant", "compresses", "kv cache");
        engine.learn_triple_with_topic(&triple, "TurboQuant");
        let triple2 = Triple::new("flash attention", "uses", "kv cache");
        engine.learn_triple_with_topic(&triple2, "FlashAttention");
        engine.flush();
    }

    // Reload into fresh engine
    {
        let mut engine = KnowledgeEngine::new(0, 50000, 100);
        engine.set_data_dir(data_dir.clone());
        let loaded = engine.load_from_file();
        assert!(loaded >= 2, "Should load at least 2 triples, got {loaded}");
        assert!(engine.num_associations() > 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p brain-spiking --test knowledge_test test_persist_and_reload_triples`
Expected: FAIL — `set_data_dir`, `learn_triple_with_topic`, `flush`, `load_from_file` don't exist yet.

- [ ] **Step 3: Add `tempfile` dev-dependency**

In `crates/brain-spiking/Cargo.toml`, add under `[dev-dependencies]`:
```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 4: Implement persistence in KnowledgeEngine**

Replace the full content of `crates/brain-spiking/src/knowledge.rs`:

```rust
use crate::concepts::{ConceptRegistry, Triple};
use crate::network::SpikingNetwork;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;

/// Knowledge engine: concept-level association matrix.
/// Learns triples INSTANTLY by strengthening directed edges between concepts.
/// Recalls by BFS through the association graph — no neuron simulation needed.
/// Persists triples to disk for cumulative cross-session knowledge.
pub struct KnowledgeEngine {
    pub registry: ConceptRegistry,
    pub concept_region: usize,
    stim_current: f32,
    /// Concept-to-concept association weights.
    /// Key: (from_concept_id, to_concept_id), Value: weight (0.0 to 2.0).
    associations: HashMap<(usize, usize), f32>,
    /// Data directory for persistence files.
    data_dir: Option<PathBuf>,
    /// Buffered writer for appending triples.
    writer: Option<std::io::BufWriter<std::fs::File>>,
    /// Unflushed triple count.
    unflushed: usize,
}

impl KnowledgeEngine {
    pub fn new(concept_region: usize, region_neurons: usize, assembly_size: usize) -> Self {
        Self {
            registry: ConceptRegistry::new(region_neurons, assembly_size),
            concept_region,
            stim_current: 5.0,
            associations: HashMap::new(),
            data_dir: None,
            writer: None,
            unflushed: 0,
        }
    }

    /// Set the data directory for persistence. Creates it if needed.
    pub fn set_data_dir(&mut self, dir: PathBuf) {
        let _ = std::fs::create_dir_all(&dir);
        self.data_dir = Some(dir);
    }

    /// Get concept ID from assembly.
    fn concept_id(asm_start: usize) -> usize {
        asm_start / 100
    }

    /// Learn a triple: INSTANT — just update the association matrix.
    /// Use learn_triple_with_topic for persistence.
    pub fn learn_triple(&mut self, _net: &mut SpikingNetwork, triple: &Triple) {
        self.learn_triple_inner(triple);
    }

    /// Learn a triple with topic provenance — persists to disk.
    pub fn learn_triple_with_topic(&mut self, triple: &Triple, topic: &str) {
        self.learn_triple_inner(triple);
        self.append_to_file(triple, topic);
        // Register topic provenance on concepts
        self.registry.add_topic(&triple.subject, topic);
        self.registry.add_topic(&triple.relation, topic);
        self.registry.add_topic(&triple.object, topic);
    }

    fn learn_triple_inner(&mut self, triple: &Triple) {
        let s_id = match self.registry.get_or_create(&triple.subject) {
            Some((a, _)) => Self::concept_id(a.start),
            None => return,
        };
        let r_id = match self.registry.get_or_create(&triple.relation) {
            Some((a, _)) => Self::concept_id(a.start),
            None => return,
        };
        let o_id = match self.registry.get_or_create(&triple.object) {
            Some((a, _)) => Self::concept_id(a.start),
            None => return,
        };

        // Strengthen directed associations (cap at 2.0 for multi-source reinforcement)
        self.strengthen(s_id, r_id, 0.8);
        self.strengthen(r_id, o_id, 0.8);
        self.strengthen(s_id, o_id, 0.4);
    }

    fn strengthen(&mut self, from: usize, to: usize, delta: f32) {
        let entry = self.associations.entry((from, to)).or_insert(0.0);
        *entry = (*entry + delta).min(2.0);
    }

    /// Append a triple to the persistence file.
    fn append_to_file(&mut self, triple: &Triple, topic: &str) {
        let Some(ref dir) = self.data_dir else { return };
        if self.writer.is_none() {
            let path = dir.join("knowledge.triples");
            let file = std::fs::OpenOptions::new()
                .create(true).append(true).open(&path);
            if let Ok(f) = file {
                self.writer = Some(std::io::BufWriter::new(f));
            }
        }
        if let Some(ref mut w) = self.writer {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_secs();
            let _ = writeln!(w, "{}|{}|{}|{}|{}", triple.subject, triple.relation, triple.object, topic, ts);
            self.unflushed += 1;
            if self.unflushed >= 10 {
                let _ = w.flush();
                self.unflushed = 0;
            }
        }
    }

    /// Flush any buffered writes.
    pub fn flush(&mut self) {
        if let Some(ref mut w) = self.writer {
            let _ = w.flush();
            self.unflushed = 0;
        }
    }

    /// Load triples from persistence file. Returns count loaded.
    pub fn load_from_file(&mut self) -> usize {
        let Some(ref dir) = self.data_dir else { return 0 };
        let path = dir.join("knowledge.triples");
        let file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        let reader = std::io::BufReader::new(file);
        let mut count = 0;
        for line in reader.lines() {
            let Ok(line) = line else { continue };
            let parts: Vec<&str> = line.splitn(5, '|').collect();
            if parts.len() >= 3 {
                let triple = Triple::new(parts[0], parts[1], parts[2]);
                let topic = if parts.len() >= 4 { parts[3] } else { "" };
                self.learn_triple_inner(&triple);
                if !topic.is_empty() {
                    self.registry.add_topic(&triple.subject, topic);
                    self.registry.add_topic(&triple.relation, topic);
                    self.registry.add_topic(&triple.object, topic);
                }
                count += 1;
            }
        }
        if count > 0 {
            tracing::info!("Loaded {count} triples from {}", path.display());
        }
        count
    }

    /// Recall: BFS through the association graph from matching concepts.
    /// INSTANT — no neuron simulation.
    pub fn recall_chain(&self, _net: &mut SpikingNetwork, query: &str, max_hops: usize) -> Vec<(String, usize)> {
        let query_words: Vec<&str> = query.split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        // Find concepts matching query words
        let mut start_ids: Vec<usize> = Vec::new();
        for name in self.registry.concept_names() {
            let name_lower = name.to_lowercase();
            for &word in &query_words {
                if name_lower.contains(word) || word.contains(&name_lower) {
                    if let Some(asm) = self.registry.get(name) {
                        start_ids.push(Self::concept_id(asm.start));
                        break;
                    }
                }
            }
        }

        if start_ids.is_empty() {
            return vec![];
        }

        // BFS through association graph
        let mut seen: std::collections::HashSet<usize> = start_ids.iter().copied().collect();
        let mut result: Vec<(String, usize)> = Vec::new();
        let mut frontier = start_ids;

        for _hop in 0..max_hops {
            let mut next: Vec<(usize, f32)> = Vec::new();

            for &src_id in &frontier {
                for (&(from, to), &weight) in &self.associations {
                    if from == src_id && !seen.contains(&to) && weight > 0.1 {
                        next.push((to, weight));
                    }
                }
            }

            if next.is_empty() { break; }

            next.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next.truncate(5);

            frontier.clear();
            for (cid, weight) in &next {
                seen.insert(*cid);
                frontier.push(*cid);
                if let Some(name) = self.id_to_name(*cid) {
                    result.push((name, (*weight * 100.0) as usize));
                }
            }
        }

        // Filter noise
        let noise_concepts = ["is", "are", "was", "were", "relates-to", "has", "have",
            "uses", "use", "called", "known", "means", "works",
            "compresses", "reduces", "enables", "provides", "creates",
            "converts", "stores", "processes", "improves", "requires",
            "replaces", "achieves", "represents", "contains", "produces",
            "maintains", "generates", "supports", "implements", "optimizes",
            "transforms", "currently talking", "numbers relate", "these numbers"];
        result.retain(|(name, _)| {
            let lower = name.to_lowercase();
            !noise_concepts.contains(&lower.as_str()) && lower.len() > 3
        });

        result.truncate(10);
        result
    }

    /// Reverse lookup: concept ID → name.
    fn id_to_name(&self, concept_id: usize) -> Option<String> {
        for name in self.registry.concept_names() {
            if let Some(asm) = self.registry.get(name) {
                if Self::concept_id(asm.start) == concept_id {
                    return Some(name.to_string());
                }
            }
        }
        None
    }

    /// Format recalled associations as structured knowledge.
    pub fn chain_to_knowledge(query: &str, chain: &[(String, usize)]) -> String {
        if chain.is_empty() { return String::new(); }
        let concepts: Vec<String> = chain.iter()
            .map(|(c, strength)| format!("{c} (strength: {strength})"))
            .collect();
        format!("{query} is associated with: {}", concepts.join(", "))
    }

    /// Number of learned associations.
    pub fn num_associations(&self) -> usize {
        self.associations.len()
    }

    /// Number of unique concepts.
    pub fn num_concepts(&self) -> usize {
        self.registry.len()
    }

    /// Get the topics a concept was learned from.
    pub fn concept_topics(&self, concept: &str) -> Vec<String> {
        self.registry.get_topics(concept)
    }

    /// Get all learned topic names.
    pub fn all_topics(&self) -> Vec<String> {
        self.registry.all_topics()
    }

    /// Find concepts shared by 2+ topics (bridge nodes).
    pub fn bridge_concepts(&self) -> Vec<(String, Vec<String>)> {
        self.registry.bridge_concepts()
    }

    /// Top N concepts by number of association edges (highest degree).
    pub fn top_connected(&self, n: usize) -> Vec<(String, usize)> {
        let mut degree: HashMap<usize, usize> = HashMap::new();
        for &(from, to) in self.associations.keys() {
            *degree.entry(from).or_insert(0) += 1;
            *degree.entry(to).or_insert(0) += 1;
        }
        let mut ranked: Vec<(usize, usize)> = degree.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.iter()
            .take(n)
            .filter_map(|(id, deg)| self.id_to_name(*id).map(|name| (name, *deg)))
            .collect()
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p brain-spiking --test knowledge_test test_persist_and_reload_triples`
Expected: FAIL — `add_topic`, `get_topics`, `all_topics`, `bridge_concepts` don't exist on ConceptRegistry yet. That's Task 2.

- [ ] **Step 6: Commit knowledge.rs persistence**

```bash
git add crates/brain-spiking/src/knowledge.rs crates/brain-spiking/tests/knowledge_test.rs crates/brain-spiking/Cargo.toml
git commit -m "feat: knowledge engine persistence — append triples to file, reload on startup"
```

---

### Task 2: Topic Provenance on ConceptRegistry

**Files:**
- Modify: `crates/brain-spiking/src/concepts.rs`

- [ ] **Step 1: Write test for topic provenance**

Append to `crates/brain-spiking/tests/knowledge_test.rs`:

```rust
#[test]
fn test_topic_provenance() {
    use brain_spiking::concepts::ConceptRegistry;

    let mut reg = ConceptRegistry::new(50000, 100);
    reg.get_or_create("kv cache");
    reg.add_topic("kv cache", "TurboQuant");
    reg.add_topic("kv cache", "FlashAttention");

    let topics = reg.get_topics("kv cache");
    assert!(topics.contains(&"TurboQuant".to_string()));
    assert!(topics.contains(&"FlashAttention".to_string()));

    let bridges = reg.bridge_concepts();
    assert!(bridges.iter().any(|(name, _)| name == "kv cache"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p brain-spiking --test knowledge_test test_topic_provenance`
Expected: FAIL — `add_topic`, `get_topics`, `bridge_concepts` not defined.

- [ ] **Step 3: Add topic provenance fields to ConceptRegistry**

In `crates/brain-spiking/src/concepts.rs`, add a new field to `ConceptRegistry` and implement the methods. After the `assembly_size` field, add:

```rust
    /// Concept name → set of topics it was learned from.
    topic_provenance: HashMap<String, std::collections::HashSet<String>>,
```

In `ConceptRegistry::new`, add to the struct literal:
```rust
            topic_provenance: HashMap::new(),
```

Add these methods to the `impl ConceptRegistry` block:

```rust
    /// Record that a concept was learned from a given topic.
    pub fn add_topic(&mut self, concept: &str, topic: &str) {
        self.topic_provenance
            .entry(concept.to_string())
            .or_default()
            .insert(topic.to_string());
    }

    /// Get topics a concept was learned from.
    pub fn get_topics(&self, concept: &str) -> Vec<String> {
        self.topic_provenance
            .get(concept)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all unique topic names across all concepts.
    pub fn all_topics(&self) -> Vec<String> {
        let mut all: std::collections::HashSet<String> = std::collections::HashSet::new();
        for topics in self.topic_provenance.values() {
            all.extend(topics.iter().cloned());
        }
        let mut v: Vec<String> = all.into_iter().collect();
        v.sort();
        v
    }

    /// Find concepts shared by 2+ topics (bridge nodes).
    pub fn bridge_concepts(&self) -> Vec<(String, Vec<String>)> {
        let mut bridges = Vec::new();
        for (concept, topics) in &self.topic_provenance {
            if topics.len() >= 2 {
                let mut t: Vec<String> = topics.iter().cloned().collect();
                t.sort();
                bridges.push((concept.clone(), t));
            }
        }
        bridges.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        bridges
    }
```

- [ ] **Step 4: Run both tests**

Run: `cargo test -p brain-spiking --test knowledge_test`
Expected: PASS for both `test_persist_and_reload_triples` and `test_topic_provenance`.

- [ ] **Step 5: Commit**

```bash
git add crates/brain-spiking/src/concepts.rs crates/brain-spiking/tests/knowledge_test.rs
git commit -m "feat: topic provenance on ConceptRegistry — track which topics each concept came from"
```

---

### Task 3: Wire Persistence Into SpikingBrain and State

**Files:**
- Modify: `crates/brain-spiking/src/lib.rs`
- Modify: `crates/brain-cognition/src/state.rs`

- [ ] **Step 1: Add `data_dir` parameter to SpikingBrain::new**

In `crates/brain-spiking/src/lib.rs`, change `SpikingBrain::new` to accept an optional data dir and pass it to KnowledgeEngine. Change the signature:

```rust
    pub fn new(scale: f32, data_dir: Option<std::path::PathBuf>) -> Self {
```

After `let knowledge = KnowledgeEngine::new(...)`, add:

```rust
        let mut knowledge = KnowledgeEngine::new(assoc_region, assoc_neurons, 100);
        if let Some(ref dir) = data_dir {
            knowledge.set_data_dir(dir.clone());
            let loaded = knowledge.load_from_file();
            if loaded > 0 {
                tracing::info!("Spiking brain loaded {loaded} persisted triples ({} associations, {} concepts)",
                    knowledge.num_associations(), knowledge.num_concepts());
            }
        }
```

Also update `new_association_only` similarly (add `data_dir: Option<std::path::PathBuf>` parameter, same pattern).

- [ ] **Step 2: Update SpikingBrain construction in state.rs**

In `crates/brain-cognition/src/state.rs`, find where `SpikingBrain::new(scale)` is called and change to:

```rust
        let data_dir = config.project_root.join("data");
        let sb = SpikingBrain::new(scale, Some(data_dir));
```

- [ ] **Step 3: Update learn_triple call to use topic**

In `crates/brain-cognition/src/state.rs`, find the tick thread section where triples are learned (the `drain` loop). The triples need a topic. Add a `topic` field to the items in the triple queue.

In `crates/brain-cognition/src/state.rs`, change the triple queue type from `Vec<brain_spiking::Triple>` to `Vec<(brain_spiking::Triple, String)>` (tuple of triple + topic).

Update the drain loop:
```rust
                    let triples: Vec<_> = {
                        let mut q = tq_clone.lock().unwrap();
                        q.drain(..).collect()
                    };
                    if !triples.is_empty() {
                        let t0 = std::time::Instant::now();
                        let count = triples.len();
                        let mut sb = sb_clone.lock().unwrap();
                        for (triple, topic) in &triples {
                            sb.knowledge.learn_triple_with_topic(triple, topic);
                        }
                        sb.knowledge.flush();
                        let elapsed = t0.elapsed().as_secs_f32();
                        tracing::info!("Learned {} triples in {:.3}s", count, elapsed);
                        drop(sb);
                        continue;
                    }
```

- [ ] **Step 4: Update autonomy.rs to enqueue (triple, topic) tuples**

In `crates/brain-cognition/src/autonomy.rs`, in the `tokio::spawn` block that enqueues triples, change:
```rust
                    queue.push(triple);
```
to:
```rust
                    queue.push((triple, topic_key_owned.clone()));
```

Also update the fallback rule-based extraction similarly.

- [ ] **Step 5: Fix any compile errors from the type change**

Run: `cargo build --release -p brain-server` (on prod-ia or locally with libtorch)
Expected: compiles cleanly.

- [ ] **Step 6: Commit**

```bash
git add crates/brain-spiking/src/lib.rs crates/brain-cognition/src/state.rs crates/brain-cognition/src/autonomy.rs
git commit -m "feat: wire persistence — SpikingBrain loads triples on startup, saves on learn"
```

---

### Task 4: Bidirectional BFS for Cross-Domain Recall

**Files:**
- Modify: `crates/brain-spiking/src/knowledge.rs`
- Modify: `crates/brain-spiking/tests/knowledge_test.rs`

- [ ] **Step 1: Write test for cross-domain recall**

Append to `crates/brain-spiking/tests/knowledge_test.rs`:

```rust
#[test]
fn test_cross_domain_recall() {
    let mut engine = KnowledgeEngine::new(0, 50000, 100);

    // Domain 1: TurboQuant
    engine.learn_triple_with_topic(
        &Triple::new("turboquant", "compresses", "kv cache"), "TurboQuant");
    engine.learn_triple_with_topic(
        &Triple::new("turboquant", "reduces", "memory usage"), "TurboQuant");

    // Domain 2: FlashAttention
    engine.learn_triple_with_topic(
        &Triple::new("flash attention", "optimizes", "kv cache"), "FlashAttention");
    engine.learn_triple_with_topic(
        &Triple::new("flash attention", "speeds up", "transformer inference"), "FlashAttention");

    // Query that spans both domains — should find bridge through "kv cache"
    let mut net = brain_spiking::network::SpikingNetwork::new();
    let chain = engine.recall_chain(&mut net, "turboquant flash attention", 10);

    // Should find concepts from BOTH domains
    let names: Vec<&str> = chain.iter().map(|(n, _)| n.as_str()).collect();
    assert!(!names.is_empty(), "Should find cross-domain associations");

    // The bridge concept "kv cache" should appear or be traversed
    // (it might be filtered as a relay node, but downstream concepts should appear)
    let has_cross_domain = names.iter().any(|n|
        n.contains("memory") || n.contains("transformer") || n.contains("kv cache")
    );
    assert!(has_cross_domain, "Should reach concepts across domains, got: {:?}", names);
}
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `cargo test -p brain-spiking --test knowledge_test test_cross_domain_recall`
Expected: may pass or fail depending on BFS reach. This establishes a baseline.

- [ ] **Step 3: Implement bidirectional BFS**

Add a new method `recall_chain_bidirectional` to `KnowledgeEngine` in `knowledge.rs`:

```rust
    /// Bidirectional BFS: when query matches 2+ concept clusters, search from both
    /// sides and find bridge paths. Falls back to unidirectional if only 1 cluster found.
    pub fn recall_chain_bidirectional(&self, _net: &mut SpikingNetwork, query: &str, max_hops: usize) -> Vec<(String, usize)> {
        let query_lower = query.to_lowercase();

        // Match against all concept names AND topic names
        let mut matched_ids: Vec<(usize, String)> = Vec::new();
        for name in self.registry.concept_names() {
            let name_lower = name.to_lowercase();
            // Multi-word phrase matching
            if query_lower.contains(&name_lower) || name_lower.split_whitespace()
                .any(|w| w.len() > 3 && query_lower.contains(w))
            {
                if let Some(asm) = self.registry.get(name) {
                    matched_ids.push((Self::concept_id(asm.start), name.to_string()));
                }
            }
        }
        // Also match single query words > 3 chars
        for word in query_lower.split_whitespace().filter(|w| w.len() > 3) {
            for name in self.registry.concept_names() {
                let name_lower = name.to_lowercase();
                if name_lower.contains(word) || word.contains(&name_lower) {
                    if let Some(asm) = self.registry.get(name) {
                        let id = Self::concept_id(asm.start);
                        if !matched_ids.iter().any(|(i, _)| *i == id) {
                            matched_ids.push((id, name.to_string()));
                        }
                    }
                }
            }
        }

        if matched_ids.is_empty() {
            return vec![];
        }

        let all_ids: Vec<usize> = matched_ids.iter().map(|(id, _)| *id).collect();

        // If only one cluster, fall back to unidirectional
        if matched_ids.len() <= 1 {
            return self.recall_chain(_net, query, max_hops);
        }

        // Split into two sides: first half vs second half of matched concepts
        let mid = matched_ids.len() / 2;
        let side_a: Vec<usize> = matched_ids[..mid].iter().map(|(id, _)| *id).collect();
        let side_b: Vec<usize> = matched_ids[mid..].iter().map(|(id, _)| *id).collect();

        // BFS from both sides
        let path_a = self.bfs_explore(&side_a, max_hops / 2 + 1);
        let path_b = self.bfs_explore(&side_b, max_hops / 2 + 1);

        // Find bridge nodes (in both explored sets)
        let set_a: std::collections::HashSet<usize> = path_a.keys().copied().collect();
        let set_b: std::collections::HashSet<usize> = path_b.keys().copied().collect();
        let bridges: Vec<usize> = set_a.intersection(&set_b).copied().collect();

        // Collect results: bridge concepts first (highest priority), then BFS from both sides
        let mut result: Vec<(String, usize)> = Vec::new();

        // Add bridge concepts with boosted weight
        for &bid in &bridges {
            if all_ids.contains(&bid) { continue; } // skip start nodes
            if let Some(name) = self.id_to_name(bid) {
                let w_a = path_a.get(&bid).copied().unwrap_or(0.0);
                let w_b = path_b.get(&bid).copied().unwrap_or(0.0);
                let combined = (w_a + w_b) * 100.0;
                result.push((name, combined as usize));
            }
        }

        // Add remaining from both sides
        let mut remaining: Vec<(usize, f32)> = Vec::new();
        for (&id, &w) in path_a.iter().chain(path_b.iter()) {
            if !all_ids.contains(&id) && !bridges.contains(&id) {
                remaining.push((id, w));
            }
        }
        remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        remaining.dedup_by_key(|(id, _)| *id);
        for (id, w) in remaining.iter().take(8) {
            if let Some(name) = self.id_to_name(*id) {
                result.push((name, (*w * 100.0) as usize));
            }
        }

        // Filter noise
        let noise_concepts = ["is", "are", "was", "were", "relates-to", "has", "have",
            "uses", "use", "called", "known", "means", "works",
            "compresses", "reduces", "enables", "provides", "creates",
            "converts", "stores", "processes", "improves", "requires",
            "replaces", "achieves", "represents", "contains", "produces",
            "maintains", "generates", "supports", "implements", "optimizes",
            "transforms", "currently talking", "numbers relate", "these numbers"];
        result.retain(|(name, _)| {
            let lower = name.to_lowercase();
            !noise_concepts.contains(&lower.as_str()) && lower.len() > 3
        });

        result.truncate(10);
        result
    }

    /// BFS explore from a set of start nodes. Returns visited node → max weight.
    fn bfs_explore(&self, start_ids: &[usize], max_hops: usize) -> HashMap<usize, f32> {
        let mut visited: HashMap<usize, f32> = HashMap::new();
        let mut frontier: Vec<usize> = start_ids.to_vec();
        for &id in start_ids {
            visited.insert(id, 1.0);
        }

        for _hop in 0..max_hops {
            let mut next: Vec<(usize, f32)> = Vec::new();
            for &src_id in &frontier {
                for (&(from, to), &weight) in &self.associations {
                    if from == src_id && !visited.contains_key(&to) && weight > 0.1 {
                        next.push((to, weight));
                    }
                    // Also follow reverse edges for bidirectional discovery
                    if to == src_id && !visited.contains_key(&from) && weight > 0.1 {
                        next.push((from, weight * 0.7)); // reverse edges weaker
                    }
                }
            }
            if next.is_empty() { break; }
            next.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next.truncate(8);

            frontier.clear();
            for (id, w) in next {
                visited.insert(id, w);
                frontier.push(id);
            }
        }
        visited
    }
```

- [ ] **Step 4: Update recall_knowledge to use bidirectional BFS**

In `crates/brain-spiking/src/lib.rs`, update `recall_knowledge`:

```rust
    pub fn recall_knowledge(&mut self, query: &str) -> (Vec<(String, usize)>, String) {
        let chain = self.knowledge.recall_chain_bidirectional(&mut self.network, query, 10);
        let knowledge = KnowledgeEngine::chain_to_knowledge(query, &chain);
        (chain, knowledge)
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p brain-spiking --test knowledge_test`
Expected: All 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/brain-spiking/src/knowledge.rs crates/brain-spiking/src/lib.rs crates/brain-spiking/tests/knowledge_test.rs
git commit -m "feat: bidirectional BFS — cross-domain recall through shared concept bridges"
```

---

### Task 5: Batch Learning Endpoint + Knowledge Stats

**Files:**
- Modify: `crates/brain-server/src/routes.rs`
- Modify: `crates/brain-server/src/app.rs`

- [ ] **Step 1: Add batch learning endpoint**

In `crates/brain-server/src/routes.rs`, add:

```rust
/// Learn from multiple YouTube videos in sequence.
pub async fn api_brain_learn_batch(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let videos = match body["videos"].as_array() {
        Some(v) => v.clone(),
        None => return Json(serde_json::json!({"error": "videos array required"})).into_response(),
    };
    if let Some(brain) = &state.brain {
        let mut learned = 0;
        let mut skipped = 0;
        let mut total_triples = 0;

        // Load existing topics to check for duplicates
        let topics_path = brain.config.project_root.join("data/topics.json");
        let existing_urls: std::collections::HashSet<String> = std::fs::read_to_string(&topics_path)
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<serde_json::Value>>(&s).ok())
            .map(|arr| arr.iter().filter_map(|v| v["url"].as_str().map(String::from)).collect())
            .unwrap_or_default();

        for video in &videos {
            let url = video["url"].as_str().unwrap_or("").to_string();
            let topic = video["topic"].as_str().unwrap_or("").to_string();
            if url.is_empty() || topic.is_empty() { continue; }

            if existing_urls.contains(&url) {
                skipped += 1;
                continue;
            }

            match brain_cognition::autonomy::youtube_learn_academic(&url, &topic, brain).await {
                Ok(pairs) => {
                    total_triples += pairs;
                    learned += 1;
                }
                Err(e) => {
                    tracing::warn!("Batch learn failed for {url}: {e}");
                }
            }
        }

        let concepts = brain.spiking_brain.as_ref()
            .map(|sb| sb.lock().unwrap().knowledge.num_concepts())
            .unwrap_or(0);

        Json(serde_json::json!({
            "topics_learned": learned,
            "triples": total_triples,
            "concepts": concepts,
            "skipped": skipped,
        })).into_response()
    } else {
        Json(serde_json::json!({"error": "Brain not initialized"})).into_response()
    }
}
```

- [ ] **Step 2: Add knowledge stats endpoint**

In `crates/brain-server/src/routes.rs`, add:

```rust
/// Knowledge graph statistics — topics, concepts, bridges.
pub async fn api_brain_knowledge_stats(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let Some(ref sb) = brain.spiking_brain {
            let sb = sb.lock().unwrap();
            let topics = sb.knowledge.all_topics();
            let bridges = sb.knowledge.bridge_concepts();
            let top_connected = sb.knowledge.top_connected(10);

            // Load topic details from topics.json
            let topics_path = brain.config.project_root.join("data/topics.json");
            let topic_details: Vec<serde_json::Value> = std::fs::read_to_string(&topics_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();

            return Json(serde_json::json!({
                "total_topics": topics.len(),
                "total_associations": sb.knowledge.num_associations(),
                "total_concepts": sb.knowledge.num_concepts(),
                "topics": topics,
                "topic_details": topic_details,
                "bridge_concepts": bridges.iter().take(10).map(|(name, topics)| {
                    serde_json::json!({"concept": name, "topics": topics})
                }).collect::<Vec<_>>(),
                "top_connected": top_connected.iter().map(|(name, deg)| {
                    serde_json::json!({"concept": name, "degree": deg})
                }).collect::<Vec<_>>(),
            })).into_response();
        }
    }
    Json(serde_json::json!({"error": "Brain not initialized"})).into_response()
}
```

- [ ] **Step 3: Register routes in app.rs**

In `crates/brain-server/src/app.rs`, add after the `learn/academic` route:

```rust
        .route("/api/brain/learn/batch", post(routes::api_brain_learn_batch))
        .route("/api/brain/knowledge/stats", get(routes::api_brain_knowledge_stats))
```

- [ ] **Step 4: Update topics.json after learning**

In `crates/brain-cognition/src/autonomy.rs`, at the end of `youtube_learn_academic` (before the return), add topic registry update:

```rust
    // Update topics.json registry
    let topics_path = brain.config.project_root.join("data/topics.json");
    let mut topic_list: Vec<serde_json::Value> = std::fs::read_to_string(&topics_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    topic_list.push(serde_json::json!({
        "topic": topic_key,
        "url": url,
        "triples_count": pairs_generated,
        "learned_at": chrono::Utc::now().to_rfc3339(),
    }));
    let _ = std::fs::create_dir_all(brain.config.project_root.join("data"));
    let _ = std::fs::write(&topics_path, serde_json::to_string_pretty(&topic_list).unwrap_or_default());
```

Note: if `chrono` is not available, use the timestamp approach already in the codebase:
```rust
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    // ... "learned_at": ts
```

- [ ] **Step 5: Build and verify**

Run: `cargo build --release -p brain-server` (on prod-ia)
Expected: compiles cleanly.

- [ ] **Step 6: Commit**

```bash
git add crates/brain-server/src/routes.rs crates/brain-server/src/app.rs crates/brain-cognition/src/autonomy.rs
git commit -m "feat: batch learning endpoint + knowledge stats API"
```

---

### Task 6: Update System Prompt to Use Chain Format

**Files:**
- Modify: `crates/brain-server/src/routes.rs`

- [ ] **Step 1: Update knowledge injection in dialogue route**

In `crates/brain-server/src/routes.rs`, find the section where `KNOWLEDGE:` is injected into the system prompt (around line 2624). Replace the knowledge formatting:

```rust
        if let Some(knowledge) = knowledge_entry {
            // Parse the association chain into a structured format
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
```

- [ ] **Step 2: Commit**

```bash
git add crates/brain-server/src/routes.rs
git commit -m "feat: structured chain format in LLM system prompt for cross-domain reasoning"
```

---

### Task 7: End-to-End Integration Test — Deploy and Validate

**Files:**
- No code changes — deployment and manual testing

- [ ] **Step 1: Sync to prod-ia and build**

```bash
rsync -az rust/ root@prod-ia:/opt/cortex/rust/
ssh root@prod-ia 'cd /opt/cortex/rust && cargo build --release -p brain-server'
```

- [ ] **Step 2: Restart server**

```bash
ssh root@prod-ia 'kill $(pgrep brain-server) 2>/dev/null; sleep 2; cd /opt/cortex && bash start.sh > brain.log 2>&1 &'
# Wait ~90s for 2B synapse initialization
```

- [ ] **Step 3: Learn 3 topics via batch endpoint**

```bash
curl -sk -X POST -H "Content-Type: application/json" -d '{
  "videos": [
    {"url": "https://www.youtube.com/watch?v=7YVrb3-ABYE", "topic": "TurboQuant"},
    {"url": "https://www.youtube.com/watch?v=XyGIGpS6urI", "topic": "FlashAttention"},
    {"url": "https://www.youtube.com/watch?v=6p4wSQRTPFs", "topic": "GGUF quantization"}
  ]
}' https://localhost:8443/api/brain/learn/batch
```

Expected: `{"topics_learned": 3, "triples": N, "concepts": N, "skipped": 0}`

- [ ] **Step 4: Check knowledge stats**

```bash
curl -sk https://localhost:8443/api/brain/knowledge/stats
```

Expected: Shows 3 topics, bridge concepts shared between them (e.g., "quantization", "memory", "inference").

- [ ] **Step 5: Ask a cross-domain question**

```bash
# Prime
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"message": "quantization attention", "session_id": "cross1"}' \
  https://localhost:8443/api/brain/dialogue/grounded > /dev/null
sleep 5
# Ask
curl -sk -X POST -H "Content-Type: application/json" \
  -d '{"message": "How are quantization and attention mechanisms related?", "session_id": "cross2"}' \
  https://localhost:8443/api/brain/dialogue/grounded
```

Expected: Brain associations include concepts from BOTH TurboQuant and FlashAttention domains. LLM answer connects both topics through shared concepts.

- [ ] **Step 6: Verify persistence — restart and check**

```bash
ssh root@prod-ia 'kill $(pgrep brain-server) 2>/dev/null; sleep 2; cd /opt/cortex && bash start.sh > brain.log 2>&1 &'
# Wait for startup, then check knowledge survived
sleep 90
curl -sk https://localhost:8443/api/brain/knowledge/stats
```

Expected: Same topic count and association count as before restart.

- [ ] **Step 7: Commit final state and push**

```bash
git add -A
git commit -m "feat: multi-topic cross-domain reasoning — persistent knowledge, bidirectional BFS, batch learning"
git push origin master
```
