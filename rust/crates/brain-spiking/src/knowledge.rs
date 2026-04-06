use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::concepts::{ConceptRegistry, Triple};
use crate::network::SpikingNetwork;

/// Knowledge engine: concept-level association matrix.
/// Learns triples INSTANTLY by strengthening directed edges between concepts.
/// Recalls by BFS through the association graph — no neuron simulation needed.
pub struct KnowledgeEngine {
    pub registry: ConceptRegistry,
    pub concept_region: usize,
    stim_current: f32,
    /// Concept-to-concept association weights.
    /// Key: (from_concept_id, to_concept_id), Value: weight (0.0 to 2.0).
    associations: HashMap<(usize, usize), f32>,
    /// Directory where knowledge is persisted.
    data_dir: Option<PathBuf>,
    /// Append-only writer for the triples log.
    writer: Option<BufWriter<File>>,
    /// Number of writes since the last explicit flush.
    unflushed: usize,
    /// Concept IDs to fire into spiking network on next tick.
    spiking_seeds: Vec<usize>,
    /// Recall mode: "focused", "broad", or "default".
    spiking_mode: String,
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
            spiking_seeds: Vec::new(),
            spiking_mode: String::new(),
        }
    }

    /// Configure persistence directory.  Opens (or creates) `triples.log` for appending.
    pub fn set_data_dir(&mut self, dir: &Path) {
        let _ = std::fs::create_dir_all(dir);
        self.data_dir = Some(dir.to_path_buf());
        let path = dir.join("triples.log");
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => {
                tracing::info!("KnowledgeEngine: persistence file opened at {}", path.display());
                self.writer = Some(BufWriter::new(f));
            }
            Err(e) => tracing::warn!("KnowledgeEngine: cannot open {}: {}", path.display(), e),
        }
    }

    /// Flush the append writer to disk.
    pub fn flush(&mut self) {
        if let Some(w) = self.writer.as_mut() {
            let _ = w.flush();
        }
        self.unflushed = 0;
    }

    /// Replay a persisted `triples.log` into this engine (no disk writes).
    /// Returns the number of triples loaded.
    pub fn load_from_file(&mut self, path: &Path) -> usize {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        let mut count = 0;
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let parts: Vec<&str> = line.splitn(5, '|').collect();
            if parts.len() < 3 { continue; }
            let triple = Triple::new(parts[0], parts[1], parts[2]);
            let topic = if parts.len() >= 4 { parts[3] } else { "" };
            self.learn_triple_inner(&triple, topic);
            count += 1;
        }
        count
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Get concept ID from assembly start index.
    fn concept_id(asm_start: usize) -> usize {
        asm_start / 100
    }

    /// Core learning logic (shared by all public learn_ methods).
    fn learn_triple_inner(&mut self, triple: &Triple, topic: &str) {
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

        // Strengthen directed associations
        self.strengthen(s_id, r_id, 0.8); // subject → relation
        self.strengthen(r_id, o_id, 0.8); // relation → object
        self.strengthen(s_id, o_id, 0.4); // subject → object (shortcut)

        // Record topic provenance for all three concepts
        if !topic.is_empty() {
            self.registry.add_topic(&triple.subject, topic);
            self.registry.add_topic(&triple.relation, topic);
            self.registry.add_topic(&triple.object, topic);
        }
    }

    fn strengthen(&mut self, from: usize, to: usize, delta: f32) {
        let entry = self.associations.entry((from, to)).or_insert(0.0);
        *entry = (*entry + delta).min(2.0);
    }

    // -----------------------------------------------------------------------
    // Public learn methods
    // -----------------------------------------------------------------------

    /// Learn a triple: INSTANT — just update the association matrix.
    /// Kept for backward compatibility (e.g. called from SpikingBrain).
    pub fn learn_triple(&mut self, _net: &mut SpikingNetwork, triple: &Triple) {
        self.learn_triple_inner(triple, "");
    }

    /// Learn a triple AND persist it to disk (no SpikingNetwork needed).
    pub fn learn_triple_with_topic(&mut self, triple: &Triple, topic: &str) {
        self.learn_triple_inner(triple, topic);

        if let Some(w) = self.writer.as_mut() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(
                w,
                "{}|{}|{}|{}|{}",
                triple.subject, triple.relation, triple.object, topic, ts
            );
            self.unflushed += 1;
            if self.unflushed >= 100 {
                let _ = w.flush();
                self.unflushed = 0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Delegate methods (registry pass-through)
    // -----------------------------------------------------------------------

    pub fn num_concepts(&self) -> usize {
        self.registry.len()
    }

    pub fn concept_topics(&self, concept: &str) -> Vec<String> {
        self.registry.get_topics(concept)
    }

    pub fn all_topics(&self) -> Vec<String> {
        self.registry.all_topics()
    }

    pub fn bridge_concepts(&self) -> Vec<String> {
        self.registry.bridge_concepts()
    }

    /// Top-N concepts by outgoing association count.
    pub fn top_connected(&self, n: usize) -> Vec<(String, usize)> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &(from, _) in self.associations.keys() {
            *counts.entry(from).or_insert(0) += 1;
        }
        let mut pairs: Vec<(usize, usize)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(n);
        pairs
            .into_iter()
            .filter_map(|(cid, cnt)| self.id_to_name(cid).map(|name| (name, cnt)))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Recall
    // -----------------------------------------------------------------------

    /// Recall: BFS through the association graph from matching concepts.
    /// INSTANT — no neuron simulation.
    pub fn recall_chain(&mut self, _net: &mut SpikingNetwork, query: &str, max_hops: usize) -> Vec<(String, usize)> {
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

        self.spiking_seeds = start_ids.clone();
        self.spiking_mode = "focused".to_string();

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

    /// BFS helper: explores forward and reverse edges from start nodes.
    /// Returns HashMap<concept_id, max_weight>.
    fn bfs_explore(&self, start_ids: &[usize], max_hops: usize) -> HashMap<usize, f32> {
        let mut visited: HashMap<usize, f32> = HashMap::new();
        let mut frontier: Vec<usize> = start_ids.to_vec();

        for id in &frontier {
            visited.insert(*id, 1.0);
        }

        for _hop in 0..max_hops {
            let mut next: Vec<(usize, f32)> = Vec::new();

            for &src_id in &frontier {
                for (&(from, to), &weight) in &self.associations {
                    if weight <= 0.1 { continue; }
                    // Forward edge: from == src_id
                    if from == src_id && !visited.contains_key(&to) {
                        next.push((to, weight));
                    }
                    // Reverse edge: to == src_id (at 0.7x weight)
                    if to == src_id && !visited.contains_key(&from) {
                        next.push((from, weight * 0.7));
                    }
                }
            }

            if next.is_empty() { break; }

            next.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next.truncate(8);

            frontier.clear();
            for (cid, w) in next {
                let entry = visited.entry(cid).or_insert(0.0);
                if w > *entry { *entry = w; }
                frontier.push(cid);
            }
        }

        visited
    }

    /// Bidirectional BFS recall: find cross-domain bridge concepts.
    /// If 2+ concept clusters found in query, BFS from both sides and boost bridge nodes.
    /// Falls back to recall_chain if fewer than 2 clusters found.
    pub fn recall_chain_bidirectional(&mut self, _net: &mut SpikingNetwork, query: &str, max_hops: usize) -> Vec<(String, usize)> {
        // Parse query for ALL recognizable concept names (including multi-word)
        let query_lower = query.to_lowercase();
        let all_names: Vec<String> = self.registry.concept_names().into_iter().map(|s| s.to_string()).collect();

        // Match concept names against query — prefer longer matches first
        let mut matched_names: Vec<String> = Vec::new();
        let mut sorted_names = all_names.clone();
        sorted_names.sort_by(|a, b| b.len().cmp(&a.len()));

        for name in &sorted_names {
            let name_lower = name.to_lowercase();
            if name_lower.len() > 3 && query_lower.contains(&name_lower) {
                matched_names.push(name.clone());
            }
        }

        // Build concept ID clusters per matched name
        let mut cluster_ids: Vec<usize> = Vec::new();
        for name in &matched_names {
            if let Some(asm) = self.registry.get(name) {
                let cid = Self::concept_id(asm.start);
                if !cluster_ids.contains(&cid) {
                    cluster_ids.push(cid);
                }
            }
        }

        if cluster_ids.len() < 2 {
            // Fall back to unidirectional recall
            return self.recall_chain(_net, query, max_hops);
        }

        // Split clusters into two sides (first half vs second half)
        let mid = cluster_ids.len() / 2;
        let side_a = &cluster_ids[..mid];
        let side_b = &cluster_ids[mid..];

        // BFS from both sides
        let visited_a = self.bfs_explore(side_a, max_hops);
        let visited_b = self.bfs_explore(side_b, max_hops);

        // Merge results: bridge nodes get boosted weight
        let mut merged: HashMap<usize, f32> = HashMap::new();

        // Start IDs don't go into results
        let start_set: std::collections::HashSet<usize> = cluster_ids.iter().copied().collect();

        for (&cid, &wa) in &visited_a {
            if start_set.contains(&cid) { continue; }
            let entry = merged.entry(cid).or_insert(0.0);
            if wa > *entry { *entry = wa; }
        }
        for (&cid, &wb) in &visited_b {
            if start_set.contains(&cid) { continue; }
            // Bridge: found by both sides — boost by 1.5x
            if visited_a.contains_key(&cid) {
                let entry = merged.entry(cid).or_insert(0.0);
                *entry = (*entry + wb * 1.5).min(3.0);
            } else {
                let entry = merged.entry(cid).or_insert(0.0);
                if wb > *entry { *entry = wb; }
            }
        }

        // Sort by weight descending
        let mut pairs: Vec<(usize, f32)> = merged.into_iter().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Store seeds for spiking propagation
        self.spiking_seeds = cluster_ids.clone();

        // Determine recall mode from topic provenance
        let mut seed_topics: std::collections::HashSet<String> = std::collections::HashSet::new();
        for name in &matched_names {
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

        // Convert to (name, weight) and filter noise
        let noise_concepts = ["is", "are", "was", "were", "relates-to", "has", "have",
            "uses", "use", "called", "known", "means", "works",
            "compresses", "reduces", "enables", "provides", "creates",
            "converts", "stores", "processes", "improves", "requires",
            "replaces", "achieves", "represents", "contains", "produces",
            "maintains", "generates", "supports", "implements", "optimizes",
            "transforms", "currently talking", "numbers relate", "these numbers"];

        let mut result: Vec<(String, usize)> = Vec::new();
        for (cid, weight) in pairs {
            if let Some(name) = self.id_to_name(cid) {
                let lower = name.to_lowercase();
                if !noise_concepts.contains(&lower.as_str()) && lower.len() > 3 {
                    result.push((name, (weight * 100.0) as usize));
                }
            }
        }

        result.truncate(10);
        result
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

    /// Stimulation current for spiking seed injection.
    pub fn stim_current(&self) -> f32 {
        self.stim_current
    }
}
