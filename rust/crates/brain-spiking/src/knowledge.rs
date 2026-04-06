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

    /// Configure persistence directory.  Opens (or creates) `triples.log` for appending.
    pub fn set_data_dir(&mut self, dir: &Path) {
        self.data_dir = Some(dir.to_path_buf());
        let path = dir.join("triples.log");
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => self.writer = Some(BufWriter::new(f)),
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
}
