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
    /// Key: (from_concept_id, to_concept_id), Value: weight (0.0 to 1.0).
    associations: std::collections::HashMap<(usize, usize), f32>,
}

impl KnowledgeEngine {
    pub fn new(concept_region: usize, region_neurons: usize, assembly_size: usize) -> Self {
        Self {
            registry: ConceptRegistry::new(region_neurons, assembly_size),
            concept_region,
            stim_current: 5.0,
            associations: std::collections::HashMap::new(),
        }
    }

    /// Get concept ID from assembly.
    fn concept_id(asm_start: usize) -> usize {
        asm_start / 100
    }

    /// Learn a triple: INSTANT — just update the association matrix.
    pub fn learn_triple(&mut self, _net: &mut SpikingNetwork, triple: &Triple) {
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
        self.strengthen(s_id, r_id, 0.8);  // subject → relation
        self.strengthen(r_id, o_id, 0.8);  // relation → object
        self.strengthen(s_id, o_id, 0.4);  // subject → object (shortcut)
    }

    fn strengthen(&mut self, from: usize, to: usize, delta: f32) {
        let entry = self.associations.entry((from, to)).or_insert(0.0);
        *entry = (*entry + delta).min(1.0);
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

        // Filter noise: relation verbs and short generic terms shouldn't appear as associations
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
