use crate::concepts::{CellAssembly, ConceptRegistry, Triple};
use crate::network::SpikingNetwork;

/// Knowledge engine: learns triples via sequential STDP, recalls via chain propagation.
/// Uses a dedicated "concept region" in the spiking network where cell assemblies live.
pub struct KnowledgeEngine {
    /// Maps concept strings to neuron populations.
    pub registry: ConceptRegistry,
    /// Which region in the network holds concept assemblies.
    pub concept_region: usize,
    /// Spike current for stimulating a cell assembly.
    stim_current: f32,
    /// Number of STDP repetitions per triple during learning.
    learn_reps: usize,
}

impl KnowledgeEngine {
    pub fn new(concept_region: usize, region_neurons: usize, assembly_size: usize) -> Self {
        Self {
            registry: ConceptRegistry::new(region_neurons, assembly_size),
            concept_region,
            stim_current: 5.0,
            learn_reps: 2,
        }
    }

    /// Learn a triple: present subject → relation → object sequentially.
    /// STDP strengthens the directional chain S→R→O.
    pub fn learn_triple(&mut self, net: &mut SpikingNetwork, triple: &Triple) {
        // Get or create cell assemblies for each concept
        let s_asm = match self.registry.get_or_create(&triple.subject) {
            Some((a, _)) => a,
            None => return,
        };
        let r_asm = match self.registry.get_or_create(&triple.relation) {
            Some((a, _)) => a,
            None => return,
        };
        let o_asm = match self.registry.get_or_create(&triple.object) {
            Some((a, _)) => a,
            None => return,
        };

        let region = self.concept_region;
        let current = self.stim_current;

        // Repeat for STDP consolidation
        for _ in 0..self.learn_reps {
            // Reset neurons between reps
            net.region_mut(region).neurons_mut().reset();

            // Phase 1: Activate subject assembly (20 steps)
            for _ in 0..10 {
                for idx in s_asm.neuron_range() {
                    net.inject_current(region, idx, current);
                }
                net.step_selective(&[region]);
            }

            // Phase 2: Activate relation assembly (20 steps)
            // Subject neurons are still decaying — STDP captures the S→R transition
            for _ in 0..10 {
                for idx in r_asm.neuron_range() {
                    net.inject_current(region, idx, current);
                }
                net.step_selective(&[region]);
            }

            // Phase 3: Activate object assembly (20 steps)
            // Relation neurons decaying — STDP captures R→O transition
            for _ in 0..10 {
                for idx in o_asm.neuron_range() {
                    net.inject_current(region, idx, current);
                }
                net.step_selective(&[region]);
            }

            // Let activity settle (10 steps, no input)
            for _ in 0..10 {
                net.step_selective(&[region]);
            }
        }
    }

    /// Recall a chain starting from a concept.
    /// Activates the start concept, then traces which populations activate in sequence.
    /// Returns an ordered list of (concept_name, activation_strength).
    pub fn recall_chain(&self, net: &mut SpikingNetwork, query: &str, max_hops: usize) -> Vec<(String, usize)> {
        // Find ALL concepts that share words with the query
        let query_words: Vec<&str> = query.split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        // Collect matching concepts to activate simultaneously
        let mut start_assemblies: Vec<(String, CellAssembly)> = Vec::new();
        for name in self.registry.concept_names() {
            let name_lower = name.to_lowercase();
            for &word in &query_words {
                if name_lower.contains(word) || word.contains(&name_lower) {
                    if let Some(asm) = self.registry.get(name) {
                        start_assemblies.push((name.to_string(), asm.clone()));
                        break;
                    }
                }
            }
        }

        if start_assemblies.is_empty() {
            tracing::info!("Chain recall: no concepts match query '{}'", query);
            return vec![];
        }
        tracing::info!("Chain recall: {} concepts match query '{}': {:?}",
            start_assemblies.len(), query,
            start_assemblies.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>());

        let region = self.concept_region;
        let current = self.stim_current;

        // Reset all neurons for clean recall
        net.region_mut(region).neurons_mut().reset();

        // Inject ALL matching concepts simultaneously
        let start_names: std::collections::HashSet<String> = start_assemblies.iter()
            .map(|(n, _)| n.clone()).collect();
        for _ in 0..10 {
            for (_, asm) in &start_assemblies {
                for idx in asm.neuron_range() {
                    net.inject_current(region, idx, current);
                }
            }
            net.step_selective(&[region]);
        }

        // Collect ALL concepts that activate during propagation (not just top-1)
        let mut all_activated: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        // Run 50 steps of free propagation — collect everything that fires
        for _ in 0..50 {
            net.step_selective(&[region]);
            for &idx in net.region(region).last_spikes() {
                if let Some(concept) = self.registry.neuron_to_concept(idx) {
                    if !start_names.contains(concept) {
                        *all_activated.entry(concept.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Sort by activation strength, return all above threshold
        let mut chain: Vec<(String, usize)> = all_activated.into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect();
        chain.sort_by(|a, b| b.1.cmp(&a.1));
        chain.truncate(10); // top 10 associations

        chain
    }

    /// Format recalled associations as structured knowledge for the LLM.
    pub fn chain_to_knowledge(query: &str, chain: &[(String, usize)]) -> String {
        if chain.is_empty() {
            return String::new();
        }

        let concepts: Vec<String> = chain.iter()
            .map(|(c, strength)| format!("{c} (strength: {strength})"))
            .collect();
        format!("{query} is associated with: {}", concepts.join(", "))
    }
}
