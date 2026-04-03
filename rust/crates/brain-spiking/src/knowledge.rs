use crate::concepts::{ConceptRegistry, Triple};
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
            learn_reps: 3,
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
            for _ in 0..20 {
                for idx in s_asm.neuron_range() {
                    net.inject_current(region, idx, current);
                }
                net.step_selective(&[region]);
            }

            // Phase 2: Activate relation assembly (20 steps)
            // Subject neurons are still decaying — STDP captures the S→R transition
            for _ in 0..20 {
                for idx in r_asm.neuron_range() {
                    net.inject_current(region, idx, current);
                }
                net.step_selective(&[region]);
            }

            // Phase 3: Activate object assembly (20 steps)
            // Relation neurons decaying — STDP captures R→O transition
            for _ in 0..20 {
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
    pub fn recall_chain(&self, net: &mut SpikingNetwork, start_concept: &str, max_hops: usize) -> Vec<(String, usize)> {
        let start_asm = match self.registry.get(start_concept) {
            Some(a) => a.clone(),
            None => return vec![],
        };

        let region = self.concept_region;
        let current = self.stim_current;

        // Reset all neurons for clean recall
        net.region_mut(region).neurons_mut().reset();

        // Inject start concept
        for _ in 0..20 {
            for idx in start_asm.neuron_range() {
                net.inject_current(region, idx, current);
            }
            net.step_selective(&[region]);
        }

        // Now let activity propagate — no more input
        let mut chain: Vec<(String, usize)> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(start_concept.to_string());

        for _hop in 0..max_hops {
            // Run 30 steps of free propagation
            let mut population_spikes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

            for _ in 0..30 {
                net.step_selective(&[region]);
                // Count spikes per concept population
                for &idx in net.region(region).last_spikes() {
                    if let Some(concept) = self.registry.neuron_to_concept(idx) {
                        if !seen.contains(concept) {
                            *population_spikes.entry(concept.to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Find the most activated new concept
            if let Some((best_concept, spike_count)) = population_spikes.into_iter()
                .max_by_key(|(_, count)| *count)
            {
                if spike_count >= 3 { // minimum activation threshold
                    chain.push((best_concept.clone(), spike_count));
                    seen.insert(best_concept.clone());

                    // Stimulate the new concept to continue the chain
                    if let Some(asm) = self.registry.get(&best_concept) {
                        for _ in 0..10 {
                            for idx in asm.neuron_range() {
                                net.inject_current(region, idx, current * 0.5);
                            }
                            net.step_selective(&[region]);
                        }
                    }
                } else {
                    break; // activation too weak — chain ends
                }
            } else {
                break; // no new concept activated
            }
        }

        chain
    }

    /// Format a recalled chain as structured knowledge for the LLM.
    pub fn chain_to_knowledge(start: &str, chain: &[(String, usize)]) -> String {
        if chain.is_empty() {
            return String::new();
        }

        let mut parts = Vec::new();
        let mut prev = start;

        for (concept, _strength) in chain {
            parts.push(format!("{prev} → {concept}"));
            prev = concept;
        }

        parts.join(". ")
    }
}
