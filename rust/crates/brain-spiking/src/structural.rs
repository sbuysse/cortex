use crate::synapse::{SynapseCSR, weight_from_i16};

/// Update structural scores based on current eligibility magnitude.
/// Called periodically (e.g., every 1000 timesteps).
/// Eligibility magnitude is accumulated into the structural score (saturating add).
pub fn accumulate_structural_scores(synapses: &mut SynapseCSR) {
    for i in 0..synapses.weights.len() {
        let elig = weight_from_i16(synapses.eligibilities[i]).abs();
        // Scale eligibility (0.0-1.0) to u8 increment (0-10)
        let increment = (elig * 10.0) as u8;
        synapses.structural_scores[i] = synapses.structural_scores[i].saturating_add(increment);
    }
}

/// Prune weak synapses: zero out weights for synapses with structural_score below threshold.
/// Returns the number of synapses pruned.
/// Note: We don't remove entries from CSR (that would require rebuilding).
/// Instead we set weight to 0, making them functionally dead.
pub fn prune_weak_synapses(synapses: &mut SynapseCSR, threshold: u8) -> usize {
    let mut pruned = 0;
    for i in 0..synapses.weights.len() {
        if synapses.structural_scores[i] < threshold && synapses.weights[i] != 0 {
            synapses.weights[i] = 0;
            synapses.weight_refs[i] = 0;
            synapses.eligibilities[i] = 0;
            synapses.structural_scores[i] = 0;
            pruned += 1;
        }
    }
    pruned
}

/// Decay all structural scores (slow forgetting of importance).
/// Called periodically. Multiplies all scores by decay_numer/decay_denom
/// using integer math. E.g., 240/256 approximates 0.9375 decay.
pub fn decay_structural_scores(synapses: &mut SynapseCSR, decay_numer: u16, decay_denom: u16) {
    for s in synapses.structural_scores.iter_mut() {
        *s = ((*s as u16 * decay_numer) / decay_denom) as u8;
    }
}

/// Count active (non-zero weight) synapses.
pub fn count_active_synapses(synapses: &SynapseCSR) -> usize {
    synapses.weights.iter().filter(|&&w| w != 0).count()
}

/// Structural health statistics for monitoring.
pub struct StructuralStats {
    pub total_synapses: usize,
    pub active_synapses: usize,
    pub pruned_synapses: usize,
    pub avg_structural_score: f32,
    pub max_structural_score: u8,
}

/// Get structural health statistics.
pub fn structural_stats(synapses: &SynapseCSR) -> StructuralStats {
    let total = synapses.weights.len();
    let active = count_active_synapses(synapses);
    let sum: u64 = synapses.structural_scores.iter().map(|&s| s as u64).sum();
    let max = synapses.structural_scores.iter().copied().max().unwrap_or(0);
    StructuralStats {
        total_synapses: total,
        active_synapses: active,
        pruned_synapses: total - active,
        avg_structural_score: if total > 0 { sum as f32 / total as f32 } else { 0.0 },
        max_structural_score: max,
    }
}
