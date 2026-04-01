use brain_spiking::synapse::{SynapseBuilder, weight_to_i16};
use brain_spiking::structural::{
    accumulate_structural_scores, prune_weak_synapses,
    decay_structural_scores, structural_stats, count_active_synapses,
};

#[test]
fn test_accumulate_scores() {
    let mut builder = SynapseBuilder::new(4);
    builder.add(0, 1, 0.5, 0);
    builder.add(0, 2, 0.3, 0);
    let mut csr = builder.freeze();

    // Set some eligibility
    csr.eligibilities[0] = weight_to_i16(0.8);  // high eligibility
    csr.eligibilities[1] = weight_to_i16(0.01); // low eligibility

    accumulate_structural_scores(&mut csr);

    assert!(csr.structural_scores[0] > csr.structural_scores[1],
        "high eligibility should give higher structural score");
}

#[test]
fn test_prune_weak() {
    let mut builder = SynapseBuilder::new(4);
    builder.add(0, 1, 0.5, 0);
    builder.add(0, 2, 0.3, 0);
    builder.add(0, 3, 0.1, 0);
    let mut csr = builder.freeze();

    // Set structural scores: first strong, rest weak
    csr.structural_scores[0] = 100;
    csr.structural_scores[1] = 5;
    csr.structural_scores[2] = 2;

    let pruned = prune_weak_synapses(&mut csr, 50);
    assert_eq!(pruned, 2, "2 synapses below threshold should be pruned");
    assert_eq!(count_active_synapses(&csr), 1, "only 1 should remain active");

    // First synapse should still have its weight
    assert_ne!(csr.weights[0], 0);
    // Pruned synapses should have zero weight
    assert_eq!(csr.weights[1], 0);
    assert_eq!(csr.weights[2], 0);
}

#[test]
fn test_decay_scores() {
    let mut builder = SynapseBuilder::new(2);
    builder.add(0, 1, 0.5, 0);
    let mut csr = builder.freeze();

    csr.structural_scores[0] = 200;
    decay_structural_scores(&mut csr, 240, 256); // ~0.9375 decay
    assert!(csr.structural_scores[0] < 200);
    assert!(csr.structural_scores[0] > 180); // 200 * 0.9375 = 187.5
}

#[test]
fn test_structural_stats() {
    let mut builder = SynapseBuilder::new(4);
    builder.add(0, 1, 0.5, 0);
    builder.add(0, 2, 0.3, 0);
    let mut csr = builder.freeze();

    csr.structural_scores[0] = 100;
    csr.structural_scores[1] = 50;

    let stats = structural_stats(&csr);
    assert_eq!(stats.total_synapses, 2);
    assert_eq!(stats.active_synapses, 2);
    assert_eq!(stats.max_structural_score, 100);
    assert!((stats.avg_structural_score - 75.0).abs() < 0.1);
}
