use brain_spiking::synapse::{SynapseBuilder, weight_to_i16, weight_from_i16};

#[test]
fn test_build_and_freeze_empty() {
    let builder = SynapseBuilder::new(10);
    let csr = builder.freeze();
    assert_eq!(csr.num_neurons(), 10);
    assert_eq!(csr.num_synapses(), 0);
}

#[test]
fn test_build_single_synapse() {
    let mut builder = SynapseBuilder::new(4);
    builder.add(0, 1, 0.5, 1);
    let csr = builder.freeze();
    assert_eq!(csr.num_synapses(), 1);
    let targets = csr.targets_of(0);
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].target, 1);
    assert!((targets[0].weight_f32() - 0.5).abs() < 0.001);
    assert_eq!(targets[0].delay, 1);
}

#[test]
fn test_build_fanout() {
    let mut builder = SynapseBuilder::new(4);
    builder.add(0, 1, 0.3, 1);
    builder.add(0, 2, 0.6, 2);
    builder.add(0, 3, -0.2, 1);
    builder.add(2, 3, 0.1, 1);
    let csr = builder.freeze();
    assert_eq!(csr.num_synapses(), 4);
    assert_eq!(csr.targets_of(0).len(), 3);
    assert_eq!(csr.targets_of(1).len(), 0);
    assert_eq!(csr.targets_of(2).len(), 1);
    assert_eq!(csr.targets_of(3).len(), 0);
}

#[test]
fn test_sorted_targets() {
    let mut builder = SynapseBuilder::new(10);
    builder.add(0, 7, 0.1, 1);
    builder.add(0, 2, 0.2, 1);
    builder.add(0, 5, 0.3, 1);
    let csr = builder.freeze();
    let targets = csr.targets_of(0);
    assert!(targets[0].target < targets[1].target);
    assert!(targets[1].target < targets[2].target);
}

#[test]
fn test_weight_quantization_roundtrip() {
    for &w in &[0.0, 0.5, -0.5, 1.0, -1.0, 0.001, -0.001] {
        let q = weight_to_i16(w);
        let back = weight_from_i16(q);
        assert!((back - w).abs() < 0.0001, "roundtrip failed for {w}: got {back}");
    }
}
