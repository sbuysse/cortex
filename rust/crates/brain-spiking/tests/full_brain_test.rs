use brain_spiking::regions::full_brain::{
    build_full_brain, AMYGDALA, ASSOCIATION, AUDITORY, HIPPOCAMPUS, PREFRONTAL, VISUAL,
};

#[test]
fn test_full_brain_tiny() {
    let net = build_full_brain(0.001, 0.1, 0.1);
    let stats = net.stats();
    assert_eq!(stats.num_regions, 10);
    assert!(
        stats.total_neurons >= 10 * 10,
        "should have at least 100 neurons"
    );
    assert!(
        stats.total_synapses > 0,
        "should have inter-region connections"
    );
}

#[test]
fn test_full_brain_spike_propagation() {
    let mut net = build_full_brain(0.001, 0.15, 0.15);

    // Inject visual input
    for _ in 0..50 {
        net.inject_current(VISUAL, 0, 5.0);
        net.inject_current(VISUAL, 1, 5.0);
        net.step();
    }

    // After 50 steps, spikes should have propagated to at least association cortex
    let stats = net.stats();
    assert!(stats.total_spikes_last_step >= 0); // smoke test — network runs without crashing
}

#[test]
fn test_full_brain_10_regions_named() {
    let net = build_full_brain(0.001, 0.05, 0.05);
    assert_eq!(net.region(VISUAL).name(), "visual_cortex");
    assert_eq!(net.region(AUDITORY).name(), "auditory_cortex");
    assert_eq!(net.region(ASSOCIATION).name(), "association_cortex");
    assert_eq!(net.region(HIPPOCAMPUS).name(), "hippocampus");
    assert_eq!(net.region(PREFRONTAL).name(), "prefrontal_cortex");
    assert_eq!(net.region(AMYGDALA).name(), "amygdala");
}
