use brain_spiking::config::{NeuronParams, RegionConfig};
use brain_spiking::region::BrainRegion;

fn test_region_config() -> RegionConfig {
    RegionConfig {
        name: "test".into(),
        num_excitatory: 80,
        num_inhibitory: 20,
        neuron_params: NeuronParams::default(),
    }
}

#[test]
fn test_region_creation() {
    let cfg = test_region_config();
    let region = BrainRegion::new(cfg);
    assert_eq!(region.num_neurons(), 100);
    assert_eq!(region.name(), "test");
}

#[test]
fn test_region_inject_and_step() {
    let cfg = test_region_config();
    let mut region = BrainRegion::new(cfg);
    region.connect(0, 1, 0.8, 0);
    region.inject_current(0, 5.0);
    let mut saw_spike_0 = false;
    let mut saw_spike_1 = false;
    for _ in 0..100 {
        region.inject_current(0, 5.0);
        let spikes = region.step();
        for &s in spikes {
            if s == 0 { saw_spike_0 = true; }
            if s == 1 { saw_spike_1 = true; }
        }
    }
    assert!(saw_spike_0, "neuron 0 should fire from injected current");
    assert!(saw_spike_1, "neuron 1 should fire from synaptic input via neuron 0");
}

#[test]
fn test_region_inhibition() {
    let cfg = test_region_config();
    let mut region = BrainRegion::new(cfg);
    // Neuron 90 is inhibitory (index >= num_excitatory=80)
    region.connect(90, 0, -0.8, 0);
    for _ in 0..100 {
        region.inject_current(0, 1.5);
        region.inject_current(90, 5.0);
        region.step();
    }
    // Smoke test: just verifying it doesn't crash with negative weights
}
