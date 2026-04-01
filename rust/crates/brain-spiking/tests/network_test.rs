use brain_spiking::config::{NeuronParams, RegionConfig};
use brain_spiking::network::{NetworkConfig, SpikingNetwork};

fn small_region(name: &str, n: usize) -> RegionConfig {
    RegionConfig {
        name: name.into(),
        num_excitatory: (n * 4) / 5,
        num_inhibitory: n / 5,
        neuron_params: NeuronParams::default(),
    }
}

#[test]
fn test_network_two_regions() {
    let config = NetworkConfig {
        regions: vec![small_region("visual", 100), small_region("association", 200)],
    };
    let mut net = SpikingNetwork::new(config);
    net.connect_inter(0, 0, 1, 50, 0.5, 1);
    net.finalize();

    let mut assoc_spikes = 0;
    for _ in 0..100 {
        net.inject_current(0, 0, 5.0);
        net.step();
        assoc_spikes += net.region(1).last_spikes().len();
    }
    assert!(
        assoc_spikes > 0,
        "association cortex should receive spikes from visual cortex"
    );
}

#[test]
fn test_network_stats() {
    let config = NetworkConfig {
        regions: vec![small_region("a", 50), small_region("b", 50)],
    };
    let net = SpikingNetwork::new(config);
    let stats = net.stats();
    assert_eq!(stats.total_neurons, 100);
    assert_eq!(stats.num_regions, 2);
}
