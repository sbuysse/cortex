use crate::config::{NeuronParams, RegionConfig};
use crate::network::{NetworkConfig, SpikingNetwork};
use rand::Rng;

/// Config-only version for use in full brain builder.
pub fn association_cortex_config(scale: f32) -> RegionConfig {
    let n = ((500_000.0 * scale) as usize).max(20);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "association_cortex".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams::default(),
    }
}

/// Build an association cortex with random initial connectivity.
/// `n`: total neurons (80% excitatory, 20% inhibitory)
/// `connection_prob`: probability of connection between any two neurons
pub fn build_association_cortex(n: usize, connection_prob: f32) -> SpikingNetwork {
    let n_exc = (n * 4) / 5;
    let n_inh = n - n_exc;

    let config = NetworkConfig {
        regions: vec![RegionConfig {
            name: "association_cortex".into(),
            num_excitatory: n_exc,
            num_inhibitory: n_inh,
            neuron_params: NeuronParams::default(),
        }],
    };

    let mut net = SpikingNetwork::new(config);
    let mut rng = rand::rng();

    // Sample connections directly: O(connections) instead of O(n^2)
    let n_connections = ((n as f64 * n as f64 * connection_prob as f64) as usize).max(1);
    for _ in 0..n_connections {
        let src = rng.random_range(0..n);
        let mut tgt = rng.random_range(0..n);
        if tgt == src { tgt = (src + 1) % n; }
        let is_inhibitory = src >= n_exc;
        let weight = if is_inhibitory {
            -rng.random_range(0.1..0.5)
        } else {
            rng.random_range(0.01..0.3)
        };
        let delay = rng.random_range(0..5_u8);
        net.connect_intra(0, src, tgt, weight, delay);
    }

    net.finalize();
    net
}
