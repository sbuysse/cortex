use crate::config::{NeuronParams, RegionConfig};
use crate::network::{SpikingNetwork, NetworkConfig};
use rand::Rng;

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

    for src in 0..n {
        for tgt in 0..n {
            if src == tgt { continue; }
            if rng.random::<f32>() < connection_prob {
                let is_inhibitory = src >= n_exc;
                let weight = if is_inhibitory {
                    -rng.random_range(0.1..0.5)
                } else {
                    rng.random_range(0.01..0.3)
                };
                let delay = rng.random_range(0..5_u8);
                net.connect_intra(0, src, tgt, weight, delay);
            }
        }
    }

    net.finalize();
    net
}
