use crate::config::{NeuronParams, RegionConfig};

/// Brainstem — arousal, energy management, neuromodulator source.
/// Uses different neuron params for tonic/burst dynamics.
pub fn brainstem_config(scale: f32) -> RegionConfig {
    let n = ((50_000.0 * scale) as usize).max(10);
    let n_exc = (n * 3) / 4;
    RegionConfig {
        name: "brainstem".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams {
            v_decay: 0.90,
            a_decay: 0.95,
            a_increment: 0.2,
            v_threshold: 0.8,
            ..NeuronParams::default()
        },
    }
}
