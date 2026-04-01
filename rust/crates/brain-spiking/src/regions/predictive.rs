use crate::config::{NeuronParams, RegionConfig};

/// Predictive cortex — anticipates next input, generates surprise signal.
/// Top-down predictions flow to sensory cortices, bottom-up errors flow here.
pub fn predictive_cortex_config(scale: f32) -> RegionConfig {
    let n = ((200_000.0 * scale) as usize).max(20);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "predictive_cortex".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams::default(),
    }
}
