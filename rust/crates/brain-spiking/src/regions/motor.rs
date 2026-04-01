use crate::config::{NeuronParams, RegionConfig};

/// Motor cortex — speech/action output generation.
pub fn motor_cortex_config(scale: f32) -> RegionConfig {
    let n = ((100_000.0 * scale) as usize).max(20);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "motor_cortex".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams::default(),
    }
}
