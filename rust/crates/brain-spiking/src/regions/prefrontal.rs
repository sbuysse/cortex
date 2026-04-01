use crate::config::{NeuronParams, RegionConfig};

/// Prefrontal cortex — working memory, goals, attention.
/// Uses NMDA-like slow recurrence (slower decay = sustained activity).
pub fn prefrontal_cortex_config(scale: f32) -> RegionConfig {
    let n = ((200_000.0 * scale) as usize).max(20);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "prefrontal_cortex".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams {
            v_decay: 0.98,
            a_decay: 0.999,
            a_increment: 0.02,
            ..NeuronParams::default()
        },
    }
}
