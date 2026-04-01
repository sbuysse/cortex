use crate::config::{NeuronParams, RegionConfig};

/// Amygdala — emotion/valence assignment.
/// Models LA (lateral, sensory input), BA (basal, association), CeA (central, output), ITC (inhibitory gating).
/// LA: first 1/4, BA: second 1/4, CeA: third 1/4, ITC: last 1/4 (mostly inhibitory).
pub fn amygdala_config(scale: f32) -> RegionConfig {
    let n = ((100_000.0 * scale) as usize).max(20);
    let n_exc = (n * 3) / 4;
    RegionConfig {
        name: "amygdala".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams {
            v_decay: 0.94,
            a_decay: 0.98,
            a_increment: 0.15,
            ..NeuronParams::default()
        },
    }
}
