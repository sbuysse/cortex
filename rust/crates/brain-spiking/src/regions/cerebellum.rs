use crate::config::{NeuronParams, RegionConfig};

/// Cerebellum — timing, sequence learning, error correction.
/// Models: granule cells (first 2/3, massive expansion), Purkinje cells (last 1/3, high fan-in).
pub fn cerebellum_config(scale: f32) -> RegionConfig {
    let n = ((150_000.0 * scale) as usize).max(20);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "cerebellum".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams {
            v_decay: 0.92,
            a_decay: 0.97,
            a_increment: 0.08,
            ..NeuronParams::default()
        },
    }
}
