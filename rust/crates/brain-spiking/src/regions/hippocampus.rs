use crate::config::{NeuronParams, RegionConfig};

/// Hippocampus — fast pattern storage and replay.
/// Internally models DG (pattern separation), CA3 (auto-associator), CA1 (output).
/// DG: first 1/3 neurons (sparse, high inhibition)
/// CA3: middle 1/3 (dense recurrent — the associative memory)
/// CA1: last 1/3 (comparator/output)
pub fn hippocampus_config(scale: f32) -> RegionConfig {
    let n = ((300_000.0 * scale) as usize).max(30);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "hippocampus".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams {
            v_decay: 0.93,
            a_decay: 0.995,
            a_increment: 0.05,
            ..NeuronParams::default()
        },
    }
}

/// Returns (dg_start, dg_end, ca3_start, ca3_end, ca1_start, ca1_end) neuron index ranges.
pub fn hippocampus_subfields(config: &RegionConfig) -> (usize, usize, usize, usize, usize, usize) {
    let n = config.total_neurons();
    let dg_end = n / 3;
    let ca3_end = 2 * n / 3;
    (0, dg_end, dg_end, ca3_end, ca3_end, n)
}
