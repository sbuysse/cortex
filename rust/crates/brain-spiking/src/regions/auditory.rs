use crate::config::{NeuronParams, RegionConfig};

/// Auditory cortex — receives Whisper embeddings.
pub fn auditory_cortex_config(scale: f32) -> RegionConfig {
    let n = ((200_000.0 * scale) as usize).max(20);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "auditory_cortex".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams::default(),
    }
}
