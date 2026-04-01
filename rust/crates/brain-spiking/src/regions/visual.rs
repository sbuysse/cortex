use crate::config::{NeuronParams, RegionConfig};

/// Visual cortex — receives DINOv2/CLIP embeddings.
/// Contains prediction and error neuron populations (predictive coding).
/// First half: prediction neurons. Second half: error neurons.
pub fn visual_cortex_config(scale: f32) -> RegionConfig {
    let n = ((200_000.0 * scale) as usize).max(20);
    let n_exc = (n * 4) / 5;
    RegionConfig {
        name: "visual_cortex".into(),
        num_excitatory: n_exc,
        num_inhibitory: n - n_exc,
        neuron_params: NeuronParams::default(),
    }
}
