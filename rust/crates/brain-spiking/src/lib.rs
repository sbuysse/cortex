pub mod config;
pub mod neuromodulation;
pub mod network;
pub mod neuron;
pub mod plasticity;
pub mod region;
pub mod regions;
pub mod sleep;
pub mod spike_decoder;
pub mod structural;
pub mod spike_encoder;
pub mod synapse;
pub mod synapse_mmap;

use network::{SpikingNetwork, NetworkStats};
use spike_encoder::LatencyEncoder;
use spike_decoder::RateDecoder;

/// High-level facade for the spiking brain.
pub struct SpikingBrain {
    pub network: SpikingNetwork,
    pub visual_encoder: LatencyEncoder,
    pub audio_encoder: LatencyEncoder,
    pub decoder: RateDecoder,
    encoding_window: u16,
}

impl SpikingBrain {
    /// Create a spiking brain with all 10 regions.
    /// `scale`: neuron count multiplier (0.01 = tiny test, 1.0 = full ~2M neurons).
    pub fn new(scale: f32) -> Self {
        let net = regions::full_brain::build_full_brain(scale, 0.05, 0.1);
        let vis_n = net.region(regions::full_brain::VISUAL).num_neurons();
        let aud_n = net.region(regions::full_brain::AUDITORY).num_neurons();
        let total_n: usize = (0..net.num_regions()).map(|i| net.region(i).num_neurons()).sum();
        Self {
            network: net,
            visual_encoder: LatencyEncoder::new(vis_n.min(512), 20),
            audio_encoder: LatencyEncoder::new(aud_n.min(512), 20),
            decoder: RateDecoder::new(total_n, 50),
            encoding_window: 20,
        }
    }

    /// Create with just the association cortex (backward compat).
    pub fn new_association_only(n_assoc: usize) -> Self {
        let net = regions::association::build_association_cortex(n_assoc, 0.05);
        let half = n_assoc / 2;
        Self {
            network: net,
            visual_encoder: LatencyEncoder::new(half.min(512), 20),
            audio_encoder: LatencyEncoder::new(half.min(512), 20),
            decoder: RateDecoder::new(n_assoc, 50),
            encoding_window: 20,
        }
    }

    pub fn process_visual(&mut self, embedding: &[f32]) {
        let region_id = if self.network.num_regions() >= 10 {
            regions::full_brain::VISUAL
        } else {
            0
        };
        let spike_times = self.visual_encoder.encode(embedding);
        for step in 0..self.encoding_window {
            for (i, &t) in spike_times.iter().enumerate() {
                if t == step { self.network.inject_current(region_id, i, 3.0); }
            }
            self.network.step();
        }
    }

    pub fn process_audio(&mut self, embedding: &[f32]) {
        let region_id = if self.network.num_regions() >= 10 {
            regions::full_brain::AUDITORY
        } else {
            0
        };
        let offset = if self.network.num_regions() < 10 {
            self.network.region(0).num_neurons() / 2
        } else {
            0
        };
        let spike_times = self.audio_encoder.encode(embedding);
        for step in 0..self.encoding_window {
            for (i, &t) in spike_times.iter().enumerate() {
                if t == step { self.network.inject_current(region_id, offset + i, 3.0); }
            }
            self.network.step();
        }
    }

    pub fn reward(&mut self, magnitude: f32) { self.network.modulators.reward(magnitude); }
    pub fn novelty(&mut self, magnitude: f32) { self.network.modulators.novelty(magnitude); }

    pub fn stats(&self) -> NetworkStats { self.network.stats() }
}
