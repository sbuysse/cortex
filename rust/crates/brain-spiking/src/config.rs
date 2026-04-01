use serde::{Deserialize, Serialize};

/// Simulation timestep in seconds (0.1ms = 0.0001s).
pub const DT: f32 = 0.0001;

/// ALIF neuron parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronParams {
    /// Membrane decay factor per dt (dimensionless, 0..1).
    pub v_decay: f32,
    /// Adaptation decay factor per dt.
    pub a_decay: f32,
    /// Adaptation increment on spike.
    pub a_increment: f32,
    /// Firing threshold (mV-like units).
    pub v_threshold: f32,
    /// Reset voltage after spike.
    pub v_reset: f32,
    /// Resting voltage.
    pub v_rest: f32,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            v_decay: 0.95,
            a_decay: 0.99,
            a_increment: 0.1,
            v_threshold: 1.0,
            v_reset: 0.0,
            v_rest: 0.0,
        }
    }
}

/// Configuration for a brain region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionConfig {
    pub name: String,
    pub num_excitatory: usize,
    pub num_inhibitory: usize,
    pub neuron_params: NeuronParams,
}

impl RegionConfig {
    pub fn total_neurons(&self) -> usize {
        self.num_excitatory + self.num_inhibitory
    }
}
