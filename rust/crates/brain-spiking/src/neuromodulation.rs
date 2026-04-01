use serde::{Deserialize, Serialize};

/// Four global neuromodulators as scalar signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuromodulators {
    pub dopamine: f32,
    pub acetylcholine: f32,
    pub norepinephrine: f32,
    pub serotonin: f32,
    da_decay: f32,
    ach_decay: f32,
    ne_decay: f32,
    ser_decay: f32,
}

impl Default for Neuromodulators {
    fn default() -> Self {
        Self {
            dopamine: 1.0, acetylcholine: 1.0, norepinephrine: 1.0, serotonin: 1.0,
            da_decay: 0.995, ach_decay: 0.99, ne_decay: 0.98, ser_decay: 0.999,
        }
    }
}

impl Neuromodulators {
    /// Combined modulator for three-factor learning.
    pub fn learning_modulator(&self) -> f32 {
        self.dopamine * (1.0 + self.acetylcholine * 0.5)
    }

    /// Global gain factor for neuronal excitability.
    pub fn gain_modulator(&self) -> f32 {
        1.0 + (self.norepinephrine - 1.0) * 0.3
    }

    /// Exploration noise level (low serotonin = more exploration).
    pub fn exploration_noise(&self) -> f32 {
        (2.0 - self.serotonin).max(0.0) * 0.1
    }

    pub fn reward(&mut self, magnitude: f32) { self.dopamine += magnitude; }
    pub fn novelty(&mut self, magnitude: f32) { self.acetylcholine += magnitude; }
    pub fn arousal(&mut self, magnitude: f32) { self.norepinephrine += magnitude; }
    pub fn set_mood(&mut self, mood_level: f32) { self.serotonin = mood_level.clamp(0.0, 2.0); }

    /// Decay all modulators toward baseline (1.0) each timestep.
    pub fn step(&mut self) {
        self.dopamine = 1.0 + (self.dopamine - 1.0) * self.da_decay;
        self.acetylcholine = 1.0 + (self.acetylcholine - 1.0) * self.ach_decay;
        self.norepinephrine = 1.0 + (self.norepinephrine - 1.0) * self.ne_decay;
        self.serotonin *= self.ser_decay;
        self.serotonin = self.serotonin.max(0.1);
    }
}
