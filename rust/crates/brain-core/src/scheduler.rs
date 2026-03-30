//! Learning rate schedulers for Hebbian training.

use serde::{Deserialize, Serialize};

/// Warmup + decay scheduler for Hebbian learning rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianScheduler {
    pub base_lr: f32,
    pub warmup_steps: usize,
    pub min_lr: f32,
    pub max_steps: usize,
    pub decay_type: DecayType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayType {
    Cosine,
    Linear,
}

impl HebbianScheduler {
    pub fn new(
        base_lr: f32,
        warmup_steps: usize,
        decay_type: DecayType,
        min_lr: f32,
        max_steps: usize,
    ) -> Self {
        Self {
            base_lr,
            warmup_steps,
            min_lr,
            max_steps,
            decay_type,
        }
    }

    /// Get the learning rate for the given step.
    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = (step + 1) as f32 / self.warmup_steps as f32;
            return self.min_lr + (self.base_lr - self.min_lr) * progress;
        }

        if self.max_steps <= self.warmup_steps {
            return self.base_lr;
        }

        let decay_steps = self.max_steps - self.warmup_steps;
        let decay_progress = (step - self.warmup_steps) as f32 / decay_steps as f32;
        let decay_progress = decay_progress.min(1.0);

        match self.decay_type {
            DecayType::Cosine => {
                let cosine = (1.0 + (std::f32::consts::PI * decay_progress).cos()) / 2.0;
                self.min_lr + (self.base_lr - self.min_lr) * cosine
            }
            DecayType::Linear => {
                self.base_lr + (self.min_lr - self.base_lr) * decay_progress
            }
        }
    }
}

/// Controls consolidation cycle frequency.
pub struct ConsolidationScheduler {
    pub initial_interval: usize,
    pub final_interval: usize,
    pub warmup_steps: usize,
}

impl ConsolidationScheduler {
    pub fn new(initial_interval: usize, final_interval: usize, warmup_steps: usize) -> Self {
        Self {
            initial_interval,
            final_interval,
            warmup_steps,
        }
    }

    pub fn get_interval(&self, step: usize) -> usize {
        if step < self.warmup_steps {
            self.initial_interval
        } else {
            self.final_interval
        }
    }

    pub fn should_consolidate(&self, step: usize) -> bool {
        if step == 0 {
            return false;
        }
        let interval = self.get_interval(step);
        step % interval == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_schedule() {
        let sched = HebbianScheduler::new(0.01, 100, DecayType::Cosine, 1e-5, 5000);
        // Warmup: step 0 should be close to min_lr
        let lr0 = sched.get_lr(0);
        assert!(lr0 < 0.01);

        // Step 100 should be at base_lr
        let lr100 = sched.get_lr(100);
        assert!((lr100 - 0.01).abs() < 1e-4);

        // Step 5000 should be at min_lr
        let lr_end = sched.get_lr(5000);
        assert!((lr_end - 1e-5).abs() < 1e-5);
    }

    #[test]
    fn test_consolidation_scheduler() {
        let sched = ConsolidationScheduler::new(100, 500, 200);
        assert!(!sched.should_consolidate(0));
        assert!(sched.should_consolidate(100));
        assert!(!sched.should_consolidate(150));
        assert!(sched.should_consolidate(500));
    }
}
