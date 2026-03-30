//! Working Memory — 7±2 slots with theta-gamma phase ordering.
//!
//! Inspired by theta-gamma coupling (PMC11211613):
//! items are held at different theta phases, creating temporal sequence order.
//! Gamma bursts boost items near the current theta phase.

use serde::Serialize;

/// A single item in working memory.
#[derive(Debug, Clone, Serialize)]
pub struct WmItem {
    pub embedding: Vec<f32>,
    pub label: String,
    pub modality: String,
    pub timestamp: f64,
    pub activation: f32,
    pub theta_phase: f32,
}

/// Working memory state.
#[derive(Debug, Clone, Serialize)]
pub struct WmState {
    pub slots_used: usize,
    pub max_slots: usize,
    pub theta_phase: f32,
    pub items: Vec<WmItemView>,
    pub focus: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WmItemView {
    pub label: String,
    pub modality: String,
    pub activation: f32,
    pub theta_phase: f32,
    pub age_seconds: f64,
}

/// Working memory system with theta-gamma oscillations.
pub struct WorkingMemory {
    items: Vec<WmItem>,
    theta_phase: f32,
    max_slots: usize,
    decay: f32,
    theta_freq: f32,
}

impl WorkingMemory {
    pub fn new(max_slots: usize, decay: f32, theta_freq: f32) -> Self {
        Self {
            items: Vec::new(),
            theta_phase: 0.0,
            max_slots,
            decay,
            theta_freq,
        }
    }

    /// Add a new perception to working memory.
    /// Returns attention score and related items.
    pub fn update(&mut self, embedding: Vec<f32>, label: String, modality: String) -> WmUpdateResult {
        let now = now_secs();

        // Advance theta phase
        self.theta_phase = (self.theta_phase + self.theta_freq * 2.0 * std::f32::consts::PI)
            % (2.0 * std::f32::consts::PI);

        // Decay existing activations + gamma boost
        for item in &mut self.items {
            item.activation *= self.decay;
            // Gamma burst: boost items near current theta phase
            let phase_dist = (item.theta_phase - self.theta_phase).abs();
            let phase_dist = phase_dist.min(2.0 * std::f32::consts::PI - phase_dist);
            let gamma_boost = (-phase_dist * phase_dist / 0.5).exp();
            item.activation = (item.activation + gamma_boost * 0.15).min(1.0);
        }

        // Compute attention scores with existing items
        let mut attention_scores = Vec::new();
        let mut related_items = Vec::new();
        let emb_norm = l2_normalize(&embedding);

        for item in &mut self.items {
            let sim = dot_product(&emb_norm, &l2_normalize(&item.embedding));
            attention_scores.append(&mut vec![sim]);
            if sim > 0.5 {
                item.activation = (item.activation + 0.2 * sim).min(1.0);
                related_items.push(item.label.clone());
            }
        }

        // Add new item
        self.items.push(WmItem {
            embedding,
            label: label.clone(),
            modality,
            timestamp: now,
            activation: 1.0,
            theta_phase: self.theta_phase,
        });

        // Prune: remove below threshold, cap at max_slots
        self.items.retain(|item| item.activation > 0.1);
        if self.items.len() > self.max_slots {
            self.items.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap());
            self.items.truncate(self.max_slots);
        }

        let avg_attention = if attention_scores.is_empty() {
            0.0
        } else {
            attention_scores.iter().sum::<f32>() / attention_scores.len() as f32
        };

        WmUpdateResult {
            attention_score: avg_attention,
            related_items,
            buffer_size: self.items.len(),
        }
    }

    /// Get the current state for API response.
    pub fn get_state(&self) -> WmState {
        let now = now_secs();
        let mut items: Vec<WmItemView> = self.items.iter().map(|item| WmItemView {
            label: item.label.clone(),
            modality: item.modality.clone(),
            activation: (item.activation * 1000.0).round() / 1000.0,
            theta_phase: (item.theta_phase * 10000.0).round() / 10000.0,
            age_seconds: ((now - item.timestamp) * 10.0).round() / 10.0,
        }).collect();
        items.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap());

        WmState {
            slots_used: self.items.len(),
            max_slots: self.max_slots,
            theta_phase: (self.theta_phase * 10000.0).round() / 10000.0,
            items: items.clone(),
            focus: items.first().map(|i| i.label.clone()),
        }
    }

    /// Get embeddings of current WM items (for fast memory store).
    pub fn get_embeddings(&self) -> Vec<&[f32]> {
        self.items.iter().map(|i| i.embedding.as_slice()).collect()
    }
}

pub struct WmUpdateResult {
    pub attention_score: f32,
    pub related_items: Vec<String>,
    pub buffer_size: usize,
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_retrieve() {
        let mut wm = WorkingMemory::new(7, 0.85, 0.15);
        let emb = vec![0.1f32; 512];
        wm.update(emb, "thunder".into(), "audio".into());
        let state = wm.get_state();
        assert_eq!(state.slots_used, 1);
        assert_eq!(state.focus, Some("thunder".into()));
    }

    #[test]
    fn test_max_slots() {
        let mut wm = WorkingMemory::new(3, 0.85, 0.15);
        for i in 0..5 {
            wm.update(vec![i as f32; 512], format!("item_{i}"), "test".into());
        }
        assert!(wm.get_state().slots_used <= 3);
    }

    #[test]
    fn test_theta_phase_advances() {
        let mut wm = WorkingMemory::new(7, 0.85, 0.15);
        let p1 = wm.theta_phase;
        wm.update(vec![0.1; 512], "a".into(), "t".into());
        let p2 = wm.theta_phase;
        assert_ne!(p1, p2, "Theta phase should advance");
    }

    #[test]
    fn test_decay() {
        let mut wm = WorkingMemory::new(7, 0.5, 0.15);
        wm.update(vec![0.1; 512], "a".into(), "t".into());
        let a1 = wm.items[0].activation;
        wm.update(vec![0.2; 512], "b".into(), "t".into());
        let a2 = wm.items[0].activation;
        assert!(a2 < a1, "First item should decay after second insert");
    }
}
