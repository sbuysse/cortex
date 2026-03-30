//! Emotion detection from audio features.
//!
//! Simple linear classifier on Whisper 512-dim embeddings → 7 emotions.
//! Weights loaded from a .bin file (512×7 matrix + 7 bias).
//! Falls back to energy-based heuristic if no trained model.

use std::path::Path;

pub const EMOTIONS: [&str; 7] = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "tired"];

/// Emotion classifier — linear head on audio embeddings.
pub struct EmotionClassifier {
    weights: Vec<f32>,  // 512 × 7 = 3584
    bias: Vec<f32>,     // 7
}

impl EmotionClassifier {
    /// Load trained weights from .bin file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        if data.len() >= (512 * 7 + 7) * 4 {
            let floats: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let weights = floats[..512 * 7].to_vec();
            let bias = floats[512 * 7..512 * 7 + 7].to_vec();
            tracing::info!("Emotion classifier loaded from {:?}", path);
            Ok(Self { weights, bias })
        } else {
            Err("Emotion model too small".into())
        }
    }

    /// Create with random/heuristic weights (no trained model).
    pub fn heuristic() -> Self {
        // Energy-based heuristic: map certain embedding dimensions to emotions
        // This is a rough approximation until we train on RAVDESS
        let mut weights = vec![0.0f32; 512 * 7];
        let bias = vec![0.0, -0.5, -0.3, -0.8, -0.8, -0.6, -0.2]; // neutral bias

        // Heuristic: higher energy → happy/angry, lower → sad/tired
        for i in 0..512 {
            // Neutral gets small positive weight everywhere
            weights[i * 7 + 0] = 0.01;
            // Happy: positive in first quadrant
            if i < 128 { weights[i * 7 + 1] = 0.05; }
            // Sad: positive in low-energy dimensions
            if i >= 256 && i < 384 { weights[i * 7 + 2] = 0.04; }
            // Angry: positive in high-energy dimensions
            if i < 64 { weights[i * 7 + 3] = 0.06; }
            // Fearful: mid-range
            if i >= 128 && i < 256 { weights[i * 7 + 4] = 0.03; }
            // Surprised: scattered
            if i % 3 == 0 { weights[i * 7 + 5] = 0.02; }
            // Tired: low overall
            if i >= 384 { weights[i * 7 + 6] = 0.04; }
        }
        Self { weights, bias }
    }

    /// Classify emotion from a 512-dim audio embedding.
    /// Returns (emotion_label, confidence, all_scores).
    pub fn classify(&self, embedding: &[f32]) -> (&'static str, f32, Vec<(String, f32)>) {
        assert!(embedding.len() >= 512);

        // Linear: scores = embedding @ weights + bias
        let mut scores = self.bias.clone();
        for i in 0..512 {
            for j in 0..7 {
                scores[j] += embedding[i] * self.weights[i * 7 + j];
            }
        }

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();

        // Find best
        let (best_idx, &best_prob) = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();

        let all: Vec<(String, f32)> = EMOTIONS.iter().zip(&probs)
            .map(|(e, p)| (e.to_string(), (*p * 1000.0).round() / 1000.0))
            .collect();

        (EMOTIONS[best_idx], best_prob, all)
    }

    /// Classify from raw audio samples using energy-based heuristics.
    /// No model needed — uses RMS energy, zero-crossing rate, spectral centroid.
    pub fn classify_from_audio(samples: &[f32]) -> (&'static str, f32) {
        if samples.is_empty() { return ("neutral", 0.5); }

        let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        let zcr: f32 = samples.windows(2).filter(|w| w[0].signum() != w[1].signum()).count() as f32 / samples.len() as f32;

        // Simple heuristic
        if rms < 0.01 { return ("tired", 0.6); }
        if rms < 0.03 { return ("sad", 0.5); }
        if rms > 0.15 && zcr > 0.1 { return ("angry", 0.5); }
        if rms > 0.1 && zcr < 0.05 { return ("happy", 0.5); }
        if zcr > 0.15 { return ("fearful", 0.4); }
        ("neutral", 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_classifier() {
        let clf = EmotionClassifier::heuristic();
        let emb = vec![0.1f32; 512];
        let (emotion, conf, all) = clf.classify(&emb);
        assert!(EMOTIONS.contains(&emotion));
        assert!(conf > 0.0 && conf <= 1.0);
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_audio_heuristic() {
        // Silence → tired
        let (e, _) = EmotionClassifier::classify_from_audio(&vec![0.001; 1600]);
        assert_eq!(e, "tired");

        // Loud → not tired
        let (e, _) = EmotionClassifier::classify_from_audio(&vec![0.2; 1600]);
        assert_ne!(e, "tired");
    }
}
