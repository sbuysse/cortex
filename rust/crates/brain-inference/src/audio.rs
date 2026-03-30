//! Audio encoder — Whisper encoder via TorchScript.

use std::path::Path;
use tch::{CModule, Device, Tensor};

/// Whisper-base audio encoder: mel spectrogram → 512-dim embedding.
pub struct WhisperEncoder {
    model: CModule,
}

impl WhisperEncoder {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let model = CModule::load(path)?;
        tracing::info!("Whisper encoder loaded from {:?}", path);
        Ok(Self { model })
    }

    /// Encode mel features (1, 80, 3000) → 512-dim L2-normalized embedding.
    /// The Whisper encoder outputs hidden states which are mean-pooled.
    pub fn encode(&self, mel_features: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let n = mel_features.len();
        // Expected: 80 * 3000 = 240000 for 30s audio, or smaller for shorter
        let frames = n / 80;
        let input = Tensor::from_slice(mel_features)
            .reshape(&[1, 80, frames as i64])
            .to_device(Device::Cpu);
        let output = self.model.forward_ts(&[input])?;
        let emb: Vec<f32> = Vec::try_from(output.flatten(0, -1))?;
        // L2 normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(emb.iter().map(|x| x / norm).collect())
    }

    /// Encode raw PCM audio samples (16kHz, mono, f32) → 512-dim embedding.
    /// This requires Whisper's mel spectrogram preprocessing which we do here.
    pub fn encode_pcm(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // For now, create a simple mel-like feature from the raw audio
        // A proper implementation would use the Whisper mel filter bank
        // This is a placeholder — the actual mel computation should match whisper's
        let n_fft = 400;
        let hop = 160;
        let n_mels = 80;
        let n_frames = 3000; // 30s at 16kHz

        // Pad or truncate to 30s
        let max_samples = sample_rate as usize * 30;
        let mut padded = vec![0.0f32; max_samples];
        let copy_len = samples.len().min(max_samples);
        padded[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Simple power spectrum → mel approximation
        // (proper implementation needs FFT + mel filterbank)
        let mut mel = vec![0.0f32; n_mels * n_frames];
        for frame in 0..n_frames {
            let start = frame * hop;
            for m in 0..n_mels {
                let mut power = 0.0f32;
                let freq_start = m * n_fft / n_mels / 2;
                let freq_end = (m + 1) * n_fft / n_mels / 2;
                for k in freq_start..freq_end.min(n_fft/2) {
                    let idx = start + k;
                    if idx < padded.len() {
                        power += padded[idx] * padded[idx];
                    }
                }
                mel[m * n_frames + frame] = (power + 1e-10).log10();
            }
        }

        self.encode(&mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_whisper_load_and_encode() {
        let path = PathBuf::from("/opt/brain/outputs/cortex/audio_encoder/whisper_encoder_ts.pt");
        if !path.exists() { return; }
        let enc = WhisperEncoder::load(&path).unwrap();
        // Create dummy mel features (80 mels, 3000 frames)
        let mel = vec![0.0f32; 80 * 3000];
        let emb = enc.encode(&mel).unwrap();
        assert_eq!(emb.len(), 512);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
