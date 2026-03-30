//! Mel spectrogram computation for Whisper.
//!
//! Whisper expects: 80-bin log-mel spectrogram, 30s padded, hop=160, n_fft=400.
//! Output shape: (80, 3000) for 30 seconds of 16kHz audio.

use rustfft::{FftPlanner, num_complex::Complex};

/// Whisper mel spectrogram parameters.
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const CHUNK_LENGTH: usize = 30; // seconds
const N_FRAMES: usize = 3000;   // 30s * 16000 / 160

/// Compute log-mel spectrogram compatible with Whisper.
/// Input: 16kHz mono float32 PCM samples.
/// Output: flat f32 array of shape (1, 80, 3000) = 240000 elements.
pub fn compute_log_mel(samples: &[f32]) -> Vec<f32> {
    let n_samples = SAMPLE_RATE * CHUNK_LENGTH; // 480000

    // Pad or truncate to 30 seconds
    let mut audio = vec![0.0f32; n_samples];
    let copy_len = samples.len().min(n_samples);
    audio[..copy_len].copy_from_slice(&samples[..copy_len]);

    // STFT
    let magnitudes = stft(&audio, N_FFT, HOP_LENGTH);

    // Mel filterbank
    let mel_filters = mel_filterbank(N_MELS, N_FFT);

    // Apply mel filterbank: (n_mels, n_fft/2+1) @ (n_fft/2+1, n_frames) → (n_mels, n_frames)
    let n_freq = N_FFT / 2 + 1;
    let n_frames = magnitudes.len() / n_freq;
    let n_frames = n_frames.min(N_FRAMES);

    let mut mel_spec = vec![0.0f32; N_MELS * N_FRAMES];
    for m in 0..N_MELS {
        for t in 0..n_frames {
            let mut sum = 0.0f32;
            for f in 0..n_freq {
                sum += mel_filters[m * n_freq + f] * magnitudes[t * n_freq + f];
            }
            // Log compression (matches Whisper's log10 clamp)
            mel_spec[m * N_FRAMES + t] = sum.max(1e-10).log10().max(-8.0);
        }
    }

    // Normalize: (mel - max) / (-max) * 4.0 + 4.0 (Whisper normalization)
    let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val > -8.0 {
        for v in &mut mel_spec {
            *v = (*v - max_val).max(-8.0);
            *v = *v / 4.0 + 1.0;  // Scale to roughly [0, 1] range
        }
    }

    mel_spec
}

/// Short-Time Fourier Transform.
/// Returns magnitude spectrum: Vec of (n_fft/2+1) values per frame.
fn stft(audio: &[f32], n_fft: usize, hop: usize) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let n_freq = n_fft / 2 + 1;

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
        .collect();

    let n_frames = (audio.len().saturating_sub(n_fft)) / hop + 1;
    let mut magnitudes = Vec::with_capacity(n_frames * n_freq);

    let mut buffer = vec![Complex::new(0.0f32, 0.0); n_fft];

    for frame in 0..n_frames {
        let start = frame * hop;
        // Apply window and copy to complex buffer
        for i in 0..n_fft {
            let sample = if start + i < audio.len() { audio[start + i] } else { 0.0 };
            buffer[i] = Complex::new(sample * window[i], 0.0);
        }

        // FFT in-place
        fft.process(&mut buffer);

        // Magnitude (power spectrum)
        for i in 0..n_freq {
            let mag = (buffer[i].re * buffer[i].re + buffer[i].im * buffer[i].im).sqrt();
            magnitudes.push(mag * mag); // Power spectrum (squared magnitude)
        }
    }

    magnitudes
}

/// Build mel filterbank: (n_mels, n_fft/2+1).
fn mel_filterbank(n_mels: usize, n_fft: usize) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let f_max = SAMPLE_RATE as f32 / 2.0;
    let f_min = 0.0f32;

    // Mel scale conversion
    let hz_to_mel = |f: f32| -> f32 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f32| -> f32 { 700.0 * (10.0f32.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 equally spaced points in mel scale
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert back to Hz then to FFT bin indices
    let bin_indices: Vec<f32> = mel_points.iter()
        .map(|&m| mel_to_hz(m) * n_fft as f32 / SAMPLE_RATE as f32)
        .collect();

    // Build triangular filters
    let mut filters = vec![0.0f32; n_mels * n_freq];
    for m in 0..n_mels {
        let left = bin_indices[m];
        let center = bin_indices[m + 1];
        let right = bin_indices[m + 2];

        for f in 0..n_freq {
            let freq = f as f32;
            if freq >= left && freq <= center && center > left {
                filters[m * n_freq + f] = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                filters[m * n_freq + f] = (right - freq) / (right - center);
            }
        }
    }

    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_shape() {
        // 1 second of silence
        let samples = vec![0.0f32; 16000];
        let mel = compute_log_mel(&samples);
        assert_eq!(mel.len(), N_MELS * N_FRAMES); // 80 * 3000 = 240000
    }

    #[test]
    fn test_mel_with_tone() {
        // 440Hz sine wave, 1 second
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let mel = compute_log_mel(&samples);
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
        // Should have non-trivial values in the mel bins corresponding to 440Hz
        let max_val = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > 0.0, "Mel should have positive values for a tone, got {max_val}");
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let filters = mel_filterbank(80, 400);
        assert_eq!(filters.len(), 80 * 201); // 80 mels × (400/2+1) freq bins
        // Each filter should have some non-zero values
        for m in 0..80 {
            let sum: f32 = filters[m * 201..(m + 1) * 201].iter().sum();
            assert!(sum > 0.0, "Mel filter {m} should have non-zero values");
        }
    }
}
