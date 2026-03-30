//! Voice Activity Detection — detect when someone is speaking.
//!
//! Uses energy-based VAD (no model needed) for initial detection,
//! with optional Silero VAD ONNX model for higher accuracy.
//! Designed to work on Raspberry Pi (minimal CPU).

/// Simple energy-based Voice Activity Detection.
/// Returns true if audio samples contain speech-like energy.
pub fn detect_voice_energy(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() { return false; }
    let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    rms > threshold
}

/// Detect speech segments in a stream of audio.
/// Returns (is_speaking, rms_energy, zero_crossing_rate).
pub fn analyze_audio_frame(samples: &[f32]) -> (bool, f32, f32) {
    if samples.is_empty() { return (false, 0.0, 0.0); }

    let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    let zcr: f32 = samples.windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count() as f32 / samples.len() as f32;

    // Speech typically: RMS > 0.01, ZCR between 0.01 and 0.2
    let is_speech = rms > 0.01 && zcr > 0.01 && zcr < 0.3;

    (is_speech, rms, zcr)
}

/// Voice Activity Detector with state tracking.
pub struct VoiceActivityDetector {
    /// Minimum consecutive speech frames to trigger
    pub min_speech_frames: usize,
    /// Minimum consecutive silence frames to end speech
    pub min_silence_frames: usize,
    /// Energy threshold for speech
    pub energy_threshold: f32,

    speech_count: usize,
    silence_count: usize,
    is_speaking: bool,
}

impl VoiceActivityDetector {
    pub fn new() -> Self {
        Self {
            min_speech_frames: 3,     // ~150ms at 50ms frames
            min_silence_frames: 10,   // ~500ms of silence to end
            energy_threshold: 0.01,
            speech_count: 0,
            silence_count: 0,
            is_speaking: false,
        }
    }

    /// Process a frame of audio (e.g., 50ms = 800 samples at 16kHz).
    /// Returns: (speech_started, speech_ended, is_currently_speaking).
    pub fn process_frame(&mut self, samples: &[f32]) -> (bool, bool, bool) {
        let (has_speech, _, _) = analyze_audio_frame(samples);
        let mut started = false;
        let mut ended = false;

        if has_speech {
            self.speech_count += 1;
            self.silence_count = 0;

            if !self.is_speaking && self.speech_count >= self.min_speech_frames {
                self.is_speaking = true;
                started = true;
            }
        } else {
            self.silence_count += 1;
            self.speech_count = 0;

            if self.is_speaking && self.silence_count >= self.min_silence_frames {
                self.is_speaking = false;
                ended = true;
            }
        }

        (started, ended, self.is_speaking)
    }

    pub fn is_speaking(&self) -> bool { self.is_speaking }

    pub fn reset(&mut self) {
        self.speech_count = 0;
        self.silence_count = 0;
        self.is_speaking = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_detection() {
        // Silence
        assert!(!detect_voice_energy(&vec![0.001; 800], 0.01));
        // Speech-like
        assert!(detect_voice_energy(&vec![0.1; 800], 0.01));
    }

    #[test]
    fn test_vad_state_machine() {
        let mut vad = VoiceActivityDetector::new();

        // Silence frames
        for _ in 0..5 {
            let (started, ended, speaking) = vad.process_frame(&vec![0.001; 800]);
            assert!(!started && !ended && !speaking);
        }

        // Speech frames
        let speech: Vec<f32> = (0..800).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
        for i in 0..5 {
            let (started, _, speaking) = vad.process_frame(&speech);
            if i == 2 { assert!(started, "Should start after 3 frames"); }
            if i >= 2 { assert!(speaking); }
        }

        // Silence → speech ends
        for i in 0..15 {
            let (_, ended, _) = vad.process_frame(&vec![0.001; 800]);
            if i == 9 { assert!(ended, "Should end after 10 silence frames"); }
        }
    }
}
