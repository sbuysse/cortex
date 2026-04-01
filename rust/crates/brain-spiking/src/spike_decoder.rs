/// Rate-based decoder: converts spike counts over a window into f32 embedding.
pub struct RateDecoder {
    dim: usize,
    window: usize,
    spike_counts: Vec<u32>,
}

impl RateDecoder {
    pub fn new(dim: usize, window: usize) -> Self {
        Self { dim, window, spike_counts: vec![0; dim] }
    }

    pub fn record_spike(&mut self, idx: usize, _t: usize) { self.spike_counts[idx] += 1; }

    pub fn decode(&self) -> Vec<f32> {
        let max_count = *self.spike_counts.iter().max().unwrap_or(&1).max(&1);
        self.spike_counts.iter().map(|&c| c as f32 / max_count as f32).collect()
    }

    pub fn reset(&mut self) { self.spike_counts.fill(0); }
}
