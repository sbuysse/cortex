/// Latency coding: converts f32 embeddings to spike times.
/// High values spike first (low spike time), low values spike last.
pub struct LatencyEncoder {
    dim: usize,
    t_max: u16,
}

impl LatencyEncoder {
    pub fn new(dim: usize, t_max: u16) -> Self { Self { dim, t_max } }

    pub fn encode(&self, embedding: &[f32]) -> Vec<u16> {
        assert_eq!(embedding.len(), self.dim);
        embedding.iter().map(|&v| {
            let clamped = v.clamp(0.0, 1.0);
            let t = ((1.0 - clamped) * self.t_max as f32) as u16;
            t.min(self.t_max)
        }).collect()
    }

    pub fn inject(&self, spike_times: &[u16], current_step: u16, currents: &mut [f32], spike_current: f32) -> usize {
        let mut count = 0;
        for (i, &t) in spike_times.iter().enumerate() {
            if t == current_step {
                currents[i] += spike_current;
                count += 1;
            }
        }
        count
    }
}
