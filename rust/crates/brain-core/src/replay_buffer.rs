//! Prioritised circular replay buffer for experience replay.
//!
//! Stores projected embeddings (visual, audio, emotion) with priority-weighted sampling.

use ndarray::Array2;
use rand::Rng;

const PRIORITY_EPS: f32 = 1e-8;

/// Fixed-capacity circular buffer with prioritised sampling.
pub struct ReplayBuffer {
    pub capacity: usize,
    pub embed_dim: usize,
    pub priority_exponent: f32,

    pub v_buffer: Array2<f32>,
    pub a_buffer: Array2<f32>,
    pub e_buffer: Array2<f32>,
    pub timestamps: Vec<f32>,
    pub priorities: Vec<f32>,

    pub position: usize,
    pub size: usize,
    has_emotion: bool,
    weights_dirty: bool,
    cached_weights: Vec<f32>,
    cached_cumsum: Vec<f32>,
}

/// Sampled batch from the replay buffer.
pub struct ReplaySample {
    pub v: Array2<f32>,
    pub a: Array2<f32>,
    pub e: Option<Array2<f32>>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, embed_dim: usize, priority_exponent: f32) -> Self {
        assert!(capacity > 0);
        assert!(embed_dim > 0);

        Self {
            capacity,
            embed_dim,
            priority_exponent,
            v_buffer: Array2::zeros((capacity, embed_dim)),
            a_buffer: Array2::zeros((capacity, embed_dim)),
            e_buffer: Array2::zeros((capacity, embed_dim)),
            timestamps: vec![0.0; capacity],
            priorities: vec![1.0; capacity],
            position: 0,
            size: 0,
            has_emotion: false,
            weights_dirty: true,
            cached_weights: Vec::new(),
            cached_cumsum: Vec::new(),
        }
    }

    /// Add experiences to the buffer. Inputs are [B, D] arrays.
    pub fn add(
        &mut self,
        v_proj: &Array2<f32>,
        a_proj: &Array2<f32>,
        e_proj: Option<&Array2<f32>>,
        timestamp: f32,
        surprise: f32,
    ) {
        let batch_size = v_proj.nrows();
        let surprise = surprise.max(PRIORITY_EPS);

        for b in 0..batch_size {
            let idx = (self.position + b) % self.capacity;
            self.v_buffer
                .row_mut(idx)
                .assign(&v_proj.row(b));
            self.a_buffer
                .row_mut(idx)
                .assign(&a_proj.row(b));
            if let Some(e) = e_proj {
                self.e_buffer.row_mut(idx).assign(&e.row(b));
                self.has_emotion = true;
            }
            self.timestamps[idx] = timestamp;
            self.priorities[idx] = surprise;
        }

        self.position = (self.position + batch_size) % self.capacity;
        self.size = (self.size + batch_size).min(self.capacity);
        self.weights_dirty = true;
    }

    /// Sample a batch with priority-weighted probabilities.
    pub fn sample(&mut self, batch_size: usize) -> ReplaySample {
        assert!(self.size > 0, "Cannot sample from empty buffer");

        let indices = self.priority_indices(batch_size);

        let d = self.embed_dim;
        let mut v = Array2::zeros((batch_size, d));
        let mut a = Array2::zeros((batch_size, d));
        let mut e = Array2::zeros((batch_size, d));

        for (i, &idx) in indices.iter().enumerate() {
            v.row_mut(i).assign(&self.v_buffer.row(idx));
            a.row_mut(i).assign(&self.a_buffer.row(idx));
            e.row_mut(i).assign(&self.e_buffer.row(idx));
        }

        ReplaySample {
            v,
            a,
            e: if self.has_emotion { Some(e) } else { None },
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    fn priority_indices(&mut self, n: usize) -> Vec<usize> {
        if self.weights_dirty || self.cached_weights.is_empty() {
            self.cached_weights.clear();
            self.cached_weights.reserve(self.size);
            self.cached_weights.extend(
                self.priorities[..self.size]
                    .iter()
                    .map(|&p| p.powf(self.priority_exponent)),
            );
            let sum: f32 = self.cached_weights.iter().sum();
            if sum > 0.0 {
                self.cached_weights.iter_mut().for_each(|w| *w /= sum);
            }

            // Cache cumulative sum
            self.cached_cumsum.clear();
            self.cached_cumsum.reserve(self.size);
            let mut acc = 0.0f32;
            for &w in &self.cached_weights {
                acc += w;
                self.cached_cumsum.push(acc);
            }

            self.weights_dirty = false;
        }

        let mut rng = rand::rng();
        (0..n)
            .map(|_| {
                let r: f32 = rng.random();
                self.cached_cumsum
                    .binary_search_by(|w| w.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(|i| i.min(self.size - 1))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_add_and_sample() {
        let mut buf = ReplayBuffer::new(100, 8, 0.6);
        let v = Array2::ones((4, 8));
        let a = Array2::ones((4, 8));
        buf.add(&v, &a, None, 0.0, 1.0);
        assert_eq!(buf.len(), 4);

        let sample = buf.sample(2);
        assert_eq!(sample.v.dim(), (2, 8));
        assert!(sample.e.is_none());
    }

    #[test]
    fn test_circular_wrap() {
        let mut buf = ReplayBuffer::new(10, 4, 0.6);
        for i in 0..5 {
            let v = Array2::from_elem((3, 4), i as f32);
            let a = Array2::zeros((3, 4));
            buf.add(&v, &a, None, i as f32, 1.0);
        }
        assert_eq!(buf.len(), 10); // capped at capacity
    }
}
