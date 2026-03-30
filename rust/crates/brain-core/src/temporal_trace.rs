//! Temporal trace modules for the Associative Core.
//!
//! Exponential moving average traces over activation vectors, enabling
//! associations between temporally proximate events.

use ndarray::{Array1, Array2};

/// Single shared trace (batch mean aggregation).
pub struct TemporalTrace {
    pub dim: usize,
    pub decay: f32,
    /// Running trace [D] internally.
    pub trace: Array1<f32>,
    /// Pre-allocated [1, D] view for returning without allocation.
    trace_2d: Array2<f32>,
    /// Pre-allocated batch mean buffer [D].
    batch_mean_buf: Array1<f32>,
}

impl TemporalTrace {
    pub fn new(dim: usize, decay: f32) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert!(
            (0.0..1.0).contains(&decay),
            "decay must be in [0, 1)"
        );

        Self {
            dim,
            decay,
            trace: Array1::zeros(dim),
            trace_2d: Array2::zeros((1, dim)),
            batch_mean_buf: Array1::zeros(dim),
        }
    }

    /// Blend current into the running trace and return as [1, D] reference.
    ///
    /// current: [B, D] — batch mean is used for the trace update.
    #[inline]
    pub fn update(&mut self, current: &Array2<f32>) -> &Array2<f32> {
        let batch_size = current.nrows() as f32;
        let alpha = 1.0 - self.decay;
        let decay = self.decay;

        // Compute batch mean in-place (no allocation)
        self.batch_mean_buf.fill(0.0);
        for row in current.rows() {
            self.batch_mean_buf.zip_mut_with(&row, |m, &v| *m += v);
        }
        let inv_bs = 1.0 / batch_size;

        // Fuse: batch_mean scale + EMA update + trace_2d copy in one pass
        let trace_slice = self.trace.as_slice_mut().unwrap();
        let mean_slice = self.batch_mean_buf.as_slice().unwrap();
        let out_slice = self.trace_2d.as_slice_mut().unwrap();
        for i in 0..self.dim {
            let b = mean_slice[i] * inv_bs;
            let t = decay * trace_slice[i] + alpha * b;
            trace_slice[i] = t;
            out_slice[i] = t;
        }
        &self.trace_2d
    }

    /// Zero out the trace.
    pub fn reset(&mut self) {
        self.trace.fill(0.0);
        self.trace_2d.fill(0.0);
    }

    /// Return the current trace as [1, D] reference.
    pub fn get_trace(&self) -> &Array2<f32> {
        &self.trace_2d
    }

    pub fn norm(&self) -> f32 {
        self.trace.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    pub fn mean(&self) -> f32 {
        self.trace.mean().unwrap_or(0.0)
    }
}

/// Per-sample temporal traces for a fixed batch size.
pub struct BatchTemporalTrace {
    pub batch_size: usize,
    pub dim: usize,
    pub decay: f32,
    /// Per-sample traces [B, D].
    pub trace: Array2<f32>,
}

impl BatchTemporalTrace {
    pub fn new(batch_size: usize, dim: usize, decay: f32) -> Self {
        assert!(batch_size > 0);
        assert!(dim > 0);
        assert!((0.0..1.0).contains(&decay));

        Self {
            batch_size,
            dim,
            decay,
            trace: Array2::zeros((batch_size, dim)),
        }
    }

    /// Update per-sample traces. current: [B, D]. Returns reference to trace.
    #[inline]
    pub fn update(&mut self, current: &Array2<f32>) -> &Array2<f32> {
        assert_eq!(current.nrows(), self.batch_size, "batch_size mismatch");

        let alpha = 1.0 - self.decay;
        self.trace.zip_mut_with(current, |t, &c| {
            *t = self.decay * *t + alpha * c;
        });
        &self.trace
    }

    pub fn reset(&mut self) {
        self.trace.fill(0.0);
    }

    pub fn get_trace(&self) -> &Array2<f32> {
        &self.trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_temporal_trace_update() {
        let mut trace = TemporalTrace::new(8, 0.95);
        let current = Array2::ones((4, 8));
        let result = trace.update(&current);
        assert_eq!(result.dim(), (1, 8));
        for &v in result.iter() {
            assert!((v - 0.05).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch_trace_shape() {
        let mut trace = BatchTemporalTrace::new(4, 8, 0.9);
        let current = Array2::ones((4, 8));
        let result = trace.update(&current);
        assert_eq!(result.dim(), (4, 8));
    }
}
