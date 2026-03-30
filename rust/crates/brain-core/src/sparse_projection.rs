//! Sparse projection layer for the shared embedding space.
//!
//! Maps encoder-specific embeddings of dimension d_in into a shared d_out-dimensional
//! space, enforcing top-k sparsity and L2 normalisation.

use ndarray::{Array1, Array2};
use rand::Rng;

/// Linear projection followed by top-k sparsification and L2 normalisation.
pub struct SparseProjection {
    /// Projection weights [d_in, d_out]
    pub proj_weight: Array2<f32>,
    /// Bias [d_out]
    pub proj_bias: Array1<f32>,
    pub k: usize,
    pub d_in: usize,
    pub d_out: usize,
}

impl SparseProjection {
    pub fn new(d_in: usize, d_out: usize, k: usize) -> Self {
        assert!(k >= 1 && k <= d_out, "k must be in [1, d_out]");

        // Xavier/Glorot uniform initialization
        let mut rng = rand::rng();
        let limit = (6.0 / (d_in + d_out) as f32).sqrt();
        let proj_weight =
            Array2::from_shape_fn((d_in, d_out), |_| rng.random_range(-limit..limit));
        let proj_bias = Array1::zeros(d_out);

        Self {
            proj_weight,
            proj_bias,
            k,
            d_in,
            d_out,
        }
    }

    /// Create with a deterministic seed for reproducibility across restarts.
    pub fn new_seeded(d_in: usize, d_out: usize, k: usize, seed: u64) -> Self {
        assert!(k >= 1 && k <= d_out, "k must be in [1, d_out]");

        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let limit = (6.0 / (d_in + d_out) as f32).sqrt();
        let proj_weight =
            Array2::from_shape_fn((d_in, d_out), |_| rng.random_range(-limit..limit));
        let proj_bias = Array1::zeros(d_out);

        Self {
            proj_weight,
            proj_bias,
            k,
            d_in,
            d_out,
        }
    }

    /// Forward pass: linear projection -> top-k sparsification -> L2 normalize.
    ///
    /// Input: [B, d_in], Output: [B, d_out]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (batch_size, d_in) = x.dim();
        assert_eq!(d_in, self.d_in, "Input dimension mismatch");

        // Linear: x @ W + b
        let mut projected = x.dot(&self.proj_weight);
        projected
            .rows_mut()
            .into_iter()
            .for_each(|mut row| row += &self.proj_bias);

        // Top-k sparsification + L2 norm, in-place where possible
        let mut output = Array2::zeros((batch_size, self.d_out));

        // Pre-allocate scratch buffer once, reuse for each row
        let mut abs_vals: Vec<(usize, f32)> = Vec::with_capacity(self.d_out);

        for (i, row) in projected.rows().into_iter().enumerate() {
            // Reuse scratch buffer
            abs_vals.clear();
            abs_vals.extend(row.iter().enumerate().map(|(j, &v)| (j, v.abs())));

            // Partial sort to find top-k (O(n) average)
            let k = self.k.min(abs_vals.len());
            abs_vals.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Scatter top-k original values (with ReLU) and compute norm in one pass
            let out_row_off = i * self.d_out;
            let out_slice = output.as_slice_mut().unwrap();
            let mut norm_sq = 0.0f32;
            for &(j, _) in &abs_vals[..k] {
                let val = row[j];
                if val > 0.0 {
                    out_slice[out_row_off + j] = val;
                    norm_sq += val * val;
                }
            }

            // L2 normalize in-place (single pass, only touching non-zero entries)
            if norm_sq > 0.0 {
                let inv_norm = 1.0 / norm_sq.sqrt();
                for &(j, _) in &abs_vals[..k] {
                    let idx = out_row_off + j;
                    // Only scale non-zero (already ReLU-filtered above)
                    if out_slice[idx] != 0.0 {
                        out_slice[idx] *= inv_norm;
                    }
                }
            }
        }

        output
    }

    /// Return the fraction of zero entries per sample.
    pub fn get_sparsity(&self) -> f32 {
        1.0 - self.k as f32 / self.d_out as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shape() {
        let proj = SparseProjection::new(384, 512, 36);
        let x = Array2::from_shape_fn((4, 384), |_| 0.1);
        let out = proj.forward(&x);
        assert_eq!(out.dim(), (4, 512));
    }

    #[test]
    fn test_sparsity() {
        let proj = SparseProjection::new(384, 512, 36);
        let x = Array2::from_shape_fn((2, 384), |_| 0.5);
        let out = proj.forward(&x);

        for row in out.rows() {
            let nonzero = row.iter().filter(|&&v| v != 0.0).count();
            assert!(nonzero <= 36, "Too many non-zero elements: {nonzero}");
        }
    }

    #[test]
    fn test_l2_normalized() {
        let proj = SparseProjection::new(64, 128, 10);
        let x = Array2::from_shape_fn((3, 64), |_| 1.0);
        let out = proj.forward(&x);

        for row in out.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "Row not L2 normalized: norm={norm}"
                );
            }
        }
    }
}
