//! Hebbian association learning — the fundamental synaptic learning unit.
//!
//! Implements a biologically-inspired Hebbian learning rule with metaplasticity:
//! synapses that have been heavily updated (consolidated) learn more slowly.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// A single Hebbian association matrix between two modalities.
///
/// Supports rectangular M of shape (d_left, d_right) for asymmetric modality dims.
///
/// Learning rule:
///   delta = lr * (x_a^T @ x_b) / batch_size
///   effective_lr = 1 / (1 + consolidation)  (metaplasticity)
///   M += delta * effective_lr
pub struct HebbianAssociation {
    pub d: usize,           // kept for backward compat (= d_left for square)
    pub d_left: usize,
    pub d_right: usize,
    pub lr: f32,
    pub decay_rate: f32,
    pub max_norm: f32,

    /// Association matrix [d_left, d_right].
    pub m: Array2<f32>,
    /// Consolidation tracker — cumulative absolute update magnitude.
    pub consolidation: Array2<f32>,
    /// Running update count.
    pub update_count: u64,
    /// Power-iteration vector for fast spectral norm approximation.
    pi_v: Array1<f32>,
    /// Pre-allocated buffer for delta computation.
    delta_buf: Array2<f32>,
    /// Cached metaplasticity modulator — recomputed every 5 updates.
    effective_lr: Array2<f32>,
    /// Pre-allocated buffer for power iteration.
    pi_u: Array1<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianStats {
    pub norm: f32,
    pub sparsity: f32,
    pub max: f32,
    pub min: f32,
    pub mean_consolidation: f32,
    pub max_consolidation: f32,
    pub update_count: u64,
}

impl HebbianAssociation {
    /// Create a square association matrix (d x d). Backward compatible.
    pub fn new(d: usize, lr: f32, decay_rate: f32, max_norm: f32) -> Self {
        Self::new_rect(d, d, lr, decay_rate, max_norm)
    }

    /// Create a rectangular association matrix (d_left x d_right).
    pub fn new_rect(d_left: usize, d_right: usize, lr: f32, decay_rate: f32, max_norm: f32) -> Self {
        assert!(d_left > 0 && d_right > 0, "Dimensions must be positive");
        assert!(
            decay_rate > 0.0 && decay_rate <= 1.0,
            "decay_rate must be in (0, 1]"
        );

        use rand::Rng;
        let mut rng = rand::rng();
        let pi_v = Array1::from_shape_fn(d_right, |_| rng.random::<f32>() * 2.0 - 1.0);

        Self {
            d: d_left, // backward compat
            d_left,
            d_right,
            lr,
            decay_rate,
            max_norm,
            m: Array2::zeros((d_left, d_right)),
            consolidation: Array2::zeros((d_left, d_right)),
            update_count: 0,
            pi_v,
            delta_buf: Array2::zeros((d_left, d_right)),
            effective_lr: Array2::ones((d_left, d_right)),
            pi_u: Array1::zeros(d_left),
        }
    }

    /// Hebbian update: strengthen associations between co-occurring patterns.
    ///
    /// x_a: [B, d_left], x_b: [B, d_right]
    #[inline]
    pub fn update(&mut self, x_a: &Array2<f32>, x_b: &Array2<f32>) {
        let (ba, da) = x_a.dim();
        let (bb, db) = x_b.dim();
        assert_eq!(da, self.d_left, "x_a last dim mismatch");
        assert_eq!(db, self.d_right, "x_b last dim mismatch");
        assert_eq!(ba, bb, "batch size mismatch");

        let batch_size = ba as f32;
        let scale = self.lr / batch_size;

        // delta = x_a^T @ x_b * (lr / batch_size)
        ndarray::linalg::general_mat_mul(scale, &x_a.t(), x_b, 0.0, &mut self.delta_buf);

        // Apply modulated update: M += delta * effective_lr
        let decay = self.decay_rate;
        let recompute_lr = self.update_count % 5 == 0;
        let total = self.d_left * self.d_right;
        let m_slice = self.m.as_slice_mut().unwrap();
        let c_slice = self.consolidation.as_slice_mut().unwrap();
        let lr_slice = self.effective_lr.as_slice_mut().unwrap();
        let d_slice = self.delta_buf.as_slice().unwrap();

        for i in 0..total {
            let d_val = d_slice[i];
            if recompute_lr {
                lr_slice[i] = 1.0 / (1.0 + (1.0 + c_slice[i]).ln());
            }
            m_slice[i] += d_val * lr_slice[i];
            c_slice[i] = (c_slice[i] + d_val.abs()) * decay;
        }

        // Spectral norm clipping every 5 updates
        if self.max_norm > 0.0 && self.update_count % 5 == 0 {
            let spectral = self.power_iteration_spectral_norm(2);
            if spectral > self.max_norm {
                let scale = self.max_norm / spectral;
                self.m.mapv_inplace(|v| v * scale);
            }
        }

        self.update_count += 1;
    }

    /// Approximate spectral norm via power iteration.
    /// Works for rectangular matrices.
    pub fn power_iteration_spectral_norm(&mut self, num_iters: usize) -> f32 {
        for _ in 0..num_iters {
            // u = M @ v  (d_left,)
            ndarray::linalg::general_mat_vec_mul(1.0, &self.m, &self.pi_v, 0.0, &mut self.pi_u);
            let u_norm = self.pi_u.dot(&self.pi_u).sqrt().max(1e-12);
            self.pi_u.mapv_inplace(|x| x / u_norm);
            // v = M^T @ u  (d_right,)
            ndarray::linalg::general_mat_vec_mul(1.0, &self.m.t(), &self.pi_u, 0.0, &mut self.pi_v);
            let v_norm = self.pi_v.dot(&self.pi_v).sqrt().max(1e-12);
            self.pi_v.mapv_inplace(|x| x / v_norm);
        }
        let result = self.m.dot(&self.pi_v);
        result.dot(&result).sqrt()
    }

    /// Pattern completion: given one modality, recall the associated pattern.
    pub fn recall(&self, x: &Array2<f32>, forward: bool) -> Array2<f32> {
        if forward {
            x.dot(&self.m)
        } else {
            x.dot(&self.m.t())
        }
    }

    /// Synaptic homeostasis: zero out weak connections.
    pub fn prune(&mut self, threshold: f32) -> usize {
        let mut pruned = 0usize;
        self.m.mapv_inplace(|v| {
            if v.abs() <= threshold {
                pruned += 1;
                0.0
            } else {
                v
            }
        });
        pruned
    }

    /// Bilinear similarity: x_a^T M x_b averaged over the batch.
    pub fn similarity(&self, x_a: &Array2<f32>, x_b: &Array2<f32>) -> f32 {
        let projected = x_a.dot(&self.m); // [B, d_right]
        let sim = (&projected * x_b).sum_axis(Axis(1)); // [B]
        sim.mean().unwrap_or(0.0)
    }

    /// Monitoring statistics.
    pub fn get_stats(&self) -> HebbianStats {
        let m_slice = self.m.as_slice().unwrap();
        let c_slice = self.consolidation.as_slice().unwrap();
        let total = m_slice.len();

        let mut norm_sq = 0.0f32;
        let mut zeros = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        let mut min_val = f32::INFINITY;
        let mut c_sum = 0.0f32;
        let mut c_max = f32::NEG_INFINITY;

        for i in 0..total {
            let x = m_slice[i];
            norm_sq += x * x;
            if x == 0.0 { zeros += 1; }
            if x > max_val { max_val = x; }
            if x < min_val { min_val = x; }

            let c = c_slice[i];
            c_sum += c;
            if c > c_max { c_max = c; }
        }

        HebbianStats {
            norm: norm_sq.sqrt(),
            sparsity: zeros as f32 / total as f32,
            max: max_val,
            min: min_val,
            mean_consolidation: c_sum / total as f32,
            max_consolidation: c_max,
            update_count: self.update_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_basic_update() {
        let mut heb = HebbianAssociation::new(16, 0.01, 0.999, 100.0);
        let x_a = Array2::from_shape_fn((4, 16), |_| 0.1);
        let x_b = Array2::from_shape_fn((4, 16), |_| 0.1);

        heb.update(&x_a, &x_b);
        assert!(heb.m.iter().any(|&x| x != 0.0), "M should be non-zero after update");
        assert_eq!(heb.update_count, 1);
    }

    #[test]
    fn test_rectangular_update() {
        let mut heb = HebbianAssociation::new_rect(384, 512, 0.01, 0.999, 100.0);
        let x_a = Array2::from_shape_fn((4, 384), |_| 0.1);
        let x_b = Array2::from_shape_fn((4, 512), |_| 0.1);

        heb.update(&x_a, &x_b);
        assert_eq!(heb.m.dim(), (384, 512));
        assert!(heb.m.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_prune() {
        let mut heb = HebbianAssociation::new(8, 0.01, 0.999, 100.0);
        heb.m.fill(0.0005);
        heb.m[[0, 0]] = 1.0;
        let pruned = heb.prune(0.001);
        assert_eq!(pruned, 8 * 8 - 1);
        assert_eq!(heb.m[[0, 0]], 1.0);
    }

    #[test]
    fn test_recall_forward_backward() {
        let mut heb = HebbianAssociation::new(8, 0.01, 0.999, 100.0);
        heb.m = Array2::eye(8);
        let x = Array2::from_shape_fn((2, 8), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let fwd = heb.recall(&x, true);
        let bwd = heb.recall(&x, false);
        assert_eq!(fwd, bwd);
    }

    #[test]
    fn test_nan_stability_50_steps() {
        let mut heb = HebbianAssociation::new(512, 0.0004, 0.992, 100.0);
        let mut rng = rand::rng();
        use rand::Rng;
        for _ in 0..50 {
            let x_a = Array2::from_shape_fn((4, 512), |_| rng.random::<f32>() - 0.5);
            let x_b = Array2::from_shape_fn((4, 512), |_| rng.random::<f32>() - 0.5);
            heb.update(&x_a, &x_b);
            assert!(
                !heb.m.iter().any(|x| x.is_nan()),
                "NaN detected in M at update {}",
                heb.update_count
            );
        }
    }

    #[test]
    fn test_rectangular_spectral_norm() {
        let mut heb = HebbianAssociation::new_rect(384, 512, 0.01, 0.999, 50.0);
        // Set M to something non-trivial
        heb.m.fill(0.1);
        let sn = heb.power_iteration_spectral_norm(5);
        assert!(sn > 0.0 && sn.is_finite());
    }
}
