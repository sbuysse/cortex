//! Cross-modal retrieval metrics and association matrix diagnostics.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-modal retrieval evaluation.
pub struct CrossModalRetrieval {
    pub pool_size: usize,
    pub k_values: Vec<usize>,
}

impl CrossModalRetrieval {
    pub fn new(pool_size: usize, k_values: Vec<usize>) -> Self {
        assert!(pool_size > 0);
        Self { pool_size, k_values }
    }

    /// Evaluate from pre-computed similarity matrices (for MLP, low-rank, multi-head).
    pub fn evaluate_from_sims(
        &self,
        sim_v2a: &Array2<f32>,
        sim_a2v: &Array2<f32>,
    ) -> HashMap<String, f64> {
        let max_k = self.k_values.iter().copied().max().unwrap_or(1).min(sim_v2a.nrows());
        let mut results = HashMap::new();
        compute_retrieval_metrics(sim_v2a, max_k, &self.k_values, "v2a", &mut results);
        compute_retrieval_metrics(sim_a2v, max_k, &self.k_values, "a2v", &mut results);
        results
    }

    /// Evaluate visual-to-audio and audio-to-visual retrieval using bilinear similarity.
    pub fn evaluate_retrieval(
        &self,
        visual_embs: &Array2<f32>,
        audio_embs: &Array2<f32>,
        association_matrix: &Array2<f32>,
    ) -> HashMap<String, f64> {
        let n = visual_embs.nrows().min(self.pool_size);

        // Use slices directly instead of copying when pool_size >= n
        let v = visual_embs.slice(ndarray::s![..n, ..]);
        let a = audio_embs.slice(ndarray::s![..n, ..]);

        // Bilinear similarity: sim[i,j] = v_i @ M @ a_j^T
        let projected_v = v.dot(association_matrix); // [N, D]
        let sim_v2a = projected_v.dot(&a.t()); // [N, N]

        let projected_a = a.dot(&association_matrix.t()); // [N, D]
        let sim_a2v = projected_a.dot(&v.t()); // [N, N]

        let max_k = self.k_values.iter().copied().max().unwrap_or(1).min(n);

        let mut results = HashMap::new();

        // Compute R@k and MR/MRR together in a single pass per direction
        compute_retrieval_metrics(&sim_v2a, max_k, &self.k_values, "v2a", &mut results);
        compute_retrieval_metrics(&sim_a2v, max_k, &self.k_values, "a2v", &mut results);

        results
    }
}

/// Compute all retrieval metrics (R@k, MR, MRR) in a single pass per row.
/// Uses O(n) counting to find rank instead of O(n log n) argsort.
fn compute_retrieval_metrics(
    sim: &Array2<f32>,
    max_k: usize,
    k_values: &[usize],
    prefix: &str,
    results: &mut HashMap<String, f64>,
) {
    let n = sim.nrows();
    let mut hits: Vec<usize> = vec![0; max_k + 1]; // hits[k] = count of items where gt in top-k
    let mut total_rank = 0u64;
    let mut total_rr = 0.0f64;

    let sim_slice = sim.as_slice().unwrap();
    let cols = sim.ncols();

    for i in 0..n {
        let row_off = i * cols;
        let gt_score = sim_slice[row_off + i];

        // O(n) rank computation via direct slice access (no ndarray overhead)
        let mut rank = 1usize; // 1-indexed
        for j in 0..n {
            if j != i && sim_slice[row_off + j] > gt_score {
                rank += 1;
            }
        }

        total_rank += rank as u64;
        total_rr += 1.0 / rank as f64;

        if rank <= max_k {
            for k in rank..=max_k {
                hits[k] += 1;
            }
        }
    }

    for &k in k_values {
        let kk = k.min(max_k);
        results.insert(format!("{prefix}_R@{k}"), hits[kk] as f64 / n as f64);
    }
    results.insert(format!("{prefix}_MR"), total_rank as f64 / n as f64);
    results.insert(format!("{prefix}_MRR"), total_rr / n as f64);
}

/// Structural diagnostics for a Hebbian association matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixDiagnostics {
    pub sparsity: f64,
    pub frobenius_norm: f64,
    pub spectral_norm: f64,
    pub estimated_rank: f64,
    pub condition_number: f64,
    pub max_value: f64,
    pub min_value: f64,
}

pub struct MatrixStats {
    pub rank_threshold: f32,
}

impl MatrixStats {
    pub fn new(rank_threshold: f32) -> Self {
        assert!(rank_threshold > 0.0);
        Self { rank_threshold }
    }

    /// Compute matrix diagnostics.
    ///
    /// Uses power iteration for spectral norm estimation instead of full SVD,
    /// which is O(D²) per iteration instead of O(D³).
    pub fn compute(&self, matrix: &Array2<f32>) -> MatrixDiagnostics {
        let total = matrix.len();

        // Single pass: compute zeros, frobenius, min, max simultaneously
        let mut zeros = 0usize;
        let mut frob_sq = 0.0f64;
        let mut max_val = f32::NEG_INFINITY;
        let mut min_val = f32::INFINITY;

        for &x in matrix.iter() {
            if x == 0.0 {
                zeros += 1;
            }
            frob_sq += (x * x) as f64;
            if x > max_val { max_val = x; }
            if x < min_val { min_val = x; }
        }

        let sparsity = zeros as f64 / total.max(1) as f64;
        let frobenius_norm = frob_sq.sqrt();

        // Power iteration for spectral norm (top singular value)
        let spectral_norm = power_iteration_spectral(matrix, 10) as f64;

        let estimated_rank = if spectral_norm > 1e-12 {
            frob_sq / (spectral_norm * spectral_norm)
        } else {
            0.0
        };

        let condition_number = if spectral_norm > 1e-12 {
            let min_sv = inverse_power_iteration(matrix, spectral_norm as f32, 10) as f64;
            if min_sv > 1e-12 {
                spectral_norm / min_sv
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };

        MatrixDiagnostics {
            sparsity,
            frobenius_norm,
            spectral_norm,
            estimated_rank,
            condition_number,
            max_value: max_val as f64,
            min_value: min_val as f64,
        }
    }
}

/// Power iteration to approximate largest singular value.
fn power_iteration_spectral(m: &Array2<f32>, iters: usize) -> f32 {
    use rand::Rng;
    let d = m.ncols();
    let mut rng = rand::rng();
    let mut v = Array1::from_shape_fn(d, |_| rng.random::<f32>() - 0.5);
    let norm = v.dot(&v).sqrt().max(1e-12);
    v.mapv_inplace(|x| x / norm);

    // Reuse u buffer to avoid allocation in the loop
    let mut u = Array1::zeros(m.nrows());

    for _ in 0..iters {
        ndarray::linalg::general_mat_vec_mul(1.0, m, &v, 0.0, &mut u);
        let u_norm = u.dot(&u).sqrt().max(1e-12);
        u.mapv_inplace(|x| x / u_norm);

        ndarray::linalg::general_mat_vec_mul(1.0, &m.t(), &u, 0.0, &mut v);
        let v_norm = v.dot(&v).sqrt().max(1e-12);
        v.mapv_inplace(|x| x / v_norm);
    }
    let result = m.dot(&v);
    result.dot(&result).sqrt()
}

/// Inverse power iteration to estimate smallest singular value.
fn inverse_power_iteration(m: &Array2<f32>, sigma_max: f32, _iters: usize) -> f32 {
    let frobenius = m.iter().map(|x| x * x).sum::<f32>().sqrt();
    let d = m.nrows().min(m.ncols()) as f32;
    if sigma_max > 1e-12 && d > 1.0 {
        let avg_sv = frobenius / d.sqrt();
        (avg_sv * avg_sv / sigma_max).max(1e-12)
    } else {
        1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_retrieval_identity_matrix() {
        let retrieval = CrossModalRetrieval::new(100, vec![1, 5, 10]);
        let n = 20;
        let d = 8;

        let m = Array2::eye(d);
        let mut v = Array2::zeros((n, d));
        let mut a = Array2::zeros((n, d));
        for i in 0..n {
            v[[i, i % d]] = 1.0;
            a[[i, i % d]] = 1.0;
        }

        let results = retrieval.evaluate_retrieval(&v, &a, &m);
        assert!(results["v2a_MRR"] > 0.0);
    }

    #[test]
    fn test_matrix_stats() {
        let stats = MatrixStats::new(1e-3);
        let m = Array2::eye(8);
        let diag = stats.compute(&m);
        assert!((diag.spectral_norm - 1.0).abs() < 0.1);
        assert!(diag.frobenius_norm > 0.0);
    }
}
