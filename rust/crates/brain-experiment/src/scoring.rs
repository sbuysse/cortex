//! Flexible scoring models: Standard, MultiHead, MLP, LowRank.
//!
//! Abstracts the association matrix so the runner can swap architectures
//! without duplicating the entire training loop.

use brain_core::association_network::{AssociationConfig, AssociationNetwork};
use ndarray::Array2;
use rand::Rng;

/// Scoring model variants that replace/augment the standard bilinear M matrix.
pub enum ScoringModel {
    /// Standard: uses assoc_net.m_va.m directly (D x D).
    Standard,

    /// Multi-head: K total heads (head 0 = assoc_net.m_va, rest in extra_nets).
    /// All trained independently, averaged for evaluation.
    MultiHead {
        extra_nets: Vec<AssociationNetwork>,
    },

    /// Two-layer MLP: v -> M1 -> ReLU -> M2 -> score with a.
    /// M1: [D, hidden], M2: [hidden, D].
    Mlp {
        m1: Array2<f32>,
        m2: Array2<f32>,
    },

    /// Low-rank factorization: M_eff = U @ V^T.
    /// U: [D, rank], V: [D, rank].
    LowRank {
        u: Array2<f32>,
        v: Array2<f32>,
    },
}

impl ScoringModel {
    /// Create a scoring model from experiment config.
    pub fn from_config(
        config: &crate::config::ExperimentConfig,
        assoc_config: &AssociationConfig,
        rng: &mut impl Rng,
    ) -> Self {
        let d = config.embed_dim;

        if config.num_heads > 1 {
            let extra: Vec<AssociationNetwork> = (1..config.num_heads)
                .map(|_| AssociationNetwork::new(assoc_config))
                .collect();
            ScoringModel::MultiHead { extra_nets: extra }
        } else if config.mlp_hidden > 0 {
            let h = config.mlp_hidden;
            let scale1 = (2.0 / (d + h) as f32).sqrt();
            let scale2 = (2.0 / (h + d) as f32).sqrt();
            let m1 = Array2::from_shape_fn((d, h), |_| (rng.random::<f32>() * 2.0 - 1.0) * scale1);
            let m2 = Array2::from_shape_fn((h, d), |_| (rng.random::<f32>() * 2.0 - 1.0) * scale2);
            ScoringModel::Mlp { m1, m2 }
        } else if config.low_rank > 0 {
            let r = config.low_rank;
            let scale = (2.0 / (d + r) as f32).sqrt();
            let u = Array2::from_shape_fn((d, r), |_| (rng.random::<f32>() * 2.0 - 1.0) * scale);
            let v = Array2::from_shape_fn((d, r), |_| (rng.random::<f32>() * 2.0 - 1.0) * scale);
            ScoringModel::LowRank { u, v }
        } else {
            ScoringModel::Standard
        }
    }

    /// Returns true if this is the standard (single M matrix) model.
    pub fn is_standard(&self) -> bool {
        matches!(self, ScoringModel::Standard)
    }

    /// Positive Hebbian-style update.
    /// Always calls assoc_net.forward_all() for temporal traces and all modalities.
    /// For non-Standard models, also updates the model-specific parameters.
    pub fn update(
        &mut self,
        assoc_net: &mut AssociationNetwork,
        v: &Array2<f32>,
        a: &Array2<f32>,
        e: Option<&Array2<f32>>,
        s: Option<&Array2<f32>>,
        p: Option<&Array2<f32>>,
        lr: f32,
        batch_size: usize,
    ) {
        // All models use assoc_net for temporal traces and all modality pairs
        assoc_net.forward_all(v, a, e, s, p);

        match self {
            ScoringModel::Standard => {
                // forward_all() already did all updates
            }
            ScoringModel::MultiHead { extra_nets } => {
                for net in extra_nets.iter_mut() {
                    net.set_lr(lr);
                    net.forward_all(v, a, e, s, p);
                }
            }
            ScoringModel::Mlp { m1, m2 } => {
                let scale = lr / batch_size as f32;
                // Forward through MLP
                let hidden = v.dot(&*m1); // [B, h]
                let relu_mask = hidden.mapv(|x| if x > 0.0 { 1.0f32 } else { 0.0 });
                let relu_hidden = &hidden * &relu_mask; // [B, h]

                // Update M2: M2 += scale * relu_hidden^T @ a
                ndarray::linalg::general_mat_mul(scale, &relu_hidden.t(), a, 1.0, m2);

                // Update M1 (backprop through relu):
                let grad_hidden = a.dot(&m2.t()) * &relu_mask; // [B, h]
                ndarray::linalg::general_mat_mul(scale, &v.t(), &grad_hidden, 1.0, m1);

                // Norm clipping
                let max = assoc_net.m_va.max_norm;
                clip_frobenius(m1, max * 2.0);
                clip_frobenius(m2, max * 2.0);
            }
            ScoringModel::LowRank { u, v: v_mat } => {
                let scale = lr / batch_size as f32;
                // U += scale * v_buf^T @ (a_buf @ V)
                let av = a.dot(&*v_mat); // [B, r]
                ndarray::linalg::general_mat_mul(scale, &v.t(), &av, 1.0, u);
                // V += scale * a_buf^T @ (v_buf @ U)
                let vu = v.dot(&*u); // [B, r]
                ndarray::linalg::general_mat_mul(scale, &a.t(), &vu, 1.0, v_mat);

                let max = assoc_net.m_va.max_norm;
                clip_frobenius(u, max);
                clip_frobenius(v_mat, max);
            }
        }
    }

    /// Compute batch similarity matrix for InfoNCE: sim[i,j] = score(v_i, a_j) / temperature.
    pub fn compute_batch_sim(
        &self,
        assoc_net: &AssociationNetwork,
        v_buf: &Array2<f32>,
        a_buf: &Array2<f32>,
        temperature: f32,
        sim_matrix: &mut Array2<f32>,
    ) {
        let inv_temp = 1.0 / temperature;
        match self {
            ScoringModel::Standard => {
                let m_a = assoc_net.m_va.m.dot(&a_buf.t()); // [D, B]
                ndarray::linalg::general_mat_mul(inv_temp, v_buf, &m_a, 0.0, sim_matrix);
            }
            ScoringModel::MultiHead { extra_nets } => {
                let avg_m = self.averaged_m(&assoc_net.m_va.m, extra_nets);
                let m_a = avg_m.dot(&a_buf.t());
                ndarray::linalg::general_mat_mul(inv_temp, v_buf, &m_a, 0.0, sim_matrix);
            }
            ScoringModel::LowRank { u, v } => {
                // (v @ U) @ (a @ V)^T / temp
                let vu = v_buf.dot(u); // [B, r]
                let av = a_buf.dot(v); // [B, r]
                ndarray::linalg::general_mat_mul(inv_temp, &vu, &av.t(), 0.0, sim_matrix);
            }
            ScoringModel::Mlp { m1, m2 } => {
                // relu(v @ M1) @ M2 @ a^T / temp
                let hidden = v_buf.dot(m1).mapv(|x| x.max(0.0)); // [B, h]
                let proj = hidden.dot(m2); // [B, D]
                ndarray::linalg::general_mat_mul(inv_temp, &proj, &a_buf.t(), 0.0, sim_matrix);
            }
        }
    }

    /// Apply InfoNCE negative gradient to the scoring model.
    pub fn apply_infonce_neg(
        &mut self,
        assoc_net: &mut AssociationNetwork,
        v_buf: &Array2<f32>,
        weighted_neg: &Array2<f32>,
        lr: f32,
        batch_size: usize,
    ) {
        let infonce_lr = lr / batch_size as f32;
        match self {
            ScoringModel::Standard => {
                let d = v_buf.ncols();
                let mut neg_delta = Array2::zeros((d, d));
                ndarray::linalg::general_mat_mul(infonce_lr, &v_buf.t(), weighted_neg, 0.0, &mut neg_delta);
                assoc_net.m_va.m -= &neg_delta;
            }
            ScoringModel::MultiHead { extra_nets } => {
                let d = v_buf.ncols();
                let mut neg_delta = Array2::zeros((d, d));
                ndarray::linalg::general_mat_mul(infonce_lr, &v_buf.t(), weighted_neg, 0.0, &mut neg_delta);
                assoc_net.m_va.m -= &neg_delta;
                for net in extra_nets.iter_mut() {
                    net.m_va.m -= &neg_delta;
                }
            }
            ScoringModel::Mlp { m1, m2 } => {
                let hidden = v_buf.dot(&*m1); // [B, h]
                let relu_mask = hidden.mapv(|x| if x > 0.0 { 1.0f32 } else { 0.0 });
                let relu_hidden = &hidden * &relu_mask;

                // M2 -= lr * relu_hidden^T @ weighted_neg
                ndarray::linalg::general_mat_mul(-infonce_lr, &relu_hidden.t(), weighted_neg, 1.0, m2);

                // M1 -= lr * v^T @ ((weighted_neg @ M2^T) * relu_mask)
                let grad_h = weighted_neg.dot(&m2.t()) * &relu_mask;
                ndarray::linalg::general_mat_mul(-infonce_lr, &v_buf.t(), &grad_h, 1.0, m1);
            }
            ScoringModel::LowRank { u, v } => {
                // U -= lr * v_buf^T @ (weighted_neg @ V)
                let wv = weighted_neg.dot(&*v); // [B, r]
                ndarray::linalg::general_mat_mul(-infonce_lr, &v_buf.t(), &wv, 1.0, u);
                // V -= lr * weighted_neg^T @ (v_buf @ U)
                let vu = v_buf.dot(&*u); // [B, r]
                ndarray::linalg::general_mat_mul(-infonce_lr, &weighted_neg.t(), &vu, 1.0, v);
            }
        }
    }

    /// Apply SigLIP gradient: M += lr * V^T @ weighted_grad  (gradient ascent on log-likelihood).
    /// weighted_grad = W @ A where W[i,j] = y*(1-sig)/tau.
    pub fn apply_siglip_grad(
        &mut self,
        assoc_net: &mut AssociationNetwork,
        v_buf: &Array2<f32>,
        weighted_grad: &Array2<f32>,
        lr: f32,
        _batch_size: usize,
    ) {
        match self {
            ScoringModel::Standard => {
                let d = v_buf.ncols();
                let mut delta = Array2::<f32>::zeros((d, d));
                ndarray::linalg::general_mat_mul(lr, &v_buf.t(), weighted_grad, 0.0, &mut delta);
                assoc_net.m_va.m += &delta;
            }
            ScoringModel::MultiHead { extra_nets } => {
                let d = v_buf.ncols();
                let mut delta = Array2::<f32>::zeros((d, d));
                ndarray::linalg::general_mat_mul(lr, &v_buf.t(), weighted_grad, 0.0, &mut delta);
                assoc_net.m_va.m += &delta;
                for net in extra_nets.iter_mut() {
                    net.m_va.m += &delta;
                }
            }
            ScoringModel::LowRank { u, v } => {
                // U += lr * v_buf^T @ (weighted_grad @ V)
                let wv = weighted_grad.dot(&*v);
                ndarray::linalg::general_mat_mul(lr, &v_buf.t(), &wv, 1.0, u);
                // V += lr * weighted_grad^T @ (v_buf @ U)
                let vu = v_buf.dot(&*u);
                ndarray::linalg::general_mat_mul(lr, &weighted_grad.t(), &vu, 1.0, v);
            }
            ScoringModel::Mlp { m1, m2 } => {
                let hidden = v_buf.dot(&*m1);
                let relu_mask = hidden.mapv(|x| if x > 0.0 { 1.0f32 } else { 0.0 });
                let relu_hidden = &hidden * &relu_mask;

                // M2 += lr * relu_hidden^T @ weighted_grad
                ndarray::linalg::general_mat_mul(lr, &relu_hidden.t(), weighted_grad, 1.0, m2);

                // M1 += lr * v^T @ ((weighted_grad @ M2^T) * relu_mask)
                let grad_h = weighted_grad.dot(&m2.t()) * &relu_mask;
                ndarray::linalg::general_mat_mul(lr, &v_buf.t(), &grad_h, 1.0, m1);
            }
        }
    }

    /// Compute full evaluation similarity matrices [N, N] for retrieval metrics.
    /// Returns (sim_v2a, sim_a2v).
    pub fn compute_eval_sims(
        &self,
        assoc_net: &AssociationNetwork,
        v: &Array2<f32>,
        a: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        match self {
            ScoringModel::Standard => {
                let pv = v.dot(&assoc_net.m_va.m);
                let sim_v2a = pv.dot(&a.t());
                let pa = a.dot(&assoc_net.m_va.m.t());
                let sim_a2v = pa.dot(&v.t());
                (sim_v2a, sim_a2v)
            }
            ScoringModel::MultiHead { extra_nets } => {
                let avg_m = self.averaged_m(&assoc_net.m_va.m, extra_nets);
                let pv = v.dot(&avg_m);
                let sim_v2a = pv.dot(&a.t());
                let pa = a.dot(&avg_m.t());
                let sim_a2v = pa.dot(&v.t());
                (sim_v2a, sim_a2v)
            }
            ScoringModel::LowRank { u, v: v_mat } => {
                let vu = v.dot(u);       // [N, r]
                let av = a.dot(v_mat);   // [N, r]
                let sim_v2a = vu.dot(&av.t()); // [N, N]
                let sim_a2v = av.dot(&vu.t());
                (sim_v2a, sim_a2v)
            }
            ScoringModel::Mlp { m1, m2 } => {
                let hidden_v = v.dot(m1).mapv(|x| x.max(0.0));
                let proj_v = hidden_v.dot(m2);
                let sim_v2a = proj_v.dot(&a.t());
                let hidden_a = a.dot(&m2.t()).mapv(|x| x.max(0.0));
                let proj_a = hidden_a.dot(&m1.t());
                let sim_a2v = proj_a.dot(&v.t());
                (sim_v2a, sim_a2v)
            }
        }
    }

    /// Get effective M matrix if possible (for compatibility with existing eval code).
    /// Returns None for MLP (nonlinear).
    pub fn effective_m<'a>(&'a self, assoc_net: &'a AssociationNetwork) -> Option<Array2<f32>> {
        match self {
            ScoringModel::Standard => Some(assoc_net.m_va.m.clone()),
            ScoringModel::MultiHead { extra_nets } => {
                Some(self.averaged_m(&assoc_net.m_va.m, extra_nets))
            }
            ScoringModel::LowRank { u, v } => Some(u.dot(&v.t())),
            ScoringModel::Mlp { .. } => None,
        }
    }

    fn averaged_m(&self, head0: &Array2<f32>, extras: &[AssociationNetwork]) -> Array2<f32> {
        let num = 1 + extras.len();
        let mut avg = head0.clone();
        for net in extras {
            avg += &net.m_va.m;
        }
        avg /= num as f32;
        avg
    }
}

/// Clip matrix by frobenius norm (conservative spectral norm bound).
fn clip_frobenius(m: &mut Array2<f32>, max_norm: f32) {
    if max_norm <= 0.0 {
        return;
    }
    let frob: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
    if frob > max_norm {
        let scale = max_norm / frob;
        m.mapv_inplace(|x| x * scale);
    }
}

/// Sort clip indices by embedding similarity (easy = high similarity first).
/// Used for curriculum learning.
pub fn curriculum_order(
    cached_v: &Array2<f32>,
    cached_a: &Array2<f32>,
) -> Vec<usize> {
    let n = cached_v.nrows();
    let mut sims: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let sim: f32 = cached_v.row(i).iter()
                .zip(cached_a.row(i).iter())
                .map(|(a, b)| a * b)
                .sum();
            (i, sim)
        })
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.into_iter().map(|(i, _)| i).collect()
}
