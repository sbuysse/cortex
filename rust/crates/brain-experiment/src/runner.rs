//! Experiment runner — replicates the full training loop from train_experiment.py.
//!
//! Accepts pre-computed embeddings and an optional MutationVtable to swap
//! any mutation target. Publishes metrics events for live SSE streaming.

use brain_core::association_network::{AssociationConfig, AssociationNetwork};
use brain_core::metrics::{CrossModalRetrieval, MatrixStats};
use brain_core::scheduler::{DecayType, HebbianScheduler};

use crate::config::ExperimentConfig;
use crate::scoring::ScoringModel;

use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Metrics emitted during training for live streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsEvent {
    pub experiment_id: i64,
    pub step: usize,
    pub metrics: HashMap<String, f64>,
}

/// Final experiment result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub metrics: HashMap<String, f64>,
    pub steps_completed: usize,
    pub epochs: usize,
    pub duration_seconds: f64,
}

/// Run a single experiment.
///
/// `cached_v`, `cached_a`, `cached_e` are the projected embeddings [N, D].
/// `cached_s`, `cached_p` are optional speech and properties embeddings.
/// These are shared read-only and already passed through SparseProjection.
///
/// When `config.skip_projection` is true, cached_v/a may have different dimensions
/// (e.g. 384 and 512 for ZCA-whitened raw embeddings), and a rectangular M is used.
pub fn run_experiment(
    config: &ExperimentConfig,
    cached_v: &Array2<f32>,
    cached_a: &Array2<f32>,
    cached_e: &Array2<f32>,
    cached_s: Option<&Array2<f32>>,
    cached_p: Option<&Array2<f32>>,
    experiment_id: i64,
    shutdown: Arc<AtomicBool>,
    live_tx: Option<&broadcast::Sender<MetricsEvent>>,
) -> ExperimentResult {
    // V2 path: skip_projection uses rectangular M with raw/whitened embeddings
    if config.skip_projection {
        return run_experiment_v2(config, cached_v, cached_a, experiment_id, shutdown, live_tx);
    }

    let start = std::time::Instant::now();

    let d = config.embed_dim;
    let n_clips = cached_v.nrows();

    // Seed RNG
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // Build association network
    let assoc_config = AssociationConfig {
        d,
        lr: config.hebbian_lr,
        decay_rate: config.decay_rate,
        temporal_decay: config.temporal_decay,
        trace_weight: config.trace_weight,
        max_norm: config.max_norm,
    };
    let mut assoc_net = AssociationNetwork::new(&assoc_config);
    let mut scoring = ScoringModel::from_config(config, &assoc_config, &mut rng);

    // Initialize M
    // Initialize association matrices
    {
        let n = cached_v.nrows() as f32;
        let blend = config.cross_correlation_blend;
        let init_scale = config.init_scale;
        let target_norm = init_scale * (d as f32).sqrt();

        // Compute cross-correlations for each modality pair
        let modality_pairs: Vec<(&str, &Array2<f32>, &Array2<f32>)> = {
            let mut pairs = vec![
                ("va", cached_v, cached_a),
                ("ve", cached_v, cached_e),
                ("ae", cached_a, cached_e),
            ];
            if let Some(cs) = cached_s {
                pairs.push(("vs", cached_v, cs));
                pairs.push(("as", cached_a, cs));
                pairs.push(("es", cached_e, cs));
            }
            if let Some(cp) = cached_p {
                pairs.push(("vp", cached_v, cp));
                pairs.push(("ap", cached_a, cp));
                pairs.push(("ep", cached_e, cp));
                if cached_s.is_some() {
                    pairs.push(("sp", cached_s.unwrap(), cp));
                }
            }
            pairs
        };

        let mut matrices = assoc_net.all_matrices_mut();

        for (name, left, right) in &modality_pairs {
            let key = format!("M_{}", name);
            if let Some(m) = matrices.iter_mut().find(|(k, _)| *k == key).map(|(_, m)| m) {
                if config.cross_correlation_init {
                    let mut cc = Array2::<f32>::zeros((d, d));
                    ndarray::linalg::general_mat_mul(1.0 / n, &left.t(), right, 0.0, &mut cc);
                    let cc_norm: f32 = cc.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if cc_norm > 1e-12 {
                        cc *= target_norm / cc_norm;
                    }
                    let m_slice = m.as_slice_mut().unwrap();
                    let cc_slice = cc.as_slice().unwrap();
                    for (idx, val) in m_slice.iter_mut().enumerate() {
                        let row = idx / d;
                        let col = idx % d;
                        let diag = if row == col { init_scale } else { 0.0 };
                        *val = blend * cc_slice[idx] + (1.0 - blend) * diag;
                    }
                } else {
                    let noise_scale = init_scale * 0.1;
                    let m_slice = m.as_slice_mut().unwrap();
                    for (idx, val) in m_slice.iter_mut().enumerate() {
                        let row = idx / d;
                        let col = idx % d;
                        let diag = if row == col { init_scale } else { 0.0 };
                        *val = diag + (rng.random::<f32>() - 0.5) * noise_scale;
                    }
                }
            }
        }
    }

    // Scheduler (warmup_factor inherited from winning mutations)
    let warmup = (100.0 * config.warmup_factor) as usize;
    let scheduler = HebbianScheduler::new(
        config.hebbian_lr,
        warmup,
        DecayType::Cosine,
        1e-5,
        config.max_steps,
    );

    // Metrics evaluator
    let eval_pool_size = if config.eval_pool_size == 0 { n_clips } else { n_clips.min(config.eval_pool_size) };
    let retrieval = CrossModalRetrieval::new(eval_pool_size, vec![1, 5, 10]);
    let matrix_stats = MatrixStats::new(1e-3);

    let eval_interval = config.eval_interval.unwrap_or_else(|| {
        50usize.max(config.max_steps / 10)
    });
    let batch_size = config.batch_size;

    // Create index for batching
    let indices: Vec<usize> = (0..n_clips).collect();

    // Pre-allocate reusable batch buffers (each matching its embedding dimension)
    let dv = cached_v.ncols();
    let da = cached_a.ncols();
    let de = cached_e.ncols();
    let mut v_buf = Array2::zeros((batch_size, dv));
    let mut a_buf = Array2::zeros((batch_size, da));
    let mut e_buf = Array2::zeros((batch_size, de));
    let mut s_buf = if let Some(s) = cached_s { Array2::zeros((batch_size, s.ncols())) } else { Array2::zeros((0, 0)) };
    let mut p_buf = if let Some(p) = cached_p { Array2::zeros((batch_size, p.ncols())) } else { Array2::zeros((0, 0)) };
    let mut perm: Vec<usize> = (0..batch_size).collect();
    let mut a_neg_buf = Array2::zeros((batch_size, da));
    let mut neg_delta = Array2::zeros((dv, da));
    let mut noise_buf_v = Array2::zeros((batch_size, dv));
    let mut noise_buf_a = Array2::zeros((batch_size, da));

    let mut step = 0usize;
    let mut epoch = 0usize;
    let mut last_metrics: HashMap<String, f64> = HashMap::new();
    let mut best_mrr = 0.0f64;
    let mut stale_evals = 0usize;

    // Hard negative mining state
    let mut hard_neg_indices: Option<Vec<Vec<usize>>> = None;
    if config.hard_negative_k > 0 {
        hard_neg_indices = Some(compute_hard_negatives(
            cached_v, cached_a, &assoc_net.m_va.m, config.hard_negative_k,
        ));
    }

    // InfoNCE buffers
    let needs_infonce_bufs = config.use_infonce || config.gradient_infonce;
    let mut sim_matrix = if needs_infonce_bufs { Array2::zeros((batch_size, batch_size)) } else { Array2::zeros((0, 0)) };
    let mut softmax_probs = if needs_infonce_bufs { Array2::zeros((batch_size, batch_size)) } else { Array2::zeros((0, 0)) };
    let mut weighted_neg = if needs_infonce_bufs { Array2::zeros((batch_size, d)) } else { Array2::zeros((0, 0)) };

    // Curriculum learning: sort indices by similarity (easy first)
    let curriculum_order = if config.curriculum {
        Some(crate::scoring::curriculum_order(cached_v, cached_a))
    } else {
        None
    };

    while step < config.max_steps && !shutdown.load(Ordering::Relaxed) {
        epoch += 1;

        // Curriculum: expand active set over training
        let active_indices = if let Some(ref order) = curriculum_order {
            let frac = config.curriculum_start_frac
                + (1.0 - config.curriculum_start_frac) * (step as f32 / config.max_steps as f32);
            let active_n = ((n_clips as f32 * frac).ceil() as usize).min(n_clips);
            &order[..active_n]
        } else {
            &indices[..]
        };

        // Copy into mutable vec for shuffling
        let mut batch_indices: Vec<usize> = active_indices.to_vec();
        batch_indices.shuffle(&mut rng);

        for chunk in batch_indices.chunks(batch_size) {
            if step >= config.max_steps || shutdown.load(Ordering::Relaxed) {
                break;
            }
            if chunk.len() < batch_size {
                continue; // drop_last
            }

            // Gather batch into pre-allocated buffers
            gather_rows_into(cached_v, chunk, &mut v_buf);
            gather_rows_into(cached_a, chunk, &mut a_buf);
            gather_rows_into(cached_e, chunk, &mut e_buf);
            if let Some(cs) = cached_s {
                gather_rows_into(cs, chunk, &mut s_buf);
            }
            if let Some(cp) = cached_p {
                gather_rows_into(cp, chunk, &mut p_buf);
            }

            // Data augmentation: add per-element noise via pre-filled buffer + scaled_add
            if config.aug_noise > 0.0 {
                let noise_scale = config.aug_noise;
                // Fill noise buffer once, apply to both v and a
                noise_buf_v.mapv_inplace(|_| rng.random::<f32>() * 2.0 - 1.0);
                v_buf.scaled_add(noise_scale, &noise_buf_v);
                noise_buf_a.mapv_inplace(|_| rng.random::<f32>() * 2.0 - 1.0);
                a_buf.scaled_add(noise_scale, &noise_buf_a);
                l2_normalize_inplace(&mut v_buf);
                l2_normalize_inplace(&mut a_buf);
            }

            // Check if optional modalities are non-zero
            let e_input = if e_buf.iter().any(|&x| x != 0.0) { Some(&e_buf as &Array2<f32>) } else { None };
            let s_input = if cached_s.is_some() { Some(&s_buf as &Array2<f32>) } else { None };
            let p_input = if cached_p.is_some() { Some(&p_buf as &Array2<f32>) } else { None };

            // Update learning rate
            let current_lr = scheduler.get_lr(step);
            assoc_net.set_lr(current_lr);

            if config.gradient_infonce && batch_size > 1 {
                // === Fused gradient descent InfoNCE ===
                // Computes the exact InfoNCE gradient w.r.t. M:
                //   dM = (lr / (B * temp)) * V^T @ (I - softmax(V M A^T / temp)) @ A
                // This replaces the separate Hebbian positive + contrastive negative
                // updates which used different inputs and learning rates.
                let infonce_temp = config.infonce_temperature;

                // 1. Compute sim matrix: sim[i,j] = v_i^T M a_j / temp
                scoring.compute_batch_sim(&assoc_net, &v_buf, &a_buf, infonce_temp, &mut sim_matrix);

                // 2. Softmax per row → P[i,j]
                for i in 0..batch_size {
                    let row = sim_matrix.row(i);
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..batch_size {
                        let exp_val = (sim_matrix[[i, j]] - max_val).exp();
                        softmax_probs[[i, j]] = exp_val;
                        sum += exp_val;
                    }
                    let inv_sum = 1.0 / sum.max(1e-12);
                    for j in 0..batch_size {
                        softmax_probs[[i, j]] *= inv_sum;
                    }
                }

                // 3. Compute (I - P) in-place: subtract identity from softmax
                for i in 0..batch_size {
                    softmax_probs[[i, i]] -= 1.0;
                    // Now softmax_probs[i,j] = P[i,j] - delta(i,j)
                    // Negate to get (I - P): gradient_weights[i,j] = delta(i,j) - P[i,j]
                    for j in 0..batch_size {
                        softmax_probs[[i, j]] = -softmax_probs[[i, j]];
                    }
                }

                // 4. weighted_target = (I - P) @ A  [B, D]
                ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);

                // 5. dM = V^T @ weighted_target  [D, D], then M += (lr / (B * temp)) * dM
                let grad_scale = current_lr / (batch_size as f32 * infonce_temp);
                ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &weighted_neg, 1.0, &mut assoc_net.m_va.m);

                // 5b. === Reverse direction: A→V InfoNCE (symmetric loss) ===
                // dM += scale * V^T @ (I - P_a2v)^T @ A
                // where P_a2v = softmax(A @ M^T @ V^T / temp)
                if config.symmetric_infonce {
                    // Compute A @ M^T → weighted_neg [B, D] (reuse buffer)
                    let m_t = assoc_net.m_va.m.t();
                    ndarray::linalg::general_mat_mul(1.0, &a_buf, &m_t, 0.0, &mut weighted_neg);
                    // sim_a2v = (A @ M^T) @ V^T / temp → sim_matrix [B, B]
                    ndarray::linalg::general_mat_mul(
                        1.0 / infonce_temp, &weighted_neg, &v_buf.t(), 0.0, &mut sim_matrix,
                    );
                    // Softmax per row → P_a2v
                    for i in 0..batch_size {
                        let max_val = sim_matrix.row(i).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for j in 0..batch_size {
                            let v = (sim_matrix[[i, j]] - max_val).exp();
                            softmax_probs[[i, j]] = v;
                            sum += v;
                        }
                        let inv = 1.0 / sum.max(1e-12);
                        for j in 0..batch_size {
                            softmax_probs[[i, j]] *= inv;
                        }
                    }
                    // R = I - P_a2v (in-place on softmax_probs)
                    for i in 0..batch_size {
                        for j in 0..batch_size {
                            softmax_probs[[i, j]] = if i == j { 1.0 } else { 0.0 } - softmax_probs[[i, j]];
                        }
                    }
                    // dM += scale * V^T @ R^T @ A
                    // Step 1: temp = R^T @ A  [B, D]
                    ndarray::linalg::general_mat_mul(1.0, &softmax_probs.t(), &a_buf, 0.0, &mut weighted_neg);
                    // Step 2: M += scale * V^T @ temp  [D, D]
                    ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &weighted_neg, 1.0, &mut assoc_net.m_va.m);
                }

                // 6. Spectral norm clipping (same schedule as HebbianAssociation)
                if assoc_net.m_va.max_norm > 0.0 && step % 5 == 0 {
                    let spectral = assoc_net.m_va.power_iteration_spectral_norm(2);
                    if spectral > assoc_net.m_va.max_norm {
                        let clip = assoc_net.m_va.max_norm / spectral;
                        assoc_net.m_va.m.mapv_inplace(|v| v * clip);
                    }
                }
                assoc_net.m_va.update_count += 1;

                // Still update other modality pairs with Hebbian (they use standard learning)
                if e_input.is_some() || s_input.is_some() || p_input.is_some() {
                    // Update non-va matrices only
                    if let Some(e) = e_input {
                        assoc_net.m_ve.update(&v_buf, e);
                        assoc_net.m_ae.update(&a_buf, e);
                    }
                    if let Some(s) = s_input {
                        assoc_net.m_vs.update(&v_buf, s);
                        assoc_net.m_as.update(&a_buf, s);
                        if e_input.is_some() {
                            assoc_net.m_es.update(&e_buf, s);
                        }
                    }
                    if let Some(p) = p_input {
                        assoc_net.m_vp.update(&v_buf, p);
                        assoc_net.m_ap.update(&a_buf, p);
                        if e_input.is_some() {
                            assoc_net.m_ep.update(&e_buf, p);
                        }
                        if s_input.is_some() {
                            assoc_net.m_sp.update(&s_buf, p);
                        }
                    }
                }
            } else {
                // === Legacy: separate Hebbian positive + contrastive negative ===
                // Positive Hebbian update (uses scoring model for non-standard architectures)
                scoring.update(&mut assoc_net, &v_buf, &a_buf, e_input, s_input, p_input, current_lr, batch_size);

                // Contrastive learning
                if config.use_siglip && batch_size > 1 {
                    let tau = config.infonce_temperature;
                    let bias = if config.siglip_bias != 0.0 {
                        config.siglip_bias
                    } else {
                        -(batch_size as f32).ln()
                    };
                    let inv_tau = 1.0 / tau;
                    scoring.compute_batch_sim(&assoc_net, &v_buf, &a_buf, 1.0, &mut sim_matrix);
                    for i in 0..batch_size {
                        for j in 0..batch_size {
                            let y: f32 = if i == j { 1.0 } else { -1.0 };
                            let logit = sim_matrix[[i, j]] * inv_tau - bias;
                            let sig = 1.0 / (1.0 + (-y * logit).exp());
                            softmax_probs[[i, j]] = y * (1.0 - sig) * inv_tau;
                        }
                    }
                    let siglip_lr = current_lr / batch_size as f32;
                    ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);
                    scoring.apply_siglip_grad(&mut assoc_net, &v_buf, &weighted_neg, siglip_lr, batch_size);
                } else if config.use_infonce && batch_size > 1 {
                    let infonce_temp = config.infonce_temperature;
                    scoring.compute_batch_sim(&assoc_net, &v_buf, &a_buf, infonce_temp, &mut sim_matrix);
                    for i in 0..batch_size {
                        let row = sim_matrix.row(i);
                        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for j in 0..batch_size {
                            let exp_val = (sim_matrix[[i, j]] - max_val).exp();
                            softmax_probs[[i, j]] = exp_val;
                            sum += exp_val;
                        }
                        let inv_sum = 1.0 / sum.max(1e-12);
                        for j in 0..batch_size {
                            softmax_probs[[i, j]] *= inv_sum;
                        }
                    }
                    ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);
                    scoring.apply_infonce_neg(&mut assoc_net, &v_buf, &weighted_neg, current_lr, batch_size);
                } else if config.neg_weight > 0.0 && batch_size > 1 {
                    if let Some(ref hard_negs) = hard_neg_indices {
                        for (i, &clip_idx) in chunk.iter().enumerate() {
                            if clip_idx < hard_negs.len() {
                                let neg_idx = hard_negs[clip_idx][rng.random_range(0..hard_negs[clip_idx].len().max(1))];
                                a_neg_buf.row_mut(i).assign(&cached_a.row(neg_idx));
                            } else {
                                let neg_idx = chunk[(i + 1) % batch_size];
                                a_neg_buf.row_mut(i).assign(&cached_a.row(neg_idx));
                            }
                        }
                    } else {
                        perm.clear();
                        perm.extend(0..batch_size);
                        perm.rotate_left(1);
                        gather_rows_into(&a_buf, &perm, &mut a_neg_buf);
                    }
                    let neg_scale = current_lr * config.neg_weight / batch_size as f32;
                    ndarray::linalg::general_mat_mul(neg_scale, &v_buf.t(), &a_neg_buf, 0.0, &mut neg_delta);
                    assoc_net.m_va.m -= &neg_delta;
                }
            }

            // Refresh hard negatives periodically
            if config.hard_negative_k > 0
                && config.hard_negative_refresh > 0
                && step > 0
                && step % config.hard_negative_refresh == 0
            {
                hard_neg_indices = Some(compute_hard_negatives(
                    cached_v, cached_a, &assoc_net.m_va.m, config.hard_negative_k,
                ));
            }

            // Post-update normalization (inherited from winning mutations)
            if config.normalize_post_update {
                normalize_rows_inplace(&mut assoc_net.m_va.m);
            }

            // Consolidation: prune
            if config.consolidation_interval > 0
                && step > 0
                && step % config.consolidation_interval == 0
            {
                if let Some(percentile) = config.percentile_prune {
                    let mut abs_vals: Vec<f32> = assoc_net.m_va.m.iter().map(|x| x.abs()).collect();
                    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let idx = ((percentile / 100.0) * abs_vals.len() as f32) as usize;
                    let threshold = abs_vals.get(idx.min(abs_vals.len() - 1)).copied().unwrap_or(0.0);
                    assoc_net.m_va.m.mapv_inplace(|v| if v.abs() <= threshold { 0.0 } else { v });
                } else {
                    assoc_net.prune(config.prune_threshold);
                }
            }

            // Evaluation
            if step > 0 && step % eval_interval == 0 {
                let eval_results = if scoring.is_standard() {
                    retrieval.evaluate_retrieval(cached_v, cached_a, &assoc_net.m_va.m)
                } else {
                    let n = cached_v.nrows().min(1000);
                    let v_pool = cached_v.slice(ndarray::s![..n, ..]).to_owned();
                    let a_pool = cached_a.slice(ndarray::s![..n, ..]).to_owned();
                    let (sv2a, sa2v) = scoring.compute_eval_sims(&assoc_net, &v_pool, &a_pool);
                    retrieval.evaluate_from_sims(&sv2a, &sa2v)
                };
                let m_stats = matrix_stats.compute(&assoc_net.m_va.m);
                let heb_stats = assoc_net.m_va.get_stats();

                let mut metrics = eval_results;
                metrics.insert("estimated_rank".to_string(), m_stats.estimated_rank);
                metrics.insert("condition_number".to_string(), m_stats.condition_number);
                metrics.insert("frobenius_norm".to_string(), m_stats.frobenius_norm);
                metrics.insert("spectral_norm".to_string(), m_stats.spectral_norm);
                metrics.insert("sparsity".to_string(), m_stats.sparsity);
                metrics.insert("norm".to_string(), heb_stats.norm as f64);
                metrics.insert("update_count".to_string(), heb_stats.update_count as f64);

                let mrr = metrics.get("v2a_MRR").copied().unwrap_or(0.0);

                // SSE broadcast (clone only when subscribers are listening)
                if let Some(tx) = live_tx {
                    if tx.receiver_count() > 0 {
                        let _ = tx.send(MetricsEvent {
                            experiment_id,
                            step,
                            metrics: metrics.clone(),
                        });
                    }
                }

                // Move into last_metrics (no clone)
                last_metrics = metrics;
                if mrr > best_mrr {
                    best_mrr = mrr;
                    stale_evals = 0;
                } else {
                    stale_evals += 1;
                }

                if config.early_stop_patience > 0 && stale_evals >= config.early_stop_patience {
                    tracing::info!(
                        experiment_id,
                        "Early stopping: no improvement for {} evals",
                        stale_evals
                    );
                    step = config.max_steps;
                    break;
                }
            }

            step += 1;
        }
    }

    // Final evaluation (move eval_results directly, no clear+re-insert)
    if n_clips >= 10 {
        let mut metrics = if scoring.is_standard() {
            retrieval.evaluate_retrieval(cached_v, cached_a, &assoc_net.m_va.m)
        } else {
            let n = cached_v.nrows().min(1000);
            let v_pool = cached_v.slice(ndarray::s![..n, ..]).to_owned();
            let a_pool = cached_a.slice(ndarray::s![..n, ..]).to_owned();
            let (sv2a, sa2v) = scoring.compute_eval_sims(&assoc_net, &v_pool, &a_pool);
            retrieval.evaluate_from_sims(&sv2a, &sa2v)
        };
        let m_stats = matrix_stats.compute(&assoc_net.m_va.m);
        let heb_stats = assoc_net.m_va.get_stats();

        metrics.insert("estimated_rank".to_string(), m_stats.estimated_rank);
        metrics.insert("condition_number".to_string(), m_stats.condition_number);
        metrics.insert("frobenius_norm".to_string(), m_stats.frobenius_norm);
        metrics.insert("spectral_norm".to_string(), m_stats.spectral_norm);
        metrics.insert("sparsity".to_string(), m_stats.sparsity);
        metrics.insert("norm".to_string(), heb_stats.norm as f64);
        metrics.insert("update_count".to_string(), heb_stats.update_count as f64);
        last_metrics = metrics;
    }

    last_metrics.insert("steps_completed".to_string(), step as f64);
    last_metrics.insert("epochs".to_string(), epoch as f64);
    let duration = start.elapsed().as_secs_f64();
    last_metrics.insert("duration_seconds".to_string(), duration);

    // Save M matrix if output_dir is provided via config
    if let Some(output_dir) = &config.output_dir {
        let m_path = std::path::Path::new(output_dir).join("m_va.bin");
        if let Some(data) = assoc_net.m_va.m.as_slice() {
            let shape = assoc_net.m_va.m.shape();
            let header = format!("{}x{}\n", shape[0], shape[1]);
            let mut bytes = header.into_bytes();
            for val in data {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            if let Err(e) = std::fs::write(&m_path, &bytes) {
                tracing::warn!("Failed to save M matrix to {:?}: {}", m_path, e);
            } else {
                tracing::info!("Saved M matrix ({:?}) to {:?}", assoc_net.m_va.m.shape(), m_path);
            }
        }
    }

    ExperimentResult {
        metrics: last_metrics,
        steps_completed: step,
        epochs: epoch,
        duration_seconds: duration,
    }
}

/// Gather specific rows from a 2D array by index via slice memcpy.
#[inline]
fn gather_rows_into(arr: &Array2<f32>, indices: &[usize], buf: &mut Array2<f32>) {
    let cols = arr.ncols();
    let src = arr.as_slice().unwrap();
    let dst = buf.as_slice_mut().unwrap();
    for (i, &idx) in indices.iter().enumerate() {
        let src_off = idx * cols;
        let dst_off = i * cols;
        dst[dst_off..dst_off + cols].copy_from_slice(&src[src_off..src_off + cols]);
    }
}

/// L2 normalize each row in-place (single-pass per row).
#[inline]
fn l2_normalize_inplace(arr: &mut Array2<f32>) {
    let cols = arr.ncols();
    let slice = arr.as_slice_mut().unwrap();
    for chunk in slice.chunks_mut(cols) {
        let mut norm_sq = 0.0f32;
        for &v in &*chunk {
            norm_sq += v * v;
        }
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
        for v in chunk {
            *v *= inv_norm;
        }
    }
}

/// Alias for L2 normalize rows (used for NormalizePostUpdate).
#[inline]
fn normalize_rows_inplace(arr: &mut Array2<f32>) {
    l2_normalize_inplace(arr);
}

/// V2 training path: rectangular M with raw/whitened embeddings (skip_projection=true).
///
/// Uses gradient InfoNCE with symmetric loss on a (dv × da) M matrix.
/// This is the pipeline that achieved MRR=0.758 with ZCA-whitened embeddings.
fn run_experiment_v2(
    config: &ExperimentConfig,
    cached_v: &Array2<f32>,
    cached_a: &Array2<f32>,
    experiment_id: i64,
    shutdown: Arc<AtomicBool>,
    live_tx: Option<&broadcast::Sender<MetricsEvent>>,
) -> ExperimentResult {
    use brain_core::hebbian::HebbianAssociation;
    use brain_core::metrics::CrossModalRetrieval;

    let start = std::time::Instant::now();
    let n_clips = cached_v.nrows();
    let dv = cached_v.ncols();
    let da = cached_a.ncols();
    let batch_size = config.batch_size;
    let infonce_temp = config.infonce_temperature;

    tracing::info!(
        experiment_id, dv, da, batch_size,
        max_steps = config.max_steps,
        "V2 training: rectangular M ({}×{})", dv, da,
    );

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    // Create rectangular M (dv × da)
    let mut m = HebbianAssociation::new_rect(dv, da, config.hebbian_lr, config.decay_rate, config.max_norm);

    // Initialize M with small noise + scaled identity-like init
    let init_scale: f32 = config.init_scale;
    let min_dim = dv.min(da);
    {
        let m_slice = m.m.as_slice_mut().unwrap();
        for idx in 0..m_slice.len() {
            let row = idx / da;
            let col = idx % da;
            let diag = if row < min_dim && col < min_dim && row == col { init_scale } else { 0.0 };
            m_slice[idx] = diag + (rng.random::<f32>() - 0.5) * init_scale * 0.1;
        }
    }

    // Pre-allocate buffers
    let mut v_buf = Array2::zeros((batch_size, dv));
    let mut a_buf = Array2::zeros((batch_size, da));
    let mut sim_matrix = Array2::zeros((batch_size, batch_size));
    let mut softmax_probs = Array2::zeros((batch_size, batch_size));
    let mut weighted_neg = Array2::zeros((batch_size, da));
    let mut proj_a_buf = Array2::zeros((batch_size, dv));
    let mut sym_temp_buf = Array2::zeros((batch_size, da));

    // Scheduler: cosine decay
    let warmup_steps = 100usize;
    let min_lr = 1e-5f32;
    let lr = config.hebbian_lr;

    let retrieval = CrossModalRetrieval::new(n_clips.min(config.eval_pool_size.max(1000)), vec![1, 5, 10]);
    let eval_interval = config.eval_interval.unwrap_or_else(|| 50usize.max(config.max_steps / 10));

    let indices: Vec<usize> = (0..n_clips).collect();
    let mut step = 0usize;
    let mut epoch = 0usize;
    let mut best_mrr = 0.0f64;
    let mut best_m: Option<Array2<f32>> = None;
    let mut last_metrics: HashMap<String, f64> = HashMap::new();

    while step < config.max_steps && !shutdown.load(Ordering::Relaxed) {
        epoch += 1;
        let mut batch_indices: Vec<usize> = indices.clone();
        batch_indices.shuffle(&mut rng);

        for chunk in batch_indices.chunks(batch_size) {
            if step >= config.max_steps || shutdown.load(Ordering::Relaxed) { break; }
            if chunk.len() < batch_size { continue; }

            // Cosine LR schedule
            let current_lr = if step < warmup_steps {
                min_lr + (lr - min_lr) * (step as f32 / warmup_steps as f32)
            } else {
                let progress = (step - warmup_steps) as f32 / (config.max_steps - warmup_steps) as f32;
                min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
            };

            // Gather batch
            gather_rows_into(cached_v, chunk, &mut v_buf);
            gather_rows_into(cached_a, chunk, &mut a_buf);
            l2_normalize_inplace(&mut v_buf);
            l2_normalize_inplace(&mut a_buf);

            // V→A: sim = V @ M @ A^T / temp
            {
                let m_at = m.m.dot(&a_buf.t());
                ndarray::linalg::general_mat_mul(1.0 / infonce_temp, &v_buf, &m_at, 0.0, &mut sim_matrix);
            }

            // Softmax per row
            for i in 0..batch_size {
                let max_val = (0..batch_size).map(|j| sim_matrix[[i, j]]).fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..batch_size {
                    let v = (sim_matrix[[i, j]] - max_val).exp();
                    softmax_probs[[i, j]] = v;
                    sum += v;
                }
                let inv = 1.0 / sum.max(1e-12);
                for j in 0..batch_size {
                    softmax_probs[[i, j]] *= inv;
                }
            }

            // target - P (standard identity target)
            for i in 0..batch_size {
                for j in 0..batch_size {
                    let target = if i == j { 1.0 } else { 0.0 };
                    softmax_probs[[i, j]] = target - softmax_probs[[i, j]];
                }
            }

            // Gradient: M += scale * V^T @ (I - P) @ A
            let grad_scale = current_lr / (batch_size as f32 * infonce_temp);
            ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);
            ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &weighted_neg, 1.0, &mut m.m);

            // Symmetric: A→V direction
            if config.symmetric_infonce {
                let m_t = m.m.t();
                ndarray::linalg::general_mat_mul(1.0, &a_buf, &m_t, 0.0, &mut proj_a_buf);
                ndarray::linalg::general_mat_mul(1.0 / infonce_temp, &proj_a_buf, &v_buf.t(), 0.0, &mut sim_matrix);

                for i in 0..batch_size {
                    let max_val = (0..batch_size).map(|j| sim_matrix[[i, j]]).fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..batch_size {
                        let v = (sim_matrix[[i, j]] - max_val).exp();
                        softmax_probs[[i, j]] = v;
                        sum += v;
                    }
                    let inv = 1.0 / sum.max(1e-12);
                    for j in 0..batch_size {
                        softmax_probs[[i, j]] *= inv;
                    }
                }
                for i in 0..batch_size {
                    for j in 0..batch_size {
                        let target = if i == j { 1.0 } else { 0.0 };
                        softmax_probs[[i, j]] = target - softmax_probs[[i, j]];
                    }
                }
                ndarray::linalg::general_mat_mul(1.0, &softmax_probs.t(), &a_buf, 0.0, &mut sym_temp_buf);
                ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &sym_temp_buf, 1.0, &mut m.m);
            }

            // Spectral norm clipping
            if m.max_norm > 0.0 && step % 5 == 0 {
                let spectral = m.power_iteration_spectral_norm(2);
                if spectral > m.max_norm {
                    let clip = m.max_norm / spectral;
                    m.m.mapv_inplace(|v| v * clip);
                }
            }
            m.update_count += 1;

            // Eval
            if step > 0 && step % eval_interval == 0 {
                let n = n_clips.min(retrieval.pool_size);
                let v_eval = cached_v.slice(ndarray::s![..n, ..]).to_owned();
                let a_eval = cached_a.slice(ndarray::s![..n, ..]).to_owned();
                let mut v_norm = v_eval.clone();
                let mut a_norm = a_eval.clone();
                l2_normalize_inplace(&mut v_norm);
                l2_normalize_inplace(&mut a_norm);
                let pv = v_norm.dot(&m.m);
                let sim_v2a = pv.dot(&a_norm.t());
                let pa = a_norm.dot(&m.m.t());
                let sim_a2v = pa.dot(&v_norm.t());
                let metrics = retrieval.evaluate_from_sims(&sim_v2a, &sim_a2v);

                let mrr = metrics.get("v2a_MRR").copied().unwrap_or(0.0);
                if mrr > best_mrr {
                    best_mrr = mrr;
                    best_m = Some(m.m.clone());
                }

                // Publish live metrics
                if let Some(tx) = live_tx {
                    let mut event_metrics: HashMap<String, f64> = metrics.clone();
                    event_metrics.insert("lr".to_string(), current_lr as f64);
                    let _ = tx.send(MetricsEvent {
                        experiment_id,
                        step,
                        metrics: event_metrics,
                    });
                }
                last_metrics = metrics;
            }

            step += 1;
        }
    }

    // Restore best checkpoint
    if let Some(best) = best_m {
        m.m = best;
    }

    // Final eval
    let n = n_clips.min(retrieval.pool_size);
    let v_eval = cached_v.slice(ndarray::s![..n, ..]).to_owned();
    let a_eval = cached_a.slice(ndarray::s![..n, ..]).to_owned();
    let mut v_norm = v_eval;
    let mut a_norm = a_eval;
    l2_normalize_inplace(&mut v_norm);
    l2_normalize_inplace(&mut a_norm);
    let pv = v_norm.dot(&m.m);
    let sim_v2a = pv.dot(&a_norm.t());
    let pa = a_norm.dot(&m.m.t());
    let sim_a2v = pa.dot(&v_norm.t());
    let final_metrics = retrieval.evaluate_from_sims(&sim_v2a, &sim_a2v);

    let duration = start.elapsed().as_secs_f64();
    let mut metrics = final_metrics;
    metrics.insert("duration_seconds".to_string(), duration);

    ExperimentResult {
        metrics,
        steps_completed: step,
        epochs: epoch,
        duration_seconds: duration,
    }
}

/// Compute hard negatives: for each clip i, find top-k most similar wrong clips.
fn compute_hard_negatives(
    cached_v: &Array2<f32>,
    cached_a: &Array2<f32>,
    m: &Array2<f32>,
    top_k: usize,
) -> Vec<Vec<usize>> {
    let n = cached_v.nrows();
    let k = top_k.min(n - 1);
    let m_a_t = m.dot(&cached_a.t()); // [D, N]

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let v_i = cached_v.row(i);
        let sims: Vec<f32> = (0..n).map(|j| v_i.dot(&m_a_t.column(j))).collect();
        let mut indexed: Vec<(usize, f32)> = sims
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, &s)| (j, s))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.push(indexed.iter().take(k).map(|(j, _)| *j).collect());
    }
    result
}
