//! Experiment runner with mutation support.
//!
//! Wraps the base training loop but applies MutationConfig modifications
//! to the Hebbian update, pruning, sparse projection, or training parameters.

use brain_core::association_network::{AssociationConfig, AssociationNetwork};
use brain_core::metrics::{CrossModalRetrieval, MatrixStats};
use brain_core::scheduler::{DecayType, HebbianScheduler};

use crate::config::ExperimentConfig;
use crate::mutations::{MutationConfig, MutationStrategy};
use crate::runner::{ExperimentResult, MetricsEvent};
use crate::scoring::ScoringModel;

use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Run experiment with an applied mutation.
pub fn run_mutated_experiment(
    config: &ExperimentConfig,
    mutation: &MutationConfig,
    cached_v: &Array2<f32>,
    cached_a: &Array2<f32>,
    cached_e: &Array2<f32>,
    cached_s: Option<&Array2<f32>>,
    cached_p: Option<&Array2<f32>>,
    experiment_id: i64,
    shutdown: Arc<AtomicBool>,
    live_tx: Option<&broadcast::Sender<MetricsEvent>>,
) -> ExperimentResult {
    // Apply training-loop-level mutations to config
    let mut cfg = config.clone();
    apply_config_mutations(&mut cfg, &mutation.strategy);

    // V2 path: skip_projection uses rectangular M — delegate to run_experiment
    // which handles the v2 path internally
    if cfg.skip_projection {
        return crate::runner::run_experiment(
            &cfg, cached_v, cached_a, cached_e, cached_s, cached_p,
            experiment_id, shutdown, live_tx,
        );
    }

    let start = std::time::Instant::now();

    let d = cfg.embed_dim;
    let n_clips = cached_v.nrows();

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    let assoc_config = AssociationConfig {
        d,
        lr: cfg.hebbian_lr,
        decay_rate: cfg.decay_rate,
        temporal_decay: cfg.temporal_decay,
        trace_weight: cfg.trace_weight,
        max_norm: cfg.max_norm,
    };
    let mut assoc_net = AssociationNetwork::new(&assoc_config);
    let mut scoring = ScoringModel::from_config(&cfg, &assoc_config, &mut rng);

    // Initialize M
    if cfg.cross_correlation_init {
        // Cross-correlation init: M = blend * (V^T @ A / N) + (1-blend) * init_scale * I
        let n = cached_v.nrows() as f32;
        let blend = cfg.cross_correlation_blend;
        let init_scale = cfg.init_scale;
        let mut cross_corr = Array2::zeros((d, d));
        ndarray::linalg::general_mat_mul(1.0 / n, &cached_v.t(), &cached_a, 0.0, &mut cross_corr);
        // Scale cross-corr to match init_scale magnitude
        let cc_norm: f32 = cross_corr.iter().map(|x| x * x).sum::<f32>().sqrt();
        let target_norm = init_scale * (d as f32).sqrt();
        if cc_norm > 1e-12 {
            cross_corr *= target_norm / cc_norm;
        }
        // Blend with identity
        for m in [&mut assoc_net.m_va.m, &mut assoc_net.m_ve.m, &mut assoc_net.m_ae.m] {
            let slice = m.as_slice_mut().unwrap();
            let cc_slice = cross_corr.as_slice().unwrap();
            for (idx, v) in slice.iter_mut().enumerate() {
                let row = idx / d;
                let col = idx % d;
                let diag = if row == col { init_scale } else { 0.0 };
                *v = blend * cc_slice[idx] + (1.0 - blend) * diag;
            }
        }
    } else {
        // Standard init: scaled identity + small noise
        let init_scale = cfg.init_scale;
        let noise_scale = init_scale * 0.1;
        for m in [&mut assoc_net.m_va.m, &mut assoc_net.m_ve.m, &mut assoc_net.m_ae.m] {
            let slice = m.as_slice_mut().unwrap();
            for (idx, v) in slice.iter_mut().enumerate() {
                let row = idx / d;
                let col = idx % d;
                let diag = if row == col { init_scale } else { 0.0 };
                *v = diag + (rng.random::<f32>() - 0.5) * noise_scale;
            }
        }
    }

    let warmup = (100.0 * cfg.warmup_factor) as usize;
    let decay_type = match &mutation.strategy {
        MutationStrategy::LinearSchedule => DecayType::Linear,
        _ => DecayType::Cosine,
    };

    let scheduler = HebbianScheduler::new(cfg.hebbian_lr, warmup, decay_type, 1e-5, cfg.max_steps);

    let eval_pool_size = if cfg.eval_pool_size == 0 { n_clips } else { n_clips.min(cfg.eval_pool_size) };
    let retrieval = CrossModalRetrieval::new(eval_pool_size, vec![1, 5, 10]);
    let matrix_stats = MatrixStats::new(1e-3);

    let eval_interval = cfg.eval_interval.unwrap_or_else(|| 50usize.max(cfg.max_steps / 10));
    let batch_size = cfg.batch_size;

    let consolidation_interval = cfg.consolidation_interval;

    let indices: Vec<usize> = (0..n_clips).collect();

    // Pre-allocate reusable batch buffers
    let mut v_buf = Array2::zeros((batch_size, d));
    let mut a_buf = Array2::zeros((batch_size, d));
    let mut e_buf = Array2::zeros((batch_size, d));
    let mut s_buf = if cached_s.is_some() { Array2::zeros((batch_size, d)) } else { Array2::zeros((0, 0)) };
    let mut p_buf = if cached_p.is_some() { Array2::zeros((batch_size, d)) } else { Array2::zeros((0, 0)) };
    let mut perm: Vec<usize> = (0..batch_size).collect();
    let mut a_neg_buf = Array2::zeros((batch_size, d));
    let mut neg_delta = Array2::zeros((d, d));
    let mut noise_buf = Array2::zeros((batch_size, d));

    let mut step = 0usize;
    let mut epoch = 0usize;
    let mut last_metrics: HashMap<String, f64> = HashMap::new();
    let mut best_mrr = 0.0f64;
    let mut stale_evals = 0usize;

    // Mutation state
    let mut momentum_buf: Option<Array2<f32>> = match &mutation.strategy {
        MutationStrategy::Momentum { .. } => Some(Array2::zeros((d, d))),
        _ => None,
    };
    let mut bcm_threshold: Option<f32> = match &mutation.strategy {
        MutationStrategy::BcmThreshold { initial_threshold, .. } => Some(*initial_threshold),
        _ => None,
    };

    // EMA shadow model for Polyak averaging
    let mut ema_m: Option<Array2<f32>> = if cfg.ema_beta > 0.0 {
        Some(Array2::zeros((d, d)))
    } else {
        None
    };
    let mut ema_initialized = false;

    // Gradient accumulation: implemented via larger effective batch size
    // (grad_accumulation > 1 scales batch_size at config level instead of buffering)

    // Hard negative mining state
    let mut hard_neg_indices: Option<Vec<Vec<usize>>> = None;
    if cfg.hard_negative_k > 0 {
        hard_neg_indices = Some(compute_hard_negatives(
            cached_v, cached_a, &assoc_net.m_va.m, cfg.hard_negative_k,
        ));
    }

    // InfoNCE pre-allocated buffers
    let needs_infonce_bufs = cfg.use_infonce || cfg.gradient_infonce;
    let mut sim_matrix = if needs_infonce_bufs { Array2::zeros((batch_size, batch_size)) } else { Array2::zeros((0, 0)) };
    let mut softmax_probs = if needs_infonce_bufs { Array2::zeros((batch_size, batch_size)) } else { Array2::zeros((0, 0)) };
    let mut weighted_neg = if needs_infonce_bufs { Array2::zeros((batch_size, d)) } else { Array2::zeros((0, 0)) };

    // Curriculum learning
    let curriculum_order = if cfg.curriculum {
        Some(crate::scoring::curriculum_order(cached_v, cached_a))
    } else {
        None
    };

    while step < cfg.max_steps && !shutdown.load(Ordering::Relaxed) {
        epoch += 1;

        let active_indices = if let Some(ref order) = curriculum_order {
            let frac = cfg.curriculum_start_frac
                + (1.0 - cfg.curriculum_start_frac) * (step as f32 / cfg.max_steps as f32);
            let active_n = ((n_clips as f32 * frac).ceil() as usize).min(n_clips);
            &order[..active_n]
        } else {
            &indices[..]
        };
        let mut batch_indices: Vec<usize> = active_indices.to_vec();
        batch_indices.shuffle(&mut rng);

        for chunk in batch_indices.chunks(batch_size) {
            if step >= cfg.max_steps || shutdown.load(Ordering::Relaxed) {
                break;
            }
            if chunk.len() < batch_size {
                continue;
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

            // Augmentation noise via pre-filled buffer + scaled_add
            if cfg.aug_noise > 0.0 {
                let noise_scale = cfg.aug_noise;
                noise_buf.mapv_inplace(|_| rng.random::<f32>() * 2.0 - 1.0);
                v_buf.scaled_add(noise_scale, &noise_buf);
                noise_buf.mapv_inplace(|_| rng.random::<f32>() * 2.0 - 1.0);
                a_buf.scaled_add(noise_scale, &noise_buf);
                l2_normalize_inplace(&mut v_buf);
                l2_normalize_inplace(&mut a_buf);
            }

            // Mixup augmentation: create interpolated positive pairs
            if cfg.mixup_alpha > 0.0 && batch_size > 1 {
                // For half the batch, interpolate with a random other sample
                let mix_n = batch_size / 2;
                for i in 0..mix_n {
                    let j = (i + 1 + rng.random_range(0..batch_size - 1)) % batch_size;
                    let lambda = 0.5 + (rng.random::<f32>() - 0.5) * cfg.mixup_alpha;
                    let one_minus = 1.0 - lambda;
                    for k in 0..d {
                        v_buf[[i, k]] = lambda * v_buf[[i, k]] + one_minus * v_buf[[j, k]];
                        a_buf[[i, k]] = lambda * a_buf[[i, k]] + one_minus * a_buf[[j, k]];
                    }
                }
            }

            // Pre-activation normalization mutation (in-place)
            if matches!(&mutation.strategy, MutationStrategy::NormalizePreActivations) {
                l2_normalize_inplace(&mut v_buf);
                l2_normalize_inplace(&mut a_buf);
            }

            let e_input = if e_buf.iter().any(|&x| x != 0.0) { Some(&e_buf as &Array2<f32>) } else { None };
            let s_input = if cached_s.is_some() { Some(&s_buf as &Array2<f32>) } else { None };
            let p_input = if cached_p.is_some() { Some(&p_buf as &Array2<f32>) } else { None };

            // Cyclic LR override
            let current_lr = if cfg.cyclic_lr_period > 0 && cfg.cyclic_lr_min > 0.0 {
                let base_lr = cfg.hebbian_lr;
                let t = (step % cfg.cyclic_lr_period) as f32 / cfg.cyclic_lr_period as f32;
                let factor = cfg.cyclic_lr_min + (cfg.cyclic_lr_max - cfg.cyclic_lr_min) * 0.5 * (1.0 + (t * std::f32::consts::PI * 2.0).cos());
                base_lr * factor
            } else {
                scheduler.get_lr(step)
            };
            assoc_net.set_lr(current_lr);

            // Warmup temperature: linearly interpolate from start_temp to end_temp
            let effective_temp = if cfg.warmup_temp_start > 0.0 && cfg.warmup_temp_end > 0.0 {
                let progress = (step as f32) / (cfg.max_steps as f32).max(1.0);
                cfg.warmup_temp_start + (cfg.warmup_temp_end - cfg.warmup_temp_start) * progress
            } else {
                cfg.infonce_temperature
            };

            if cfg.gradient_infonce && batch_size > 1 {
                // === Fused gradient descent InfoNCE ===
                let infonce_temp = effective_temp;
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
                // (I - P): negate softmax and add identity
                for i in 0..batch_size {
                    for j in 0..batch_size {
                        softmax_probs[[i, j]] = if i == j { 1.0 } else { 0.0 } - softmax_probs[[i, j]];
                    }
                }
                // weighted_target = (I - P) @ A
                ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);
                // M += (lr / (B * temp)) * V^T @ weighted_target
                let grad_scale = current_lr / (batch_size as f32 * infonce_temp);
                ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &weighted_neg, 1.0, &mut assoc_net.m_va.m);

                // === Reverse direction: A→V InfoNCE (symmetric loss) ===
                if cfg.symmetric_infonce {
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
                    // R = I - P_a2v
                    for i in 0..batch_size {
                        for j in 0..batch_size {
                            softmax_probs[[i, j]] = if i == j { 1.0 } else { 0.0 } - softmax_probs[[i, j]];
                        }
                    }
                    // dM += scale * V^T @ R^T @ A
                    ndarray::linalg::general_mat_mul(1.0, &softmax_probs.t(), &a_buf, 0.0, &mut weighted_neg);
                    ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &weighted_neg, 1.0, &mut assoc_net.m_va.m);
                }

                // Spectral norm clipping
                if assoc_net.m_va.max_norm > 0.0 && step % 5 == 0 {
                    let spectral = assoc_net.m_va.power_iteration_spectral_norm(2);
                    if spectral > assoc_net.m_va.max_norm {
                        let clip = assoc_net.m_va.max_norm / spectral;
                        assoc_net.m_va.m.mapv_inplace(|v| v * clip);
                    }
                }
                assoc_net.m_va.update_count += 1;
            } else {
                // === Legacy: separate Hebbian positive + contrastive negative ===
                scoring.update(&mut assoc_net, &v_buf, &a_buf, e_input, s_input, p_input, current_lr, batch_size);

                // Apply post-update Hebbian mutations
                apply_hebbian_mutations(
                    &mutation.strategy,
                    &mut assoc_net,
                    &v_buf,
                    &a_buf,
                    current_lr,
                    batch_size,
                    d,
                    &mut momentum_buf,
                    &mut bcm_threshold,
                );

                // Contrastive learning
                if cfg.use_siglip && batch_size > 1 {
                    let tau = effective_temp;
                    let bias = if cfg.siglip_bias != 0.0 { cfg.siglip_bias } else { -(batch_size as f32).ln() };
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
                } else if cfg.use_infonce && batch_size > 1 {
                    let infonce_temp = effective_temp;
                    scoring.compute_batch_sim(&assoc_net, &v_buf, &a_buf, infonce_temp, &mut sim_matrix);
                    let eps = cfg.label_smoothing;
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
                        if eps > 0.0 {
                            let smooth_pos = 1.0 - eps;
                            let smooth_neg = eps / (batch_size - 1) as f32;
                            for j in 0..batch_size {
                                let target = if i == j { smooth_pos } else { smooth_neg };
                                softmax_probs[[i, j]] -= target;
                            }
                        }
                    }
                    ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);
                    scoring.apply_infonce_neg(&mut assoc_net, &v_buf, &weighted_neg, current_lr, batch_size);
                } else if cfg.neg_weight > 0.0 && batch_size > 1 {
                    // Standard anti-Hebbian with random permutation (or hard negatives)
                    let neg_weight = cfg.neg_weight;

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

                    let neg_scale = current_lr * neg_weight / batch_size as f32;
                    ndarray::linalg::general_mat_mul(neg_scale, &v_buf.t(), &a_neg_buf, 0.0, &mut neg_delta);
                    assoc_net.m_va.m -= &neg_delta;
                }
            }

            // Refresh hard negatives periodically
            if cfg.hard_negative_k > 0
                && cfg.hard_negative_refresh > 0
                && step > 0
                && step % cfg.hard_negative_refresh == 0
            {
                hard_neg_indices = Some(compute_hard_negatives(
                    cached_v, cached_a, &assoc_net.m_va.m, cfg.hard_negative_k,
                ));
            }

            // Spectral regularization: add alpha * I to M to prevent rank collapse
            if cfg.spectral_reg_alpha > 0.0 && step % 10 == 0 {
                let alpha = cfg.spectral_reg_alpha;
                let m_slice = assoc_net.m_va.m.as_slice_mut().unwrap();
                for i in 0..d {
                    m_slice[i * d + i] += alpha;
                }
            }

            // Diagonal boost: periodically add identity to counteract rank collapse
            if cfg.diagonal_boost > 0.0
                && cfg.diagonal_boost_interval > 0
                && step > 0
                && step % cfg.diagonal_boost_interval == 0
            {
                let boost = cfg.diagonal_boost;
                let m_slice = assoc_net.m_va.m.as_slice_mut().unwrap();
                for i in 0..d {
                    m_slice[i * d + i] += boost;
                }
            }

            // EMA update: shadow_M = beta * shadow_M + (1-beta) * M
            if let Some(ref mut shadow) = ema_m {
                let beta = cfg.ema_beta;
                if !ema_initialized {
                    shadow.assign(&assoc_net.m_va.m);
                    ema_initialized = true;
                } else {
                    let one_minus = 1.0 - beta;
                    let s_slice = shadow.as_slice_mut().unwrap();
                    let m_slice = assoc_net.m_va.m.as_slice().unwrap();
                    for i in 0..(d * d) {
                        s_slice[i] = beta * s_slice[i] + one_minus * m_slice[i];
                    }
                }
            }

            // Post-update normalization (from mutation or inherited config)
            if cfg.normalize_post_update {
                normalize_rows_inplace(&mut assoc_net.m_va.m);
            }

            // Pruning (with possible mutations or inherited config)
            if consolidation_interval > 0 && step > 0 && step % consolidation_interval == 0 {
                if let Some(percentile) = cfg.percentile_prune {
                    // Percentile pruning (from mutation or inherited config)
                    let mut abs_vals: Vec<f32> = assoc_net.m_va.m.iter().map(|x| x.abs()).collect();
                    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let idx = ((percentile / 100.0) * abs_vals.len() as f32) as usize;
                    let threshold = abs_vals.get(idx.min(abs_vals.len() - 1)).copied().unwrap_or(0.0);
                    assoc_net.m_va.m.mapv_inplace(|v| if v.abs() <= threshold { 0.0 } else { v });
                } else {
                    apply_prune_mutation(
                        &mutation.strategy,
                        &mut assoc_net,
                        cfg.prune_threshold,
                    );
                }
            }

            // Evaluation
            if step > 0 && step % eval_interval == 0 {
                // If EMA enabled, swap in shadow M for evaluation
                let saved_m = if let Some(ref shadow) = ema_m {
                    let saved = assoc_net.m_va.m.clone();
                    assoc_net.m_va.m.assign(shadow);
                    Some(saved)
                } else {
                    None
                };

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
                metrics.insert("estimated_rank".into(), m_stats.estimated_rank);
                metrics.insert("condition_number".into(), m_stats.condition_number);
                metrics.insert("frobenius_norm".into(), m_stats.frobenius_norm);
                metrics.insert("spectral_norm".into(), m_stats.spectral_norm);
                metrics.insert("sparsity".into(), m_stats.sparsity);
                metrics.insert("norm".into(), heb_stats.norm as f64);
                metrics.insert("update_count".into(), heb_stats.update_count as f64);

                // Restore training M if we swapped in EMA
                if let Some(saved) = saved_m {
                    assoc_net.m_va.m = saved;
                }

                let mrr = metrics.get("v2a_MRR").copied().unwrap_or(0.0);

                if let Some(tx) = live_tx {
                    if tx.receiver_count() > 0 {
                        let _ = tx.send(MetricsEvent {
                            experiment_id,
                            step,
                            metrics: metrics.clone(),
                        });
                    }
                }

                last_metrics = metrics;
                if mrr > best_mrr {
                    best_mrr = mrr;
                    stale_evals = 0;
                } else {
                    stale_evals += 1;
                }

                if cfg.early_stop_patience > 0 && stale_evals >= cfg.early_stop_patience {
                    step = cfg.max_steps;
                    break;
                }
            }

            step += 1;
        }
    }

    // Final evaluation (move eval_results directly)
    // If EMA, use shadow M for final eval
    if let Some(ref shadow) = ema_m {
        assoc_net.m_va.m.assign(shadow);
    }
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

        metrics.insert("estimated_rank".into(), m_stats.estimated_rank);
        metrics.insert("condition_number".into(), m_stats.condition_number);
        metrics.insert("frobenius_norm".into(), m_stats.frobenius_norm);
        metrics.insert("spectral_norm".into(), m_stats.spectral_norm);
        metrics.insert("sparsity".into(), m_stats.sparsity);
        metrics.insert("norm".into(), heb_stats.norm as f64);
        metrics.insert("update_count".into(), heb_stats.update_count as f64);
        last_metrics = metrics;
    }

    last_metrics.insert("steps_completed".into(), step as f64);
    last_metrics.insert("epochs".into(), epoch as f64);
    let duration = start.elapsed().as_secs_f64();
    last_metrics.insert("duration_seconds".into(), duration);

    ExperimentResult {
        metrics: last_metrics,
        steps_completed: step,
        epochs: epoch,
        duration_seconds: duration,
    }
}

/// Compute the effective config after applying a mutation strategy.
/// This is the config that should be stored in the DB so improvements accumulate.
pub fn compute_mutated_config(base: &ExperimentConfig, strategy: &MutationStrategy) -> ExperimentConfig {
    let mut cfg = base.clone();
    apply_config_mutations(&mut cfg, strategy);
    cfg
}

/// Apply config-level mutations (training loop parameters).
fn apply_config_mutations(cfg: &mut ExperimentConfig, strategy: &MutationStrategy) {
    match strategy {
        MutationStrategy::ScaleLearningRate { factor } => {
            cfg.hebbian_lr *= factor;
        }
        MutationStrategy::ScaleAugNoise { factor } => {
            cfg.aug_noise *= factor;
        }
        MutationStrategy::ScaleInitScale { factor } => {
            cfg.init_scale *= factor;
        }
        MutationStrategy::ScaleK { factor } => {
            cfg.sparsity_k = (cfg.sparsity_k as f32 * factor).round().max(1.0) as usize;
        }
        MutationStrategy::ScaleTemporalDecay { factor } => {
            cfg.temporal_decay = (cfg.temporal_decay * factor).clamp(0.01, 0.999);
        }
        MutationStrategy::ScaleTraceWeight { factor } => {
            cfg.trace_weight = (cfg.trace_weight * factor).clamp(0.0, 1.0);
        }
        MutationStrategy::ScaleNegWeight { factor } => {
            cfg.neg_weight *= factor;
        }
        MutationStrategy::ScalePruneThreshold { factor } => {
            cfg.prune_threshold *= factor;
        }
        MutationStrategy::ScaleConsolidationInterval { factor } => {
            cfg.consolidation_interval = (cfg.consolidation_interval as f32 * factor).max(1.0) as usize;
        }
        MutationStrategy::ScaleWarmup { factor } => {
            cfg.warmup_factor *= factor;
        }
        MutationStrategy::NormalizePostUpdate => {
            cfg.normalize_post_update = true;
        }
        MutationStrategy::PercentilePrune { percentile } => {
            cfg.percentile_prune = Some(*percentile);
        }
        MutationStrategy::ScaleBatchSize { target_size } => {
            cfg.batch_size = *target_size;
            // Hebbian forward already divides by batch_size, so no LR scaling needed.
            // Larger batch = more negatives per step but same gradient magnitude.
        }
        MutationStrategy::InfoNce { temperature } => {
            cfg.use_infonce = true;
            cfg.infonce_temperature = *temperature;
            // InfoNCE needs larger batches to be effective
            if cfg.batch_size < 32 {
                cfg.batch_size = 64;
            }
        }
        MutationStrategy::CrossCorrelationInit { blend } => {
            cfg.cross_correlation_init = true;
            cfg.cross_correlation_blend = *blend;
        }
        MutationStrategy::HardNegativeMining { top_k, refresh_interval } => {
            cfg.hard_negative_k = *top_k;
            cfg.hard_negative_refresh = *refresh_interval;
        }
        MutationStrategy::ScaleMaxSteps { target_steps } => {
            cfg.max_steps = *target_steps;
        }
        MutationStrategy::MultiHead { num_heads } => {
            cfg.num_heads = *num_heads;
        }
        MutationStrategy::MlpAssociation { hidden_dim } => {
            cfg.mlp_hidden = *hidden_dim;
        }
        MutationStrategy::LowRank { rank } => {
            cfg.low_rank = *rank;
        }
        MutationStrategy::CurriculumLearning { start_frac } => {
            cfg.curriculum = true;
            cfg.curriculum_start_frac = *start_frac;
        }
        MutationStrategy::SpectralRegularization { alpha } => {
            cfg.spectral_reg_alpha = *alpha;
        }
        MutationStrategy::LabelSmoothing { epsilon } => {
            cfg.label_smoothing = *epsilon;
        }
        MutationStrategy::GradientAccumulation { accumulate_steps } => {
            // Simulate gradient accumulation by scaling batch size
            cfg.batch_size = (cfg.batch_size * accumulate_steps).min(2048);
        }
        MutationStrategy::Mixup { alpha } => {
            cfg.mixup_alpha = *alpha;
        }
        MutationStrategy::WarmupTemperature { start_temp, end_temp } => {
            cfg.warmup_temp_start = *start_temp;
            cfg.warmup_temp_end = *end_temp;
        }
        MutationStrategy::EmaEval { beta } => {
            cfg.ema_beta = *beta;
        }
        MutationStrategy::DiagonalBoost { boost, interval } => {
            cfg.diagonal_boost = *boost;
            cfg.diagonal_boost_interval = *interval;
        }
        MutationStrategy::CyclicLr { min_factor, max_factor, cycle_steps } => {
            cfg.cyclic_lr_min = *min_factor;
            cfg.cyclic_lr_max = *max_factor;
            cfg.cyclic_lr_period = *cycle_steps;
        }
        MutationStrategy::GradientInfoNce => {
            cfg.gradient_infonce = true;
        }
        MutationStrategy::SymmetricInfoNce => {
            cfg.gradient_infonce = true;
            cfg.symmetric_infonce = true;
        }
        MutationStrategy::ScaleMaxStepsLong { max_steps } => {
            cfg.max_steps = *max_steps;
        }
        MutationStrategy::FullPoolEval => {
            cfg.eval_pool_size = 0; // 0 = use all clips
        }
        MutationStrategy::Composite { mutations } => {
            for m in mutations {
                apply_config_mutations(cfg, m);
            }
        }
        _ => {}
    }
}

/// Apply Hebbian update mutations after the standard forward pass.
fn apply_hebbian_mutations(
    strategy: &MutationStrategy,
    assoc_net: &mut AssociationNetwork,
    v: &Array2<f32>,
    a: &Array2<f32>,
    lr: f32,
    batch_size: usize,
    d: usize,
    momentum_buf: &mut Option<Array2<f32>>,
    bcm_threshold: &mut Option<f32>,
) {
    match strategy {
        MutationStrategy::OjasRule { strength } => {
            // Oja's rule: M -= strength * lr * (M @ a^T) @ a / batch
            let proj = assoc_net.m_va.m.dot(&a.t()); // [D, B]
            let correction = proj.dot(a); // [D, D]
            let scale = strength * lr / batch_size as f32;
            assoc_net.m_va.m.scaled_add(-scale, &correction);
        }
        MutationStrategy::WeightDecay { lambda } => {
            assoc_net.m_va.m.mapv_inplace(|v| v * (1.0 - lambda));
        }
        MutationStrategy::WeightClip { max_val } => {
            assoc_net.m_va.m.mapv_inplace(|v| v.clamp(-max_val, *max_val));
        }
        MutationStrategy::CompetitiveTopK { k_fraction } => {
            // Only keep updates for most active rows
            let k = (d as f32 * k_fraction).max(1.0) as usize;
            let mut row_norms: Vec<(usize, f32)> = (0..d)
                .map(|i| {
                    let norm: f32 = assoc_net.m_va.m.row(i).iter().map(|x| x * x).sum();
                    (i, norm)
                })
                .collect();
            row_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            // Zero out non-top-k rows' recent updates (approximate by scaling down)
            for &(i, _) in &row_norms[k..] {
                assoc_net.m_va.m.row_mut(i).mapv_inplace(|v| v * 0.99);
            }
        }
        MutationStrategy::Momentum { beta } => {
            if let Some(buf) = momentum_buf {
                // Exponential moving average of M
                let m = &assoc_net.m_va.m;
                buf.zip_mut_with(m, |b, &m_val| {
                    *b = *b * beta + m_val * (1.0 - beta);
                });
            }
        }
        MutationStrategy::BcmThreshold { sliding_rate, .. } => {
            if let Some(theta) = bcm_threshold {
                // Compute mean post-synaptic activity
                let proj = v.dot(&assoc_net.m_va.m); // [B, D]
                let mean_activity: f32 = proj.iter().map(|x| x.abs()).sum::<f32>()
                    / (proj.len() as f32).max(1.0);
                // Slide threshold toward mean activity
                *theta = *theta * (1.0 - sliding_rate) + mean_activity * sliding_rate;
                // Modulate M by (activity - threshold)
                // This is approximate: scale rows by their deviation from threshold
                let row_activities: Vec<f32> = (0..d)
                    .map(|j| {
                        proj.column(j).iter().map(|x| x.abs()).sum::<f32>() / batch_size as f32
                    })
                    .collect();
                for j in 0..d {
                    let mod_factor = ((row_activities[j] - *theta) * 0.01).clamp(-0.1, 0.1);
                    assoc_net.m_va.m.column_mut(j).mapv_inplace(|v| v * (1.0 + mod_factor));
                }
            }
        }
        _ => {}
    }
}

/// Apply pruning mutations.
fn apply_prune_mutation(
    strategy: &MutationStrategy,
    assoc_net: &mut AssociationNetwork,
    base_threshold: f32,
) {
    match strategy {
        MutationStrategy::ScalePruneThreshold { factor } => {
            assoc_net.prune(base_threshold * factor);
        }
        MutationStrategy::ConsolidationAwarePrune { consolidation_threshold } => {
            // Only prune weights where consolidation is below threshold
            let d = assoc_net.m_va.d;
            let m = assoc_net.m_va.m.as_slice_mut().unwrap();
            let c = assoc_net.m_va.consolidation.as_slice().unwrap();
            for i in 0..(d * d) {
                if m[i].abs() <= base_threshold && c[i] < *consolidation_threshold {
                    m[i] = 0.0;
                }
            }
        }
        MutationStrategy::PercentilePrune { percentile } => {
            let mut abs_vals: Vec<f32> = assoc_net.m_va.m.iter().map(|x| x.abs()).collect();
            abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((percentile / 100.0) * abs_vals.len() as f32) as usize;
            let threshold = abs_vals.get(idx.min(abs_vals.len() - 1)).copied().unwrap_or(0.0);
            assoc_net.m_va.m.mapv_inplace(|v| if v.abs() <= threshold { 0.0 } else { v });
        }
        _ => {
            assoc_net.prune(base_threshold);
        }
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

#[inline]
fn normalize_rows_inplace(arr: &mut Array2<f32>) {
    l2_normalize_inplace(arr);
}

/// Compute hard negatives: for each clip i, find the top-k most similar wrong clips.
/// Returns Vec<Vec<usize>> where result[i] = indices of hardest negatives for clip i.
fn compute_hard_negatives(
    cached_v: &Array2<f32>,
    cached_a: &Array2<f32>,
    m: &Array2<f32>,
    top_k: usize,
) -> Vec<Vec<usize>> {
    let n = cached_v.nrows();
    let k = top_k.min(n - 1);

    // Compute similarity: S = V @ M @ A^T  [N, N]
    // Do this in chunks to avoid N*N memory for large N
    let m_a_t = m.dot(&cached_a.t()); // [D, N]

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let v_i = cached_v.row(i);
        // sim[j] = v_i^T @ M @ a_j for all j
        let sims: Vec<f32> = (0..n)
            .map(|j| v_i.dot(&m_a_t.column(j)))
            .collect();

        // Find top-k most similar (excluding self)
        let mut indexed: Vec<(usize, f32)> = sims
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, &s)| (j, s))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let hard: Vec<usize> = indexed.iter().take(k).map(|(j, _)| *j).collect();
        result.push(hard);
    }
    result
}
