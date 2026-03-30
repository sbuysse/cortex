//! V2 training pipeline — implements 5 ideas to push MRR from 0.55 to 0.7+:
//!
//! A. Direct non-square M (384×512) — skip SparseProjection, use raw embeddings
//! B. ZCA whitening — decorrelate embedding dimensions before training
//! C. Multi-positive InfoNCE — clips with same VGGSound category are soft positives
//! D. Ensemble of N random projections — average similarity matrices at eval
//! E. Quadratic scoring — v^T M₁ a + (v⊙v)^T M₂ (a⊙a)
//!
//! Usage:
//!   MODE=whitened_rect train-v2          # A+B: whitened raw embeddings, 384×512 M
//!   MODE=multi_positive train-v2         # A+B+C: + multi-positive InfoNCE
//!   MODE=quadratic train-v2              # A+B+C+E: + quadratic scoring
//!   MODE=ensemble train-v2               # D: ensemble of 5 projected M matrices
//!   MODE=all train-v2                    # A+B+C+E combined

use brain_core::hebbian::HebbianAssociation;
use brain_core::metrics::CrossModalRetrieval;
use brain_experiment::embed_cache::{CachedEmbeddings, CategoryLabels, WhiteningTransforms};

use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::path::Path;

fn main() {
    tracing_subscriber::fmt::init();

    let mode = std::env::var("MODE").unwrap_or_else(|_| "all".to_string());
    let embed_path = std::env::var("EMBED_CACHE")
        .unwrap_or_else(|_| "/opt/brain/data/vggsound/.embed_cache/expanded_embeddings.safetensors".to_string());
    let whitening_path = std::env::var("WHITENING_PATH")
        .unwrap_or_else(|_| "/opt/brain/data/vggsound/.embed_cache/whitening.safetensors".to_string());
    let labels_path = std::env::var("LABELS_PATH")
        .unwrap_or_else(|_| "/opt/brain/data/vggsound/.embed_cache/labels.safetensors".to_string());
    let output_dir = std::env::var("OUTPUT_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/best_model_v2".to_string());
    let max_steps: usize = std::env::var("MAX_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let batch_size: usize = std::env::var("BATCH_SIZE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);

    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");
    let out = Path::new(&output_dir);

    eprintln!("=== Train V2 — Mode: {mode} ===");
    eprintln!("Loading embeddings...");
    let cache = CachedEmbeddings::load(Path::new(&embed_path)).expect("Failed to load embeddings");
    let n_clips = cache.n_clips;
    eprintln!("Loaded {} clips (v={}, a={})", n_clips, cache.v_emb.ncols(), cache.a_emb.ncols());

    // Load whitening transforms
    let whitening = if mode != "ensemble" && Path::new(&whitening_path).exists() {
        eprintln!("Loading ZCA whitening...");
        Some(WhiteningTransforms::load(Path::new(&whitening_path)).expect("Failed to load whitening"))
    } else {
        None
    };

    // Load labels for multi-positive
    let labels = if mode == "multi_positive" || mode == "all" {
        if Path::new(&labels_path).exists() {
            eprintln!("Loading category labels...");
            Some(CategoryLabels::load(Path::new(&labels_path)).expect("Failed to load labels"))
        } else {
            eprintln!("WARNING: No labels file found, multi-positive disabled");
            None
        }
    } else {
        None
    };

    let use_quadratic = mode == "quadratic" || mode == "all";

    match mode.as_str() {
        "ensemble" => {
            run_ensemble(&cache, max_steps, batch_size, labels.as_ref(), out);
        }
        _ => {
            // A+B: Use whitened raw embeddings with rectangular M
            let (v_train, a_train) = if let Some(ref w) = whitening {
                eprintln!("Using ZCA-whitened embeddings (v={}, a={})", w.v_white.ncols(), w.a_white.ncols());
                (w.v_white.clone(), w.a_white.clone())
            } else {
                eprintln!("Using raw embeddings (no whitening)");
                (cache.v_emb.clone(), cache.a_emb.clone())
            };

            let dv = v_train.ncols();
            let da = a_train.ncols();
            eprintln!("Training rectangular M: {}×{}", dv, da);

            let result = train_direct(
                &v_train, &a_train,
                labels.as_ref(),
                use_quadratic,
                max_steps, batch_size,
                dv, da,
            );

            // Save M matrix
            save_matrix(&result.m.m, &out.join("m_va.bin"));
            eprintln!("Saved M ({}×{}) to m_va.bin", dv, da);

            // Save quadratic M2 if present
            if let Some(ref m2) = result.m2 {
                save_matrix(&m2.m, &out.join("m2_va.bin"));
                eprintln!("Saved M2 ({}×{}) to m2_va.bin", dv, da);
            }

            // Save the whitened/raw embeddings for brain voice
            save_matrix(&v_train, &out.join("v_proj.bin"));
            save_matrix(&a_train, &out.join("a_proj.bin"));
            eprintln!("Saved v_proj.bin and a_proj.bin");

            // Print final metrics
            eprintln!("\n=== Final Results ===");
            for (k, v) in &result.metrics {
                if k.contains("MRR") || k.contains("R@") || k.contains("MR") || k == "duration_seconds" {
                    eprintln!("  {}: {:.6}", k, v);
                }
            }
        }
    }
}

struct TrainResult {
    m: HebbianAssociation,
    m2: Option<HebbianAssociation>,
    metrics: HashMap<String, f64>,
    best_mrr: f64,
}

/// Train with direct rectangular M (ideas A+B+C+E).
fn train_direct(
    v_emb: &Array2<f32>,
    a_emb: &Array2<f32>,
    labels: Option<&CategoryLabels>,
    use_quadratic: bool,
    max_steps: usize,
    batch_size: usize,
    dv: usize,
    da: usize,
) -> TrainResult {
    let start = std::time::Instant::now();
    let n_clips = v_emb.nrows();

    // Best config params from exp 20013
    let lr: f32 = 0.06742;
    let decay_rate: f32 = 0.992;
    let max_norm: f32 = 22.165;
    let infonce_temp: f32 = 0.006394;
    let seed: u64 = 42;

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Create rectangular M (dv × da)
    let mut m = HebbianAssociation::new_rect(dv, da, lr, decay_rate, max_norm);

    // Initialize M with small noise + scaled identity-like init
    let init_scale: f32 = 0.01;
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

    // Quadratic M2 for (v⊙v)^T M₂ (a⊙a) — idea E
    let mut m2 = if use_quadratic {
        let mut m2 = HebbianAssociation::new_rect(dv, da, lr * 0.5, decay_rate, max_norm);
        // Initialize M2 small
        m2.m.mapv_inplace(|_| (rng.random::<f32>() - 0.5) * init_scale * 0.05);
        Some(m2)
    } else {
        None
    };

    // Precompute label-based positive mask for multi-positive InfoNCE (idea C)
    // We compute this per-batch at runtime since batch indices vary

    // Pre-allocate buffers
    let mut v_buf = Array2::zeros((batch_size, dv));
    let mut a_buf = Array2::zeros((batch_size, da));
    let mut sim_matrix = Array2::zeros((batch_size, batch_size));
    let mut softmax_probs = Array2::zeros((batch_size, batch_size));
    let mut weighted_neg = Array2::zeros((batch_size, da));

    // Quadratic buffers
    let mut v_sq_buf = if use_quadratic { Array2::zeros((batch_size, dv)) } else { Array2::zeros((0, 0)) };
    let mut a_sq_buf = if use_quadratic { Array2::zeros((batch_size, da)) } else { Array2::zeros((0, 0)) };
    let mut sim2_matrix = if use_quadratic { Array2::zeros((batch_size, batch_size)) } else { Array2::zeros((0, 0)) };
    let mut weighted2 = if use_quadratic { Array2::zeros((batch_size, da)) } else { Array2::zeros((0, 0)) };

    // Pre-allocate symmetric direction buffers (avoid alloc inside loop)
    let mut proj_a_buf = Array2::zeros((batch_size, dv));
    let mut sym_temp_buf = Array2::zeros((batch_size, da));

    // Scheduler: cosine decay
    let warmup_steps = 100usize;
    let min_lr = 1e-5f32;

    let retrieval = CrossModalRetrieval::new(n_clips.min(1000), vec![1, 5, 10]);
    let eval_interval = 50usize.max(max_steps / 10);

    let indices: Vec<usize> = (0..n_clips).collect();
    let mut step = 0usize;
    let mut best_mrr = 0.0f64;
    let mut best_m: Option<Array2<f32>> = None;
    let mut best_m2: Option<Array2<f32>> = None;
    let mut last_metrics: HashMap<String, f64> = HashMap::new();

    while step < max_steps {
        let mut batch_indices: Vec<usize> = indices.clone();
        batch_indices.shuffle(&mut rng);

        for chunk in batch_indices.chunks(batch_size) {
            if step >= max_steps { break; }
            if chunk.len() < batch_size { continue; }

            // Cosine LR schedule
            let current_lr = if step < warmup_steps {
                min_lr + (lr - min_lr) * (step as f32 / warmup_steps as f32)
            } else {
                let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
                min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
            };

            // Gather batch
            gather_rows_into(v_emb, chunk, &mut v_buf);
            gather_rows_into(a_emb, chunk, &mut a_buf);

            // L2 normalize
            l2_normalize_inplace(&mut v_buf);
            l2_normalize_inplace(&mut a_buf);

            // Compute similarity: sim[i,j] = v_i^T M a_j / temp
            {
                // projected_v = V @ M  [B, da]
                // sim = projected_v @ A^T / temp  [B, B]
                let m_at = m.m.dot(&a_buf.t()); // [dv, B]
                ndarray::linalg::general_mat_mul(1.0 / infonce_temp, &v_buf, &m_at, 0.0, &mut sim_matrix);
            }

            // Add quadratic term if enabled (idea E)
            if use_quadratic {
                if let Some(ref m2_mat) = m2 {
                    // v_sq = v ⊙ v
                    v_sq_buf.assign(&v_buf);
                    v_sq_buf.mapv_inplace(|x| x * x);
                    // a_sq = a ⊙ a
                    a_sq_buf.assign(&a_buf);
                    a_sq_buf.mapv_inplace(|x| x * x);
                    // sim2[i,j] = v_sq_i^T M₂ a_sq_j / temp
                    let m2_at = m2_mat.m.dot(&a_sq_buf.t());
                    ndarray::linalg::general_mat_mul(1.0 / infonce_temp, &v_sq_buf, &m2_at, 0.0, &mut sim2_matrix);
                    // sim_matrix += sim2_matrix
                    sim_matrix += &sim2_matrix;
                }
            }

            // Softmax per row → P[i,j]
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

            // Build target: I (standard) or label-based soft positives (idea C)
            // gradient = target - P  (positive direction: increase probability of positives)
            if let Some(ref lab) = labels {
                // Multi-positive: target[i,j] = 1/|P_i| if label[chunk[i]] == label[chunk[j]]
                for i in 0..batch_size {
                    let li = lab.labels[chunk[i]];
                    let mut n_pos = 0usize;
                    for j in 0..batch_size {
                        if lab.labels[chunk[j]] == li {
                            n_pos += 1;
                        }
                    }
                    let target_val = 1.0 / n_pos.max(1) as f32;
                    for j in 0..batch_size {
                        let target = if lab.labels[chunk[j]] == li { target_val } else { 0.0 };
                        softmax_probs[[i, j]] = target - softmax_probs[[i, j]];
                    }
                }
            } else {
                // Standard: target = I
                for i in 0..batch_size {
                    for j in 0..batch_size {
                        let target = if i == j { 1.0 } else { 0.0 };
                        softmax_probs[[i, j]] = target - softmax_probs[[i, j]];
                    }
                }
            }

            // Gradient for M: dM = V^T @ (target - P) @ A * scale
            let grad_scale = current_lr / (batch_size as f32 * infonce_temp);

            // weighted_target = (target - P) @ A  [B, da]
            ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);
            // M += scale * V^T @ weighted_target  [dv, da]
            ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &weighted_neg, 1.0, &mut m.m);

            // Symmetric: A→V direction
            {
                let m_t = m.m.t();
                // projected_a = A @ M^T  [B, dv]
                ndarray::linalg::general_mat_mul(1.0, &a_buf, &m_t, 0.0, &mut proj_a_buf);
                // sim_a2v = proj_a @ V^T / temp  [B, B]
                ndarray::linalg::general_mat_mul(1.0 / infonce_temp, &proj_a_buf, &v_buf.t(), 0.0, &mut sim_matrix);

                // Softmax
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

                // Build R = target - P
                if let Some(ref lab) = labels {
                    for i in 0..batch_size {
                        let li = lab.labels[chunk[i]];
                        let mut n_pos = 0usize;
                        for j in 0..batch_size {
                            if lab.labels[chunk[j]] == li { n_pos += 1; }
                        }
                        let target_val = 1.0 / n_pos.max(1) as f32;
                        for j in 0..batch_size {
                            let target = if lab.labels[chunk[j]] == li { target_val } else { 0.0 };
                            softmax_probs[[i, j]] = target - softmax_probs[[i, j]];
                        }
                    }
                } else {
                    for i in 0..batch_size {
                        for j in 0..batch_size {
                            let target = if i == j { 1.0 } else { 0.0 };
                            softmax_probs[[i, j]] = target - softmax_probs[[i, j]];
                        }
                    }
                }

                // dM += scale * V^T @ R^T @ A
                ndarray::linalg::general_mat_mul(1.0, &softmax_probs.t(), &a_buf, 0.0, &mut sym_temp_buf);
                ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &sym_temp_buf, 1.0, &mut m.m);
            }

            // Quadratic gradient for M2 (idea E)
            if use_quadratic {
                if let Some(ref mut m2_mat) = m2 {
                    // Recompute gradient weights from total sim (stored in softmax_probs already as target-P)
                    // But we need separate gradient for M2 through v_sq, a_sq path
                    // dM2 = scale * v_sq^T @ (target - P) @ a_sq
                    ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_sq_buf, 0.0, &mut weighted2);
                    ndarray::linalg::general_mat_mul(grad_scale * 0.5, &v_sq_buf.t(), &weighted2, 1.0, &mut m2_mat.m);
                }
            }

            // Spectral norm clipping every 5 steps
            if step % 5 == 0 {
                if m.max_norm > 0.0 {
                    let spectral = m.power_iteration_spectral_norm(2);
                    if spectral > m.max_norm {
                        let clip = m.max_norm / spectral;
                        m.m.mapv_inplace(|v| v * clip);
                    }
                }
                if let Some(ref mut m2_mat) = m2 {
                    if m2_mat.max_norm > 0.0 {
                        let spectral = m2_mat.power_iteration_spectral_norm(2);
                        if spectral > m2_mat.max_norm {
                            let clip = m2_mat.max_norm / spectral;
                            m2_mat.m.mapv_inplace(|v| v * clip);
                        }
                    }
                }
            }

            m.update_count += 1;

            // Eval
            if step > 0 && step % eval_interval == 0 {
                let metrics = evaluate(&retrieval, v_emb, a_emb, &m.m, m2.as_ref().map(|m| &m.m));
                let mrr = metrics.get("v2a_MRR").copied().unwrap_or(0.0);
                let r1 = metrics.get("v2a_R@1").copied().unwrap_or(0.0);
                if mrr > best_mrr {
                    best_mrr = mrr;
                    best_m = Some(m.m.clone());
                    best_m2 = m2.as_ref().map(|m2| m2.m.clone());
                }
                eprintln!("  step {step}/{max_steps}: v2a_MRR={mrr:.6} v2a_R@1={r1:.4} (best={best_mrr:.6})");
                last_metrics = metrics;
            }

            step += 1;
        }
    }

    // Restore best checkpoint
    if let Some(best) = best_m {
        eprintln!("Restoring best checkpoint (MRR={best_mrr:.6})");
        m.m = best;
        if let Some(best2) = best_m2 {
            if let Some(ref mut m2_mat) = m2 {
                m2_mat.m = best2;
            }
        }
    }

    // Final eval on best checkpoint
    let metrics = evaluate(&retrieval, v_emb, a_emb, &m.m, m2.as_ref().map(|m| &m.m));
    let duration = start.elapsed().as_secs_f64();
    let mut final_metrics = metrics;
    final_metrics.insert("duration_seconds".to_string(), duration);

    TrainResult { m, m2, metrics: final_metrics, best_mrr }
}

/// Ensemble of N projected M matrices (idea D).
fn run_ensemble(
    cache: &CachedEmbeddings,
    max_steps: usize,
    batch_size: usize,
    labels: Option<&CategoryLabels>,
    out: &Path,
) {
    let n_ensemble = 5;
    let embed_dim = 512;
    let sparsity_k = 36;

    eprintln!("=== Ensemble of {} projections ===", n_ensemble);
    let mut all_m: Vec<Array2<f32>> = Vec::new();
    let mut all_v_proj: Vec<Array2<f32>> = Vec::new();
    let mut all_a_proj: Vec<Array2<f32>> = Vec::new();

    for i in 0..n_ensemble {
        let v_seed = 100 + i as u64 * 1000;
        let a_seed = 200 + i as u64 * 1000;
        eprintln!("\n--- Ensemble member {}/{} (v_seed={}, a_seed={}) ---", i+1, n_ensemble, v_seed, a_seed);

        let proj_v = brain_core::sparse_projection::SparseProjection::new_seeded(
            cache.v_emb.ncols(), embed_dim, sparsity_k, v_seed);
        let proj_a = brain_core::sparse_projection::SparseProjection::new_seeded(
            cache.a_emb.ncols(), embed_dim, sparsity_k, a_seed);

        let v_proj = proj_v.forward(&cache.v_emb);
        let a_proj = proj_a.forward(&cache.a_emb);

        let result = train_direct(
            &v_proj, &a_proj,
            labels,
            false, // no quadratic for ensemble
            max_steps, batch_size,
            embed_dim, embed_dim,
        );

        all_m.push(result.m.m);
        all_v_proj.push(v_proj);
        all_a_proj.push(a_proj);
    }

    // Evaluate ensemble: average similarity matrices
    let pool_size = cache.n_clips.min(1000);
    let retrieval = CrossModalRetrieval::new(pool_size, vec![1, 5, 10]);
    let n = cache.n_clips.min(pool_size);

    let mut avg_sim_v2a = Array2::<f32>::zeros((n, n));
    let mut avg_sim_a2v = Array2::<f32>::zeros((n, n));

    for i in 0..n_ensemble {
        let v = all_v_proj[i].slice(ndarray::s![..n, ..]).to_owned();
        let a = all_a_proj[i].slice(ndarray::s![..n, ..]).to_owned();
        let m = &all_m[i];

        let pv = v.dot(m);
        let sim_v2a = pv.dot(&a.t());
        let pa = a.dot(&m.t());
        let sim_a2v = pa.dot(&v.t());

        avg_sim_v2a += &sim_v2a;
        avg_sim_a2v += &sim_a2v;
    }
    avg_sim_v2a /= n_ensemble as f32;
    avg_sim_a2v /= n_ensemble as f32;

    let results = retrieval.evaluate_from_sims(&avg_sim_v2a, &avg_sim_a2v);
    eprintln!("\n=== Ensemble Results ({} members) ===", n_ensemble);
    for (k, v) in &results {
        if k.contains("MRR") || k.contains("R@") || k.contains("MR") {
            eprintln!("  {}: {:.6}", k, v);
        }
    }

    // Save best single M (member 0 uses same seeds as production)
    save_matrix(&all_m[0], &out.join("m_va.bin"));
    save_matrix(&all_v_proj[0], &out.join("v_proj.bin"));
    save_matrix(&all_a_proj[0], &out.join("a_proj.bin"));
    eprintln!("Saved member 0 matrices for production use");
}

/// Evaluate retrieval with optional quadratic scoring.
fn evaluate(
    retrieval: &CrossModalRetrieval,
    v_emb: &Array2<f32>,
    a_emb: &Array2<f32>,
    m: &Array2<f32>,
    m2: Option<&Array2<f32>>,
) -> HashMap<String, f64> {
    let n = v_emb.nrows().min(retrieval.pool_size);
    let v = v_emb.slice(ndarray::s![..n, ..]).to_owned();
    let a = a_emb.slice(ndarray::s![..n, ..]).to_owned();

    // L2 normalize for eval
    let mut v_norm = v.clone();
    let mut a_norm = a.clone();
    l2_normalize_inplace(&mut v_norm);
    l2_normalize_inplace(&mut a_norm);

    // Linear term: v @ M @ a^T
    let pv = v_norm.dot(m);
    let mut sim_v2a = pv.dot(&a_norm.t());
    let pa = a_norm.dot(&m.t());
    let mut sim_a2v = pa.dot(&v_norm.t());

    // Add quadratic term if M2 exists
    if let Some(m2) = m2 {
        let v_sq = v_norm.mapv(|x| x * x);
        let a_sq = a_norm.mapv(|x| x * x);
        let pv2 = v_sq.dot(m2);
        sim_v2a += &pv2.dot(&a_sq.t());
        let pa2 = a_sq.dot(&m2.t());
        sim_a2v += &pa2.dot(&v_sq.t());
    }

    retrieval.evaluate_from_sims(&sim_v2a, &sim_a2v)
}

fn save_matrix(m: &Array2<f32>, path: &Path) {
    let shape = m.shape();
    let header = format!("{}x{}\n", shape[0], shape[1]);
    let mut bytes = header.into_bytes();
    for val in m.as_slice().unwrap() {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    std::fs::write(path, &bytes).expect("Failed to write matrix");
}

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
