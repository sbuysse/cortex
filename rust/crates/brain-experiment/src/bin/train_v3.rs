//! V3 training: hard negative mining + warm-start from v2 model.
//!
//! Key improvements over v2:
//! - Warm-start from saved M matrix (fine-tune, don't retrain)
//! - Hard negative mining: bias batches toward confusing pairs
//! - Full-pool eval (24K clips) for real metrics
//! - Cosine restarts for better optima
//!
//! Usage:
//!   OPENBLAS_NUM_THREADS=8 train-v3

use brain_core::hebbian::HebbianAssociation;
use brain_core::metrics::CrossModalRetrieval;
use brain_experiment::embed_cache::WhiteningTransforms;

use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::path::Path;

fn main() {
    let whitening_path = std::env::var("WHITENING_PATH")
        .unwrap_or_else(|_| "/opt/brain/data/vggsound/.embed_cache/whitening.safetensors".to_string());
    let model_dir = std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/v2_whitened_rect".to_string());
    let output_dir = std::env::var("OUTPUT_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/v3_hard_neg".to_string());
    let max_steps: usize = std::env::var("MAX_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let batch_size: usize = std::env::var("BATCH_SIZE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let hard_neg_k: usize = std::env::var("HARD_NEG_K")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let hard_neg_refresh: usize = std::env::var("HARD_NEG_REFRESH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let hard_neg_frac: f32 = std::env::var("HARD_NEG_FRAC")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);

    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    eprintln!("=== Train V3 — Hard Negative Mining ===");
    eprintln!("Loading whitened embeddings...");
    let w = WhiteningTransforms::load(Path::new(&whitening_path)).expect("Failed to load whitening");
    let v_emb = &w.v_white;
    let a_emb = &w.a_white;
    let n_clips = v_emb.nrows();
    let dv = v_emb.ncols();
    let da = a_emb.ncols();
    eprintln!("Clips: {}, v={}, a={}", n_clips, dv, da);

    // Load pre-trained M from v2
    eprintln!("Loading v2 model from {}...", model_dir);
    let m_init = load_matrix(&Path::new(&model_dir).join("m_va.bin"));
    eprintln!("M shape: {}×{}", m_init.nrows(), m_init.ncols());

    // Training params — lower LR for fine-tuning
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let max_norm: f32 = 22.165;
    let infonce_temp: f32 = std::env::var("TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.006394);
    let temp_start: f32 = std::env::var("TEMP_START").ok().and_then(|s| s.parse().ok()).unwrap_or(infonce_temp);
    let temp_end: f32 = std::env::var("TEMP_END").ok().and_then(|s| s.parse().ok()).unwrap_or(infonce_temp);
    let use_temp_anneal = (temp_start - temp_end).abs() > 1e-8;

    eprintln!("Config: lr={}, temp={}, batch={}, max_steps={}", lr, infonce_temp, batch_size, max_steps);
    if use_temp_anneal {
        eprintln!("Temperature annealing: {} → {}", temp_start, temp_end);
    }
    eprintln!("Hard negatives: K={}, refresh_every={}, frac={}", hard_neg_k, hard_neg_refresh, hard_neg_frac);

    let start = std::time::Instant::now();
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Initialize M from v2 checkpoint
    let mut m = HebbianAssociation::new_rect(dv, da, lr, 0.992, max_norm);
    m.m.assign(&m_init);

    // Pre-allocate buffers
    let mut v_buf = Array2::zeros((batch_size, dv));
    let mut a_buf = Array2::zeros((batch_size, da));
    let mut sim_matrix = Array2::zeros((batch_size, batch_size));
    let mut softmax_probs = Array2::zeros((batch_size, batch_size));
    let mut weighted_neg = Array2::zeros((batch_size, da));
    let mut proj_a_buf = Array2::zeros((batch_size, dv));
    let mut sym_temp_buf = Array2::zeros((batch_size, da));

    // Cosine LR schedule with restarts
    let warmup_steps = 50usize;
    let min_lr = 1e-6f32;
    let restart_period: usize = std::env::var("RESTART_PERIOD")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(max_steps); // single period by default

    // Eval: 1K for speed + full pool periodically
    let retrieval_1k = CrossModalRetrieval::new(1000, vec![1, 5, 10]);
    let retrieval_full = CrossModalRetrieval::new(n_clips, vec![1, 5, 10]);
    let eval_interval = 50usize.max(max_steps / 20);

    // Hard negative index: for each clip i, top-K wrong clips by similarity
    let n_hard = (batch_size as f32 * hard_neg_frac) as usize;
    let n_random = batch_size - n_hard;
    let mut hard_negatives: Vec<Vec<usize>> = Vec::new(); // [n_clips][K]

    let indices: Vec<usize> = (0..n_clips).collect();
    let mut step = 0usize;
    let mut best_mrr_1k = 0.0f64;
    let mut best_mrr_full = 0.0f64;
    let mut best_m: Option<Array2<f32>> = None;

    while step < max_steps {
        // Refresh hard negatives
        if step % hard_neg_refresh == 0 {
            eprintln!("  [step {}] Refreshing hard negatives (K={})...", step, hard_neg_k);
            hard_negatives = compute_hard_negatives(v_emb, a_emb, &m.m, hard_neg_k);
            eprintln!("  [step {}] Hard negatives computed", step);
        }

        // Build batch: n_random random clips + for each, include some of its hard negatives
        let mut batch_indices: Vec<usize> = Vec::with_capacity(batch_size);

        // Start with random anchor clips
        let mut anchors: Vec<usize> = indices.clone();
        anchors.shuffle(&mut rng);
        anchors.truncate(n_random);
        batch_indices.extend_from_slice(&anchors);

        // Add hard negatives of random anchors
        if !hard_negatives.is_empty() && n_hard > 0 {
            let per_anchor = (n_hard / n_random.max(1)).max(1);
            for &anchor in &anchors {
                if batch_indices.len() >= batch_size { break; }
                let negs = &hard_negatives[anchor];
                for &neg in negs.iter().take(per_anchor) {
                    if batch_indices.len() >= batch_size { break; }
                    batch_indices.push(neg);
                }
            }
        }

        // Fill remaining with random if needed
        while batch_indices.len() < batch_size {
            batch_indices.push(rng.random_range(0..n_clips));
        }
        batch_indices.truncate(batch_size);

        // Cosine LR with restarts
        let cycle_step = step % restart_period;
        let current_lr = if cycle_step < warmup_steps {
            min_lr + (lr - min_lr) * (cycle_step as f32 / warmup_steps as f32)
        } else {
            let progress = (cycle_step - warmup_steps) as f32 / (restart_period - warmup_steps) as f32;
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        };

        let chunk = &batch_indices[..];

        // Gather batch
        gather_rows_into(v_emb, chunk, &mut v_buf);
        gather_rows_into(a_emb, chunk, &mut a_buf);
        l2_normalize_inplace(&mut v_buf);
        l2_normalize_inplace(&mut a_buf);

        // Temperature (possibly annealed)
        let current_temp = if use_temp_anneal {
            let progress = step as f32 / max_steps as f32;
            temp_start + (temp_end - temp_start) * progress
        } else {
            infonce_temp
        };

        // V→A: sim = V @ M @ A^T / temp
        {
            let m_at = m.m.dot(&a_buf.t());
            ndarray::linalg::general_mat_mul(1.0 / current_temp, &v_buf, &m_at, 0.0, &mut sim_matrix);
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

        // target - P
        for i in 0..batch_size {
            for j in 0..batch_size {
                let target = if i == j { 1.0 } else { 0.0 };
                softmax_probs[[i, j]] = target - softmax_probs[[i, j]];
            }
        }

        // Gradient: M += scale * V^T @ (I - P) @ A
        let grad_scale = current_lr / (batch_size as f32 * current_temp);
        ndarray::linalg::general_mat_mul(1.0, &softmax_probs, &a_buf, 0.0, &mut weighted_neg);
        ndarray::linalg::general_mat_mul(grad_scale, &v_buf.t(), &weighted_neg, 1.0, &mut m.m);

        // Symmetric: A→V
        {
            let m_t = m.m.t();
            ndarray::linalg::general_mat_mul(1.0, &a_buf, &m_t, 0.0, &mut proj_a_buf);
            ndarray::linalg::general_mat_mul(1.0 / current_temp, &proj_a_buf, &v_buf.t(), 0.0, &mut sim_matrix);

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
            let metrics_1k = evaluate(&retrieval_1k, v_emb, a_emb, &m.m);
            let mrr_1k = metrics_1k.get("v2a_MRR").copied().unwrap_or(0.0);
            let r1_1k = metrics_1k.get("v2a_R@1").copied().unwrap_or(0.0);

            // Full pool eval every 5 intervals
            let do_full = step % (eval_interval * 5) == 0 || step + eval_interval >= max_steps;
            let (mrr_full, r1_full, r10_full) = if do_full {
                let mf = evaluate(&retrieval_full, v_emb, a_emb, &m.m);
                let mrr = mf.get("v2a_MRR").copied().unwrap_or(0.0);
                let r1 = mf.get("v2a_R@1").copied().unwrap_or(0.0);
                let r10 = mf.get("v2a_R@10").copied().unwrap_or(0.0);
                if mrr > best_mrr_full {
                    best_mrr_full = mrr;
                    best_m = Some(m.m.clone());
                }
                (mrr, r1, r10)
            } else {
                if mrr_1k > best_mrr_1k {
                    best_mrr_1k = mrr_1k;
                    // Only save best_m if we haven't done a full eval yet
                    if best_m.is_none() {
                        best_m = Some(m.m.clone());
                    }
                }
                (-1.0, -1.0, -1.0)
            };

            if do_full {
                eprintln!("  step {}/{}: 1K MRR={:.4} R@1={:.4} | FULL MRR={:.4} R@1={:.4} R@10={:.4} (best_full={:.4})",
                    step, max_steps, mrr_1k, r1_1k, mrr_full, r1_full, r10_full, best_mrr_full);
            } else {
                eprintln!("  step {}/{}: 1K MRR={:.4} R@1={:.4} (best_1k={:.4})",
                    step, max_steps, mrr_1k, r1_1k, best_mrr_1k);
            }
        }

        step += 1;
    }

    // Restore best checkpoint
    if let Some(best) = best_m {
        eprintln!("Restoring best checkpoint (full MRR={:.4})", best_mrr_full);
        m.m = best;
    }

    // Final full eval
    let final_metrics = evaluate(&retrieval_full, v_emb, a_emb, &m.m);
    let duration = start.elapsed().as_secs_f64();

    eprintln!("\n=== Final Results (full pool = {}) ===", n_clips);
    let mut keys: Vec<_> = final_metrics.keys().collect();
    keys.sort();
    for k in &keys {
        eprintln!("  {}: {:.6}", k, final_metrics[*k]);
    }
    eprintln!("  duration_seconds: {:.1}", duration);

    // Also eval on 1K for comparison
    let metrics_1k = evaluate(&retrieval_1k, v_emb, a_emb, &m.m);
    eprintln!("\n=== 1K pool comparison ===");
    for k in &keys {
        if let Some(v) = metrics_1k.get(*k) {
            eprintln!("  {}: {:.6}", k, v);
        }
    }

    // Save
    let out = Path::new(&output_dir);
    save_matrix(&m.m, &out.join("m_va.bin"));
    eprintln!("\nSaved M to {}", output_dir);
}

/// Compute top-K hard negatives for each clip using current M.
fn compute_hard_negatives(
    v_emb: &Array2<f32>,
    a_emb: &Array2<f32>,
    m: &Array2<f32>,
    top_k: usize,
) -> Vec<Vec<usize>> {
    let n = v_emb.nrows();

    // L2 normalize
    let mut v_norm = v_emb.to_owned();
    let mut a_norm = a_emb.to_owned();
    l2_normalize_inplace(&mut v_norm);
    l2_normalize_inplace(&mut a_norm);

    // Compute projected_v = V @ M  [N, da]
    let projected_v = v_norm.dot(m);

    // For memory efficiency, compute similarities in blocks
    let block_size = 1000;
    let mut result = Vec::with_capacity(n);

    for i_start in (0..n).step_by(block_size) {
        let i_end = (i_start + block_size).min(n);
        let v_block = projected_v.slice(ndarray::s![i_start..i_end, ..]);
        // sim_block[i, j] = projected_v[i] . a_norm[j]
        let sim_block = v_block.dot(&a_norm.t()); // [block, N]

        for i_local in 0..(i_end - i_start) {
            let i_global = i_start + i_local;
            let row = sim_block.row(i_local);

            // Find top-K indices excluding the correct match (i_global)
            let mut indexed: Vec<(usize, f32)> = row.iter()
                .enumerate()
                .filter(|&(j, _)| j != i_global)
                .map(|(j, &s)| (j, s))
                .collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            result.push(indexed.iter().take(top_k).map(|(j, _)| *j).collect());
        }
    }

    result
}

fn evaluate(
    retrieval: &CrossModalRetrieval,
    v_emb: &Array2<f32>,
    a_emb: &Array2<f32>,
    m: &Array2<f32>,
) -> HashMap<String, f64> {
    let n = v_emb.nrows().min(retrieval.pool_size);
    let v = v_emb.slice(ndarray::s![..n, ..]).to_owned();
    let a = a_emb.slice(ndarray::s![..n, ..]).to_owned();
    let mut v_norm = v;
    let mut a_norm = a;
    l2_normalize_inplace(&mut v_norm);
    l2_normalize_inplace(&mut a_norm);

    let pv = v_norm.dot(m);
    let sim_v2a = pv.dot(&a_norm.t());
    let pa = a_norm.dot(&m.t());
    let sim_a2v = pa.dot(&v_norm.t());
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

fn load_matrix(path: &Path) -> Array2<f32> {
    let data = std::fs::read(path).expect("Failed to read matrix file");
    let newline = data.iter().position(|&b| b == b'\n').expect("No header newline");
    let header = std::str::from_utf8(&data[..newline]).expect("Invalid header");
    let parts: Vec<usize> = header.split('x').map(|s| s.parse().expect("Bad dim")).collect();
    let (rows, cols) = (parts[0], parts[1]);
    let float_data = &data[newline + 1..];
    let floats: Vec<f32> = float_data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(floats.len(), rows * cols);
    Array2::from_shape_vec((rows, cols), floats).expect("Shape error")
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
