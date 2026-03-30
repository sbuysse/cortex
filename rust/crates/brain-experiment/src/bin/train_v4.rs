//! V4 training: MLP dual-encoder with hard negative mining.
//!
//! Replaces bilinear v^T M a with nonlinear projections:
//!   v_proj = ReLU(V @ W_v)    [B, d_hidden]
//!   a_proj = ReLU(A @ W_a)    [B, d_hidden]
//!   sim = v_proj @ a_proj^T / temp
//!
//! Manual gradient descent through ReLU + linear layers.
//! Warm-starts W_v, W_a from the v2/v3 M matrix where possible.

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
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/v3f_anneal".to_string());
    let output_dir = std::env::var("OUTPUT_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/v4_mlp".to_string());
    let max_steps: usize = std::env::var("MAX_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let batch_size: usize = std::env::var("BATCH_SIZE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let d_hidden: usize = std::env::var("D_HIDDEN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.005);
    let temp_start: f32 = std::env::var("TEMP_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let temp_end: f32 = std::env::var("TEMP_END").ok().and_then(|s| s.parse().ok()).unwrap_or(0.005);
    let hard_neg_k: usize = std::env::var("HARD_NEG_K")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let hard_neg_refresh: usize = std::env::var("HARD_NEG_REFRESH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let hard_neg_frac: f32 = std::env::var("HARD_NEG_FRAC")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let max_norm: f32 = std::env::var("MAX_NORM").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0);

    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    eprintln!("=== Train V4 — MLP Dual Encoder ===");
    eprintln!("Loading whitened embeddings...");
    let w = WhiteningTransforms::load(Path::new(&whitening_path)).expect("Failed to load whitening");
    let v_emb = &w.v_white;
    let a_emb = &w.a_white;
    let n_clips = v_emb.nrows();
    let dv = v_emb.ncols(); // 384
    let da = a_emb.ncols(); // 512
    eprintln!("Clips: {}, v={}, a={}, d_hidden={}", n_clips, dv, da, d_hidden);

    // Initialize W_v [dv, d_hidden] and W_a [da, d_hidden]
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Try to warm-start from saved M matrix
    let m_path = Path::new(&model_dir).join("m_va.bin");
    let mut w_v = Array2::<f32>::zeros((dv, d_hidden));
    let mut w_a = Array2::<f32>::zeros((da, d_hidden));

    if m_path.exists() {
        let m = load_matrix(&m_path);
        eprintln!("Warm-starting from M ({}×{})", m.nrows(), m.ncols());
        // M is [dv, da]. Use SVD-like initialization:
        // W_v = M[:, :d_hidden] (first d_hidden cols of M)
        // W_a = I[:da, :d_hidden] (identity-like)
        let cols_to_copy = d_hidden.min(da);
        for i in 0..dv {
            for j in 0..cols_to_copy {
                w_v[[i, j]] = m[[i, j]];
            }
        }
        // W_a: identity-like initialization
        let min_da_d = da.min(d_hidden);
        for i in 0..min_da_d {
            w_a[[i, i]] = 1.0;
        }
        // Add small noise
        w_v.mapv_inplace(|x| x + (rng.random::<f32>() - 0.5) * 0.001);
        w_a.mapv_inplace(|x| x + (rng.random::<f32>() - 0.5) * 0.001);
    } else {
        eprintln!("No saved M found, random initialization");
        let scale_v = (2.0 / dv as f32).sqrt();
        let scale_a = (2.0 / da as f32).sqrt();
        w_v.mapv_inplace(|_| (rng.random::<f32>() - 0.5) * scale_v);
        w_a.mapv_inplace(|_| (rng.random::<f32>() - 0.5) * scale_a);
    }

    eprintln!("Config: lr={}, temp={}→{}, batch={}, max_steps={}, max_norm={}",
        lr, temp_start, temp_end, batch_size, max_steps, max_norm);
    eprintln!("Hard negatives: K={}, refresh={}, frac={}", hard_neg_k, hard_neg_refresh, hard_neg_frac);

    let start = std::time::Instant::now();

    // Pre-allocate buffers
    let mut v_buf = Array2::zeros((batch_size, dv));
    let mut a_buf = Array2::zeros((batch_size, da));
    let mut v_pre = Array2::zeros((batch_size, d_hidden)); // pre-ReLU
    let mut a_pre = Array2::zeros((batch_size, d_hidden));
    let mut v_proj = Array2::zeros((batch_size, d_hidden)); // post-ReLU
    let mut a_proj = Array2::zeros((batch_size, d_hidden));
    let mut sim_matrix = Array2::zeros((batch_size, batch_size));
    let mut softmax_probs = Array2::zeros((batch_size, batch_size));
    let mut grad_v_proj = Array2::zeros((batch_size, d_hidden));
    let mut grad_a_proj = Array2::zeros((batch_size, d_hidden));

    // Symmetric direction buffers
    let mut sym_sim = Array2::zeros((batch_size, batch_size));
    let mut sym_softmax = Array2::zeros((batch_size, batch_size));

    let warmup_steps = 50usize;
    let min_lr = 1e-6f32;

    let retrieval_1k = CrossModalRetrieval::new(1000, vec![1, 5, 10]);
    let retrieval_full = CrossModalRetrieval::new(n_clips, vec![1, 5, 10]);
    let eval_interval = 50usize.max(max_steps / 20);

    let n_hard = (batch_size as f32 * hard_neg_frac) as usize;
    let n_random = batch_size - n_hard;
    let mut hard_negatives: Vec<Vec<usize>> = Vec::new();

    let indices: Vec<usize> = (0..n_clips).collect();
    let mut step = 0usize;
    let mut best_mrr_full = 0.0f64;
    let mut best_w_v: Option<Array2<f32>> = None;
    let mut best_w_a: Option<Array2<f32>> = None;

    while step < max_steps {
        // Refresh hard negatives using current MLP projections
        if step % hard_neg_refresh == 0 {
            eprintln!("  [step {}] Refreshing hard negatives...", step);
            hard_negatives = compute_hard_negatives_mlp(v_emb, a_emb, &w_v, &w_a, hard_neg_k);
            eprintln!("  [step {}] Done", step);
        }

        // Build batch with hard negatives
        let mut batch_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut anchors: Vec<usize> = indices.clone();
        anchors.shuffle(&mut rng);
        anchors.truncate(n_random);
        batch_indices.extend_from_slice(&anchors);

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
        while batch_indices.len() < batch_size {
            batch_indices.push(rng.random_range(0..n_clips));
        }
        batch_indices.truncate(batch_size);

        // Cosine LR
        let cycle_step = step;
        let current_lr = if cycle_step < warmup_steps {
            min_lr + (lr - min_lr) * (cycle_step as f32 / warmup_steps as f32)
        } else {
            let progress = (cycle_step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        };

        // Temperature annealing
        let current_temp = temp_start + (temp_end - temp_start) * (step as f32 / max_steps as f32);

        let chunk = &batch_indices[..];

        // Gather batch + L2 normalize
        gather_rows_into(v_emb, chunk, &mut v_buf);
        gather_rows_into(a_emb, chunk, &mut a_buf);
        l2_normalize_inplace(&mut v_buf);
        l2_normalize_inplace(&mut a_buf);

        // === Forward pass ===
        // v_pre = V @ W_v  [B, d_hidden]
        ndarray::linalg::general_mat_mul(1.0, &v_buf, &w_v, 0.0, &mut v_pre);
        // v_proj = ReLU(v_pre)
        v_proj.assign(&v_pre);
        v_proj.mapv_inplace(|x| x.max(0.0));

        // a_pre = A @ W_a  [B, d_hidden]
        ndarray::linalg::general_mat_mul(1.0, &a_buf, &w_a, 0.0, &mut a_pre);
        // a_proj = ReLU(a_pre)
        a_proj.assign(&a_pre);
        a_proj.mapv_inplace(|x| x.max(0.0));

        // sim = v_proj @ a_proj^T / temp  [B, B]
        ndarray::linalg::general_mat_mul(1.0 / current_temp, &v_proj, &a_proj.t(), 0.0, &mut sim_matrix);

        // === V→A InfoNCE loss + gradient ===
        // Softmax per row
        softmax_rows(&sim_matrix, &mut softmax_probs, batch_size);
        // target - P
        for i in 0..batch_size {
            for j in 0..batch_size {
                softmax_probs[[i, j]] = if i == j { 1.0 } else { 0.0 } - softmax_probs[[i, j]];
            }
        }

        let grad_scale = current_lr / (batch_size as f32 * current_temp);

        // dL/d_v_proj = grad_scale * (I-P) @ a_proj  [B, d_hidden]
        ndarray::linalg::general_mat_mul(grad_scale, &softmax_probs, &a_proj, 0.0, &mut grad_v_proj);
        // dL/d_a_proj = grad_scale * (I-P)^T @ v_proj  [B, d_hidden]
        ndarray::linalg::general_mat_mul(grad_scale, &softmax_probs.t(), &v_proj, 0.0, &mut grad_a_proj);

        // === A→V symmetric direction ===
        // sim_a2v = a_proj @ v_proj^T / temp
        ndarray::linalg::general_mat_mul(1.0 / current_temp, &a_proj, &v_proj.t(), 0.0, &mut sym_sim);
        softmax_rows(&sym_sim, &mut sym_softmax, batch_size);
        for i in 0..batch_size {
            for j in 0..batch_size {
                sym_softmax[[i, j]] = if i == j { 1.0 } else { 0.0 } - sym_softmax[[i, j]];
            }
        }
        // Accumulate symmetric gradients
        // dL/d_a_proj += grad_scale * (I-P_a2v) @ v_proj
        ndarray::linalg::general_mat_mul(grad_scale, &sym_softmax, &v_proj, 1.0, &mut grad_a_proj);
        // dL/d_v_proj += grad_scale * (I-P_a2v)^T @ a_proj
        ndarray::linalg::general_mat_mul(grad_scale, &sym_softmax.t(), &a_proj, 1.0, &mut grad_v_proj);

        // === Backprop through ReLU ===
        // dL/d_v_pre = dL/d_v_proj ⊙ (v_pre > 0)
        for i in 0..batch_size {
            for j in 0..d_hidden {
                if v_pre[[i, j]] <= 0.0 {
                    grad_v_proj[[i, j]] = 0.0;
                }
            }
        }
        // dL/d_a_pre = dL/d_a_proj ⊙ (a_pre > 0)
        for i in 0..batch_size {
            for j in 0..d_hidden {
                if a_pre[[i, j]] <= 0.0 {
                    grad_a_proj[[i, j]] = 0.0;
                }
            }
        }

        // === Update weights ===
        // W_v += V^T @ dL/d_v_pre  [dv, d_hidden]
        ndarray::linalg::general_mat_mul(1.0, &v_buf.t(), &grad_v_proj, 1.0, &mut w_v);
        // W_a += A^T @ dL/d_a_pre  [da, d_hidden]
        ndarray::linalg::general_mat_mul(1.0, &a_buf.t(), &grad_a_proj, 1.0, &mut w_a);

        // Spectral norm clipping on both weight matrices
        if step % 5 == 0 {
            clip_spectral_norm(&mut w_v, max_norm);
            clip_spectral_norm(&mut w_a, max_norm);
        }

        // Eval
        if step > 0 && step % eval_interval == 0 {
            let metrics_1k = evaluate_mlp(&retrieval_1k, v_emb, a_emb, &w_v, &w_a);
            let mrr_1k = metrics_1k.get("v2a_MRR").copied().unwrap_or(0.0);
            let r1_1k = metrics_1k.get("v2a_R@1").copied().unwrap_or(0.0);

            let do_full = step % (eval_interval * 5) == 0 || step + eval_interval >= max_steps;
            if do_full {
                let mf = evaluate_mlp(&retrieval_full, v_emb, a_emb, &w_v, &w_a);
                let mrr = mf.get("v2a_MRR").copied().unwrap_or(0.0);
                let r1 = mf.get("v2a_R@1").copied().unwrap_or(0.0);
                let r10 = mf.get("v2a_R@10").copied().unwrap_or(0.0);
                if mrr > best_mrr_full {
                    best_mrr_full = mrr;
                    best_w_v = Some(w_v.clone());
                    best_w_a = Some(w_a.clone());
                }
                eprintln!("  step {}/{}: 1K MRR={:.4} R@1={:.4} | FULL MRR={:.4} R@1={:.4} R@10={:.4} (best_full={:.4}) temp={:.4}",
                    step, max_steps, mrr_1k, r1_1k, mrr, r1, r10, best_mrr_full, current_temp);
            } else {
                eprintln!("  step {}/{}: 1K MRR={:.4} R@1={:.4} temp={:.4}",
                    step, max_steps, mrr_1k, r1_1k, current_temp);
            }
        }

        step += 1;
    }

    // Restore best
    if let (Some(bv), Some(ba)) = (best_w_v, best_w_a) {
        eprintln!("Restoring best checkpoint (full MRR={:.4})", best_mrr_full);
        w_v = bv;
        w_a = ba;
    }

    // Final eval
    let final_metrics = evaluate_mlp(&retrieval_full, v_emb, a_emb, &w_v, &w_a);
    let duration = start.elapsed().as_secs_f64();
    eprintln!("\n=== Final Results (full pool = {}) ===", n_clips);
    let mut keys: Vec<_> = final_metrics.keys().collect();
    keys.sort();
    for k in &keys {
        eprintln!("  {}: {:.6}", k, final_metrics[*k]);
    }
    eprintln!("  duration_seconds: {:.1}", duration);

    let metrics_1k = evaluate_mlp(&retrieval_1k, v_emb, a_emb, &w_v, &w_a);
    eprintln!("\n=== 1K pool comparison ===");
    for k in &keys {
        if let Some(v) = metrics_1k.get(*k) {
            eprintln!("  {}: {:.6}", k, v);
        }
    }

    // Save
    let out = Path::new(&output_dir);
    save_matrix(&w_v, &out.join("w_v.bin"));
    save_matrix(&w_a, &out.join("w_a.bin"));
    eprintln!("\nSaved W_v and W_a to {}", output_dir);
}

fn softmax_rows(sim: &Array2<f32>, out: &mut Array2<f32>, n: usize) {
    for i in 0..n {
        let max_val = (0..n).map(|j| sim[[i, j]]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..n {
            let v = (sim[[i, j]] - max_val).exp();
            out[[i, j]] = v;
            sum += v;
        }
        let inv = 1.0 / sum.max(1e-12);
        for j in 0..n {
            out[[i, j]] *= inv;
        }
    }
}

/// Power-iteration spectral norm clipping for a matrix.
fn clip_spectral_norm(w: &mut Array2<f32>, max_norm: f32) {
    let (rows, cols) = w.dim();
    // One step of power iteration
    let mut v = ndarray::Array1::<f32>::ones(cols);
    let norm = v.dot(&v).sqrt();
    v /= norm;

    let u = w.dot(&v);
    let sigma = u.dot(&u).sqrt();
    if sigma > max_norm && sigma > 1e-12 {
        let clip = max_norm / sigma;
        w.mapv_inplace(|x| x * clip);
    }
}

fn compute_hard_negatives_mlp(
    v_emb: &Array2<f32>,
    a_emb: &Array2<f32>,
    w_v: &Array2<f32>,
    w_a: &Array2<f32>,
    top_k: usize,
) -> Vec<Vec<usize>> {
    let n = v_emb.nrows();
    let mut v_norm = v_emb.to_owned();
    let mut a_norm = a_emb.to_owned();
    l2_normalize_inplace(&mut v_norm);
    l2_normalize_inplace(&mut a_norm);

    // Project through MLP
    let v_pre = v_norm.dot(w_v);
    let v_proj = v_pre.mapv(|x| x.max(0.0));
    let a_pre = a_norm.dot(w_a);
    let a_proj = a_pre.mapv(|x| x.max(0.0));

    let block_size = 1000;
    let mut result = Vec::with_capacity(n);

    for i_start in (0..n).step_by(block_size) {
        let i_end = (i_start + block_size).min(n);
        let v_block = v_proj.slice(ndarray::s![i_start..i_end, ..]);
        let sim_block = v_block.dot(&a_proj.t());

        for i_local in 0..(i_end - i_start) {
            let i_global = i_start + i_local;
            let row = sim_block.row(i_local);
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

fn evaluate_mlp(
    retrieval: &CrossModalRetrieval,
    v_emb: &Array2<f32>,
    a_emb: &Array2<f32>,
    w_v: &Array2<f32>,
    w_a: &Array2<f32>,
) -> HashMap<String, f64> {
    let n = v_emb.nrows().min(retrieval.pool_size);
    let v = v_emb.slice(ndarray::s![..n, ..]).to_owned();
    let a = a_emb.slice(ndarray::s![..n, ..]).to_owned();
    let mut v_norm = v;
    let mut a_norm = a;
    l2_normalize_inplace(&mut v_norm);
    l2_normalize_inplace(&mut a_norm);

    let v_proj = v_norm.dot(w_v).mapv(|x| x.max(0.0));
    let a_proj = a_norm.dot(w_a).mapv(|x| x.max(0.0));

    let sim_v2a = v_proj.dot(&a_proj.t());
    let sim_a2v = a_proj.dot(&v_proj.t());
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
