//! V5: 2-layer MLP dual encoder with hard negative mining + temp annealing.
//!
//! Architecture:
//!   v: 384 → W_v1 (384×512) → ReLU → W_v2 (512×d_out) → ReLU → d_out
//!   a: 512 → W_a1 (512×512) → ReLU → W_a2 (512×d_out) → ReLU → d_out
//!   sim = v_proj @ a_proj^T / temp

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
    let output_dir = std::env::var("OUTPUT_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/v5_deep_mlp".to_string());
    let max_steps: usize = std::env::var("MAX_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let batch_size: usize = std::env::var("BATCH_SIZE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let d_hidden: usize = 512;
    let d_out: usize = std::env::var("D_OUT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.003);
    let temp_start: f32 = std::env::var("TEMP_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let temp_end: f32 = std::env::var("TEMP_END").ok().and_then(|s| s.parse().ok()).unwrap_or(0.005);
    let hard_neg_k: usize = 64;
    let hard_neg_refresh: usize = 2000;
    let hard_neg_frac: f32 = 0.5;
    let max_norm: f32 = 30.0;

    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    eprintln!("=== Train V5 — 2-Layer MLP Dual Encoder ===");
    let w = WhiteningTransforms::load(Path::new(&whitening_path)).expect("Failed to load whitening");
    let v_emb = &w.v_white;
    let a_emb = &w.a_white;
    let n_clips = v_emb.nrows();
    let dv = v_emb.ncols(); // 384
    let da = a_emb.ncols(); // 512
    eprintln!("Clips: {}, v={}, a={}, hidden={}, out={}", n_clips, dv, da, d_hidden, d_out);

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Initialize weights — He initialization
    let he_v1 = (2.0 / dv as f32).sqrt();
    let he_a1 = (2.0 / da as f32).sqrt();
    let he_2 = (2.0 / d_hidden as f32).sqrt();

    // Try warm-start from v4 weights
    let v4_dir = std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/v4_mlp".to_string());
    let (mut w_v1, mut w_a1);
    let v4_wv = load_matrix_opt(&Path::new(&v4_dir).join("w_v.bin"));
    let v4_wa = load_matrix_opt(&Path::new(&v4_dir).join("w_a.bin"));

    if let (Some(wv), Some(wa)) = (&v4_wv, &v4_wa) {
        eprintln!("Warm-starting layer 1 from v4 weights");
        w_v1 = wv.clone();
        w_a1 = wa.clone();
    } else {
        eprintln!("Random initialization");
        w_v1 = Array2::from_shape_fn((dv, d_hidden), |_| (rng.random::<f32>() - 0.5) * he_v1);
        w_a1 = Array2::from_shape_fn((da, d_hidden), |_| (rng.random::<f32>() - 0.5) * he_a1);
    }
    let mut w_v2 = Array2::from_shape_fn((d_hidden, d_out), |_| (rng.random::<f32>() - 0.5) * he_2);
    let mut w_a2 = Array2::from_shape_fn((d_hidden, d_out), |_| (rng.random::<f32>() - 0.5) * he_2);

    eprintln!("Config: lr={}, temp={}→{}, batch={}, steps={}", lr, temp_start, temp_end, batch_size, max_steps);

    let start = std::time::Instant::now();

    // Buffers
    let mut v_buf = Array2::zeros((batch_size, dv));
    let mut a_buf = Array2::zeros((batch_size, da));
    // Layer 1 outputs
    let mut v_h1_pre = Array2::zeros((batch_size, d_hidden));
    let mut a_h1_pre = Array2::zeros((batch_size, d_hidden));
    let mut v_h1 = Array2::zeros((batch_size, d_hidden));
    let mut a_h1 = Array2::zeros((batch_size, d_hidden));
    // Layer 2 outputs
    let mut v_h2_pre = Array2::zeros((batch_size, d_out));
    let mut a_h2_pre = Array2::zeros((batch_size, d_out));
    let mut v_proj = Array2::zeros((batch_size, d_out));
    let mut a_proj = Array2::zeros((batch_size, d_out));
    // Sim + softmax
    let mut sim = Array2::zeros((batch_size, batch_size));
    let mut probs = Array2::zeros((batch_size, batch_size));
    // Gradients
    let mut grad_v_proj = Array2::zeros((batch_size, d_out));
    let mut grad_a_proj = Array2::zeros((batch_size, d_out));
    let mut grad_v_h1 = Array2::zeros((batch_size, d_hidden));
    let mut grad_a_h1 = Array2::zeros((batch_size, d_hidden));

    let min_lr = 1e-6f32;
    let warmup_steps = 50usize;
    let retrieval_1k = CrossModalRetrieval::new(1000, vec![1, 5, 10]);
    let retrieval_full = CrossModalRetrieval::new(n_clips, vec![1, 5, 10]);
    let eval_interval = 50usize.max(max_steps / 20);

    let n_hard = (batch_size as f32 * hard_neg_frac) as usize;
    let n_random = batch_size - n_hard;
    let mut hard_negatives: Vec<Vec<usize>> = Vec::new();

    let indices: Vec<usize> = (0..n_clips).collect();
    let mut step = 0usize;
    let mut best_mrr_full = 0.0f64;
    let mut best_weights: Option<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> = None;

    while step < max_steps {
        if step % hard_neg_refresh == 0 {
            eprintln!("  [step {}] Refreshing hard negatives...", step);
            hard_negatives = compute_hard_negatives_2layer(v_emb, a_emb, &w_v1, &w_v2, &w_a1, &w_a2);
            eprintln!("  [step {}] Done", step);
        }

        // Build batch
        let mut batch_idx: Vec<usize> = Vec::with_capacity(batch_size);
        let mut anchors: Vec<usize> = indices.clone();
        anchors.shuffle(&mut rng);
        anchors.truncate(n_random);
        batch_idx.extend_from_slice(&anchors);
        if !hard_negatives.is_empty() && n_hard > 0 {
            let per = (n_hard / n_random.max(1)).max(1);
            for &a in &anchors {
                if batch_idx.len() >= batch_size { break; }
                for &neg in hard_negatives[a].iter().take(per) {
                    if batch_idx.len() >= batch_size { break; }
                    batch_idx.push(neg);
                }
            }
        }
        while batch_idx.len() < batch_size { batch_idx.push(rng.random_range(0..n_clips)); }
        batch_idx.truncate(batch_size);

        let current_lr = if step < warmup_steps {
            min_lr + (lr - min_lr) * (step as f32 / warmup_steps as f32)
        } else {
            let p = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
            min_lr + 0.5 * (lr - min_lr) * (1.0 + (std::f32::consts::PI * p).cos())
        };
        let temp = temp_start + (temp_end - temp_start) * (step as f32 / max_steps as f32);

        gather_rows_into(v_emb, &batch_idx, &mut v_buf);
        gather_rows_into(a_emb, &batch_idx, &mut a_buf);
        l2_normalize_inplace(&mut v_buf);
        l2_normalize_inplace(&mut a_buf);

        // === Forward: 2-layer MLP ===
        // V path: h1 = ReLU(V @ W_v1), proj = ReLU(h1 @ W_v2)
        ndarray::linalg::general_mat_mul(1.0, &v_buf, &w_v1, 0.0, &mut v_h1_pre);
        v_h1.assign(&v_h1_pre); v_h1.mapv_inplace(|x| x.max(0.0));
        ndarray::linalg::general_mat_mul(1.0, &v_h1, &w_v2, 0.0, &mut v_h2_pre);
        v_proj.assign(&v_h2_pre); v_proj.mapv_inplace(|x| x.max(0.0));

        // A path
        ndarray::linalg::general_mat_mul(1.0, &a_buf, &w_a1, 0.0, &mut a_h1_pre);
        a_h1.assign(&a_h1_pre); a_h1.mapv_inplace(|x| x.max(0.0));
        ndarray::linalg::general_mat_mul(1.0, &a_h1, &w_a2, 0.0, &mut a_h2_pre);
        a_proj.assign(&a_h2_pre); a_proj.mapv_inplace(|x| x.max(0.0));

        // sim = v_proj @ a_proj^T / temp
        ndarray::linalg::general_mat_mul(1.0 / temp, &v_proj, &a_proj.t(), 0.0, &mut sim);

        // === V→A InfoNCE ===
        softmax_rows(&sim, &mut probs, batch_size);
        for i in 0..batch_size { for j in 0..batch_size {
            probs[[i, j]] = if i == j { 1.0 } else { 0.0 } - probs[[i, j]];
        }}
        let scale = current_lr / (batch_size as f32 * temp);

        // grad_v_proj = scale * (I-P) @ a_proj
        ndarray::linalg::general_mat_mul(scale, &probs, &a_proj, 0.0, &mut grad_v_proj);
        // grad_a_proj = scale * (I-P)^T @ v_proj
        ndarray::linalg::general_mat_mul(scale, &probs.t(), &v_proj, 0.0, &mut grad_a_proj);

        // === A→V symmetric ===
        ndarray::linalg::general_mat_mul(1.0 / temp, &a_proj, &v_proj.t(), 0.0, &mut sim);
        softmax_rows(&sim, &mut probs, batch_size);
        for i in 0..batch_size { for j in 0..batch_size {
            probs[[i, j]] = if i == j { 1.0 } else { 0.0 } - probs[[i, j]];
        }}
        ndarray::linalg::general_mat_mul(scale, &probs, &v_proj, 1.0, &mut grad_a_proj);
        ndarray::linalg::general_mat_mul(scale, &probs.t(), &a_proj, 1.0, &mut grad_v_proj);

        // === Backprop through layer 2 ===
        // mask by ReLU: grad *= (pre > 0)
        for i in 0..batch_size { for j in 0..d_out {
            if v_h2_pre[[i, j]] <= 0.0 { grad_v_proj[[i, j]] = 0.0; }
            if a_h2_pre[[i, j]] <= 0.0 { grad_a_proj[[i, j]] = 0.0; }
        }}
        // W_v2 += h1^T @ grad_v_proj
        ndarray::linalg::general_mat_mul(1.0, &v_h1.t(), &grad_v_proj, 1.0, &mut w_v2);
        ndarray::linalg::general_mat_mul(1.0, &a_h1.t(), &grad_a_proj, 1.0, &mut w_a2);

        // === Backprop through layer 1 ===
        // grad_h1 = grad_proj @ W2^T * (h1_pre > 0)
        ndarray::linalg::general_mat_mul(1.0, &grad_v_proj, &w_v2.t(), 0.0, &mut grad_v_h1);
        ndarray::linalg::general_mat_mul(1.0, &grad_a_proj, &w_a2.t(), 0.0, &mut grad_a_h1);
        for i in 0..batch_size { for j in 0..d_hidden {
            if v_h1_pre[[i, j]] <= 0.0 { grad_v_h1[[i, j]] = 0.0; }
            if a_h1_pre[[i, j]] <= 0.0 { grad_a_h1[[i, j]] = 0.0; }
        }}
        // W_v1 += V^T @ grad_v_h1
        ndarray::linalg::general_mat_mul(1.0, &v_buf.t(), &grad_v_h1, 1.0, &mut w_v1);
        ndarray::linalg::general_mat_mul(1.0, &a_buf.t(), &grad_a_h1, 1.0, &mut w_a1);

        // Spectral norm clipping
        if step % 5 == 0 {
            clip_spectral(&mut w_v1, max_norm);
            clip_spectral(&mut w_v2, max_norm);
            clip_spectral(&mut w_a1, max_norm);
            clip_spectral(&mut w_a2, max_norm);
        }

        // Eval
        if step > 0 && step % eval_interval == 0 {
            let m1k = eval_2layer(&retrieval_1k, v_emb, a_emb, &w_v1, &w_v2, &w_a1, &w_a2);
            let mrr1k = m1k.get("v2a_MRR").copied().unwrap_or(0.0);
            let r1_1k = m1k.get("v2a_R@1").copied().unwrap_or(0.0);

            let do_full = step % (eval_interval * 5) == 0 || step + eval_interval >= max_steps;
            if do_full {
                let mf = eval_2layer(&retrieval_full, v_emb, a_emb, &w_v1, &w_v2, &w_a1, &w_a2);
                let mrr = mf.get("v2a_MRR").copied().unwrap_or(0.0);
                let r1 = mf.get("v2a_R@1").copied().unwrap_or(0.0);
                let r10 = mf.get("v2a_R@10").copied().unwrap_or(0.0);
                if mrr > best_mrr_full {
                    best_mrr_full = mrr;
                    best_weights = Some((w_v1.clone(), w_v2.clone(), w_a1.clone(), w_a2.clone()));
                }
                eprintln!("  step {}/{}: 1K MRR={:.4} R@1={:.4} | FULL MRR={:.4} R@1={:.4} R@10={:.4} (best={:.4}) t={:.4}",
                    step, max_steps, mrr1k, r1_1k, mrr, r1, r10, best_mrr_full, temp);
            } else {
                eprintln!("  step {}/{}: 1K MRR={:.4} R@1={:.4} t={:.4}",
                    step, max_steps, mrr1k, r1_1k, temp);
            }
        }
        step += 1;
    }

    if let Some((bv1, bv2, ba1, ba2)) = best_weights {
        eprintln!("Restoring best (full MRR={:.4})", best_mrr_full);
        w_v1 = bv1; w_v2 = bv2; w_a1 = ba1; w_a2 = ba2;
    }

    let mf = eval_2layer(&retrieval_full, v_emb, a_emb, &w_v1, &w_v2, &w_a1, &w_a2);
    let dur = start.elapsed().as_secs_f64();
    eprintln!("\n=== Final (24K pool) ===");
    let mut keys: Vec<_> = mf.keys().collect(); keys.sort();
    for k in &keys { eprintln!("  {}: {:.6}", k, mf[*k]); }
    eprintln!("  duration: {:.1}s", dur);

    let m1k = eval_2layer(&retrieval_1k, v_emb, a_emb, &w_v1, &w_v2, &w_a1, &w_a2);
    eprintln!("\n=== 1K comparison ===");
    for k in &keys { if let Some(v) = m1k.get(*k) { eprintln!("  {}: {:.6}", k, v); } }

    let out = Path::new(&output_dir);
    save_matrix(&w_v1, &out.join("w_v1.bin"));
    save_matrix(&w_v2, &out.join("w_v2.bin"));
    save_matrix(&w_a1, &out.join("w_a1.bin"));
    save_matrix(&w_a2, &out.join("w_a2.bin"));
    eprintln!("\nSaved to {}", output_dir);
}

fn softmax_rows(sim: &Array2<f32>, out: &mut Array2<f32>, n: usize) {
    for i in 0..n {
        let max_v = (0..n).map(|j| sim[[i, j]]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..n { let v = (sim[[i, j]] - max_v).exp(); out[[i, j]] = v; sum += v; }
        let inv = 1.0 / sum.max(1e-12);
        for j in 0..n { out[[i, j]] *= inv; }
    }
}

fn clip_spectral(w: &mut Array2<f32>, max_norm: f32) {
    let mut v = ndarray::Array1::<f32>::ones(w.ncols());
    v /= v.dot(&v).sqrt().max(1e-12);
    let u = w.dot(&v);
    let sigma = u.dot(&u).sqrt();
    if sigma > max_norm { w.mapv_inplace(|x| x * max_norm / sigma); }
}

fn project_2layer(emb: &Array2<f32>, w1: &Array2<f32>, w2: &Array2<f32>) -> Array2<f32> {
    let h1 = emb.dot(w1).mapv(|x| x.max(0.0));
    h1.dot(w2).mapv(|x| x.max(0.0))
}

fn eval_2layer(ret: &CrossModalRetrieval, v: &Array2<f32>, a: &Array2<f32>,
    wv1: &Array2<f32>, wv2: &Array2<f32>, wa1: &Array2<f32>, wa2: &Array2<f32>,
) -> HashMap<String, f64> {
    let n = v.nrows().min(ret.pool_size);
    let mut vn = v.slice(ndarray::s![..n, ..]).to_owned();
    let mut an = a.slice(ndarray::s![..n, ..]).to_owned();
    l2_normalize_inplace(&mut vn); l2_normalize_inplace(&mut an);
    let vp = project_2layer(&vn, wv1, wv2);
    let ap = project_2layer(&an, wa1, wa2);
    let s_v2a = vp.dot(&ap.t());
    let s_a2v = ap.dot(&vp.t());
    ret.evaluate_from_sims(&s_v2a, &s_a2v)
}

fn compute_hard_negatives_2layer(v: &Array2<f32>, a: &Array2<f32>,
    wv1: &Array2<f32>, wv2: &Array2<f32>, wa1: &Array2<f32>, wa2: &Array2<f32>,
) -> Vec<Vec<usize>> {
    let n = v.nrows();
    let mut vn = v.to_owned(); let mut an = a.to_owned();
    l2_normalize_inplace(&mut vn); l2_normalize_inplace(&mut an);
    let vp = project_2layer(&vn, wv1, wv2);
    let ap = project_2layer(&an, wa1, wa2);
    let block = 1000;
    let mut result = Vec::with_capacity(n);
    for s in (0..n).step_by(block) {
        let e = (s + block).min(n);
        let vb = vp.slice(ndarray::s![s..e, ..]);
        let sim_b = vb.dot(&ap.t());
        for il in 0..(e - s) {
            let ig = s + il;
            let row = sim_b.row(il);
            let mut idx: Vec<(usize, f32)> = row.iter().enumerate()
                .filter(|&(j, _)| j != ig).map(|(j, &s)| (j, s)).collect();
            idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            result.push(idx.iter().take(64).map(|(j, _)| *j).collect());
        }
    }
    result
}

fn save_matrix(m: &Array2<f32>, path: &std::path::Path) {
    let s = m.shape();
    let mut b = format!("{}x{}\n", s[0], s[1]).into_bytes();
    for v in m.as_slice().unwrap() { b.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(path, &b).expect("write failed");
}

fn load_matrix_opt(path: &std::path::Path) -> Option<Array2<f32>> {
    let data = std::fs::read(path).ok()?;
    let nl = data.iter().position(|&b| b == b'\n')?;
    let hdr = std::str::from_utf8(&data[..nl]).ok()?;
    let parts: Vec<usize> = hdr.split('x').filter_map(|s| s.parse().ok()).collect();
    if parts.len() != 2 { return None; }
    let floats: Vec<f32> = data[nl+1..].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    if floats.len() != parts[0] * parts[1] { return None; }
    Array2::from_shape_vec((parts[0], parts[1]), floats).ok()
}

#[inline]
fn gather_rows_into(arr: &Array2<f32>, idx: &[usize], buf: &mut Array2<f32>) {
    let c = arr.ncols(); let src = arr.as_slice().unwrap(); let dst = buf.as_slice_mut().unwrap();
    for (i, &ix) in idx.iter().enumerate() {
        dst[i*c..(i+1)*c].copy_from_slice(&src[ix*c..(ix+1)*c]);
    }
}

#[inline]
fn l2_normalize_inplace(arr: &mut Array2<f32>) {
    let c = arr.ncols();
    for ch in arr.as_slice_mut().unwrap().chunks_mut(c) {
        let n = ch.iter().map(|&v| v*v).sum::<f32>().sqrt().max(1e-12);
        let inv = 1.0 / n;
        ch.iter_mut().for_each(|v| *v *= inv);
    }
}
