//! Evaluate saved v2 model on arbitrary pool sizes.
//! Usage: POOL_SIZE=24604 eval-v2

use brain_core::metrics::CrossModalRetrieval;
use brain_experiment::embed_cache::WhiteningTransforms;
use ndarray::Array2;
use std::path::Path;

fn main() {
    let model_dir = std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/v2_whitened_rect".to_string());
    let whitening_path = std::env::var("WHITENING_PATH")
        .unwrap_or_else(|_| "/opt/brain/data/vggsound/.embed_cache/whitening.safetensors".to_string());
    let pool_size: usize = std::env::var("POOL_SIZE")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 = all clips

    eprintln!("Loading whitened embeddings...");
    let w = WhiteningTransforms::load(Path::new(&whitening_path)).expect("Failed to load whitening");
    let n_clips = w.v_white.nrows();
    let pool = if pool_size == 0 || pool_size >= n_clips { n_clips } else { pool_size };
    eprintln!("Clips: {}, eval pool: {}", n_clips, pool);

    eprintln!("Loading M matrix from {}...", model_dir);
    let m = load_matrix(&Path::new(&model_dir).join("m_va.bin"));
    eprintln!("M shape: {}×{}", m.nrows(), m.ncols());

    // Evaluate at multiple pool sizes for comparison
    let pools = if pool == n_clips {
        vec![1000, 5000, 10000, n_clips]
    } else {
        vec![pool]
    };

    for p in pools {
        let n = p.min(n_clips);
        let retrieval = CrossModalRetrieval::new(n, vec![1, 5, 10]);

        let v = w.v_white.slice(ndarray::s![..n, ..]).to_owned();
        let a = w.a_white.slice(ndarray::s![..n, ..]).to_owned();

        let mut v_norm = v;
        let mut a_norm = a;
        l2_normalize_inplace(&mut v_norm);
        l2_normalize_inplace(&mut a_norm);

        eprintln!("\nComputing similarities for pool={}...", n);
        let pv = v_norm.dot(&m);
        let sim_v2a = pv.dot(&a_norm.t());
        let pa = a_norm.dot(&m.t());
        let sim_a2v = pa.dot(&v_norm.t());

        let results = retrieval.evaluate_from_sims(&sim_v2a, &sim_a2v);

        eprintln!("=== Pool {} ===", n);
        let mut keys: Vec<_> = results.keys().collect();
        keys.sort();
        for k in keys {
            let v = results[k];
            eprintln!("  {}: {:.6}", k, v);
        }
    }
}

fn load_matrix(path: &Path) -> Array2<f32> {
    let data = std::fs::read(path).expect("Failed to read matrix file");
    // Format: "rows×cols\n" header followed by f32 LE bytes
    let newline = data.iter().position(|&b| b == b'\n').expect("No header newline");
    let header = std::str::from_utf8(&data[..newline]).expect("Invalid header");
    let parts: Vec<usize> = header.split('x').map(|s| s.parse().expect("Bad dim")).collect();
    let (rows, cols) = (parts[0], parts[1]);
    let float_data = &data[newline + 1..];
    let floats: Vec<f32> = float_data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(floats.len(), rows * cols, "Matrix size mismatch");
    Array2::from_shape_vec((rows, cols), floats).expect("Shape error")
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
