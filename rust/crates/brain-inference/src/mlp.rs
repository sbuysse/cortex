//! MLP dual encoder — loads V6 weights and projects embeddings.
//!
//! Architecture: ReLU(emb @ W), L2-normalized
//! V_dim=384 → hidden=512, A_dim=512 → hidden=512

use ndarray::{Array1, Array2};
use std::io::{BufRead, Read};
use std::path::Path;

/// MLP dual encoder for cross-modal projection.
pub struct MlpEncoder {
    pub w_v: Array2<f32>, // (384, 512)
    pub w_a: Array2<f32>, // (512, 512)
    pub sparse_k: usize,  // 0 = disabled, >0 = keep top-K activations
}

impl MlpEncoder {
    /// Load MLP weights from Rust binary format (header: RxC\n + f32 LE data).
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let w_v = load_bin_matrix(&dir.join("w_v.bin"))?;
        let w_a = load_bin_matrix(&dir.join("w_a.bin"))?;
        tracing::info!(
            "Loaded MLP: W_v={:?}, W_a={:?}",
            w_v.dim(),
            w_a.dim()
        );
        Ok(Self {
            w_v,
            w_a,
            sparse_k: 0,
        })
    }

    /// Project a visual embedding (384-dim) to the shared 512-dim space.
    pub fn project_visual(&self, emb: &[f32]) -> Vec<f32> {
        project(emb, &self.w_v, self.sparse_k)
    }

    /// Project an audio embedding (512-dim) to the shared 512-dim space.
    pub fn project_audio(&self, emb: &[f32]) -> Vec<f32> {
        project(emb, &self.w_a, self.sparse_k)
    }

    /// Project a batch of visual embeddings.
    pub fn project_visual_batch(&self, embs: &Array2<f32>) -> Array2<f32> {
        project_batch(embs, &self.w_v, self.sparse_k)
    }

    /// Project a batch of audio embeddings.
    pub fn project_audio_batch(&self, embs: &Array2<f32>) -> Array2<f32> {
        project_batch(embs, &self.w_a, self.sparse_k)
    }

    /// Cosine similarity between two projected embeddings.
    pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-12)
    }
}

/// ReLU(emb @ W), L2-normalized, with optional top-K sparsification.
pub fn project(emb: &[f32], w: &Array2<f32>, sparse_k: usize) -> Vec<f32> {
    let (rows, cols) = w.dim();
    assert_eq!(emb.len(), rows, "Embedding dim mismatch");

    // Matrix multiply: emb (1, rows) @ W (rows, cols) → (1, cols)
    let mut proj = vec![0.0f32; cols];
    for j in 0..cols {
        let mut sum = 0.0f32;
        for i in 0..rows {
            sum += emb[i] * w[[i, j]];
        }
        // ReLU
        proj[j] = sum.max(0.0);
    }

    // Top-K sparsification
    if sparse_k > 0 && sparse_k < cols {
        let mut sorted = proj.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[sparse_k];
        for v in proj.iter_mut() {
            if *v < threshold {
                *v = 0.0;
            }
        }
    }

    // L2 normalize
    let norm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for v in proj.iter_mut() {
        *v /= norm;
    }

    proj
}

/// Batch projection: (N, D_in) @ W (D_in, D_out) → (N, D_out), ReLU + L2 norm.
fn project_batch(embs: &Array2<f32>, w: &Array2<f32>, sparse_k: usize) -> Array2<f32> {
    let n = embs.nrows();
    let d_out = w.ncols();

    // Matrix multiply
    let mut result = embs.dot(w);

    // ReLU
    result.mapv_inplace(|v| v.max(0.0));

    // Top-K sparsification per row
    if sparse_k > 0 && sparse_k < d_out {
        for mut row in result.rows_mut() {
            let mut sorted: Vec<f32> = row.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let threshold = sorted[sparse_k.min(sorted.len() - 1)];
            for v in row.iter_mut() {
                if *v < threshold {
                    *v = 0.0;
                }
            }
        }
    }

    // L2 normalize per row
    for mut row in result.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }

    result
}

/// Train one step of gradient InfoNCE on a batch of (visual, audio) pairs.
/// Returns the number of pairs trained and the mean loss.
pub fn train_infonce(
    w_v: &mut Array2<f32>,
    w_a: &mut Array2<f32>,
    v_data: &Array2<f32>,  // (n, 384)
    a_data: &Array2<f32>,  // (n, 512)
    lr: f32,
    temp: f32,
    n_steps: usize,
) -> (usize, f32) {
    let n = v_data.nrows();
    if n < 2 { return (0, 0.0); }
    let mut total_loss = 0.0f32;

    for _step in 0..n_steps {
        // Forward: ReLU(X @ W)
        let v_pre = v_data.dot(w_v);
        let a_pre = a_data.dot(w_a);
        let v_proj = v_pre.mapv(|x| x.max(0.0));
        let a_proj = a_pre.mapv(|x| x.max(0.0));

        // L2 normalize projections for similarity
        let v_norm = l2_normalize_rows(&v_proj);
        let a_norm = l2_normalize_rows(&a_proj);

        // Similarity matrix: (n, n)
        let sim = v_norm.dot(&a_norm.t()) / temp;

        // Softmax per row (numerically stable)
        let sim_max = row_max(&sim);
        let mut exp_sim = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                exp_sim[[i, j]] = (sim[[i, j]] - sim_max[i]).exp();
            }
        }
        let row_sums = exp_sim.sum_axis(ndarray::Axis(1));
        let mut probs = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                probs[[i, j]] = exp_sim[[i, j]] / (row_sums[i] + 1e-12);
            }
        }

        // Diagonal similarities (matched pair quality)
        let mut diag_sims = vec![0.0f32; n];
        for i in 0..n {
            diag_sims[i] = (0..v_norm.ncols()).map(|j| v_norm[[i, j]] * a_norm[[i, j]]).sum::<f32>().clamp(0.0, 1.0);
        }

        // Congruence-weighted target
        let mut target = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            target[[i, i]] = (diag_sims[i] * 2.0).exp();
        }
        let target_sums = target.sum_axis(ndarray::Axis(1));
        for i in 0..n {
            if target_sums[i] > 0.0 { target.row_mut(i).mapv_inplace(|v| v / target_sums[i]); }
        }

        // Gradient: target - probs
        let grad = &target - &probs;

        // Loss: -mean(log(probs[i,i]))
        let loss: f32 = -(0..n).map(|i| probs[[i, i]].max(1e-12).ln()).sum::<f32>() / n as f32;
        total_loss += loss;

        // Surprise weighting
        let surprise: Vec<f32> = diag_sims.iter().map(|&s| 1.0 - s).collect();
        let surprise_mean: f32 = surprise.iter().sum::<f32>() / n as f32 + 1e-12;

        // Backprop: gradient for projections
        let scale = lr / (n as f32 * temp);
        let grad_v_proj = {
            let mut g = grad.dot(&a_proj); // (n, 512)
            for i in 0..n {
                let sw = surprise[i] / surprise_mean;
                g.row_mut(i).mapv_inplace(|v| v * scale * sw);
            }
            // ReLU mask
            for i in 0..n {
                for j in 0..g.ncols() {
                    if v_pre[[i, j]] <= 0.0 { g[[i, j]] = 0.0; }
                }
            }
            g
        };
        let grad_a_proj = {
            let mut g = grad.t().dot(&v_proj); // (n, 512)
            for i in 0..n {
                let sw = surprise[i] / surprise_mean;
                g.row_mut(i).mapv_inplace(|v| v * scale * sw);
            }
            for i in 0..n {
                for j in 0..g.ncols() {
                    if a_pre[[i, j]] <= 0.0 { g[[i, j]] = 0.0; }
                }
            }
            g
        };

        // Weight update: W += X^T @ grad
        *w_v = &*w_v + &v_data.t().dot(&grad_v_proj);
        *w_a = &*w_a + &a_data.t().dot(&grad_a_proj);
    }

    (n, total_loss / n_steps as f32)
}

fn l2_normalize_rows(m: &Array2<f32>) -> Array2<f32> {
    let mut result = m.clone();
    for mut row in result.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|x| x / norm);
    }
    result
}

fn row_max(m: &Array2<f32>) -> Vec<f32> {
    (0..m.nrows()).map(|i| m.row(i).iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect()
}

/// Load a binary matrix in Rust format (header: "RxC\n" + f32 LE data).
pub fn load_bin_matrix(path: &Path) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    // Read header line
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let header = header.trim();
    let parts: Vec<&str> = header.split('x').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid matrix header: {header}").into());
    }
    let rows: usize = parts[0].parse()?;
    let cols: usize = parts[1].parse()?;

    // Read f32 LE data
    let mut data = vec![0u8; rows * cols * 4];
    reader.read_exact(&mut data)?;

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(Array2::from_shape_vec((rows, cols), floats)?)
}

/// Save a matrix in Rust binary format.
pub fn save_bin_matrix(m: &Array2<f32>, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    let (rows, cols) = m.dim();
    write!(file, "{}x{}\n", rows, cols)?;
    for &val in m.iter() {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_project_shape() {
        let w = Array2::from_elem((384, 512), 0.01f32);
        let emb = vec![1.0f32; 384];
        let result = project(&emb, &w, 0);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_project_l2_normalized() {
        let w = Array2::from_elem((384, 512), 0.01f32);
        let emb = vec![1.0f32; 384];
        let result = project(&emb, &w, 0);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "L2 norm should be 1.0, got {norm}");
    }

    #[test]
    fn test_project_relu_nonnegative() {
        let w = Array2::from_elem((384, 512), 0.01f32);
        let emb = vec![1.0f32; 384];
        let result = project(&emb, &w, 0);
        assert!(result.iter().all(|&v| v >= 0.0), "All values should be >= 0 after ReLU");
    }

    #[test]
    fn test_project_sparse() {
        // Use varied weights so activations differ
        let mut w = Array2::zeros((384, 512));
        for i in 0..384 {
            for j in 0..512 {
                w[[i, j]] = ((i * 512 + j) as f32 * 0.001).sin() * 0.01;
            }
        }
        let emb: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).cos()).collect();
        let result = project(&emb, &w, 50);
        let nonzero = result.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero <= 51, "Expected <=51 nonzero, got {nonzero}"); // +1 for boundary
    }

    #[test]
    fn test_load_bin_matrix() {
        // Create a test matrix and save it
        let m = Array2::from_elem((3, 4), 1.5f32);
        let path = std::path::PathBuf::from("/tmp/test_matrix.bin");
        save_bin_matrix(&m, &path).unwrap();

        // Load it back
        let loaded = load_bin_matrix(&path).unwrap();
        assert_eq!(loaded.dim(), (3, 4));
        assert!((loaded[[0, 0]] - 1.5).abs() < 1e-6);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((MlpEncoder::similarity(&a, &b) - 1.0).abs() < 1e-5);

        let c = vec![0.0, 1.0, 0.0];
        assert!(MlpEncoder::similarity(&a, &c).abs() < 1e-5);
    }

    #[test]
    fn test_load_real_weights() {
        let dir = std::path::Path::new("/opt/brain/outputs/cortex/v6_mlp");
        if dir.exists() {
            let encoder = MlpEncoder::load(dir).unwrap();
            assert_eq!(encoder.w_v.dim(), (384, 512));
            assert_eq!(encoder.w_a.dim(), (512, 512));

            // Test projection
            let v_emb = vec![0.1f32; 384];
            let result = encoder.project_visual(&v_emb);
            assert_eq!(result.len(), 512);
            let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_train_infonce() {
        use rand::Rng;
        let mut rng = rand::rng();
        // Random weights so projections differ per sample
        let mut w_v = Array2::from_shape_fn((384, 512), |_| rng.random::<f32>() * 0.02 - 0.01);
        let mut w_a = Array2::from_shape_fn((512, 512), |_| rng.random::<f32>() * 0.02 - 0.01);
        let w_v_orig = w_v.clone();

        let n = 8;
        let mut v_data = Array2::zeros((n, 384));
        let mut a_data = Array2::zeros((n, 512));
        for i in 0..n {
            for j in 0..384 { v_data[[i, j]] = rng.random::<f32>(); }
            for j in 0..512 { a_data[[i, j]] = rng.random::<f32>(); }
        }

        let (trained, loss) = train_infonce(&mut w_v, &mut w_a, &v_data, &a_data, 0.001, 0.01, 5);
        assert_eq!(trained, n);
        assert!(loss > 0.0 && loss.is_finite(), "Loss should be positive finite, got {loss}");

        // Weights should have changed
        let diff: f32 = w_v.iter().zip(w_v_orig.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "w_v should update, total diff={diff}");
    }
}
