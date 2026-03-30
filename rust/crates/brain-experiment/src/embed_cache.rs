//! Embedding cache loader using safetensors format.
//!
//! Loads pre-computed encoder embeddings (DINOv2, Whisper, Emotion) from
//! safetensors files. These are shared read-only across all experiments.

use ndarray::Array2;
use safetensors::SafeTensors;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("Shape mismatch: expected 2D tensor, got {0:?}")]
    ShapeMismatch(Vec<usize>),
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
}

/// Shared embedding cache — immutable after construction.
pub struct CachedEmbeddings {
    /// Visual encoder outputs [N, D_visual]
    pub v_emb: Array2<f32>,
    /// Audio encoder outputs [N, D_audio]
    pub a_emb: Array2<f32>,
    /// Emotion encoder outputs [N, D_emotion] (audio wav2vec2 + face ResNet-18)
    pub e_emb: Array2<f32>,
    /// Speech content outputs [N, D_speech] (Whisper ASR → sentence embedding)
    pub s_emb: Option<Array2<f32>>,
    /// Scene/audio properties [N, D_props] (color, motion, edges, loudness, pitch, tempo)
    pub p_emb: Option<Array2<f32>>,
    /// Number of clips
    pub n_clips: usize,
}

impl CachedEmbeddings {
    /// Load embeddings from a safetensors file.
    pub fn load(path: &Path) -> Result<Self, CacheError> {
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let v_emb = load_f32_tensor(&tensors, "v_emb")?;
        let a_emb = load_f32_tensor(&tensors, "a_emb")?;
        let e_emb = load_f32_tensor(&tensors, "e_emb")?;
        let s_emb = load_f32_tensor(&tensors, "s_emb").ok();
        let p_emb = load_f32_tensor(&tensors, "p_emb").ok();

        let n_clips = v_emb.nrows();

        Ok(Self {
            v_emb,
            a_emb,
            e_emb,
            s_emb,
            p_emb,
            n_clips,
        })
    }

    /// Load from a PyTorch .pt cache file that was converted to safetensors.
    ///
    /// If the safetensors file doesn't exist but a .pt file does,
    /// returns an error suggesting conversion.
    pub fn load_or_convert(safetensors_path: &Path, pt_path: &Path) -> Result<Self, CacheError> {
        if safetensors_path.exists() {
            return Self::load(safetensors_path);
        }
        if pt_path.exists() {
            return Err(CacheError::MissingTensor(format!(
                "Safetensors file not found at {}. Run the conversion script: \
                 python scripts/convert_pt_to_safetensors.py {}",
                safetensors_path.display(),
                pt_path.display()
            )));
        }
        Err(CacheError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("No embedding cache found at {} or {}", safetensors_path.display(), pt_path.display()),
        )))
    }

    /// Wrap in Arc for shared access across experiments.
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

/// Precomputed ZCA whitening transforms.
pub struct WhiteningTransforms {
    /// Visual mean [D_visual]
    pub v_mean: ndarray::Array1<f32>,
    /// Visual ZCA matrix [D_visual, D_visual]
    pub v_zca: Array2<f32>,
    /// Audio mean [D_audio]
    pub a_mean: ndarray::Array1<f32>,
    /// Audio ZCA matrix [D_audio, D_audio]
    pub a_zca: Array2<f32>,
    /// Pre-whitened visual embeddings [N, D_visual]
    pub v_white: Array2<f32>,
    /// Pre-whitened audio embeddings [N, D_audio]
    pub a_white: Array2<f32>,
}

impl WhiteningTransforms {
    /// Load precomputed whitening from safetensors.
    pub fn load(path: &Path) -> Result<Self, CacheError> {
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let v_mean = load_f32_1d(&tensors, "v_mean")?;
        let v_zca = load_f32_tensor(&tensors, "v_zca")?;
        let a_mean = load_f32_1d(&tensors, "a_mean")?;
        let a_zca = load_f32_tensor(&tensors, "a_zca")?;
        let v_white = load_f32_tensor(&tensors, "v_white")?;
        let a_white = load_f32_tensor(&tensors, "a_white")?;

        Ok(Self { v_mean, v_zca, a_mean, a_zca, v_white, a_white })
    }
}

/// Category labels for multi-positive InfoNCE.
pub struct CategoryLabels {
    /// Label index per clip [N]
    pub labels: Vec<i32>,
    /// Number of unique categories
    pub n_categories: usize,
}

impl CategoryLabels {
    /// Load labels from safetensors.
    pub fn load(path: &Path) -> Result<Self, CacheError> {
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let tensor = tensors
            .tensor("labels")
            .map_err(|_| CacheError::MissingTensor("labels".to_string()))?;
        let shape = tensor.shape();
        if shape.len() != 1 {
            return Err(CacheError::ShapeMismatch(shape.to_vec()));
        }

        let labels: Vec<i32> = tensor.data()
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let n_categories = labels.iter().copied().max().unwrap_or(0) as usize + 1;

        Ok(Self { labels, n_categories })
    }
}

fn load_f32_1d(tensors: &SafeTensors, name: &str) -> Result<ndarray::Array1<f32>, CacheError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| CacheError::MissingTensor(name.to_string()))?;
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(CacheError::ShapeMismatch(shape.to_vec()));
    }
    let floats: Vec<f32> = tensor.data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(ndarray::Array1::from_vec(floats))
}

fn load_f32_tensor(tensors: &SafeTensors, name: &str) -> Result<Array2<f32>, CacheError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| CacheError::MissingTensor(name.to_string()))?;

    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(CacheError::ShapeMismatch(shape.to_vec()));
    }

    let (rows, cols) = (shape[0], shape[1]);

    // Convert bytes to f32
    let data = tensor.data();
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Array2::from_shape_vec((rows, cols), floats)
        .map_err(|_| CacheError::ShapeMismatch(vec![rows, cols]))
}
