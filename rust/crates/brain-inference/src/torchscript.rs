//! TorchScript model loading and inference via tch-rs.
//!
//! Loads exported .pt TorchScript models and runs inference on CPU.

use std::path::Path;
use tch::{CModule, Device, Tensor};

/// A loaded TorchScript model for single-input inference.
pub struct TorchScriptModel {
    module: CModule,
    name: String,
}

impl TorchScriptModel {
    /// Load a TorchScript model from a .pt file.
    pub fn load(path: &Path, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let module = CModule::load(path)?;
        tracing::info!("Loaded TorchScript model '{}' from {:?}", name, path);
        Ok(Self {
            module,
            name: name.to_string(),
        })
    }

    /// Run inference: single tensor input → single tensor output.
    pub fn forward(&self, input: &[f32], shape: &[i64]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let tensor = Tensor::from_slice(input).reshape(shape).to_device(Device::Cpu);
        let output = self.module.forward_ts(&[tensor])?;
        let output_vec: Vec<f32> = Vec::try_from(output.flatten(0, -1))?;
        Ok(output_vec)
    }

    /// Run inference with two tensor inputs (e.g., brain decoder: embedding + tokens).
    pub fn forward2(
        &self,
        input1: &[f32],
        shape1: &[i64],
        input2: &[i64],
        shape2: &[i64],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let t1 = Tensor::from_slice(input1).reshape(shape1).to_device(Device::Cpu);
        let t2 = Tensor::from_slice(input2).reshape(shape2).to_device(Device::Cpu);
        let output = self.module.forward_ts(&[t1, t2])?;
        let output_vec: Vec<f32> = Vec::try_from(output.flatten(0, -1))?;
        Ok(output_vec)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// World model: predict audio embedding from visual embedding (512→512 + residual).
pub struct WorldModel {
    model: TorchScriptModel,
}

impl WorldModel {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            model: TorchScriptModel::load(path, "world_model_v2")?,
        })
    }

    /// Predict audio embedding from a 512-dim visual embedding.
    pub fn predict(&self, visual_emb: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        assert_eq!(visual_emb.len(), 512);
        let result = self.model.forward(visual_emb, &[1, 512])?;
        // L2 normalize the output
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(result.iter().map(|x| x / norm).collect())
    }
}

/// Confidence predictor: predict confidence score from audio embedding (512→scalar).
pub struct ConfidencePredictor {
    model: TorchScriptModel,
}

impl ConfidencePredictor {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            model: TorchScriptModel::load(path, "confidence_predictor")?,
        })
    }

    /// Returns sigmoid confidence score [0, 1].
    pub fn predict(&self, emb: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        let logit = self.model.forward(emb, &[1, 512])?;
        let sigmoid = 1.0 / (1.0 + (-logit[0]).exp());
        Ok(sigmoid)
    }
}

/// Temporal predictor: predict next event embedding from a sequence.
pub struct TemporalPredictor {
    model: TorchScriptModel,
}

impl TemporalPredictor {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            model: TorchScriptModel::load(path, "temporal_predictor")?,
        })
    }

    /// Predict the next 512-dim embedding from a sequence of embeddings.
    pub fn predict(&self, sequence: &[Vec<f32>]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let seq_len = sequence.len();
        assert!(seq_len > 0 && seq_len <= 8);
        let flat: Vec<f32> = sequence.iter().flatten().copied().collect();
        let result = self.model.forward(&flat, &[1, seq_len as i64, 512])?;
        // L2 normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(result.iter().map(|x| x / norm).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_world_model_load_and_predict() {
        let path = PathBuf::from("/opt/brain/outputs/cortex/world_model/predictor_v2_ts.pt");
        if !path.exists() {
            return; // skip if not exported
        }
        let wm = WorldModel::load(&path).unwrap();
        let input = vec![0.1f32; 512];
        let output = wm.predict(&input).unwrap();
        assert_eq!(output.len(), 512);
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "Output should be L2 normalized, got norm={norm}");
    }

    #[test]
    fn test_confidence_predictor() {
        let path = PathBuf::from("/opt/brain/outputs/cortex/self_model/confidence_predictor_ts.pt");
        if !path.exists() {
            return;
        }
        let cp = ConfidencePredictor::load(&path).unwrap();
        let input = vec![0.1f32; 512];
        let score = cp.predict(&input).unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Confidence should be [0,1], got {score}");
    }

    #[test]
    fn test_temporal_predictor() {
        let path = PathBuf::from("/opt/brain/outputs/cortex/temporal_model/model_ts.pt");
        if !path.exists() {
            return;
        }
        let tp = TemporalPredictor::load(&path).unwrap();
        let seq = vec![vec![0.1f32; 512]; 3]; // 3-step sequence
        let pred = tp.predict(&seq).unwrap();
        assert_eq!(pred.len(), 512);
    }
}
