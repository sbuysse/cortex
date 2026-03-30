//! Visual encoders — DINOv2 and CLIP via TorchScript.

use std::path::Path;
use tch::{CModule, Device, Tensor};

/// DINOv2 visual encoder: image tensor → 384-dim embedding.
pub struct DINOv2Encoder {
    model: CModule,
}

impl DINOv2Encoder {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let model = CModule::load(path)?;
        tracing::info!("DINOv2 encoder loaded from {:?}", path);
        Ok(Self { model })
    }

    /// Encode a preprocessed image tensor (1, 3, 224, 224) → 384-dim L2-normalized embedding.
    pub fn encode(&self, image_tensor: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        assert_eq!(image_tensor.len(), 3 * 224 * 224);
        let input = Tensor::from_slice(image_tensor)
            .reshape(&[1, 3, 224, 224])
            .to_device(Device::Cpu);
        let output = self.model.forward_ts(&[input])?;
        let emb: Vec<f32> = Vec::try_from(output.flatten(0, -1))?;
        // L2 normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(emb.iter().map(|x| x / norm).collect())
    }
}

/// CLIP image encoder: image tensor → 512-dim embedding for scene classification.
pub struct CLIPEncoder {
    model: CModule,
}

impl CLIPEncoder {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let model = CModule::load(path)?;
        tracing::info!("CLIP encoder loaded from {:?}", path);
        Ok(Self { model })
    }

    /// Encode a preprocessed image tensor (1, 3, 224, 224) → 512-dim L2-normalized embedding.
    pub fn encode(&self, image_tensor: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        assert_eq!(image_tensor.len(), 3 * 224 * 224);
        let input = Tensor::from_slice(image_tensor)
            .reshape(&[1, 3, 224, 224])
            .to_device(Device::Cpu);
        let output = self.model.forward_ts(&[input])?;
        let emb: Vec<f32> = Vec::try_from(output.flatten(0, -1))?;
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(emb.iter().map(|x| x / norm).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_dinov2_load_and_encode() {
        let path = PathBuf::from("/opt/brain/outputs/cortex/visual_encoder/dinov2_ts.pt");
        if !path.exists() { return; }
        let enc = DINOv2Encoder::load(&path).unwrap();
        let input = vec![0.5f32; 3 * 224 * 224];
        let emb = enc.encode(&input).unwrap();
        assert_eq!(emb.len(), 384);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clip_load_and_encode() {
        let path = PathBuf::from("/opt/brain/outputs/cortex/visual_encoder/clip_ts.pt");
        if !path.exists() { return; }
        let enc = CLIPEncoder::load(&path).unwrap();
        let input = vec![0.5f32; 3 * 224 * 224];
        let emb = enc.encode(&input).unwrap();
        assert_eq!(emb.len(), 512);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
