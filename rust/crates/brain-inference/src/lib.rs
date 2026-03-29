//! Brain Inference Engine — all model loading and inference via libtorch (tch-rs).
//!
//! Models:
//! - MLP V6 dual encoder (384→512, 512→512)
//! - World model (visual→audio prediction, TorchScript)
//! - Confidence predictor (embedding→score, TorchScript)
//! - Temporal predictor (sequence→next, TorchScript)
//! - Text encoder MiniLM (text→384-dim, TorchScript)
//! - DINOv2 visual encoder (image→384-dim, TorchScript)
//! - Whisper audio encoder (mel→512-dim, TorchScript)
//! - CLIP scene classifier (image→512-dim, TorchScript)

pub mod mlp;
pub mod torchscript;
pub mod text;
pub mod visual;
pub mod audio;
pub mod mel;
pub mod emotion;
pub mod vad;
pub mod faces;
pub mod companion_decoder;
pub mod quantized_index;
pub use quantized_index::QuantizedIndex;

pub use mlp::MlpEncoder;
pub use companion_decoder::CompanionDecoder;
pub use torchscript::{WorldModel, ConfidencePredictor, TemporalPredictor, TorchScriptModel};
pub use text::TextEncoder;
pub use visual::{DINOv2Encoder, CLIPEncoder};
pub use audio::WhisperEncoder;
