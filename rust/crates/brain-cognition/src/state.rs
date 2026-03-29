//! BrainState — the central cognitive state object.
//!
//! Owns all subsystems: working memory, fast memory, grid encoder,
//! SSE bus, memory database, inference engine, and configuration.

use std::sync::Arc;
use crate::config::BrainConfig;
use crate::sse::SseBus;
use crate::working_memory::WorkingMemory;
use crate::fast_memory::HopfieldMemory;
use crate::grid_cells::GridCellEncoder;
use crate::memory_db::MemoryDb;
use crate::personal_memory::PersonalMemory;
use crate::concepts::ConceptCodebook;

/// The brain's complete cognitive state.
pub struct BrainState {
    pub config: BrainConfig,
    pub sse: SseBus,
    pub working_memory: std::sync::Mutex<WorkingMemory>,
    pub fast_memory: std::sync::Mutex<HopfieldMemory>,
    pub grid_encoder: std::sync::Mutex<GridCellEncoder>,
    pub memory_db: MemoryDb,
    pub personal_memory: std::sync::Mutex<PersonalMemory>,
    pub inference: Option<brain_inference::MlpEncoder>,
    pub text_encoder: Option<brain_inference::TextEncoder>,
    pub codebook: std::sync::Mutex<Option<ConceptCodebook>>,
    pub world_model: Option<brain_inference::WorldModel>,
    pub confidence_model: Option<brain_inference::ConfidencePredictor>,
    pub temporal_model: Option<brain_inference::TemporalPredictor>,
    pub visual_encoder: Option<brain_inference::DINOv2Encoder>,
    pub clip_encoder: Option<brain_inference::CLIPEncoder>,
    pub audio_encoder: Option<brain_inference::WhisperEncoder>,
    pub companion_decoder: Option<brain_inference::CompanionDecoder>,
    pub online_pairs: std::sync::Mutex<Vec<(Vec<f32>, Vec<f32>)>>,
    pub online_learning_count: std::sync::atomic::AtomicI64,
    pub autonomy_running: std::sync::atomic::AtomicBool,
    pub autonomy_cycles: std::sync::atomic::AtomicI64,
    pub autonomy_videos: std::sync::atomic::AtomicI64,
    pub dream_count: std::sync::atomic::AtomicI64,
    pub prediction_error_history: std::sync::Mutex<Vec<f32>>,
    pub start_time: std::time::Instant,
}

impl BrainState {
    /// Create a new BrainState from configuration.
    pub fn new(config: BrainConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let sse = SseBus::new(256);

        let wm = WorkingMemory::new(config.wm_slots, config.wm_decay, config.wm_theta_freq);
        let fast = HopfieldMemory::new(512, config.fast_memory_capacity);
        let grid = GridCellEncoder::new(config.grid_scales.clone());

        let db = MemoryDb::open(&config.memory_db_path)?;
        tracing::info!("Memory DB opened: {:?}", config.memory_db_path);

        let personal_mem_path = config.project_root.join("data/brain_personal.json");
        let personal_mem = PersonalMemory::load(personal_mem_path);
        tracing::info!("Personal memory loaded ({} facts, {} conversations)", personal_mem.facts.len(), personal_mem.conversations.len());

        // Load MLP: prefer online-trained weights, fallback to base model
        let online_dir = config.project_root.join("outputs/cortex/v6_mlp_online");
        let inference = if online_dir.join("w_v.bin").exists() {
            match brain_inference::MlpEncoder::load(&online_dir) {
                Ok(mlp) => {
                    tracing::info!("MLP encoder loaded from ONLINE weights: {:?}", online_dir);
                    Some(mlp)
                }
                Err(e) => {
                    tracing::warn!("Failed to load online MLP ({e}), falling back to base");
                    brain_inference::MlpEncoder::load(&config.model_dir).ok()
                }
            }
        } else {
            let mlp = brain_inference::MlpEncoder::load(&config.model_dir).ok();
            if mlp.is_some() { tracing::info!("MLP encoder loaded from {:?}", config.model_dir); }
            mlp
        };

        // Load TorchScript models (lazy — only if files exist)
        let root = &config.project_root;
        let world_model = match brain_inference::WorldModel::load(
            &root.join("outputs/cortex/world_model/predictor_v2_ts.pt")) {
            Ok(m) => { tracing::info!("World model (TorchScript) loaded"); Some(m) }
            Err(e) => { tracing::warn!("World model failed: {e}"); None }
        };

        let confidence_model = brain_inference::ConfidencePredictor::load(
            &root.join("outputs/cortex/self_model/confidence_predictor_ts.pt")).ok();
        if confidence_model.is_some() { tracing::info!("Confidence predictor (TorchScript) loaded"); }

        let temporal_model = brain_inference::TemporalPredictor::load(
            &root.join("outputs/cortex/temporal_model/model_ts.pt")).ok();
        if temporal_model.is_some() { tracing::info!("Temporal predictor (TorchScript) loaded"); }

        // Perception encoders (DINOv2, CLIP, Whisper)
        let visual_encoder = match brain_inference::DINOv2Encoder::load(
            &root.join("outputs/cortex/visual_encoder/dinov2_ts.pt")) {
            Ok(m) => { tracing::info!("DINOv2 visual encoder loaded"); Some(m) }
            Err(e) => { tracing::warn!("DINOv2 failed: {e}"); None }
        };
        let clip_encoder = match brain_inference::CLIPEncoder::load(
            &root.join("outputs/cortex/visual_encoder/clip_ts.pt")) {
            Ok(m) => { tracing::info!("CLIP encoder loaded"); Some(m) }
            Err(e) => { tracing::warn!("CLIP failed: {e}"); None }
        };
        let audio_encoder = match brain_inference::WhisperEncoder::load(
            &root.join("outputs/cortex/audio_encoder/whisper_encoder_ts.pt")) {
            Ok(m) => { tracing::info!("Whisper encoder loaded"); Some(m) }
            Err(e) => { tracing::warn!("Whisper failed: {e}"); None }
        };

        let text_encoder = match brain_inference::TextEncoder::load(
            &root.join("outputs/cortex/text_encoder")) {
            Ok(te) => { tracing::info!("Text encoder loaded: {} labels", te.label_count()); Some(te) }
            Err(e) => { tracing::warn!("Text encoder failed: {e}"); None }
        };

        let companion_decoder = {
            let model_path = config.project_root
                .join("outputs/cortex/hope_companion/hope_companion_ts.pt");
            if model_path.exists() {
                match brain_inference::CompanionDecoder::load(&model_path) {
                    Ok(dec) => {
                        tracing::info!("Loaded HOPE CompanionDecoder");
                        Some(dec)
                    }
                    Err(e) => {
                        tracing::warn!("CompanionDecoder load failed: {}", e);
                        None
                    }
                }
            } else {
                tracing::info!("HOPE CompanionDecoder model not found — Ollama fallback active");
                None
            }
        };

        // Build concept codebook if we have text encoder + MLP
        // Concept codebook: built from pre-encoded label embeddings (fast, no TorchScript calls)
        let codebook = match (&text_encoder, &inference) {
            (Some(te), Some(mlp)) if te.has_labels() => {
                // Use the pre-loaded label embeddings from the .npy file (already in TextEncoder)
                // These are 384-dim text embeddings, project through MLP to 512-dim
                let mut label_embs = Vec::new();
                if let (Some(labels), Some(embs)) = (&te.labels, &te.label_embeddings) {
                    for (label, emb) in labels.iter().zip(embs.iter()) {
                        label_embs.push((label.clone(), emb.clone()));
                    }
                }
                if !label_embs.is_empty() {
                    Some(ConceptCodebook::build(&label_embs, &mlp.w_v))
                } else { None }
            }
            _ => None,
        };

        // Restore counters from DB
        let online_count = db.get_stat("online_learning_count").ok()
            .flatten().and_then(|v| v.parse().ok()).unwrap_or(0i64);
        let autonomy_cycles = db.get_stat("autonomy_cycles").ok()
            .flatten().and_then(|v| v.parse().ok()).unwrap_or(0i64);
        let autonomy_videos = db.get_stat("autonomy_videos").ok()
            .flatten().and_then(|v| v.parse().ok()).unwrap_or(0i64);
        let dream_count = db.get_stat("dream_count").ok()
            .flatten().and_then(|v| v.parse().ok()).unwrap_or(0i64);

        Ok(Self {
            config,
            sse,
            working_memory: std::sync::Mutex::new(wm),
            fast_memory: std::sync::Mutex::new(fast),
            grid_encoder: std::sync::Mutex::new(grid),
            memory_db: db,
            personal_memory: std::sync::Mutex::new(personal_mem),
            inference,
            text_encoder,
            codebook: std::sync::Mutex::new(codebook),
            world_model,
            confidence_model,
            temporal_model,
            visual_encoder,
            clip_encoder,
            audio_encoder,
            companion_decoder,
            online_pairs: std::sync::Mutex::new(Vec::new()),
            online_learning_count: std::sync::atomic::AtomicI64::new(online_count),
            autonomy_running: std::sync::atomic::AtomicBool::new(false),
            autonomy_cycles: std::sync::atomic::AtomicI64::new(autonomy_cycles),
            autonomy_videos: std::sync::atomic::AtomicI64::new(autonomy_videos),
            dream_count: std::sync::atomic::AtomicI64::new(dream_count),
            prediction_error_history: std::sync::Mutex::new(Vec::new()),
            start_time: std::time::Instant::now(),
        })
    }

    /// Uptime in seconds.
    pub fn uptime_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Health check: which components are loaded?
    pub fn health(&self) -> serde_json::Value {
        let grid_fitted = self.grid_encoder.lock().unwrap().is_fitted();
        let pairs = self.online_pairs.lock().unwrap().len();
        serde_json::json!({
            "status": "ok",
            "uptime_seconds": (self.uptime_secs() * 10.0).round() / 10.0,
            "components": {
                "mlp": self.inference.is_some(),
                "world_model": self.world_model.is_some(),
                "confidence_model": self.confidence_model.is_some(),
                "temporal_model": self.temporal_model.is_some(),
                "text_encoder": self.text_encoder.is_some(),
                "visual_encoder": self.visual_encoder.is_some(),
                "clip_encoder": self.clip_encoder.is_some(),
                "audio_encoder": self.audio_encoder.is_some(),
                "grid_encoder": grid_fitted,
                "memory_db": true,
            },
            "memory": {
                "working_memory": self.working_memory.lock().unwrap().get_state().slots_used,
                "fast_memory": self.fast_memory.lock().unwrap().count(),
                "prototypes": self.memory_db.prototype_count().unwrap_or(0),
                "perceptions": self.memory_db.perception_count().unwrap_or(0),
                "episodes": self.memory_db.episode_count().unwrap_or(0),
                "kg_edges": self.memory_db.edge_count().unwrap_or(0),
                "sse_subscribers": self.sse.subscriber_count(),
                "online_pairs_buffer": pairs,
            },
            "autonomy": {
                "running": self.autonomy_running.load(std::sync::atomic::Ordering::Relaxed),
                "cycles": self.autonomy_cycles.load(std::sync::atomic::Ordering::Relaxed),
                "videos": self.autonomy_videos.load(std::sync::atomic::Ordering::Relaxed),
            },
            "online_learning": {
                "total_learned": self.online_learning_count.load(std::sync::atomic::Ordering::Relaxed),
                "buffer_size": pairs,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_state_creation() {
        let mut config = BrainConfig::from_env();
        // Use temp DB for testing
        config.memory_db_path = std::path::PathBuf::from(":memory:");
        config.model_dir = std::path::PathBuf::from("/nonexistent"); // MLP won't load

        let state = BrainState::new(config).unwrap();
        assert!(state.inference.is_none()); // no MLP at /nonexistent
        assert!(state.uptime_secs() < 1.0);

        let health = state.health();
        assert_eq!(health["status"], "ok");
        assert_eq!(health["components"]["mlp"], false);
        assert_eq!(health["components"]["memory_db"], true);
    }

    #[test]
    fn test_brain_state_with_real_model() {
        let config = BrainConfig::from_env();
        if !config.model_dir.join("w_v.bin").exists() {
            return; // skip if no model
        }
        let state = BrainState::new(config).unwrap();
        assert!(state.inference.is_some());

        let health = state.health();
        assert_eq!(health["components"]["mlp"], true);
    }
}
