//! BrainState — the central cognitive state object.
//!
//! Owns all subsystems: working memory, fast memory, grid encoder,
//! SSE bus, memory database, inference engine, and configuration.

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
    /// Learned text concepts — (label, 384-dim embedding) pairs from academic learning.
    /// Searched alongside text_encoder labels during associative recall.
    pub learned_concepts: std::sync::Mutex<Vec<(String, Vec<f32>)>>,
    /// Triple queue for knowledge learning — separate from brain mutex.
    pub triple_queue: std::sync::Arc<std::sync::Mutex<Vec<(brain_spiking::Triple, String, i32)>>>,
    /// Recall queue — concept names to recall via chain propagation.
    pub recall_queue: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    pub codebook: std::sync::Mutex<Option<ConceptCodebook>>,
    pub world_model: Option<brain_inference::WorldModel>,
    pub confidence_model: Option<brain_inference::ConfidencePredictor>,
    pub temporal_model: Option<brain_inference::TemporalPredictor>,
    pub visual_encoder: Option<brain_inference::DINOv2Encoder>,
    pub clip_encoder: Option<brain_inference::CLIPEncoder>,
    pub audio_encoder: Option<brain_inference::WhisperEncoder>,
    pub companion_decoder: Option<brain_inference::CompanionDecoder>,
    pub spiking_brain: Option<std::sync::Arc<std::sync::Mutex<brain_spiking::SpikingBrain>>>,
    /// Latest brain snapshot — updated by tick thread, read by dialogue route without locking the brain.
    pub spiking_snapshot: std::sync::Arc<std::sync::Mutex<brain_spiking::BrainSnapshot>>,
    pub emotion_table: Vec<[f32; 512]>,
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

        let spiking_brain = {
            // SPIKING_SCALE=0.01 for tiny test, 1.0 for full ~2M neurons
            let scale: f32 = std::env::var("SPIKING_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            if scale > 0.0 {
                tracing::info!("Initializing spiking brain (10 regions, scale={scale})");
                let data_dir = config.project_root.join("data");
                let mut sb = brain_spiking::SpikingBrain::new(scale, Some(data_dir));
                let save_dir = config.project_root.join("outputs/cortex");
                let loaded = sb.load(&save_dir);
                if loaded > 0 {
                    tracing::info!("Restored saved weights for {loaded} spiking brain regions");
                }
                Some(std::sync::Arc::new(std::sync::Mutex::new(sb)))
            } else {
                None
            }
        };

        let spiking_snapshot = std::sync::Arc::new(std::sync::Mutex::new(brain_spiking::BrainSnapshot::default()));

        let triple_queue = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(brain_spiking::Triple, String, i32)>::new()));
        let recall_queue: std::sync::Arc<std::sync::Mutex<Option<String>>> = std::sync::Arc::new(std::sync::Mutex::new(None));

        // Start spiking brain background tick thread
        if let Some(ref sb) = spiking_brain {
            let sb_clone = std::sync::Arc::clone(sb);
            let snap_clone = std::sync::Arc::clone(&spiking_snapshot);
            let tq_clone = std::sync::Arc::clone(&triple_queue);
            let rq_clone = std::sync::Arc::clone(&recall_queue);
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(2));

                    // Check for recall request (no brain lock needed to check)
                    let recall_concept = rq_clone.lock().unwrap().take();
                    if let Some(concept) = recall_concept {
                        let t0 = std::time::Instant::now();
                        tracing::info!("Chain recall starting for: {}", concept);

                        let mut sb = sb_clone.lock().unwrap();

                        // Phase 1: BFS recall (instant)
                        let (bfs_chain, knowledge) = sb.recall_knowledge(&concept);
                        let bfs_labels: Vec<String> = bfs_chain.iter().map(|(n, _)| n.clone()).collect();

                        // Phase 2: Spiking recall (fire seeds into network)
                        let spiking_result = sb.run_spiking_recall();

                        // Build snapshot
                        let mut snap = brain_spiking::BrainSnapshot::default();
                        snap.has_data = true;

                        if let Some((spiking_direct, spiking_predicted, mode)) = spiking_result {
                            snap.recall_mode = mode;
                            snap.spiking_associations = spiking_direct.clone();
                            snap.predicted_associations = spiking_predicted.clone();

                            let all_spiking: Vec<(String, usize)> = spiking_direct.iter()
                                .chain(spiking_predicted.iter())
                                .cloned().collect();
                            let spiking_names: std::collections::HashSet<String> =
                                all_spiking.iter().map(|(n, _)| n.clone()).collect();
                            let bfs_names: std::collections::HashSet<String> =
                                bfs_chain.iter().map(|(n, _)| n.clone()).collect();
                            let predicted_names: std::collections::HashSet<String> =
                                spiking_predicted.iter().map(|(n, _)| n.clone()).collect();

                            // Spiking-primary merge: spiking results are the primary source,
                            // BFS confirms but doesn't dominate. This is the brain-like approach —
                            // knowledge lives in synaptic weights, not in a symbolic database.
                            let mut merged: Vec<(String, usize, &str)> = Vec::new();

                            // Confirmed: found by both spiking and BFS (highest confidence)
                            for (name, spike_w) in spiking_direct.iter().chain(spiking_predicted.iter()) {
                                if bfs_names.contains(name) {
                                    let bfs_w = bfs_chain.iter()
                                        .find(|(n, _)| n == name)
                                        .map(|(_, w)| *w).unwrap_or(0);
                                    let weight = ((*spike_w).max(bfs_w) as f32 * 1.5) as usize;
                                    merged.push((name.clone(), weight, "confirmed"));
                                }
                            }
                            // Predicted: spiking window 2 (chain-following, full weight)
                            for (name, w) in &spiking_predicted {
                                if !bfs_names.contains(name) {
                                    merged.push((name.clone(), *w, "predicted"));
                                }
                            }
                            // Emergent: spiking direct, not in BFS (full weight — these are real neural discoveries)
                            for (name, w) in &spiking_direct {
                                if !bfs_names.contains(name) && !predicted_names.contains(name) {
                                    merged.push((name.clone(), *w, "emergent"));
                                }
                            }
                            // BFS-only: supplement with explicit facts the spiking network missed
                            for (name, w) in &bfs_chain {
                                if !spiking_names.contains(name) {
                                    merged.push((name.clone(), *w, "explicit"));
                                }
                            }

                            merged.sort_by(|a, b| b.1.cmp(&a.1));
                            merged.truncate(12);

                            if !merged.is_empty() {
                                let tagged: Vec<String> = merged.iter()
                                    .map(|(name, strength, tag)| format!("[{tag}] {name} (strength: {strength})"))
                                    .collect();
                                let knowledge_tagged = format!("{} is associated with: {}",
                                    concept, tagged.join(", "));
                                snap.associated_labels.push(format!("KNOWLEDGE: {knowledge_tagged}"));
                            }
                            for (name, _, tag) in &merged {
                                snap.associated_labels.push(format!("[{}] {}", tag, name));
                            }
                        } else {
                            // No spiking results — BFS only (same as before)
                            if !knowledge.is_empty() {
                                snap.associated_labels.push(format!("KNOWLEDGE: {knowledge}"));
                            }
                            snap.associated_labels.extend(bfs_labels);
                        }

                        drop(sb);
                        *snap_clone.lock().unwrap() = snap;
                        tracing::info!("Recall done in {:.1}s", t0.elapsed().as_secs_f32());
                        continue;
                    }

                    // Drain ALL triples from external queue in one batch
                    let triples: Vec<_> = {
                        let mut q = tq_clone.lock().unwrap();
                        q.drain(..).collect()
                    };
                    if !triples.is_empty() {
                        let t0 = std::time::Instant::now();
                        let count = triples.len();
                        let mut sb = sb_clone.lock().unwrap();
                        let mut imprinted = 0;
                        for (triple, topic, seq_idx) in &triples {
                            sb.knowledge.learn_triple_with_topic(triple, topic, *seq_idx);
                            imprinted += sb.imprint_synapses(triple);
                        }
                        // Chain imprint consecutive triples via STDP
                        let chain_count = sb.imprint_chain_stdp(&triples);
                        sb.knowledge.flush();
                        let elapsed = t0.elapsed().as_secs_f32();
                        tracing::info!("Learned {} triples in {:.3}s, imprinted {} synapses, {} chain links",
                            count, elapsed, imprinted, chain_count);
                        drop(sb);
                        continue;
                    }

                    // Check for pending brain work (needs brain lock)
                    let has_pending = {
                        let sb = sb_clone.lock().unwrap();
                        sb.has_pending_work()
                    };
                    if has_pending {
                        let t0 = std::time::Instant::now();
                        let mut sb = sb_clone.lock().unwrap();
                        sb.tick();
                        let snap = sb.get_snapshot().clone();
                        let total_spikes: usize = snap.region_activity.iter().map(|(_, c)| *c).sum();
                        let region_detail: String = snap.region_activity.iter()
                            .filter(|(_, c)| *c > 0)
                            .map(|(n, c)| format!("{}:{}", n, c))
                            .collect::<Vec<_>>()
                            .join(", ");
                        drop(sb);
                        *snap_clone.lock().unwrap() = snap;
                        tracing::info!("Spiking brain tick: {:.1}s, {} spikes [{}]",
                            t0.elapsed().as_secs_f32(), total_spikes, region_detail);
                    }
                }
            });
            tracing::info!("Spiking brain background tick thread started (5s interval)");
        }

        let emotion_table_path = config.project_root
            .join("outputs/cortex/hope_companion/emotion_table.bin");
        let emotion_table = crate::brain_state::load_emotion_table(&emotion_table_path);
        let all_zero = emotion_table.iter().all(|row| row.iter().all(|&x| x == 0.0));
        if all_zero {
            tracing::info!("Emotion table not found — grounded mode uses zero emotion bias");
        } else {
            tracing::info!("Emotion table loaded ({} emotions)", emotion_table.len());
        }

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
            learned_concepts: std::sync::Mutex::new(Vec::new()),
            triple_queue: triple_queue,
            recall_queue,
            codebook: std::sync::Mutex::new(codebook),
            world_model,
            confidence_model,
            temporal_model,
            visual_encoder,
            clip_encoder,
            audio_encoder,
            companion_decoder,
            spiking_brain,
            spiking_snapshot,
            emotion_table,
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
                "companion_decoder": self.companion_decoder.is_some(),
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
