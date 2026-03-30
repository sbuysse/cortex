//! Application state shared across all request handlers.

use brain_db::KnowledgeBase;
use brain_experiment::runner::MetricsEvent;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::sync::broadcast;

use crate::resources::{ResourceManager, ResourceSnapshot};

/// Projected embeddings + clip labels for interactive retrieval.
pub struct InteractState {
    /// Projected visual embeddings [N, 512]
    pub v: Array2<f32>,
    /// Projected audio embeddings [N, 512]
    pub a: Array2<f32>,
    /// Projected emotion embeddings [N, 512]
    pub e: Array2<f32>,
    /// Projected speech embeddings [N, 512] (optional)
    pub s: Option<Array2<f32>>,
    /// Projected property embeddings [N, 512] (optional)
    pub p: Option<Array2<f32>>,
    /// Clip labels (index → label string)
    pub labels: Vec<String>,
    /// Number of clips
    pub n_clips: usize,
    /// Unique labels sorted
    pub unique_labels: Vec<String>,
    /// Label → list of clip indices
    pub label_index: HashMap<String, Vec<usize>>,
}

impl InteractState {
    pub fn new(
        v: Array2<f32>,
        a: Array2<f32>,
        e: Array2<f32>,
        s: Option<Array2<f32>>,
        p: Option<Array2<f32>>,
        labels: Vec<String>,
    ) -> Self {
        let n_clips = v.nrows();
        let mut label_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            label_index.entry(label.clone()).or_default().push(i);
        }
        let mut unique_labels: Vec<String> = label_index.keys().cloned().collect();
        unique_labels.sort();
        Self { v, a, e, s, p, labels, n_clips, unique_labels, label_index }
    }

    /// Get the embedding matrix for a modality by name.
    pub fn get_modality(&self, name: &str) -> Option<&Array2<f32>> {
        match name {
            "v" | "vision" => Some(&self.v),
            "a" | "audio" => Some(&self.a),
            "e" | "emotion" => Some(&self.e),
            "s" | "speech" => self.s.as_ref(),
            "p" | "properties" => self.p.as_ref(),
            _ => None,
        }
    }

    /// List available modalities.
    pub fn available_modalities(&self) -> Vec<&'static str> {
        let mut mods = vec!["vision", "audio", "emotion"];
        if self.s.is_some() { mods.push("speech"); }
        if self.p.is_some() { mods.push("properties"); }
        mods
    }
}

/// Pre-computed brain visualization data.
pub struct BrainViz {
    /// 2D PCA coordinates for sample clips [N_sample, 2] per modality
    pub pca_v: Vec<[f32; 2]>,
    pub pca_a: Vec<[f32; 2]>,
    /// Labels for PCA sample clips
    pub pca_labels: Vec<String>,
    /// Category → color index for the scatter plot
    pub category_colors: HashMap<String, u8>,
    /// Category indices for each sample clip
    pub pca_category_idx: Vec<u8>,
    /// 32x32 downsampled cross-correlation heatmaps per modality pair
    pub cross_corr: HashMap<String, Vec<Vec<f32>>>,
    /// Weight distribution histograms (50 bins) per modality pair
    pub weight_histograms: HashMap<String, Vec<u32>>,
    /// Histogram bin edges
    pub hist_min: f32,
    pub hist_max: f32,
    /// Per-clip similarity: for a random sample, the similarity to its true match
    pub sample_sims: Vec<f32>,
    /// Total clips
    pub n_clips: usize,
}

/// Shared application state.
pub struct AppState {
    pub db: Arc<KnowledgeBase>,
    pub templates: tera::Tera,
    pub live_tx: broadcast::Sender<MetricsEvent>,
    pub resource_mgr: Arc<RwLock<ResourceManager>>,
    pub project_root: PathBuf,
    pub output_dir: PathBuf,
    pub interact: Option<Arc<InteractState>>,
    pub brain_viz: Option<Arc<BrainViz>>,
    /// Cognitive architecture state (Rust-native brain)
    pub brain: Option<Arc<brain_cognition::BrainState>>,
}

impl AppState {
    pub fn new(
        db: KnowledgeBase,
        templates_dir: &std::path::Path,
        project_root: PathBuf,
        output_dir: PathBuf,
        interact: Option<Arc<InteractState>>,
        brain_viz: Option<Arc<BrainViz>>,
    ) -> Self {
        let mut tera = tera::Tera::new(
            &templates_dir.join("**/*.html").to_string_lossy(),
        )
        .expect("Failed to load templates");

        // Register custom filters
        register_filters(&mut tera);

        let (live_tx, _) = broadcast::channel(256);

        // Initialize BrainState (cognitive architecture)
        let brain_config = brain_cognition::BrainConfig::from_env();
        let brain = match brain_cognition::BrainState::new(brain_config) {
            Ok(bs) => {
                tracing::info!("BrainState initialized (Rust-native cognition)");
                Some(Arc::new(bs))
            }
            Err(e) => {
                tracing::warn!("BrainState init failed: {e} — cognitive endpoints will proxy to Python");
                None
            }
        };

        Self {
            db: Arc::new(db),
            templates: tera,
            live_tx,
            resource_mgr: Arc::new(RwLock::new(ResourceManager::new())),
            project_root,
            output_dir,
            interact,
            brain_viz,
            brain,
        }
    }

    pub fn get_resources(&self) -> ResourceSnapshot {
        self.resource_mgr
            .write()
            .unwrap()
            .get_snapshot()
    }
}

fn register_filters(tera: &mut tera::Tera) {
    tera.register_filter(
        "round4",
        |value: &tera::Value, _: &std::collections::HashMap<String, tera::Value>| {
            match value {
                tera::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        Ok(tera::Value::String(format!("{:.4}", f)))
                    } else {
                        Ok(tera::Value::String("—".to_string()))
                    }
                }
                _ => Ok(tera::Value::String("—".to_string())),
            }
        },
    );

    tera.register_filter(
        "round2",
        |value: &tera::Value, _: &std::collections::HashMap<String, tera::Value>| {
            match value {
                tera::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        Ok(tera::Value::String(format!("{:.2}", f)))
                    } else {
                        Ok(tera::Value::String("—".to_string()))
                    }
                }
                _ => Ok(tera::Value::String("—".to_string())),
            }
        },
    );

    tera.register_filter(
        "pct",
        |value: &tera::Value, _: &std::collections::HashMap<String, tera::Value>| {
            match value {
                tera::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        Ok(tera::Value::String(format!("{:.1}%", f * 100.0)))
                    } else {
                        Ok(tera::Value::String("—".to_string()))
                    }
                }
                _ => Ok(tera::Value::String("—".to_string())),
            }
        },
    );
}
