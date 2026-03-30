//! Brain configuration — all constants from environment variables.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct BrainConfig {
    pub project_root: PathBuf,
    pub model_dir: PathBuf,
    pub memory_db_path: PathBuf,
    pub audioset_dir: PathBuf,

    // Working memory
    pub wm_slots: usize,
    pub wm_decay: f32,
    pub wm_theta_freq: f32,

    // Neuroscience features
    pub sparse_k: usize,
    pub ach_window: usize,
    pub ach_lr_min: f32,
    pub ach_lr_max: f32,
    pub fast_memory_capacity: usize,
    pub episode_gap_seconds: f64,

    // Grid cells
    pub grid_scales: Vec<f32>,

    // Autonomy
    pub autonomy_interval_secs: u64,

    // LLM
    pub ollama_url: String,
    pub ollama_model: String,
}

impl BrainConfig {
    pub fn from_env() -> Self {
        let root = PathBuf::from(
            std::env::var("BRAIN_PROJECT_ROOT").unwrap_or_else(|_| "/opt/brain".into()),
        );
        Self {
            model_dir: root.join("outputs/cortex/v6_mlp"),
            memory_db_path: root.join("outputs/cortex/brain_memory.db"),
            audioset_dir: root.join("outputs/cortex/audioset_brain"),
            project_root: root,

            wm_slots: env_usize("WM_SLOTS", 7),
            wm_decay: env_f32("WM_DECAY", 0.85),
            wm_theta_freq: env_f32("WM_THETA_FREQ", 0.15),

            sparse_k: env_usize("SPARSE_K", 0),
            ach_window: env_usize("ACH_WINDOW", 50),
            ach_lr_min: env_f32("ACH_LR_MIN", 0.0002),
            ach_lr_max: env_f32("ACH_LR_MAX", 0.005),
            fast_memory_capacity: env_usize("FAST_MEMORY_CAPACITY", 2000),
            episode_gap_seconds: env_f64("EPISODE_GAP_SECONDS", 30.0),

            grid_scales: vec![0.05, 0.15, 0.5],

            autonomy_interval_secs: env_u64("AUTONOMY_INTERVAL", 300),

            ollama_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434/api/generate".into()),
            ollama_model: std::env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "qwen2.5:1.5b".into()),
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
