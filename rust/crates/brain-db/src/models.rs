//! Data models matching the Python dataclasses.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: Option<i64>,
    pub config_json: String,
    pub status: String,
    pub hypothesis: String,
    pub final_metrics: String,
    pub parent_id: Option<i64>,
    pub error_msg: String,
    pub code_patch: String,
    pub created_at: f64,
    pub finished_at: f64,
}

impl Default for Experiment {
    fn default() -> Self {
        Self {
            id: None,
            config_json: "{}".to_string(),
            status: "pending".to_string(),
            hypothesis: String::new(),
            final_metrics: "{}".to_string(),
            parent_id: None,
            error_msg: String::new(),
            code_patch: String::new(),
            created_at: 0.0,
            finished_at: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    pub id: Option<i64>,
    pub experiment_id: i64,
    pub step: i64,
    pub metrics_json: String,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: Option<i64>,
    pub timestamp: f64,
    pub subsystem: String,
    pub action: String,
    pub reasoning: String,
    pub context_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInventoryItem {
    pub id: Option<i64>,
    pub dataset: String,
    pub clip_id: String,
    pub path: String,
    pub size_bytes: i64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMutation {
    pub id: Option<i64>,
    pub target_file: String,
    pub target_name: String,
    pub original_code: String,
    pub mutated_code: String,
    pub diff: String,
    pub llm_prompt: String,
    pub llm_response: String,
    pub experiment_id: Option<i64>,
    pub score_delta: f64,
    pub accepted: bool,
    pub created_at: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentCounts {
    pub total: i64,
    pub completed: i64,
    pub running: i64,
    pub failed: i64,
    pub pending: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationStats {
    pub total: i64,
    pub accepted: i64,
    pub acceptance_rate: f64,
    pub by_target: Vec<TargetMutationStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMutationStats {
    pub target_name: String,
    pub cnt: i64,
    pub accepted_cnt: i64,
    pub avg_delta: f64,
}
