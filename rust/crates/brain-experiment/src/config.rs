//! Experiment configuration.

use serde::{Deserialize, Serialize};

/// Full experiment configuration matching the Python config JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,
    #[serde(default = "default_sparsity_k")]
    pub sparsity_k: usize,
    #[serde(default = "default_hebbian_lr")]
    pub hebbian_lr: f32,
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f32,
    #[serde(default = "default_temporal_decay")]
    pub temporal_decay: f32,
    #[serde(default = "default_trace_weight")]
    pub trace_weight: f32,
    #[serde(default = "default_max_norm")]
    pub max_norm: f32,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_init_scale")]
    pub init_scale: f32,
    #[serde(default = "default_neg_weight")]
    pub neg_weight: f32,
    #[serde(default = "default_aug_noise")]
    pub aug_noise: f32,
    #[serde(default = "default_log_interval")]
    pub log_interval: usize,
    #[serde(default)]
    pub eval_interval: Option<usize>,
    #[serde(default)]
    pub early_stop_patience: usize,
    #[serde(default = "default_consolidation_interval")]
    pub consolidation_interval: usize,
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: f32,
    /// If true, L2-normalize M rows after each Hebbian update.
    /// Works well with loose max_norm (100+), conflicts with tight max_norm (<30).
    #[serde(default)]
    pub normalize_post_update: bool,
    /// If Some, prune by percentile instead of absolute threshold.
    #[serde(default)]
    pub percentile_prune: Option<f32>,
    /// Warmup steps factor (multiplied by 100).
    #[serde(default = "default_warmup_factor")]
    pub warmup_factor: f32,
    /// If true, use InfoNCE contrastive loss instead of anti-Hebbian.
    #[serde(default)]
    pub use_infonce: bool,
    /// Temperature for InfoNCE softmax (lower = sharper).
    #[serde(default = "default_infonce_temperature")]
    pub infonce_temperature: f32,
    /// If true, initialize M from cross-correlation of embeddings.
    #[serde(default)]
    pub cross_correlation_init: bool,
    /// Blend factor for cross-correlation init (1.0 = pure cross-corr, 0.0 = identity).
    #[serde(default = "default_cc_blend")]
    pub cross_correlation_blend: f32,
    /// If >0, use hard negative mining with this many negatives.
    #[serde(default)]
    pub hard_negative_k: usize,
    /// Refresh interval for hard negative mining.
    #[serde(default = "default_hard_neg_refresh")]
    pub hard_negative_refresh: usize,
    /// Number of association heads (1 = standard, >1 = multi-head average).
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    /// MLP hidden dimension (0 = disabled, >0 = two-layer MLP association).
    #[serde(default)]
    pub mlp_hidden: usize,
    /// Low-rank factorization rank (0 = full rank, >0 = M = U @ V^T).
    #[serde(default)]
    pub low_rank: usize,
    /// Enable curriculum learning (start with easy pairs, expand).
    #[serde(default)]
    pub curriculum: bool,
    /// Fraction of data to start with in curriculum learning.
    #[serde(default = "default_curriculum_start")]
    pub curriculum_start_frac: f32,
    /// If true, use SigLIP contrastive loss (sigmoid pairwise, no softmax).
    /// Works much better than InfoNCE with small-medium batch sizes.
    #[serde(default)]
    pub use_siglip: bool,
    /// Learnable bias for SigLIP (initialized from -ln(batch_size)).
    #[serde(default)]
    pub siglip_bias: f32,
    /// Spectral regularization alpha (0 = disabled). Adds alpha*I to M periodically.
    #[serde(default)]
    pub spectral_reg_alpha: f32,
    /// Label smoothing epsilon for InfoNCE (0 = hard labels).
    #[serde(default)]
    pub label_smoothing: f32,
    /// Gradient accumulation steps (1 = no accumulation).
    #[serde(default = "default_grad_accum")]
    pub grad_accumulation: usize,
    /// Mixup alpha (0 = disabled). Controls Beta distribution shape.
    #[serde(default)]
    pub mixup_alpha: f32,
    /// Warmup temperature for InfoNCE (0 = use fixed infonce_temperature).
    #[serde(default)]
    pub warmup_temp_start: f32,
    /// End temperature for warmup schedule.
    #[serde(default)]
    pub warmup_temp_end: f32,
    /// EMA beta for Polyak averaging (0 = disabled).
    #[serde(default)]
    pub ema_beta: f32,
    /// Diagonal boost magnitude (0 = disabled).
    #[serde(default)]
    pub diagonal_boost: f32,
    /// Diagonal boost interval in steps.
    #[serde(default = "default_diag_boost_interval")]
    pub diagonal_boost_interval: usize,
    /// Cyclic LR min factor (0 = disabled, use cosine decay).
    #[serde(default)]
    pub cyclic_lr_min: f32,
    /// Cyclic LR max factor.
    #[serde(default)]
    pub cyclic_lr_max: f32,
    /// Cyclic LR period in steps.
    #[serde(default)]
    pub cyclic_lr_period: usize,
    /// If true, use proper gradient descent for InfoNCE instead of separate
    /// Hebbian positive + contrastive negative updates. This computes the
    /// fused gradient dM = (lr/(B*temp)) * V^T @ (I - softmax(V M A^T / temp)) @ A
    /// and applies it directly to M, bypassing metaplasticity and temporal traces.
    #[serde(default)]
    pub gradient_infonce: bool,
    /// If true (default when gradient_infonce is active), also train the reverse
    /// direction A→V with InfoNCE. The total gradient becomes:
    ///   dM = scale * V^T @ (I - P_v2a) @ A  +  scale * V^T @ (I - P_a2v)^T @ A
    /// where P_a2v = softmax(A @ M^T @ V^T / temp).
    #[serde(default = "default_symmetric_infonce")]
    pub symmetric_infonce: bool,
    /// Evaluation pool size. 0 means use all clips. Default: 1000.
    #[serde(default = "default_eval_pool_size")]
    pub eval_pool_size: usize,
    /// Output directory for saving trained model weights.
    #[serde(default)]
    pub output_dir: Option<String>,
    /// Skip SparseProjection — use raw embeddings with non-square M.
    #[serde(default)]
    pub skip_projection: bool,
    /// Use ZCA-whitened embeddings instead of raw/projected.
    #[serde(default)]
    pub use_whitening: bool,
    /// Use multi-positive InfoNCE (clips with same category label are soft positives).
    #[serde(default)]
    pub multi_positive: bool,
    /// Number of ensemble projections (1 = no ensemble).
    #[serde(default = "default_ensemble_count")]
    pub ensemble_count: usize,
    /// Use quadratic scoring: v^T M₁ a + (v⊙v)^T M₂ (a⊙a).
    #[serde(default)]
    pub quadratic_scoring: bool,
}

fn default_embed_dim() -> usize { 512 }
fn default_sparsity_k() -> usize { 36 }
fn default_hebbian_lr() -> f32 { 0.00035 }
fn default_decay_rate() -> f32 { 0.992 }
fn default_temporal_decay() -> f32 { 0.86 }
fn default_trace_weight() -> f32 { 0.26 }
fn default_max_norm() -> f32 { 100.0 }
fn default_max_steps() -> usize { 20000 }
fn default_batch_size() -> usize { 2 }
fn default_seed() -> u64 { 42 }
fn default_init_scale() -> f32 { 0.01 }
fn default_neg_weight() -> f32 { 0.5 }
fn default_aug_noise() -> f32 { 0.02 }
fn default_log_interval() -> usize { 50 }
fn default_consolidation_interval() -> usize { 500 }
fn default_prune_threshold() -> f32 { 0.001 }
fn default_warmup_factor() -> f32 { 1.0 }
fn default_infonce_temperature() -> f32 { 0.07 }
fn default_cc_blend() -> f32 { 0.5 }
fn default_hard_neg_refresh() -> usize { 500 }
fn default_num_heads() -> usize { 1 }
fn default_curriculum_start() -> f32 { 0.2 }
fn default_grad_accum() -> usize { 1 }
fn default_diag_boost_interval() -> usize { 1000 }
fn default_symmetric_infonce() -> bool { true }
fn default_eval_pool_size() -> usize { 1000 }
fn default_ensemble_count() -> usize { 1 }

impl Default for ExperimentConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}
