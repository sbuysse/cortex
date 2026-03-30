//! Parameterized mutation strategies applied at runtime.
//!
//! Instead of compiling .so mutations, the LLM suggests structured JSON
//! that modifies the training loop behavior. This covers the same search
//! space as the Python code_evolver but is safer and faster.

use serde::{Deserialize, Serialize};

/// A complete mutation specification returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    /// Which component this mutation targets.
    pub target: MutationTarget,
    /// Human-readable hypothesis for why this helps.
    pub hypothesis: String,
    /// The actual mutation to apply.
    pub strategy: MutationStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationTarget {
    HebbianUpdate,
    HebbianPrune,
    SparseProjection,
    TemporalTrace,
    TrainingLoop,
}

impl std::fmt::Display for MutationTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HebbianUpdate => write!(f, "HebbianAssociation.update"),
            Self::HebbianPrune => write!(f, "HebbianAssociation.prune"),
            Self::SparseProjection => write!(f, "SparseProjection.forward"),
            Self::TemporalTrace => write!(f, "TemporalTrace.update"),
            Self::TrainingLoop => write!(f, "TrainingLoop"),
        }
    }
}

/// All available mutation strategies, organized by target.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MutationStrategy {
    // === Hebbian Update Mutations ===

    /// Oja's rule: subtract lr * (y @ y^T) @ W to prevent explosion.
    OjasRule { strength: f32 },

    /// BCM sliding threshold: modulate updates by (post - threshold).
    BcmThreshold { initial_threshold: f32, sliding_rate: f32 },

    /// Add weight decay term: M -= lambda * M each step.
    WeightDecay { lambda: f32 },

    /// L2-normalize activations before computing the Hebbian delta.
    NormalizePreActivations,

    /// L2-normalize M rows after each update.
    NormalizePostUpdate,

    /// Scale the learning rate by a factor.
    ScaleLearningRate { factor: f32 },

    /// Adjust anti-Hebbian (contrastive) weight.
    ScaleNegWeight { factor: f32 },

    /// Only update the top-k most active rows of M (competitive learning).
    CompetitiveTopK { k_fraction: f32 },

    /// Clip individual weight magnitudes.
    WeightClip { max_val: f32 },

    /// Add momentum to the Hebbian delta (running average of deltas).
    Momentum { beta: f32 },

    // === Pruning Mutations ===

    /// Scale the pruning threshold.
    ScalePruneThreshold { factor: f32 },

    /// Only prune weights with low consolidation (unconsolidated synapses).
    ConsolidationAwarePrune { consolidation_threshold: f32 },

    /// Prune by percentile rather than absolute threshold.
    PercentilePrune { percentile: f32 },

    /// More frequent or less frequent pruning.
    ScaleConsolidationInterval { factor: f32 },

    // === Sparse Projection Mutations ===

    /// Multiply k (number of active dimensions) by a factor.
    ScaleK { factor: f32 },

    /// Use soft thresholding instead of hard top-k.
    SoftThreshold { temperature: f32 },

    /// Apply L2 normalization before top-k instead of after.
    NormBeforeTopK,

    /// Use top-k by absolute value then apply sign (instead of ReLU).
    AbsTopKWithSign,

    // === Temporal Trace Mutations ===

    /// Scale the temporal decay rate.
    ScaleTemporalDecay { factor: f32 },

    /// Scale the trace blending weight.
    ScaleTraceWeight { factor: f32 },

    // === Training Loop Mutations ===

    /// Scale augmentation noise.
    ScaleAugNoise { factor: f32 },

    /// Adjust initial weight scale.
    ScaleInitScale { factor: f32 },

    /// Use different learning rate schedule.
    LinearSchedule,

    /// Adjust warmup duration.
    ScaleWarmup { factor: f32 },

    /// Composite: apply multiple mutations.
    Composite { mutations: Vec<MutationStrategy> },

    // === NEW PLATEAU-BREAKING Mutations ===

    /// Spectral regularization: penalize dominant singular value to prevent rank collapse.
    /// After each update, blend M toward (M + alpha * I) to spread energy across dims.
    /// Addresses estimated_rank ≈ 1.3 collapse in 512x512 M.
    SpectralRegularization { alpha: f32 },

    /// Label smoothing for InfoNCE: soft targets (positive = 1-eps, negative = eps/(B-1)).
    /// Prevents overconfident matching and encourages richer representations.
    LabelSmoothing { epsilon: f32 },

    /// Gradient accumulation: accumulate Hebbian deltas over N batches before applying.
    /// Produces smoother updates, especially useful with large batch sizes.
    GradientAccumulation { accumulate_steps: usize },

    /// Mixup augmentation: interpolate pairs (v_i, a_i) and (v_j, a_j) with random lambda.
    /// Creates synthetic positive pairs to expand effective training set.
    Mixup { alpha: f32 },

    /// Warmup temperature: start InfoNCE at high temp (broad), decay to target over training.
    /// Prevents early overcommitment to noisy similarity structure.
    WarmupTemperature { start_temp: f32, end_temp: f32 },

    /// EMA (Exponential Moving Average) of M for evaluation.
    /// Training M updates normally, but eval uses shadow_M = beta * shadow_M + (1-beta) * M.
    /// Polyak averaging produces smoother, better-generalizing models.
    EmaEval { beta: f32 },

    /// Diagonal boost: periodically add small identity matrix to M to counteract rank collapse.
    /// Every `interval` steps: M += boost * I.
    DiagonalBoost { boost: f32, interval: usize },

    /// Cyclic learning rate: oscillate LR between bounds instead of monotonic decay.
    /// Can escape local optima that cosine decay converges to.
    CyclicLr { min_factor: f32, max_factor: f32, cycle_steps: usize },

    // === HIGH-IMPACT Architectural Mutations ===

    /// Scale batch size (default 2 → 32/64/128).
    /// Larger batches provide more negatives per step — the single biggest bottleneck.
    ScaleBatchSize { target_size: usize },

    /// InfoNCE contrastive loss replaces anti-Hebbian with proper softmax contrastive.
    /// Requires larger batch size to be effective (>=32).
    InfoNce { temperature: f32 },

    /// Initialize M from cross-correlation V^T @ A / N instead of identity.
    /// Gives a warm start encoding global correlation structure.
    CrossCorrelationInit { blend: f32 },

    /// Hard negative mining: find most confusing negatives instead of random.
    /// Refresh hard negatives every `refresh_interval` steps.
    HardNegativeMining { top_k: usize, refresh_interval: usize },

    /// Scale max training steps (more epochs over data).
    ScaleMaxSteps { target_steps: usize },

    /// Multi-head association: K separate M matrices, averaged for retrieval.
    MultiHead { num_heads: usize },

    /// MLP association: M1 [D, hidden] -> ReLU -> M2 [hidden, D].
    MlpAssociation { hidden_dim: usize },

    /// Low-rank factorization: M = U [D, r] @ V^T [r, D].
    LowRank { rank: usize },

    /// Curriculum learning: start with easy pairs, expand over training.
    CurriculumLearning { start_frac: f32 },

    /// Fused gradient descent InfoNCE: replaces separate Hebbian pos + contrastive neg
    /// with the exact InfoNCE gradient. Dramatically better optimization.
    GradientInfoNce,

    /// Symmetric InfoNCE: train both V→A and A→V directions simultaneously.
    SymmetricInfoNce,

    /// Scale up max_steps for longer training runs.
    ScaleMaxStepsLong { max_steps: usize },

    /// Full-pool evaluation: test against all 24K clips instead of 1K subset.
    FullPoolEval,
}

/// Weighted micro-mutation: favors strategies with historical acceptance.
///
/// Plateau-breakers (weight 24): spectral_reg=4, label_smooth=3, grad_accum=3,
///   mixup=3, warmup_temp=3, ema=3, diag_boost=3, cyclic_lr=2
/// Architecture (weight 18): multi_head=5, mlp=5, low_rank=5, curriculum=3
/// Existing arch (weight 14): batch_size=2, infonce=4, cross_corr=2, hard_neg=3, max_steps=2, infonce2=1
/// Parametric (weight 19): neg_weight=3, temporal_decay=2, trace_weight=2, prune=2,
///   lr=2, init_scale=2, aug_noise=2, consol=2, norm_post=1, pctl=1
/// Total = 75
pub fn random_micro_mutation(rng: &mut impl rand::Rng) -> MutationConfig {
    let choice = rng.random_range(0u32..75);

    let (target, hypothesis, strategy) = match choice {
        // === PLATEAU-BREAKING MUTATIONS (weight 24, range 0..24) ===

        // SpectralRegularization: fight rank collapse (weight 4, range 0..4)
        0..4 => {
            let alpha = 0.0001 + rng.random::<f32>() * 0.005; // 0.0001-0.005
            (MutationTarget::HebbianUpdate, "Spectral regularization to prevent rank-1 collapse", MutationStrategy::SpectralRegularization { alpha })
        }
        // LabelSmoothing (weight 3, range 4..7)
        4..7 => {
            let epsilon = 0.05 + rng.random::<f32>() * 0.2; // 0.05-0.25
            (MutationTarget::HebbianUpdate, "Label smoothing for softer contrastive targets", MutationStrategy::LabelSmoothing { epsilon })
        }
        // GradientAccumulation (weight 3, range 7..10)
        7..10 => {
            let steps = [2, 4, 8, 16];
            let accumulate_steps = steps[rng.random_range(0..steps.len())];
            (MutationTarget::TrainingLoop, "Gradient accumulation for smoother updates", MutationStrategy::GradientAccumulation { accumulate_steps })
        }
        // Mixup (weight 3, range 10..13)
        10..13 => {
            let alpha = 0.1 + rng.random::<f32>() * 0.4; // 0.1-0.5
            (MutationTarget::TrainingLoop, "Mixup augmentation for synthetic positive pairs", MutationStrategy::Mixup { alpha })
        }
        // WarmupTemperature (weight 3, range 13..16)
        13..16 => {
            let start_temp = 0.05 + rng.random::<f32>() * 0.15; // 0.05-0.2
            let end_temp = 0.003 + rng.random::<f32>() * 0.01; // 0.003-0.013
            (MutationTarget::TrainingLoop, "Temperature warmup: broad exploration then sharp focusing", MutationStrategy::WarmupTemperature { start_temp, end_temp })
        }
        // EmaEval (weight 3, range 16..19)
        16..19 => {
            let beta = 0.99 + rng.random::<f32>() * 0.009; // 0.99-0.999
            (MutationTarget::TrainingLoop, "EMA Polyak averaging for smoother evaluation model", MutationStrategy::EmaEval { beta })
        }
        // DiagonalBoost (weight 3, range 19..22)
        19..22 => {
            let boost = 0.0001 + rng.random::<f32>() * 0.002; // 0.0001-0.002
            let intervals = [500, 1000, 2000, 5000];
            let interval = intervals[rng.random_range(0..intervals.len())];
            (MutationTarget::HebbianUpdate, "Diagonal identity boost to counteract rank collapse", MutationStrategy::DiagonalBoost { boost, interval })
        }
        // CyclicLr (weight 2, range 22..24)
        22..24 => {
            let min_factor = 0.1 + rng.random::<f32>() * 0.3; // 0.1-0.4
            let max_factor = 1.5 + rng.random::<f32>() * 1.5; // 1.5-3.0
            let cycles = [5000, 10000, 20000];
            let cycle_steps = cycles[rng.random_range(0..cycles.len())];
            (MutationTarget::TrainingLoop, "Cyclic learning rate to escape local optima", MutationStrategy::CyclicLr { min_factor, max_factor, cycle_steps })
        }

        // === EXISTING ARCHITECTURAL (weight 18, range 24..42) ===

        // MultiHead (weight 5, range 24..29)
        24..29 => {
            let heads = [2, 4, 8];
            let num_heads = heads[rng.random_range(0..heads.len())];
            (MutationTarget::TrainingLoop, "Multi-head association for ensemble diversity", MutationStrategy::MultiHead { num_heads })
        }
        // MLP association (weight 5, range 29..34)
        29..34 => {
            let dims = [64, 128, 256, 512];
            let hidden_dim = dims[rng.random_range(0..dims.len())];
            (MutationTarget::HebbianUpdate, "MLP nonlinear association for increased capacity", MutationStrategy::MlpAssociation { hidden_dim })
        }
        // Low-rank factorization (weight 5, range 34..39)
        34..39 => {
            let ranks = [32, 64, 128, 256];
            let rank = ranks[rng.random_range(0..ranks.len())];
            (MutationTarget::HebbianUpdate, "Low-rank factorization for structured learning", MutationStrategy::LowRank { rank })
        }
        // Curriculum learning (weight 3, range 39..42)
        39..42 => {
            let start_frac = 0.1 + rng.random::<f32>() * 0.3;
            (MutationTarget::TrainingLoop, "Curriculum learning: easy pairs first", MutationStrategy::CurriculumLearning { start_frac })
        }

        // === EXISTING ARCH MUTATIONS (weight 14, range 42..56) ===

        // ScaleBatchSize (weight 2, range 42..44)
        42..44 => {
            let sizes = [128, 256, 512, 1024];
            let target_size = sizes[rng.random_range(0..sizes.len())];
            (MutationTarget::TrainingLoop, "Scale batch size for more negatives per step", MutationStrategy::ScaleBatchSize { target_size })
        }
        // InfoNCE temperature tuning (weight 4, range 44..48)
        44..48 => {
            let temperature = 0.005 + rng.random::<f32>() * 0.06;
            (MutationTarget::HebbianUpdate, "InfoNCE contrastive loss with softmax", MutationStrategy::InfoNce { temperature })
        }
        // CrossCorrelationInit (weight 2, range 48..50)
        48..50 => {
            let blend = 0.3 + rng.random::<f32>() * 0.5;
            (MutationTarget::TrainingLoop, "Cross-correlation init for warm start", MutationStrategy::CrossCorrelationInit { blend })
        }
        // HardNegativeMining (weight 3, range 50..53)
        50..53 => {
            let top_k = 10 + rng.random_range(0usize..91);
            let refresh_interval = 200 + rng.random_range(0usize..600);
            (MutationTarget::HebbianUpdate, "Hard negative mining for focused contrastive signal", MutationStrategy::HardNegativeMining { top_k, refresh_interval })
        }
        // ScaleMaxSteps (weight 2, range 53..55)
        53..55 => {
            let steps = [40000, 60000, 80000, 100000];
            let target_steps = steps[rng.random_range(0..steps.len())];
            (MutationTarget::TrainingLoop, "Scale training steps for more epochs", MutationStrategy::ScaleMaxSteps { target_steps })
        }
        // InfoNCE wider search (weight 1, range 55..56)
        55..56 => {
            let temperature = 0.003 + rng.random::<f32>() * 0.15;
            (MutationTarget::HebbianUpdate, "InfoNCE temperature search", MutationStrategy::InfoNce { temperature })
        }

        // === PARAMETRIC (weight 19, range 56..75) ===

        // scale_neg_weight (weight 3, range 56..59)
        56..59 => {
            let factor = 0.85 + rng.random::<f32>() * 0.4;
            (MutationTarget::HebbianUpdate, "Scale anti-Hebbian weight", MutationStrategy::ScaleNegWeight { factor })
        }
        // scale_temporal_decay (weight 2, range 59..61)
        59..61 => {
            let factor = 0.7 + rng.random::<f32>() * 0.2;
            (MutationTarget::TemporalTrace, "Scale temporal decay", MutationStrategy::ScaleTemporalDecay { factor })
        }
        // scale_trace_weight (weight 2, range 61..63)
        61..63 => {
            let factor = 0.5 + rng.random::<f32>() * 1.0;
            (MutationTarget::TemporalTrace, "Scale trace weight", MutationStrategy::ScaleTraceWeight { factor })
        }
        // scale_prune_threshold (weight 2, range 63..65)
        63..65 => {
            let factor = 0.8 + rng.random::<f32>() * 0.8;
            (MutationTarget::HebbianPrune, "Scale prune threshold", MutationStrategy::ScalePruneThreshold { factor })
        }
        // scale_learning_rate (weight 2, range 65..67)
        65..67 => {
            let factor = 0.8 + rng.random::<f32>() * 0.4;
            (MutationTarget::HebbianUpdate, "Scale learning rate", MutationStrategy::ScaleLearningRate { factor })
        }
        // scale_init_scale (weight 2, range 67..69)
        67..69 => {
            let factor = 0.2 + rng.random::<f32>() * 1.0;
            (MutationTarget::TrainingLoop, "Scale init scale", MutationStrategy::ScaleInitScale { factor })
        }
        // scale_aug_noise (weight 2, range 69..71)
        69..71 => {
            let factor = 0.3 + rng.random::<f32>() * 2.0;
            (MutationTarget::TrainingLoop, "Scale augmentation noise", MutationStrategy::ScaleAugNoise { factor })
        }
        // scale_consolidation_interval (weight 2, range 71..73)
        71..73 => {
            let factor = 0.5 + rng.random::<f32>() * 1.0;
            (MutationTarget::HebbianPrune, "Scale consolidation interval", MutationStrategy::ScaleConsolidationInterval { factor })
        }
        // normalize_post_update (weight 1, range 73..74)
        73..74 => {
            (MutationTarget::HebbianUpdate, "L2-normalize M rows after update", MutationStrategy::NormalizePostUpdate)
        }
        // percentile_prune (weight 1, range 74..75)
        _ => {
            let percentile = 5.0 + rng.random::<f32>() * 20.0;
            (MutationTarget::HebbianPrune, "Percentile-based pruning", MutationStrategy::PercentilePrune { percentile })
        }
    };

    MutationConfig {
        target,
        hypothesis: hypothesis.to_string(),
        strategy,
    }
}

/// Generate the diff description of a mutation for DB storage.
pub fn mutation_diff(config: &MutationConfig) -> String {
    format!(
        "Target: {}\nStrategy: {}\nHypothesis: {}",
        config.target,
        serde_json::to_string_pretty(&config.strategy).unwrap_or_default(),
        config.hypothesis,
    )
}
