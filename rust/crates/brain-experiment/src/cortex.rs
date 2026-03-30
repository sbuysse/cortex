//! Cortex — autonomous training daemon.
//!
//! Continuously proposes mutations via Ollama, runs paired experiments
//! (baseline vs mutated), and accepts improvements to the knowledge base.

use brain_core::sparse_projection::SparseProjection;
use brain_db::KnowledgeBase;
use ndarray::Array2;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::config::ExperimentConfig;
use crate::embed_cache::CachedEmbeddings;
use crate::mutations::{self, MutationConfig, MutationStrategy, MutationTarget};
use crate::mutated_runner::run_mutated_experiment;
use crate::ollama::OllamaClient;
use crate::runner::{run_experiment, MetricsEvent};

/// Acceptance threshold: mutation must improve v2a_MRR + v2a_R@1 by this much.
/// Most parametric mutations have deltas 0.0001-0.0005, so 0.001 was too aggressive.
/// Lowered to 0.0002 to capture real signal above noise (noise floor is ~0.0001).
const ACCEPT_THRESHOLD: f64 = 0.0002;

/// Number of mutations to propose per cycle.
const MUTATIONS_PER_CYCLE: usize = 8;

/// Fraction of mutations that are micro (no LLM).
/// Micro mutations are fast and weighted toward proven strategies.
const MICRO_FRACTION: f32 = 0.5;

/// Cortex configuration.
#[derive(Debug, Clone)]
pub struct CortexConfig {
    pub max_steps: usize,
    pub max_cycles: Option<usize>,
    pub concurrent_experiments: usize,
    /// Path to sweep queue JSON. If present, cortex runs sweep mutations before normal mode.
    pub sweep_file: Option<String>,
    /// Steps to use for sweep screening (default: 20000).
    pub sweep_screen_steps: usize,
}

impl Default for CortexConfig {
    fn default() -> Self {
        Self {
            max_steps: 20000,
            max_cycles: None,
            concurrent_experiments: 4,
            sweep_file: None,
            sweep_screen_steps: 20000,
        }
    }
}

/// Projected embeddings for all modalities.
pub struct ProjectedEmbeddings {
    pub v: Array2<f32>,
    pub a: Array2<f32>,
    pub e: Array2<f32>,
    pub s: Option<Array2<f32>>,
    pub p: Option<Array2<f32>>,
}

/// Project raw encoder embeddings through SparseProjection to shared dim.
pub fn project_embeddings(
    raw: &CachedEmbeddings,
    embed_dim: usize,
    sparsity_k: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let projected = project_all_embeddings(raw, embed_dim, sparsity_k);
    (projected.v, projected.a, projected.e)
}

/// Project all modalities (including optional speech and properties).
pub fn project_all_embeddings(
    raw: &CachedEmbeddings,
    embed_dim: usize,
    sparsity_k: usize,
) -> ProjectedEmbeddings {
    let d_visual = raw.v_emb.ncols();
    let d_audio = raw.a_emb.ncols();
    let d_emotion = raw.e_emb.ncols();

    tracing::info!(d_visual, d_audio, d_emotion, embed_dim, sparsity_k, "Projecting embeddings");

    // Fixed seeds ensure identical projections across server restarts
    let proj_v = SparseProjection::new_seeded(d_visual, embed_dim, sparsity_k, 100);
    let proj_a = SparseProjection::new_seeded(d_audio, embed_dim, sparsity_k, 200);
    let proj_e = SparseProjection::new_seeded(d_emotion, embed_dim, sparsity_k, 300);

    let v = proj_v.forward(&raw.v_emb);
    let a = proj_a.forward(&raw.a_emb);
    let e = proj_e.forward(&raw.e_emb);

    let s = raw.s_emb.as_ref().map(|s_raw| {
        let d_speech = s_raw.ncols();
        tracing::info!(d_speech, "Projecting speech embeddings");
        let proj_s = SparseProjection::new_seeded(d_speech, embed_dim, sparsity_k, 400);
        proj_s.forward(s_raw)
    });

    let p = raw.p_emb.as_ref().map(|p_raw| {
        let d_props = p_raw.ncols();
        tracing::info!(d_props, "Projecting property embeddings");
        let proj_p = SparseProjection::new_seeded(d_props, embed_dim, sparsity_k, 500);
        proj_p.forward(p_raw)
    });

    tracing::info!(
        v_shape = ?v.dim(), a_shape = ?a.dim(), e_shape = ?e.dim(),
        has_speech = s.is_some(), has_props = p.is_some(),
        "Embeddings projected"
    );

    ProjectedEmbeddings { v, a, e, s, p }
}

/// Optional whitened embeddings for the v2 pipeline (skip_projection + use_whitening).
/// When present, the cortex uses these instead of SparseProjection-projected embeddings
/// for experiments with skip_projection=true.
pub struct WhitenedEmbeddings {
    pub v: Arc<Array2<f32>>,  // [N, 384]
    pub a: Arc<Array2<f32>>,  // [N, 512]
}

/// Run the cortex autonomous loop.
///
/// This is the main entry point — call from a tokio::spawn.
pub async fn run_cortex(
    db: Arc<KnowledgeBase>,
    cached_v: Arc<Array2<f32>>,
    cached_a: Arc<Array2<f32>>,
    cached_e: Arc<Array2<f32>>,
    cached_s: Option<Arc<Array2<f32>>>,
    cached_p: Option<Arc<Array2<f32>>>,
    whitened: Option<WhitenedEmbeddings>,
    live_tx: broadcast::Sender<MetricsEvent>,
    shutdown: Arc<AtomicBool>,
    cortex_config: CortexConfig,
) {
    let ollama = OllamaClient::default();
    let mut cycle = 0u64;

    tracing::info!("Cortex starting autonomous loop");

    // Log startup decision
    let _ = db.log_decision(
        "cortex",
        "startup",
        "Starting Rust autonomous training loop",
        Some(&serde_json::json!({
            "max_steps": cortex_config.max_steps,
            "concurrent": cortex_config.concurrent_experiments,
        })),
    );

    loop {
        if shutdown.load(Ordering::Relaxed) {
            tracing::info!("Cortex received shutdown signal");
            break;
        }

        if let Some(max) = cortex_config.max_cycles {
            if cycle >= max as u64 {
                tracing::info!("Cortex completed {max} cycles");
                break;
            }
        }

        cycle += 1;
        tracing::info!(cycle, "Starting cortex cycle");

        match run_cycle(
            &db,
            &ollama,
            &cached_v,
            &cached_a,
            &cached_e,
            &cached_s,
            &cached_p,
            &whitened,
            &live_tx,
            shutdown.clone(),
            &cortex_config,
            cycle,
        )
        .await
        {
            Ok(accepted) => {
                tracing::info!(cycle, accepted, "Cycle complete");
            }
            Err(e) => {
                tracing::error!(cycle, error = %e, "Cycle failed");
                let _ = db.log_decision(
                    "cortex",
                    "cycle_error",
                    &format!("Cycle {cycle} failed: {e}"),
                    None,
                );
                // Wait before retrying
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
        }
    }

    let _ = db.log_decision("cortex", "shutdown", "Cortex loop exiting", None);
    tracing::info!("Cortex shutdown complete");
}

async fn run_cycle(
    db: &Arc<KnowledgeBase>,
    ollama: &OllamaClient,
    cached_v: &Arc<Array2<f32>>,
    cached_a: &Arc<Array2<f32>>,
    cached_e: &Arc<Array2<f32>>,
    cached_s: &Option<Arc<Array2<f32>>>,
    cached_p: &Option<Arc<Array2<f32>>>,
    whitened: &Option<WhitenedEmbeddings>,
    live_tx: &broadcast::Sender<MetricsEvent>,
    shutdown: Arc<AtomicBool>,
    cortex_config: &CortexConfig,
    cycle: u64,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    // Get best config from DB or use defaults (shared across all tasks via Arc)
    let mut base_config_inner = get_best_config(db)?;

    // 1. Propose mutations — check sweep queue first
    let mutations = if let Some(ref sweep_path) = cortex_config.sweep_file {
        let sweep_mutations = load_sweep_batch(sweep_path, cycle);
        if !sweep_mutations.is_empty() {
            // Use reduced steps for sweep screening
            base_config_inner.max_steps = cortex_config.sweep_screen_steps;
            tracing::info!(
                count = sweep_mutations.len(),
                screen_steps = cortex_config.sweep_screen_steps,
                "Running sweep batch"
            );
            sweep_mutations
        } else {
            propose_mutations(db, ollama, cycle).await
        }
    } else {
        propose_mutations(db, ollama, cycle).await
    };
    let base_config = Arc::new(base_config_inner);
    if mutations.is_empty() {
        tracing::warn!("No mutations proposed this cycle");
        return Ok(0);
    }

    tracing::info!(count = mutations.len(), "Proposed mutations");

    // 2. Run paired experiments for all mutations concurrently (bounded by semaphore)
    let semaphore = Arc::new(tokio::sync::Semaphore::new(cortex_config.concurrent_experiments));

    // Prepare all mutation tasks, then spawn concurrently
    let mut handles = Vec::with_capacity(mutations.len());

    for (i, mutation) in mutations.into_iter().enumerate() {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let mutation_diff = mutations::mutation_diff(&mutation);
        let target_name = mutation.target.to_string();

        tracing::info!(
            idx = i,
            target = %mutation.target,
            hypothesis = %mutation.hypothesis,
            "Evaluating mutation"
        );

        // Create DB records (cheap, sequential)
        // Baseline uses base config; mutated uses effective config after mutation
        let base_json = serde_json::to_value(base_config.as_ref())?;
        let mutated_cfg = crate::mutated_runner::compute_mutated_config(base_config.as_ref(), &mutation.strategy);
        let mutated_json = serde_json::to_value(&mutated_cfg)?;
        let baseline_id = db.create_experiment(
            &base_json,
            &format!("Baseline for cycle {cycle} mutation {i}"),
            None,
            "",
        )?;
        let mutated_id = db.create_experiment(
            &mutated_json,
            &mutation.hypothesis,
            Some(baseline_id),
            &mutation_diff,
        )?;

        let mutation_record = brain_db::models::CodeMutation {
            id: None,
            target_file: String::new(),
            target_name: target_name.clone(),
            original_code: String::new(),
            mutated_code: serde_json::to_string_pretty(&mutation.strategy).unwrap_or_default(),
            diff: mutation_diff.clone(),
            llm_prompt: String::new(),
            llm_response: mutation.hypothesis.clone(),
            experiment_id: Some(mutated_id),
            score_delta: 0.0,
            accepted: false,
            created_at: 0.0,
        };
        let mutation_db_id = db.add_code_mutation(&mutation_record)?;

        db.update_experiment_status(baseline_id, "running", None, "")?;
        db.update_experiment_status(mutated_id, "running", None, "")?;

        // Spawn concurrent evaluation task (Arc clones are cheap — no deep copy)
        // Choose embeddings: if skip_projection + whitened available, use whitened (v2 pipeline)
        let use_v2 = base_config.skip_projection && whitened.is_some();
        let sem = semaphore.clone();
        let db_arc = db.clone();
        let bl_config = base_config.clone();  // Arc clone
        let mut_config = base_config.clone(); // Arc clone
        let bl_v = if use_v2 { whitened.as_ref().unwrap().v.clone() } else { cached_v.clone() };
        let bl_a = if use_v2 { whitened.as_ref().unwrap().a.clone() } else { cached_a.clone() };
        let bl_e = cached_e.clone();
        let bl_s = cached_s.clone();
        let bl_p = cached_p.clone();
        let mut_v = if use_v2 { whitened.as_ref().unwrap().v.clone() } else { cached_v.clone() };
        let mut_a = if use_v2 { whitened.as_ref().unwrap().a.clone() } else { cached_a.clone() };
        let mut_e = cached_e.clone();
        let mut_s = cached_s.clone();
        let mut_p = cached_p.clone();
        let bl_tx = live_tx.clone();
        let mut_tx = live_tx.clone();
        let shutdown_bl = shutdown.clone();
        let shutdown_mut = shutdown.clone();
        let mutation_clone = mutation.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.ok()?;

            let bl_handle = tokio::task::spawn_blocking(move || {
                run_experiment(
                    bl_config.as_ref(), &bl_v, &bl_a, &bl_e,
                    bl_s.as_deref(), bl_p.as_deref(),
                    baseline_id, shutdown_bl, Some(&bl_tx),
                )
            });
            let mut_handle = tokio::task::spawn_blocking(move || {
                run_mutated_experiment(
                    mut_config.as_ref(), &mutation_clone, &mut_v, &mut_a, &mut_e,
                    mut_s.as_deref(), mut_p.as_deref(),
                    mutated_id, shutdown_mut, Some(&mut_tx),
                )
            });

            let (baseline_result, mutated_result) = tokio::try_join!(bl_handle, mut_handle).ok()?;

            // Update DB with results
            let bl_metrics_json = serde_json::to_value(&baseline_result.metrics).ok()?;
            let mut_metrics_json = serde_json::to_value(&mutated_result.metrics).ok()?;

            db_arc.update_experiment_status(baseline_id, "completed", Some(&bl_metrics_json), "").ok()?;
            db_arc.update_experiment_status(mutated_id, "completed", Some(&mut_metrics_json), "").ok()?;

            let bl_score = composite_score(&baseline_result.metrics);
            let mut_score = composite_score(&mutated_result.metrics);
            let delta = mut_score - bl_score;
            let is_accepted = delta >= ACCEPT_THRESHOLD;

            db_arc.update_mutation_result(mutation_db_id, mutated_id, delta, is_accepted).ok()?;

            let bl_r1 = baseline_result.metrics.get("v2a_R@1").copied().unwrap_or(0.0);
            let mut_r1 = mutated_result.metrics.get("v2a_R@1").copied().unwrap_or(0.0);

            let decision = if is_accepted { "accepted" } else { "rejected" };
            tracing::info!(
                target = %target_name,
                bl_r1 = format!("{:.4}", bl_r1),
                mut_r1 = format!("{:.4}", mut_r1),
                delta = format!("{:.6}", delta),
                decision,
                "Mutation evaluation complete"
            );

            let _ = db_arc.log_decision(
                "cortex",
                &format!("mutation_{decision}"),
                &format!(
                    "{}: {} (Δ={:.6}, bl_R@1={:.4}, mut_R@1={:.4})",
                    target_name, mutation.hypothesis, delta, bl_r1, mut_r1
                ),
                Some(&serde_json::json!({
                    "mutation_id": mutation_db_id,
                    "baseline_id": baseline_id,
                    "mutated_id": mutated_id,
                    "delta": delta,
                    "baseline_r1": bl_r1,
                    "mutated_r1": mut_r1,
                })),
            );

            Some(is_accepted)
        });

        handles.push(handle);
    }

    // Await all concurrent evaluations
    let mut accepted = 0usize;
    for handle in handles {
        if let Ok(Some(is_accepted)) = handle.await {
            if is_accepted {
                accepted += 1;
            }
        }
    }

    Ok(accepted)
}

/// Propose a batch of mutations (mix of micro + LLM).
///
/// Micro mutations use weighted sampling favoring proven strategies.
/// LLM mutations are deduplicated — if the LLM proposes the same strategy
/// type more than once, extras are replaced with micro mutations.
async fn propose_mutations(
    db: &Arc<KnowledgeBase>,
    ollama: &OllamaClient,
    cycle: u64,
) -> Vec<MutationConfig> {
    let mut mutations = Vec::new();
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::from_os_rng();

    let n_micro = (MUTATIONS_PER_CYCLE as f32 * MICRO_FRACTION).ceil() as usize;
    let n_llm = MUTATIONS_PER_CYCLE - n_micro;

    // Micro mutations (fast, weighted toward proven strategies)
    for _ in 0..n_micro {
        mutations.push(mutations::random_micro_mutation(&mut rng));
    }

    // LLM mutations with deduplication
    if ollama.is_available().await {
        let mut seen_strategies = std::collections::HashSet::new();
        let temps = [0.4, 0.7, 0.9, 0.5, 1.0, 0.3, 0.8, 1.2];

        for i in 0..n_llm {
            let temp = temps[i % temps.len()];

            match propose_llm_mutation(db, ollama, temp, cycle).await {
                Ok(m) => {
                    // Dedup: check strategy type
                    let strat_key = format!("{:?}", std::mem::discriminant(&m.strategy));
                    if seen_strategies.insert(strat_key) {
                        mutations.push(m);
                    } else {
                        tracing::info!("LLM proposed duplicate strategy, replacing with micro");
                        mutations.push(mutations::random_micro_mutation(&mut rng));
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LLM mutation proposal failed");
                    mutations.push(mutations::random_micro_mutation(&mut rng));
                }
            }
        }
    } else {
        tracing::warn!("Ollama not available, using only micro mutations");
        for _ in 0..n_llm {
            mutations.push(mutations::random_micro_mutation(&mut rng));
        }
    }

    mutations
}

/// Propose a mutation via LLM.
async fn propose_llm_mutation(
    db: &Arc<KnowledgeBase>,
    ollama: &OllamaClient,
    temperature: f32,
    cycle: u64,
) -> Result<MutationConfig, Box<dyn std::error::Error + Send + Sync>> {
    // Pick target weighted by post-deploy acceptance rates (30min, 22.3% overall).
    // SparseProjection removed (0/124). HebbianUpdate dominates (neg_weight 38.5%).
    let weighted_targets = [
        (MutationTarget::HebbianUpdate, 5),      // neg_weight 38.5%, lr 27.3%
        (MutationTarget::HebbianPrune, 4),       // consol_interval 50%, prune_thresh 17.2%
        (MutationTarget::TemporalTrace, 3),      // temporal_decay 19.5%, trace_weight 19.2%
        (MutationTarget::TrainingLoop, 2),       // init_scale 12.8%, aug_noise 11.1%
    ];
    let total_w: usize = weighted_targets.iter().map(|(_, w)| *w).sum();
    let mut pick = (cycle as usize) % total_w;
    let mut target = MutationTarget::TrainingLoop;
    for (t, w) in &weighted_targets {
        if pick < *w {
            target = *t;
            break;
        }
        pick -= w;
    }

    // Get recent mutation history for context
    let recent = db.get_mutations_for_target(Some(&target.to_string()), 10)?;
    let mut history = String::new();
    for m in &recent {
        let status = if m.accepted { "ACCEPTED" } else { "rejected" };
        history.push_str(&format!(
            "- [{status}] delta={:.6}: {}\n",
            m.score_delta,
            m.llm_response.chars().take(100).collect::<String>()
        ));
    }

    // Get best performance
    let best_config = db.get_best_config("v2a_R@1").unwrap_or_default();
    let best_r1 = best_config.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let best_mrr = best_config.get("v2a_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let prompt = build_llm_prompt(target, &history, best_r1, best_mrr);
    let response = ollama.generate(&prompt, temperature).await?;

    // Parse JSON from response
    parse_mutation_response(&response, target)
}

fn build_llm_prompt(
    target: MutationTarget,
    history: &str,
    best_r1: f64,
    best_mrr: f64,
) -> String {
    // Only show strategies that have non-zero acceptance or are unexplored.
    // Dead strategies (0% after many tries) are excluded from the prompt.
    let strategies = match target {
        MutationTarget::HebbianUpdate => r#"Available strategies (pick ONE and fill in parameters):
- {"type": "spectral_regularization", "alpha": 0.001} — add alpha*I to M every 10 steps to prevent rank collapse (NEW, try 0.0001-0.005)
- {"type": "label_smoothing", "epsilon": 0.1} — soft InfoNCE targets to prevent overconfident matching (NEW, try 0.05-0.25)
- {"type": "diagonal_boost", "boost": 0.001, "interval": 1000} — periodic identity addition to fight rank collapse (NEW)
- {"type": "scale_neg_weight", "factor": 1.1} — adjust anti-Hebbian weight (try 0.85-1.25)
- {"type": "scale_learning_rate", "factor": 0.9} — multiply learning rate (try 0.8-1.2)

AVOID (0% acceptance): ojas_rule, bcm_threshold, normalize_pre_activations, weight_decay, weight_clip, momentum, competitive_top_k"#,
        MutationTarget::HebbianPrune => r#"Available strategies (pick ONE and fill in parameters):
- {"type": "scale_consolidation_interval", "factor": 0.7} — change pruning frequency (50% acceptance! try 0.5-1.5)
- {"type": "scale_prune_threshold", "factor": 1.2} — multiply pruning threshold (try 0.8-1.6)
- {"type": "percentile_prune", "percentile": 10.0} — prune by percentile (try 5-25)

AVOID (0% acceptance): consolidation_aware_prune"#,
        MutationTarget::SparseProjection => r#"Available strategies (pick ONE and fill in parameters):
- {"type": "scale_k", "factor": 1.3} — multiply number of active dims (try 0.5-2.0)

NOTE: All SparseProjection strategies have 0% acceptance (0/124 trials). Try extreme scale_k values."#,
        MutationTarget::TemporalTrace => r#"Available strategies (pick ONE and fill in parameters):
- {"type": "scale_temporal_decay", "factor": 0.75} — adjust temporal decay (try 0.7-0.85)
- {"type": "scale_trace_weight", "factor": 0.8} — adjust trace blending (try 0.5-1.5)"#,
        MutationTarget::TrainingLoop => r#"Available strategies (pick ONE and fill in parameters):
- {"type": "warmup_temperature", "start_temp": 0.1, "end_temp": 0.006} — start broad, focus sharp over training (NEW, high impact)
- {"type": "ema_eval", "beta": 0.995} — Polyak averaging for smoother evaluation model (NEW, try 0.99-0.999)
- {"type": "mixup", "alpha": 0.3} — interpolate pairs for synthetic positives (NEW, try 0.1-0.5)
- {"type": "gradient_accumulation", "accumulate_steps": 4} — smoother updates (NEW, try 2-16)
- {"type": "cyclic_lr", "min_factor": 0.2, "max_factor": 2.0, "cycle_steps": 10000} — oscillating LR to escape local optima (NEW)
- {"type": "scale_max_steps", "target_steps": 80000} — more training epochs (try 60000-100000)
- {"type": "scale_init_scale", "factor": 0.4} — adjust initial weight scale (try 0.2-1.2)
- {"type": "scale_aug_noise", "factor": 1.5} — adjust augmentation noise (try 0.3-2.5)

AVOID: linear_schedule, scale_warmup (both 0% acceptance)"#,
    };

    format!(
        r#"You are an expert in Hebbian learning and cross-modal retrieval.

SYSTEM: AMN learning audio-visual associations via Hebbian matrices (512×512).
Current best: R@1={best_r1:.4}, MRR={best_mrr:.6}
Target: {target}

{strategies}

Recent history for this target:
{history}

IMPORTANT:
- Pick a strategy with DIFFERENT parameter values than what's been tried
- Strategies marked "BEST" have high acceptance — explore different parameter ranges
- Strategies marked "AVOID" have 0% acceptance — do NOT pick them

Respond with EXACTLY this JSON (no markdown):
{{"strategy": <strategy JSON>, "hypothesis": "<1 sentence>"}}"#,
    )
}

/// Parse LLM response into a MutationConfig.
fn parse_mutation_response(
    response: &str,
    target: MutationTarget,
) -> Result<MutationConfig, Box<dyn std::error::Error + Send + Sync>> {
    // Try to find JSON in the response
    let json_str = extract_json(response)
        .ok_or_else(|| "No JSON found in LLM response")?;

    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

    let strategy_val = parsed
        .get("strategy")
        .ok_or("Missing 'strategy' field")?;
    let hypothesis = parsed
        .get("hypothesis")
        .and_then(|v| v.as_str())
        .unwrap_or("LLM-proposed mutation")
        .to_string();

    let strategy: MutationStrategy = serde_json::from_value(strategy_val.clone())?;

    Ok(MutationConfig {
        target,
        hypothesis,
        strategy,
    })
}

/// Extract JSON object from LLM response text.
fn extract_json(text: &str) -> Option<String> {
    // Try to find { ... } in the response
    let mut depth = 0i32;
    let mut start = None;

    for (i, ch) in text.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        return Some(text[s..=i].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Composite score for mutation acceptance.
fn composite_score(metrics: &std::collections::HashMap<String, f64>) -> f64 {
    let mrr = metrics.get("v2a_MRR").copied().unwrap_or(0.0);
    let r1 = metrics.get("v2a_R@1").copied().unwrap_or(0.0);
    mrr + r1
}

/// Get the best experiment config from the DB using composite score (MRR + R@1).
/// This matches the acceptance criterion so winning mutations become the new base.
fn get_best_config(db: &KnowledgeBase) -> Result<ExperimentConfig, Box<dyn std::error::Error + Send + Sync>> {
    let conn = db.pool().get()?;
    let result: Result<(String, String), _> = conn.query_row(
        "SELECT config_json, final_metrics FROM experiments \
         WHERE status='completed' \
         AND json_extract(final_metrics, '$.v2a_MRR') IS NOT NULL \
         ORDER BY (CAST(json_extract(final_metrics, '$.v2a_MRR') AS REAL) \
                 + CAST(COALESCE(json_extract(final_metrics, '$.v2a_R@1'), '0') AS REAL)) DESC \
         LIMIT 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );
    match result {
        Ok((config_json, _metrics_json)) => {
            match serde_json::from_str::<ExperimentConfig>(&config_json) {
                Ok(mut config) => {
                    // Cap max_steps: best experiment (19514, MRR=0.0422) needed 60K steps.
                    config.max_steps = config.max_steps.clamp(5000, 100000);
                    tracing::info!(
                        batch_size = config.batch_size,
                        max_steps = config.max_steps,
                        use_siglip = config.use_siglip,
                        infonce_temperature = config.infonce_temperature,
                        "Loaded best config"
                    );
                    Ok(config)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to deserialize best config, using defaults");
                    Ok(ExperimentConfig::default())
                }
            }
        }
        Err(_) => Ok(ExperimentConfig::default()),
    }
}

/// Load a batch of mutations from the sweep queue file.
/// Each cycle reads the next 8 mutations. Returns empty when exhausted.
/// File format: JSON array of MutationConfig objects.
fn load_sweep_batch(path: &str, cycle: u64) -> Vec<MutationConfig> {
    let batch_size = 8; // mutations per cycle
    let offset = ((cycle - 1) as usize) * batch_size;

    match std::fs::read_to_string(path) {
        Ok(content) => {
            match serde_json::from_str::<Vec<MutationConfig>>(&content) {
                Ok(all) => {
                    if offset >= all.len() {
                        tracing::info!(total = all.len(), "Sweep queue exhausted");
                        return Vec::new();
                    }
                    let end = (offset + batch_size).min(all.len());
                    tracing::info!(
                        offset, end, total = all.len(),
                        "Loading sweep mutations"
                    );
                    all[offset..end].to_vec()
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to parse sweep file");
                    Vec::new()
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, path, "No sweep file found, using normal mode");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json() {
        let text = r#"Here is my suggestion: {"strategy": {"type": "weight_decay", "lambda": 0.001}, "hypothesis": "test"} done"#;
        let json = extract_json(text).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("strategy").is_some());
    }

    #[test]
    fn test_parse_mutation_response() {
        let response = r#"{"strategy": {"type": "weight_decay", "lambda": 0.001}, "hypothesis": "Prevents weight explosion"}"#;
        let result = parse_mutation_response(response, MutationTarget::HebbianUpdate);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.target, MutationTarget::HebbianUpdate);
        assert!(matches!(config.strategy, MutationStrategy::WeightDecay { .. }));
    }
}
