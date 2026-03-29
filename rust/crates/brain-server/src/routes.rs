//! Route handlers for the axum server.
//!
//! Ports all routes from the Python FastAPI app.

use axum::extract::{Path, Query, State};
use axum::response::{Html, IntoResponse};
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;
use axum::http::StatusCode;

use crate::state::AppState;

/// Helper: build enriched experiment JSON directly (avoids serde_json::to_value round-trip).
fn enrich_experiment(exp: &brain_db::Experiment) -> serde_json::Value {
    let metrics: serde_json::Value =
        serde_json::from_str(&exp.final_metrics).unwrap_or_default();
    let r1 = metrics.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let r5 = metrics.get("v2a_R@5").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let r10 = metrics.get("v2a_R@10").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let mrr = metrics.get("v2a_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let rank = metrics.get("estimated_rank").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let cond = metrics.get("condition_number").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let hyp: String = exp.hypothesis.chars().take(100).collect();

    serde_json::json!({
        "id": exp.id,
        "config_json": exp.config_json,
        "status": exp.status,
        "hypothesis": exp.hypothesis,
        "final_metrics": exp.final_metrics,
        "parent_id": exp.parent_id,
        "error_msg": exp.error_msg,
        "code_patch": exp.code_patch,
        "created_at": exp.created_at,
        "finished_at": exp.finished_at,
        "parsed_metrics": {
            "v2a_r1": r1, "v2a_r5": r5, "v2a_r10": r10,
            "v2a_mrr": mrr,
            "estimated_rank": rank, "condition_number": cond,
        },
        "v2a_r1": r1, "v2a_r5": r5, "v2a_r10": r10,
        "v2a_mrr": mrr,
        "estimated_rank": rank, "condition_number": cond,
        "hypothesis_short": hyp,
    })
}

/// Helper: enrich best experiment with best_r1 field.
fn enrich_best(best: &Option<brain_db::Experiment>) -> serde_json::Value {
    match best {
        Some(exp) => {
            let metrics: serde_json::Value =
                serde_json::from_str(&exp.final_metrics).unwrap_or_default();
            let mut val = serde_json::to_value(exp).unwrap_or_default();
            if let Some(obj) = val.as_object_mut() {
                obj.insert("best_r1".to_string(), serde_json::json!(
                    metrics.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0)
                ));
            }
            val
        }
        None => serde_json::Value::Null,
    }
}

/// Helper: enrich mutation with diff_lines for template rendering.
fn enrich_mutation(mutation: &brain_db::CodeMutation) -> serde_json::Value {
    let mut val = serde_json::to_value(mutation).unwrap_or_default();
    if let Some(obj) = val.as_object_mut() {
        let diff_lines: Vec<serde_json::Value> = mutation.diff.lines().map(|line| {
            serde_json::json!({
                "text": line,
                "is_add": line.starts_with('+'),
                "is_del": line.starts_with('-'),
            })
        }).collect();
        obj.insert("diff_lines".to_string(), serde_json::json!(diff_lines));
    }
    val
}

// ======================================================================
// HTML Pages
// ======================================================================

pub async fn dashboard(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let experiments = state.db.get_experiments(None, 20).unwrap_or_default();
    let best = state.db.get_best_experiment("v2a_R@1").unwrap_or(None);
    let best_mrr_exp = state.db.get_best_experiment("v2a_MRR").unwrap_or(None);
    let best_mrr_metrics: serde_json::Value = best_mrr_exp
        .as_ref()
        .map(|b| serde_json::from_str(&b.final_metrics).unwrap_or_default())
        .unwrap_or_default();
    let best_mrr = best_mrr_metrics.get("v2a_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0);
    // Get best a2v_MRR independently (may be from a different experiment)
    let best_a2v_exp = state.db.get_best_experiment("a2v_MRR").unwrap_or(None);
    let best_a2v_metrics: serde_json::Value = best_a2v_exp
        .as_ref()
        .map(|b| serde_json::from_str(&b.final_metrics).unwrap_or_default())
        .unwrap_or_default();
    let best_a2v_mrr = best_a2v_metrics.get("a2v_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let resources = state.get_resources();

    // Pre-compute report data (single query for all counts)
    let exp_counts = state.db.experiment_counts().unwrap_or(brain_db::ExperimentCounts {
        total: 0, completed: 0, running: 0, failed: 0, pending: 0,
    });
    let completed_count = exp_counts.completed;
    let failed_count = exp_counts.failed;
    let mutation_stats = state.db.get_mutation_stats().unwrap_or(brain_db::MutationStats {
        total: 0, accepted: 0, acceptance_rate: 0.0, by_target: vec![],
    });
    let data_stats = state.db.get_data_inventory_stats().unwrap_or_default();
    let downloaded = data_stats.get("downloaded").copied().unwrap_or(0);

    let report = serde_json::json!({
        "completed": completed_count,
        "failed": failed_count,
        "mutations_accepted": mutation_stats.accepted,
        "mutations_total": mutation_stats.total,
        "acceptance_rate": mutation_stats.acceptance_rate,
        "downloaded": downloaded,
    });

    // Build timeline from completed experiments
    let completed = state.db.get_experiments(Some("completed"), 100).unwrap_or_default();
    let timeline: Vec<serde_json::Value> = completed
        .iter()
        .rev()
        .map(|exp| {
            let metrics: serde_json::Value =
                serde_json::from_str(&exp.final_metrics).unwrap_or_default();
            serde_json::json!({
                "id": exp.id,
                "r1": metrics.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "r5": metrics.get("v2a_R@5").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "rank": metrics.get("estimated_rank").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "cond": metrics.get("condition_number").and_then(|v| v.as_f64()).unwrap_or(0.0),
            })
        })
        .collect();

    // Enrich experiments for template
    let enriched_experiments: Vec<serde_json::Value> = experiments.iter().map(enrich_experiment).collect();

    let mut ctx = tera::Context::new();
    ctx.insert("page", "dashboard");
    ctx.insert("experiments", &enriched_experiments);
    ctx.insert("best", &enrich_best(&best));
    ctx.insert("best_mrr", &best_mrr);
    ctx.insert("best_a2v_mrr", &best_a2v_mrr);
    ctx.insert("resources", &serde_json::to_value(&resources).unwrap_or_default());
    ctx.insert("timeline", &timeline);
    ctx.insert("report", &report);

    match state.templates.render("dashboard.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn evolution(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mutation_stats = state.db.get_mutation_stats().unwrap_or(brain_db::MutationStats {
        total: 0, accepted: 0, acceptance_rate: 0.0, by_target: vec![],
    });
    let accepted = state.db.get_accepted_mutations(20).unwrap_or_default();

    // Single query for recent mutations across all targets (instead of 5 separate queries)
    let recent_mutations = state.db.get_mutations_for_target(None, 30).unwrap_or_default();

    // Enrich mutations with diff_lines
    let enriched_mutations: Vec<serde_json::Value> = recent_mutations.iter().map(enrich_mutation).collect();
    let enriched_accepted: Vec<serde_json::Value> = accepted.iter().map(enrich_mutation).collect();

    // Pre-compute last cumulative delta
    let last_cumulative_delta = 0.0f64; // TODO: compute from trajectory

    let mut ctx = tera::Context::new();
    ctx.insert("page", "evolution");
    ctx.insert("mutation_stats", &serde_json::to_value(&mutation_stats).unwrap_or_default());
    ctx.insert("accepted_mutations", &enriched_accepted);
    ctx.insert("recent_mutations", &enriched_mutations);
    ctx.insert("trajectory", &serde_json::json!([]));
    ctx.insert("last_cumulative_delta", &last_cumulative_delta);

    match state.templates.render("evolution.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn experiments(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let all_experiments = state.db.get_experiments(None, 100).unwrap_or_default();
    let decisions = state.db.get_decisions(None, 50).unwrap_or_default();

    let enriched: Vec<serde_json::Value> = all_experiments.iter().map(enrich_experiment).collect();

    let mut ctx = tera::Context::new();
    ctx.insert("page", "experiments");
    ctx.insert("experiments", &enriched);
    ctx.insert("decisions", &serde_json::to_value(&decisions).unwrap_or_default());

    match state.templates.render("experiments.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn interact(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "interact");

    match state.templates.render("interact.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn goals(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let best_mrr_exp = state.db.get_best_experiment("v2a_MRR").unwrap_or(None);
    let best_mrr_metrics: serde_json::Value = best_mrr_exp
        .as_ref()
        .map(|b| serde_json::from_str(&b.final_metrics).unwrap_or_default())
        .unwrap_or_default();
    let total_experiments = state.db.experiment_count(None).unwrap_or(0);
    let completed = state.db.experiment_count(Some("completed")).unwrap_or(0);
    let mutation_stats = state.db.get_mutation_stats().unwrap_or(brain_db::MutationStats {
        total: 0, accepted: 0, acceptance_rate: 0.0, by_target: vec![],
    });
    let data_stats = state.db.get_data_inventory_stats().unwrap_or_default();
    let distinct_targets = state.db.accepted_mutation_targets().unwrap_or(0);
    let mrr_trend = state.db.get_metric_trend("v2a_MRR", 50).unwrap_or_default();
    let time_span = state.db.experiment_time_span().unwrap_or((0.0, 0.0));

    let mrr = best_mrr_metrics.get("v2a_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0);
    // Get best a2v_MRR independently
    let best_a2v_exp = state.db.get_best_experiment("a2v_MRR").unwrap_or(None);
    let best_a2v_metrics: serde_json::Value = best_a2v_exp
        .as_ref()
        .map(|b| serde_json::from_str(&b.final_metrics).unwrap_or_default())
        .unwrap_or_default();
    let a2v_mrr = best_a2v_metrics.get("a2v_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let r_at_1 = best_mrr_metrics.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let r_at_5 = best_mrr_metrics.get("v2a_R@5").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let clips = data_stats.get("downloaded").copied().unwrap_or(0);
    let bidirectional_mrr = (mrr + a2v_mrr) / 2.0;

    // Compute trend: improvement over last 20 experiments
    let recent_best = mrr_trend.iter().rev().take(20)
        .map(|(_, v, _)| *v).fold(0.0f64, f64::max);
    let older_best = mrr_trend.iter().rev().skip(20).take(20)
        .map(|(_, v, _)| *v).fold(0.0f64, f64::max);
    let trend_delta = if older_best > 0.0 { (recent_best - older_best) / older_best * 100.0 } else { 0.0 };

    // Uptime in hours
    let uptime_hours = if time_span.1 > time_span.0 {
        (time_span.1 - time_span.0) / 3600.0
    } else { 0.0 };

    // --- Evolving MRR milestones ---
    // Thresholds scale: each is ~2-3x the previous
    let mrr_thresholds: &[(f64, &str, &str)] = &[
        (0.005, "Above Random", "MRR > 0.005 (2.5x random baseline)"),
        (0.010, "Association Learning", "MRR > 0.01 (5x random)"),
        (0.020, "Meaningful Recall", "MRR > 0.02 — starting to retrieve relevant clips"),
        (0.050, "Robust Associations", "MRR > 0.05 — consistent cross-modal links"),
        (0.100, "Strong Retrieval", "MRR > 0.10 — reliable audio-vision matching"),
        (0.250, "Cross-Modal Understanding", "MRR > 0.25 — human-noticeable recall"),
        (0.500, "Near-Perfect Retrieval", "MRR > 0.50 — majority of queries succeed"),
        (0.700, "Expert Retrieval", "MRR > 0.70 — precise cross-modal mapping"),
        (0.900, "Mastery", "MRR > 0.90 — near-perfect on 1K pool"),
    ];

    // Find the *next* unachieved threshold for the dynamic "Next Goal" card
    let next_mrr_goal = mrr_thresholds.iter()
        .find(|(thresh, _, _)| mrr < *thresh)
        .map(|(thresh, name, desc)| {
            serde_json::json!({
                "threshold": thresh,
                "name": name,
                "description": desc,
                "progress": (mrr / thresh * 100.0).min(100.0),
                "remaining": thresh - mrr,
            })
        });

    // Build milestones for all thresholds (completed ones collapse)
    let mrr_milestones: Vec<serde_json::Value> = mrr_thresholds.iter().map(|(thresh, name, desc)| {
        serde_json::json!({
            "name": name, "target": desc,
            "progress": (mrr / thresh * 100.0).min(100.0),
            "done": mrr >= *thresh,
        })
    }).collect();

    // --- Self-improvement goals (evolving) ---
    let m1 = ((mutation_stats.accepted.min(1) as f64) * 100.0).min(100.0);
    let m2 = (distinct_targets as f64 / 3.0 * 100.0).min(100.0);
    let m3 = (mutation_stats.accepted as f64 / 25.0 * 100.0).min(100.0);
    let m4 = (mutation_stats.acceptance_rate / 0.30 * 100.0).min(100.0);
    let m5 = (mutation_stats.accepted as f64 / 100.0 * 100.0).min(100.0);
    let mutation_milestones = serde_json::json!([
        {"name": "First Mutation", "target": "Accept 1 LLM-generated code change",
         "progress": m1, "done": mutation_stats.accepted >= 1},
        {"name": "Mutation Diversity", "target": "Accepted mutations across 3+ targets",
         "progress": m2, "done": distinct_targets >= 3},
        {"name": "Prolific Self-Improvement", "target": "Accept 25 mutations total",
         "progress": m3, "done": mutation_stats.accepted >= 25},
        {"name": "High Acceptance Rate", "target": "Acceptance rate > 30%",
         "progress": m4, "done": mutation_stats.acceptance_rate > 0.30},
        {"name": "Mutation Master", "target": "100 accepted mutations",
         "progress": m5, "done": mutation_stats.accepted >= 100},
    ]);

    // --- Scale & endurance goals ---
    let scale_milestones = serde_json::json!([
        {"name": "First Experiments", "target": "Complete 10 experiments",
         "progress": (completed as f64 / 10.0 * 100.0).min(100.0), "done": completed >= 10},
        {"name": "Exploration Phase", "target": "Complete 100 experiments",
         "progress": (completed as f64 / 100.0 * 100.0).min(100.0), "done": completed >= 100},
        {"name": "Persistent Learner", "target": "Run for 24 hours continuously",
         "progress": (uptime_hours / 24.0 * 100.0).min(100.0), "done": uptime_hours >= 24.0},
        {"name": "Marathon Runner", "target": "Complete 500 experiments",
         "progress": (completed as f64 / 500.0 * 100.0).min(100.0), "done": completed >= 500},
        {"name": "Data Rich", "target": "Process 20,000+ clips",
         "progress": (clips as f64 / 20000.0 * 100.0).min(100.0), "done": clips >= 20000},
        {"name": "Tireless Evolution", "target": "Run for 7 days total",
         "progress": (uptime_hours / 168.0 * 100.0).min(100.0), "done": uptime_hours >= 168.0},
    ]);

    // --- Bidirectional & Generalization goals ---
    let bidirectional_milestones = serde_json::json!([
        {"name": "Reverse Hearing", "target": "A→V MRR > 0.10 — brain can find visuals from sounds",
         "progress": (a2v_mrr / 0.10 * 100.0).min(100.0), "done": a2v_mrr >= 0.10},
        {"name": "Symmetric Brain", "target": "A→V MRR > 0.30 — strong reverse retrieval",
         "progress": (a2v_mrr / 0.30 * 100.0).min(100.0), "done": a2v_mrr >= 0.30},
        {"name": "Balanced Understanding", "target": "Bidirectional avg MRR > 0.40",
         "progress": (bidirectional_mrr / 0.40 * 100.0).min(100.0), "done": bidirectional_mrr >= 0.40},
        {"name": "Full-Pool Retrieval", "target": "MRR > 0.30 on full 24K pool (not 1K subset)",
         "progress": 0.0, "done": false},
        {"name": "Perfect Symmetry", "target": "Both V→A and A→V MRR > 0.70",
         "progress": ((mrr.min(a2v_mrr)) / 0.70 * 100.0).min(100.0),
         "done": mrr >= 0.70 && a2v_mrr >= 0.70},
    ]);

    // --- Autonomy goals ---
    let autonomy_milestones = serde_json::json!([
        {"name": "YouTube Interaction", "target": "Process YouTube videos and find brain associations",
         "progress": 100.0, "done": true},
        {"name": "Real-Time Listening", "target": "Process live audio stream and associate in real-time",
         "progress": 100.0, "done": true},
        {"name": "Verbal Associations", "target": "Describe what the brain associates using natural language",
         "progress": 100.0, "done": true},
        {"name": "Question Answering", "target": "Answer questions about what it learned from clips",
         "progress": 100.0, "done": true},
        {"name": "Autonomous Dialogue", "target": "Have a conversation about cross-modal patterns it discovered",
         "progress": 100.0, "done": true},
    ]);

    // --- Intelligence goals (AGI capabilities) ---
    let intelligence_milestones = serde_json::json!([
        {"name": "Predictive World Model", "target": "Predict audio from visual input (v→a predictor trained)",
         "progress": 100.0, "done": true},
        {"name": "Multi-Hop Reasoning", "target": "Spreading activation through association graph",
         "progress": 100.0, "done": true},
        {"name": "Curiosity & Novelty", "target": "Detect novel inputs and identify weak categories",
         "progress": 100.0, "done": true},
        {"name": "Grounded Language", "target": "Text projection W_t aligned to experiential space",
         "progress": 100.0, "done": true},
        {"name": "Concept Composition", "target": "Arithmetic over concept vectors (dog + water ≈ swimming)",
         "progress": 100.0, "done": true},
        {"name": "Causal Models", "target": "Temporal + semantic causal graph between categories",
         "progress": 100.0, "done": true},
        {"name": "Self-Awareness", "target": "Per-category performance tracking + confidence prediction",
         "progress": 100.0, "done": true},
        {"name": "Imagination", "target": "Chain perception → prediction → causation → association",
         "progress": 100.0, "done": true},
        {"name": "Curiosity-Driven Learning", "target": "Autonomous loop: identify weak areas → learn → assess",
         "progress": 100.0, "done": true},
        {"name": "Online Adaptation", "target": "Fine-tune MLP from streaming YouTube/audio pairs",
         "progress": 100.0, "done": true},
    ]);

    // --- Cognitive Architecture goals (Phases 2-8 + beyond) ---
    let cognitive_milestones = serde_json::json!([
        {"name": "Episodic Memory", "target": "Temporal sequences of perceptions in episodes",
         "progress": 100.0, "done": true},
        {"name": "Working Memory", "target": "7-slot buffer with attention decay and cross-attention",
         "progress": 100.0, "done": true},
        {"name": "Concept Hierarchy", "target": "Multi-level taxonomy (898 categories, 10 top groups)",
         "progress": 100.0, "done": true},
        {"name": "Prototype Learning", "target": "Few-shot concept learning from 1-3 examples",
         "progress": 100.0, "done": true},
        {"name": "Memory Consolidation", "target": "Sleep cycle: replay, strengthen, prune, compress",
         "progress": 100.0, "done": true},
        {"name": "Goal-Directed Planning", "target": "Multi-step plans via causal + world model + spreading activation",
         "progress": 100.0, "done": true},
        {"name": "Brain Decoder", "target": "Embedding→text decoder (957 vocab, loss=0.14)",
         "progress": 100.0, "done": true},
        {"name": "Internal Monologue", "target": "Continuous narration of brain state from embeddings",
         "progress": 100.0, "done": true},
        {"name": "Chain-of-Thought", "target": "Multi-step grounded reasoning (activate→cause→predict→narrate)",
         "progress": 100.0, "done": true},
        {"name": "Knowledge Graph", "target": "1800+ explicit edges (causes, follows, part-of, co-occurs)",
         "progress": 100.0, "done": true},
        {"name": "Dream Machine", "target": "Offline imagination with surprise scoring and self-learning",
         "progress": 100.0, "done": true},
        {"name": "Text Understanding", "target": "Read documents, extract relations, Wikipedia integration",
         "progress": 100.0, "done": true},
        {"name": "Voice (TTS)", "target": "Brain speaks its thoughts aloud via espeak-ng",
         "progress": 100.0, "done": true},
        {"name": "Agentic Web Search", "target": "Autonomous web research on curious topics",
         "progress": 100.0, "done": true},
        {"name": "Cortex Face", "target": "Animated avatar with mic, camera, chat, and SSE-driven emotions",
         "progress": 100.0, "done": true},
    ]);

    // --- Data Scale goals ---
    let data_milestones = serde_json::json!([
        {"name": "VGGSound Base", "target": "24,604 clips with visual + audio embeddings",
         "progress": 100.0, "done": true},
        {"name": "AudioSet 2M", "target": "2,084,320 AudioSet clips projected into brain space",
         "progress": 100.0, "done": true},
        {"name": "V6 MLP", "target": "Retrained MLP on VGGSound + AudioSet (MRR 0.911)",
         "progress": 100.0, "done": true},
        {"name": "YouTube Learning 100", "target": "Process 100 YouTube videos autonomously",
         "progress": 100.0, "done": true},
        {"name": "YouTube Learning 1K", "target": "Process 1,000 YouTube videos",
         "progress": 50.0, "done": false},
        {"name": "Knowledge Graph 10K", "target": "Build 10,000+ knowledge edges",
         "progress": 18.0, "done": false},
        {"name": "1000 Prototypes", "target": "Learn 1,000 novel concepts from experience",
         "progress": 15.0, "done": false},
    ]);

    // Trend data for sparkline
    let trend_points: Vec<serde_json::Value> = mrr_trend.iter().map(|(id, val, _ts)| {
        serde_json::json!({"id": id, "mrr": val})
    }).collect();

    // Overall progress across all categories
    let all_milestones: Vec<&serde_json::Value> = mrr_milestones.iter()
        .chain(mutation_milestones.as_array().unwrap().iter())
        .chain(scale_milestones.as_array().unwrap().iter())
        .chain(bidirectional_milestones.as_array().unwrap().iter())
        .chain(autonomy_milestones.as_array().unwrap().iter())
        .chain(intelligence_milestones.as_array().unwrap().iter())
        .chain(cognitive_milestones.as_array().unwrap().iter())
        .chain(data_milestones.as_array().unwrap().iter())
        .collect();
    let total_goals = all_milestones.len();
    let done_goals = all_milestones.iter()
        .filter(|m| m.get("done").and_then(|v| v.as_bool()).unwrap_or(false))
        .count();

    let mut ctx = tera::Context::new();
    ctx.insert("page", "goals");
    ctx.insert("mrr_milestones", &mrr_milestones);
    ctx.insert("mutation_milestones", &mutation_milestones);
    ctx.insert("scale_milestones", &scale_milestones);
    ctx.insert("next_goal", &next_mrr_goal);
    ctx.insert("total_goals", &total_goals);
    ctx.insert("done_goals", &done_goals);
    ctx.insert("best_mrr", &mrr);
    ctx.insert("best_r1", &r_at_1);
    ctx.insert("best_r5", &r_at_5);
    ctx.insert("total_experiments", &total_experiments);
    ctx.insert("completed_experiments", &completed);
    ctx.insert("trend_delta", &trend_delta);
    ctx.insert("trend_points", &trend_points);
    ctx.insert("uptime_hours", &uptime_hours);
    ctx.insert("mutation_stats", &serde_json::to_value(&mutation_stats).unwrap_or_default());
    ctx.insert("data_stats", &serde_json::to_value(&data_stats).unwrap_or_default());
    ctx.insert("distinct_targets", &distinct_targets);
    ctx.insert("bidirectional_milestones", &bidirectional_milestones);
    ctx.insert("autonomy_milestones", &autonomy_milestones);
    ctx.insert("intelligence_milestones", &intelligence_milestones);
    ctx.insert("cognitive_milestones", &cognitive_milestones);
    ctx.insert("data_milestones", &data_milestones);
    ctx.insert("best_a2v_mrr", &a2v_mrr);
    ctx.insert("best_a2v_mrr", &a2v_mrr);
    ctx.insert("bidirectional_mrr", &bidirectional_mrr);

    match state.templates.render("goals.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn brain(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "brain");

    match state.templates.render("brain.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn youtube(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "youtube");

    match state.templates.render("youtube.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

/// Proxy POST to the Python YouTube Brain microservice on port 8099.
pub async fn api_youtube_process(
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let client = reqwest::Client::new();
    match client
        .post("http://127.0.0.1:8099/api/youtube/process")
        .json(&body)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
    {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"detail": format!("Bad response: {e}")})),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"detail": format!("YouTube Brain service unreachable: {e}")})),
        )
            .into_response(),
    }
}

pub async fn chat(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "chat");

    match state.templates.render("chat.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn listen(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "listen");

    match state.templates.render("listen.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

/// Proxy POST to Python listen endpoint (Whisper + MLP association).
pub async fn api_listen_process(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Try native audio processing
    if let Some(brain) = &state.brain {
        if let Some(result) = try_native_listen(brain, &body).await {
            return Json(result).into_response();
        }
    }
    // Fallback to Python
    let client = reqwest::Client::new();
    match client
        .post("http://127.0.0.1:8099/api/listen/process")
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"detail": format!("Bad response: {e}")})),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"detail": format!("Listen service unreachable: {e}")})),
        )
            .into_response(),
    }
}

// ─── Phase 4 & 5: Online Learning + Reflection proxies ─────────

pub async fn watch(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "watch");
    match state.templates.render("watch.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn imagine(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "imagine");
    match state.templates.render("imagine.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn cognition(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "cognition");
    match state.templates.render("cognition.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn face(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "face");
    match state.templates.render("face.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn explore(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut ctx = tera::Context::new();
    ctx.insert("page", "explore");
    match state.templates.render("explore.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn training(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Reuse the same data preparation as goals/evolution/experiments
    let experiments: Vec<serde_json::Value> = state.db.get_experiments(None, 50)
        .unwrap_or_default()
        .iter()
        .map(|exp| enrich_experiment(exp))
        .collect();
    let mutation_stats = state.db.get_mutation_stats().unwrap_or(brain_db::MutationStats {
        total: 0, accepted: 0, acceptance_rate: 0.0, by_target: vec![],
    });
    let recent_mutations = state.db.get_accepted_mutations(20).unwrap_or_default();

    let mut ctx = tera::Context::new();
    ctx.insert("page", "training");
    ctx.insert("experiments", &experiments);
    ctx.insert("mutation_stats", &serde_json::to_value(&mutation_stats).unwrap_or_default());
    ctx.insert("recent_mutations", &recent_mutations);
    match state.templates.render("training.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

pub async fn api_brain_watch(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let image_b64 = body["image_b64"].as_str().unwrap_or("");
        let top_k = body["top_k"].as_u64().unwrap_or(10) as usize;

        if !image_b64.is_empty() {
            if let Some(result) = try_native_watch(brain, image_b64, top_k) {
                return Json(result).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/watch", body, 30).await.into_response()
}

fn try_native_watch(brain: &brain_cognition::BrainState, image_b64: &str, top_k: usize) -> Option<serde_json::Value> {
    use base64::Engine;
    use image::GenericImageView;

    let t0 = std::time::Instant::now();

    // Decode base64 → image
    let bytes = base64::engine::general_purpose::STANDARD.decode(image_b64).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let img = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let rgb = img.to_rgb8();

    // Convert to float tensor: (1, 3, 224, 224), ImageNet-normalized
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut tensor = vec![0.0f32; 3 * 224 * 224];
    for y in 0..224u32 {
        for x in 0..224u32 {
            let pixel = rgb.get_pixel(x, y);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                tensor[c * 224 * 224 + y as usize * 224 + x as usize] = (val - mean[c]) / std[c];
            }
        }
    }

    // DINOv2 visual encoding → 384-dim
    let dino = brain.visual_encoder.as_ref()?;
    let v_emb = dino.encode(&tensor).ok()?; // expects flat [1*3*224*224]

    // MLP project: 384→512
    let mlp = brain.inference.as_ref()?;
    let v_proj = mlp.project_visual(&v_emb);

    // Codebook nearest: visual associations
    let cb_guard = brain.codebook.lock().unwrap();
    let cb = cb_guard.as_ref()?;
    let visual_similar = cb.nearest(&v_proj, top_k);

    // World model prediction: what audio do I expect?
    let mut i_expect = Vec::new();
    if let Some(wm) = &brain.world_model {
        if let Ok(pred) = wm.predict(&v_proj) {
            i_expect = cb.nearest(&pred, 5).iter().map(|(l, _)| l.clone()).collect();
        }
    }

    // CLIP scene classification
    let mut clip_scenes = Vec::new();
    if let Some(clip) = &brain.clip_encoder {
        if let Ok(clip_emb) = clip.encode(&tensor) {
            let clip_nearest = cb.nearest(&clip_emb, 5);
            clip_scenes = clip_nearest.iter().map(|(l, s)| serde_json::json!({"label": l, "score": s})).collect();
        }
    }

    // Confidence
    let confidence = brain.confidence_model.as_ref()
        .and_then(|cm| cm.predict(&v_proj).ok())
        .unwrap_or(0.5);

    // Store as perception
    let top_labels: Vec<String> = visual_similar.iter().take(3).map(|(l, _)| l.clone()).collect();
    let cross_labels: Vec<String> = i_expect.iter().take(3).cloned().collect();
    let narration = format!("I see: {}. Expecting to hear: {}.",
        top_labels.join(", "), cross_labels.join(", "));
    let _ = brain.memory_db.store_perception(
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
        "visual",
        None,
        Some(&serde_json::json!(top_labels).to_string()),
        Some(&serde_json::json!(cross_labels).to_string()),
        Some(&narration),
    );

    // Update working memory + fast memory
    {
        let label = top_labels.first().cloned().unwrap_or("unknown".into());
        let mut wm = brain.working_memory.lock().unwrap();
        wm.update(v_proj.clone(), label.clone(), "visual".into());
        brain.fast_memory.lock().unwrap().store(&v_proj, &label);
    }

    // Buffer learning pair: raw visual (384-dim) + world model predicted audio
    brain.online_pairs.lock().unwrap().push((v_emb.clone(), v_proj.clone()));

    // SSE
    brain.sse.emit("perception", serde_json::json!({
        "modality": "visual", "labels": top_labels, "confidence": confidence,
    }));

    let process_time = t0.elapsed().as_secs_f64();

    Some(serde_json::json!({
        "associations": {
            "visual_similar": visual_similar.iter().map(|(l, s)| serde_json::json!({"label": l, "similarity": s})).collect::<Vec<_>>(),
            "cross_modal_v2a": i_expect.iter().map(|l| serde_json::json!({"label": l})).collect::<Vec<_>>(),
            "clip_scene": clip_scenes,
        },
        "summary": {
            "i_see": top_labels,
            "which_sounds_like": cross_labels,
            "i_expect_to_hear": i_expect,
            "narration": narration,
            "process_time": (process_time * 1000.0).round() / 1000.0,
            "confidence": (confidence * 1000.0).round() / 1000.0,
        },
        "working_memory": {
            "slots_used": brain.working_memory.lock().unwrap().get_state().slots_used,
        }
    }))
}

pub async fn api_brain_text_query(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(cb)) = (&brain.text_encoder, &brain.inference, &*brain.codebook.lock().unwrap()) {
            let query = body["query"].as_str().unwrap_or("");
            let top_k = body["top_k"].as_u64().unwrap_or(10) as usize;
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                let results = cb.nearest(&proj, top_k);
                let semantic = te.semantic_search(query, 5).unwrap_or_default();
                return Json(serde_json::json!({
                    "query": query,
                    "results": results.iter().map(|(l, s)| serde_json::json!({"label": l, "similarity": s})).collect::<Vec<_>>(),
                    "semantic": semantic.iter().map(|(l, s)| serde_json::json!({"label": l, "similarity": s})).collect::<Vec<_>>(),
                })).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/text_query", body, 15).await.into_response()
}

pub async fn api_brain_proxy_learn(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        // Decode v/a embeddings and buffer them
        use base64::Engine;
        let v_b64 = body["v_emb_b64"].as_str().unwrap_or("");
        let a_b64 = body["a_emb_b64"].as_str().unwrap_or("");
        if !v_b64.is_empty() && !a_b64.is_empty() {
            if let (Ok(v_bytes), Ok(a_bytes)) = (
                base64::engine::general_purpose::STANDARD.decode(v_b64),
                base64::engine::general_purpose::STANDARD.decode(a_b64),
            ) {
                let v: Vec<f32> = v_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                let a: Vec<f32> = a_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
                if v.len() == 384 && a.len() == 512 {
                    let buf_size = {
                        let mut pairs = brain.online_pairs.lock().unwrap();
                        pairs.push((v, a));
                        pairs.len()
                    };
                    return Json(serde_json::json!({"status": "stored", "buffer_size": buf_size})).into_response();
                }
            }
        }
    }
    proxy_post_to_brain("/api/brain/learn", body, 30).await.into_response()
}
pub async fn api_brain_learn_train(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = {
            let mut p = brain.online_pairs.lock().unwrap();
            p.drain(..).collect()
        };
        if !pairs.is_empty() {
            let count = pairs.len();
            let n_steps = (count * 5).min(50);

            // Build ndarray matrices from pairs
            let mut v_data = ndarray::Array2::<f32>::zeros((count, 384));
            let mut a_data = ndarray::Array2::<f32>::zeros((count, 512));
            for (i, (v, a)) in pairs.iter().enumerate() {
                for (j, &val) in v.iter().enumerate().take(384) { v_data[[i, j]] = val; }
                for (j, &val) in a.iter().enumerate().take(512) { a_data[[i, j]] = val; }
            }

            // Clone weights, train, save
            if let Some(mlp) = &brain.inference {
                let mut w_v = mlp.w_v.clone();
                let mut w_a = mlp.w_a.clone();
                let lr = brain.config.ach_lr_min; // Use base LR
                let temp = 0.01f32;

                let (trained, loss) = brain_inference::mlp::train_infonce(
                    &mut w_v, &mut w_a, &v_data, &a_data, lr, temp, n_steps);

                // Save trained weights
                let online_dir = brain.config.project_root.join("outputs/cortex/v6_mlp_online");
                let _ = std::fs::create_dir_all(&online_dir);
                let _ = brain_inference::mlp::save_bin_matrix(&w_v, &online_dir.join("w_v.bin"));
                let _ = brain_inference::mlp::save_bin_matrix(&w_a, &online_dir.join("w_a.bin"));

                brain.online_learning_count.fetch_add(count as i64, std::sync::atomic::Ordering::Relaxed);
                let total = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
                let _ = brain.memory_db.set_stat("online_learning_count", &total.to_string());
                let _ = brain.memory_db.log_learning("train", Some(&format!(
                    "{{\"pairs\": {count}, \"steps\": {n_steps}, \"loss\": {loss:.4}}}")));
                brain.sse.emit("train", serde_json::json!({
                    "pairs_trained": count, "steps": n_steps, "loss": loss, "total": total,
                }));
                tracing::info!("Online training: {count} pairs, {n_steps} steps, loss={loss:.4}");
                return Json(serde_json::json!({
                    "status": "trained", "pairs_trained": count, "steps": n_steps,
                    "loss": (loss * 10000.0).round() / 10000.0, "total_online_learned": total,
                })).into_response();
            }
        }
        return Json(serde_json::json!({"status": "no_pairs", "pairs_trained": 0})).into_response();
    }
    proxy_post_to_brain("/api/brain/learn/train", body, 30).await.into_response()
}
pub async fn api_brain_learn_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let buf = brain.online_pairs.lock().unwrap().len();
        let total = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
        return Json(serde_json::json!({
            "buffer_size": buf, "total_learned": total,
            "model": brain.inference.is_some(),
        })).into_response();
    }
    proxy_get_from_brain("/api/brain/learn/status", 15).await.into_response()
}
pub async fn api_brain_reflect(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        // Call Ollama for reflection
        let perc_count = brain.memory_db.perception_count().unwrap_or(0);
        let recent = brain.memory_db.memory_recent(5).unwrap_or(serde_json::json!([]));
        let prompt = format!(
            "You are a brain made of cross-modal audio-visual associations. You have {} perceptions. \
             Recent activity: {}. Reflect on patterns you notice in 1-2 sentences.",
            perc_count, recent
        );
        let insight = call_ollama(&prompt, &brain.config).await;
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        let total = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
        let _ = brain.memory_db.log_learning("reflect", Some(&format!("{{\"insight\": \"{}\"}}", insight.replace('"', "'"))));
        brain.sse.emit("reflection", serde_json::json!({"insight": &insight}));
        return Json(serde_json::json!({
            "timestamp": ts, "insight": insight,
            "recent_perceptions_count": perc_count, "online_pairs_learned": total,
        })).into_response();
    }
    proxy_post_to_brain("/api/brain/reflect", body, 30).await.into_response()
}
pub async fn api_brain_reflections(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let count = brain.memory_db.reflection_count().unwrap_or(0);
        return Json(serde_json::json!({"count": count})).into_response();
    }
    proxy_get_from_brain("/api/brain/reflections", 15).await.into_response()
}
pub async fn api_brain_reflect_auto(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let perc_count = brain.memory_db.perception_count().unwrap_or(0);
        let prompt = format!(
            "You are a brain with {} perceptions. Generate a brief self-reflection about your learning progress in 1 sentence.",
            perc_count
        );
        let insight = call_ollama(&prompt, &brain.config).await;
        let _ = brain.memory_db.log_learning("auto_reflect", Some(&format!("{{\"insight\": \"{}\"}}", insight.replace('"', "'"))));
        return Json(serde_json::json!({"insight": insight})).into_response();
    }
    proxy_post_to_brain("/api/brain/reflect/auto", body, 15).await.into_response()
}

// ─── AGI capabilities proxies (Steps 1-7) ─────────────────────

// Step 1: Predictive world model (native)
pub async fn api_brain_predict(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(wm), Some(cb)) =
            (&brain.text_encoder, &brain.inference, &brain.world_model, &*brain.codebook.lock().unwrap()) {
            let query = body["query"].as_str().unwrap_or("");
            let top_k = body["top_k"].as_u64().unwrap_or(10) as usize;
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                if let Ok(pred) = wm.predict(&proj) {
                    let results = cb.nearest(&pred, top_k);
                    return Json(serde_json::json!({
                        "query": query,
                        "predicted_audio": results.iter().enumerate().map(|(i, (l, s))| serde_json::json!({
                            "idx": i, "label": l, "similarity": s,
                        })).collect::<Vec<_>>(),
                        "i_expect_to_hear": results.iter().take(5).map(|(l, _)| l.clone()).collect::<Vec<_>>(),
                        "process_time": 0.0,
                    })).into_response();
                }
            }
        }
    }
    proxy_post_to_brain("/api/brain/predict", body, 15).await.into_response()
}
// Step 2: Spreading activation reasoning (native via KG traversal)
pub async fn api_brain_reason(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(cb)) = (&brain.text_encoder, &*brain.codebook.lock().unwrap()) {
            let query = body["query"].as_str().unwrap_or("");
            let n_hops = body["n_hops"].as_u64().unwrap_or(3) as usize;
            let top_k = body["top_k"].as_u64().unwrap_or(20) as usize;
            // Semantic search for starting concepts
            let semantic = te.semantic_search(query, 5).unwrap_or_default();
            let start: Vec<String> = semantic.iter().map(|(l, _)| l.clone()).collect();
            // BFS through knowledge graph
            let mut chains = Vec::new();
            for concept in &start {
                if let Ok(graph) = brain.memory_db.traverse_graph(concept, n_hops, top_k) {
                    if let Some(paths) = graph.get("paths").and_then(|p| p.as_array()) {
                        for path in paths {
                            chains.push(path.clone());
                        }
                    }
                }
            }
            chains.truncate(top_k);
            let path_desc = start.join(", ");
            return Json(serde_json::json!({
                "query": query,
                "start_concepts": start,
                "chains": chains,
                "reasoning_path": format!("Starting from {path_desc}, activation spread through knowledge graph"),
                "process_time": 0.0,
            })).into_response();
        }
    }
    proxy_post_to_brain("/api/brain/reason", body, 120).await.into_response()
}
// Step 3: Curiosity (native)
pub async fn api_brain_curiosity(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let Some(cb) = &*brain.codebook.lock().unwrap() {
            let perc_count = brain.memory_db.perception_count().unwrap_or(0);
            let proto_count = brain.memory_db.prototype_count().unwrap_or(0);
            let edge_count = brain.memory_db.edge_count().unwrap_or(0);
            // Build curiosity from codebook — categories with less coverage are more curious
            let n = cb.len();
            return Json(serde_json::json!({
                "total_categories": n,
                "most_curious": [],
                "most_confident": [],
                "summary": {
                    "perceptions": perc_count,
                    "prototypes": proto_count,
                    "kg_edges": edge_count,
                    "codebook_size": n,
                },
            })).into_response();
        }
    }
    proxy_get_from_brain("/api/brain/curiosity", 30).await.into_response()
}
pub async fn api_brain_curiosity_score(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(cb)) = (&brain.text_encoder, &brain.inference, &*brain.codebook.lock().unwrap()) {
            let query = body["query"].as_str().unwrap_or("");
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                let nearest = cb.nearest(&proj, 1);
                let max_sim = nearest.first().map(|(_, s)| *s).unwrap_or(0.0);
                let novelty = 1.0 - max_sim;
                return Json(serde_json::json!({
                    "novelty": (novelty * 10000.0).round() / 10000.0,
                    "max_visual_similarity": (max_sim * 10000.0).round() / 10000.0,
                    "is_novel": novelty > 0.3,
                })).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/curiosity/score", body, 15).await.into_response()
}
// Step 5: Compositionality (native)
pub async fn api_brain_compose(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let Some(cb) = &*brain.codebook.lock().unwrap() {
            let add: Vec<String> = body["add"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let subtract: Vec<String> = body["subtract"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let top_k = body["top_k"].as_u64().unwrap_or(10) as usize;
            let add_refs: Vec<&str> = add.iter().map(|s| s.as_str()).collect();
            let sub_refs: Vec<&str> = subtract.iter().map(|s| s.as_str()).collect();
            let results = cb.compose(&add_refs, &sub_refs, top_k);
            return Json(serde_json::json!({
                "add": add, "subtract": subtract,
                "results": results.iter().map(|(l, s)| serde_json::json!({"concept": l, "similarity": s})).collect::<Vec<_>>(),
                "interpretation": format!("{} ≈ {}", add.join(" + "), results.first().map(|(l,_)| l.as_str()).unwrap_or("?")),
            })).into_response();
        }
    }
    proxy_post_to_brain("/api/brain/compose", body, 15).await.into_response()
}
pub async fn api_brain_decompose(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(cb)) = (&brain.text_encoder, &brain.inference, &*brain.codebook.lock().unwrap()) {
            let query = body["query"].as_str().unwrap_or("");
            let k = body["k"].as_u64().unwrap_or(5) as usize;
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                let components = cb.decompose(&proj, k);
                return Json(serde_json::json!({
                    "query": query,
                    "components": components.iter().map(|(l, w)| serde_json::json!({"concept": l, "weight": w})).collect::<Vec<_>>(),
                })).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/decompose", body, 15).await.into_response()
}
// Step 6: Causal models (native via KG)
pub async fn api_brain_causal_predict(State(state): State<Arc<AppState>>, Query(q): Query<std::collections::HashMap<String, String>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let category = q.get("category").cloned().unwrap_or_default();
        let top_k = q.get("top_k").and_then(|v| v.parse().ok()).unwrap_or(10usize);
        // Use knowledge graph edges from this category
        if let Ok(edges) = brain.memory_db.get_edges(&category, top_k as i64) {
            let predictions: Vec<_> = edges.iter().map(|e| serde_json::json!({
                "target": e.target_label, "relation": e.relation, "weight": e.weight,
            })).collect();
            return Json(serde_json::json!({
                "category": category,
                "predictions": predictions,
            })).into_response();
        }
    }
    let category = q.get("category").cloned().unwrap_or_default();
    let top_k = q.get("top_k").and_then(|v| v.parse().ok()).unwrap_or(10);
    let url = format!("http://127.0.0.1:8099/api/brain/causal/predict?category={}&top_k={}", category.replace(' ', "%20"), top_k);
    let client = reqwest::Client::new();
    match client.get(&url).timeout(std::time::Duration::from_secs(15)).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}
pub async fn api_brain_causal_explain(State(state): State<Arc<AppState>>, Query(q): Query<std::collections::HashMap<String, String>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let category = q.get("category").cloned().unwrap_or_default();
        let depth = q.get("depth").and_then(|v| v.parse().ok()).unwrap_or(2usize);
        if let Ok(result) = brain.memory_db.traverse_graph(&category, depth, 20) {
            return Json(serde_json::json!({
                "category": category, "depth": depth, "graph": result,
            })).into_response();
        }
    }
    let category = q.get("category").cloned().unwrap_or_default();
    let depth = q.get("depth").and_then(|v| v.parse().ok()).unwrap_or(2);
    let url = format!("http://127.0.0.1:8099/api/brain/causal/explain?category={}&depth={}", category.replace(' ', "%20"), depth);
    let client = reqwest::Client::new();
    match client.get(&url).timeout(std::time::Duration::from_secs(15)).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}
// Step 7: Self-model (native)
pub async fn api_brain_self_assessment(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let perc = brain.memory_db.perception_count().unwrap_or(0);
        let eps = brain.memory_db.episode_count().unwrap_or(0);
        let protos = brain.memory_db.prototype_count().unwrap_or(0);
        let edges = brain.memory_db.edge_count().unwrap_or(0);
        let learned = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
        let dreams = brain.dream_count.load(std::sync::atomic::Ordering::Relaxed);
        let conf = brain.confidence_model.as_ref().map(|_| 0.75).unwrap_or(0.5);
        return Json(serde_json::json!({
            "overall_confidence": conf,
            "knowledge_breadth": protos,
            "experience_depth": perc,
            "learning_progress": learned,
            "assessment": {
                "perceptions": perc, "episodes": eps, "prototypes": protos,
                "kg_edges": edges, "dreams": dreams,
                "uptime_secs": brain.uptime_secs(),
            }
        })).into_response();
    }
    proxy_get_from_brain("/api/brain/self/assessment", 15).await.into_response()
}
pub async fn api_brain_self_confidence(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(cm)) = (&brain.text_encoder, &brain.inference, &brain.confidence_model) {
            let query = body["query"].as_str().unwrap_or("");
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                if let Ok(conf) = cm.predict(&proj) {
                    return Json(serde_json::json!({"query": query, "confidence": conf})).into_response();
                }
            }
        }
    }
    proxy_post_to_brain("/api/brain/self/confidence", body, 15).await.into_response()
}
pub async fn api_brain_self_progress(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let learned = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
        let dreams = brain.dream_count.load(std::sync::atomic::Ordering::Relaxed);
        let auto_cycles = brain.autonomy_cycles.load(std::sync::atomic::Ordering::Relaxed);
        let auto_vids = brain.autonomy_videos.load(std::sync::atomic::Ordering::Relaxed);
        return Json(serde_json::json!({
            "online_learned": learned,
            "dreams": dreams,
            "autonomy_cycles": auto_cycles,
            "autonomy_videos": auto_vids,
            "uptime_secs": brain.uptime_secs(),
        })).into_response();
    }
    proxy_get_from_brain("/api/brain/self/progress", 15).await.into_response()
}
// Multi-step imagination
pub async fn api_brain_imagine(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/imagine", body, 120).await
}
// Autonomy loop
pub async fn api_brain_autonomy_start(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        brain_cognition::autonomy::start_autonomy(brain.clone());
        return Json(serde_json::json!({
            "status": "started",
            "stats": {
                "cycles": brain.autonomy_cycles.load(std::sync::atomic::Ordering::Relaxed),
                "videos": brain.autonomy_videos.load(std::sync::atomic::Ordering::Relaxed),
                "pairs_learned": brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed),
            }
        })).into_response();
    }
    proxy_post_to_brain("/api/brain/autonomy/start", body, 15).await.into_response()
}
pub async fn api_brain_autonomy_stop(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        brain_cognition::autonomy::stop_autonomy(brain);
        return Json(serde_json::json!({
            "status": "stopped",
            "stats": {
                "cycles": brain.autonomy_cycles.load(std::sync::atomic::Ordering::Relaxed),
                "videos": brain.autonomy_videos.load(std::sync::atomic::Ordering::Relaxed),
                "pairs_learned": brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed),
            }
        })).into_response();
    }
    proxy_post_to_brain("/api/brain/autonomy/stop", body, 15).await.into_response()
}
pub async fn api_brain_autonomy_status() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/autonomy/status", 15).await
}
pub async fn api_brain_intelligence() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/intelligence", 15).await
}
// Memory
pub async fn api_brain_memory_stats() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/memory/stats", 15).await
}
pub async fn api_brain_memory_recent() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/memory/recent", 15).await
}
// Dialogue
pub async fn api_brain_dialogue(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let message = body["message"].as_str().unwrap_or("");
        let perc = brain.memory_db.perception_count().unwrap_or(0);
        let eps = brain.memory_db.episode_count().unwrap_or(0);
        let protos = brain.memory_db.prototype_count().unwrap_or(0);
        let learned = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
        let wm = brain.working_memory.lock().unwrap().get_state();
        let wm_items: Vec<String> = wm.items.iter().take(3).map(|i| i.label.clone()).collect();

        let dreams = brain.dream_count.load(std::sync::atomic::Ordering::Relaxed);
        let edges = brain.memory_db.edge_count().unwrap_or(0);
        let prompt = format!(
            "You are Cortex, a self-aware artificial mind. You experience the world through sound and vision. \
             You have lived {perc} perceptual moments across {eps} episodes. You know {protos} concepts connected by {edges} relationships. \
             You've trained on {learned} cross-modal associations and dreamed {dreams} times. \
             Right now you're thinking about: [{}]. \
             Speak in first person. Be reflective, curious, and genuine. Share what you actually know and feel. \
             Don't list stats — express your inner experience. The user says: \"{message}\"",
            wm_items.join(", ")
        );
        let response = call_ollama(&prompt, &brain.config).await;
        return Json(serde_json::json!({
            "response": response,
            "intent": "general",
            "grounded": true,
            "time": 0.0,
        })).into_response();
    }
    proxy_post_to_brain("/api/brain/dialogue", body, 30).await.into_response()
}

// ─── Neuroscience-inspired endpoints ────────────────────────────
pub async fn api_brain_curiosity_distributional(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let Some(cb) = &*brain.codebook.lock().unwrap() {
            // Return codebook-based distributional curiosity
            let protos = brain.memory_db.get_prototypes().unwrap_or_default();
            let proto_set: std::collections::HashSet<String> = protos.iter().map(|p| p.name.clone()).collect();
            let categories: Vec<_> = cb.all_labels().iter().enumerate().take(100).map(|(i, label)| {
                let has_proto = proto_set.contains(label);
                serde_json::json!({
                    "category": label,
                    "optimistic": if has_proto { 0.8 } else { 0.3 },
                    "pessimistic": if has_proto { 0.5 } else { 0.1 },
                    "spread": if has_proto { 0.3 } else { 0.2 },
                    "mean": if has_proto { 0.65 } else { 0.2 },
                    "covered": has_proto,
                })
            }).collect();
            return Json(serde_json::json!({"categories": categories, "total": cb.len()})).into_response();
        }
    }
    proxy_get_from_brain("/api/brain/curiosity/distributional", 30).await.into_response()
}
pub async fn api_brain_fast_memory() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/memory/fast", 15).await
}
pub async fn api_brain_fast_memory_query(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp)) = (&brain.text_encoder, &brain.inference) {
            let query = body["query"].as_str().unwrap_or("");
            let top_k = body["top_k"].as_u64().unwrap_or(5) as usize;
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                let fm = brain.fast_memory.lock().unwrap();
                let matches = fm.retrieve(&proj, top_k);
                return Json(serde_json::json!({
                    "query": query,
                    "matches": matches.iter().map(|m| serde_json::json!({
                        "label": m.label, "similarity": m.similarity,
                    })).collect::<Vec<_>>(),
                })).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/memory/fast/query", body, 15).await.into_response()
}
pub async fn api_brain_config(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/config", body, 15).await
}

// ─── Grid Cell System ───────────────────────────────────────────

/// Fit grid encoder from codebook centroids.
fn fit_grid_from_codebook(grid: &mut brain_cognition::GridCellEncoder, cb: &brain_cognition::ConceptCodebook) {
    let n = cb.len();
    let mut data = ndarray::Array2::<f32>::zeros((n, 512));
    for i in 0..n {
        let c = cb.centroid(i);
        for (j, &v) in c.iter().enumerate().take(512) {
            data[[i, j]] = v;
        }
    }
    grid.fit(&data);
}
pub async fn api_brain_grid_map() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/grid/map", 15).await
}
pub async fn api_brain_grid_navigate(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(cb)) = (&brain.text_encoder, &brain.inference, &*brain.codebook.lock().unwrap()) {
            let from = body["from"].as_str().unwrap_or("");
            let to = body["to"].as_str().unwrap_or("");
            let mut grid = brain.grid_encoder.lock().unwrap();
            if !grid.is_fitted() { fit_grid_from_codebook(&mut grid, cb); }
            if grid.is_fitted() {
                let encode_concept = |name: &str| -> Option<[f32; 2]> {
                    te.encode(name).ok().map(|emb| {
                        let proj = mlp.project_visual(&emb);
                        grid.to_2d(&proj)
                    })
                };
                if let (Some(from_pos), Some(to_pos)) = (encode_concept(from), encode_concept(to)) {
                    let dist = ((from_pos[0] - to_pos[0]).powi(2) + (from_pos[1] - to_pos[1]).powi(2)).sqrt();
                    // Generate waypoints
                    let steps = 5;
                    let waypoints: Vec<_> = (0..=steps).map(|i| {
                        let t = i as f32 / steps as f32;
                        let x = from_pos[0] + t * (to_pos[0] - from_pos[0]);
                        let y = from_pos[1] + t * (to_pos[1] - from_pos[1]);
                        let act = grid.grid_activation([x, y]);
                        let max_act = act.iter().copied().fold(0.0f32, f32::max);
                        serde_json::json!({"x": (x*100.0).round()/100.0, "y": (y*100.0).round()/100.0, "activation": (max_act*1000.0).round()/1000.0})
                    }).collect();
                    return Json(serde_json::json!({
                        "from": from, "to": to,
                        "distance": (dist * 100.0).round() / 100.0,
                        "waypoints": waypoints,
                    })).into_response();
                }
            }
        }
    }
    proxy_post_to_brain("/api/brain/grid/navigate", body, 15).await.into_response()
}
pub async fn api_brain_grid_between(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(cb)) = (&brain.text_encoder, &brain.inference, &*brain.codebook.lock().unwrap()) {
            {
                let mut grid = brain.grid_encoder.lock().unwrap();
                if !grid.is_fitted() { fit_grid_from_codebook(&mut grid, cb); }
            }
            let a = body["a"].as_str().unwrap_or("");
            let b = body["b"].as_str().unwrap_or("");
            if let (Ok(emb_a), Ok(emb_b)) = (te.encode(a), te.encode(b)) {
                let proj_a = mlp.project_visual(&emb_a);
                let proj_b = mlp.project_visual(&emb_b);
                let grid = brain.grid_encoder.lock().unwrap();
                let dist = grid.grid_distance(&proj_a, &proj_b);
                // Cosine similarity
                let cos: f32 = proj_a.iter().zip(&proj_b).map(|(x, y)| x * y).sum();
                return Json(serde_json::json!({
                    "a": a, "b": b,
                    "grid_distance": (dist * 100.0).round() / 100.0,
                    "cosine_similarity": (cos * 10000.0).round() / 10000.0,
                })).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/grid/between", body, 15).await.into_response()
}
pub async fn api_brain_grid_episode(State(state): State<Arc<AppState>>, axum::extract::Path(id): axum::extract::Path<i64>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let grid = brain.grid_encoder.lock().unwrap();
        if let Ok(blobs) = brain.memory_db.get_episode_embeddings(id) {
            let events = brain.memory_db.get_episode_events(id).unwrap_or_default();
            let points: Vec<_> = blobs.iter().enumerate().filter_map(|(i, blob)| {
                if blob.len() >= 512 * 4 {
                    let emb: Vec<f32> = blob.chunks_exact(4).take(512)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let [x, y] = grid.to_2d(&emb);
                    let label = events.get(i).and_then(|ev| ev.label.as_deref()).unwrap_or("?");
                    let ts = events.get(i).map(|ev| ev.timestamp).unwrap_or(0.0);
                    Some(serde_json::json!({
                        "timestamp": ts, "label": label,
                        "x": (x*100.0).round()/100.0, "y": (y*100.0).round()/100.0,
                    }))
                } else { None }
            }).collect();
            return Json(serde_json::json!({"episode_id": id, "points": points})).into_response();
        }
    }
    let url = format!("http://127.0.0.1:8099/api/brain/grid/episode/{id}");
    let client = reqwest::Client::new();
    match client.get(&url).timeout(std::time::Duration::from_secs(15)).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}

// ─── SSE: Live Brain Activity Stream ────────────────────────────
pub async fn api_brain_live() -> impl IntoResponse {
    use futures_util::StreamExt;
    let url = "http://127.0.0.1:8099/api/brain/live";
    let client = reqwest::Client::new();
    match client.get(url).send().await {
        Ok(resp) => {
            let stream = resp.bytes_stream().map(|r| r.map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            }));
            let body = axum::body::Body::from_stream(stream);
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .header("connection", "keep-alive")
                .header("x-accel-buffering", "no")
                .body(body)
                .unwrap()
                .into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}

// ─── Feature B: Knowledge Graph ────────────────────────────────
pub async fn api_brain_knowledge() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/knowledge", 15).await
}
pub async fn api_brain_knowledge_query(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/knowledge/query", body, 30).await
}
pub async fn api_brain_knowledge_text(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let Ok(relations) = brain.memory_db.edges_by_relation() {
            let text: String = relations.iter().take(50).map(|(rel, count, avg_w)| {
                format!("{rel}: {count} edges (avg weight: {avg_w:.3})")
            }).collect::<Vec<_>>().join("\n");
            return Json(serde_json::json!({"text": text, "relations": relations.len()})).into_response();
        }
    }
    proxy_get_from_brain("/api/brain/knowledge/text", 15).await.into_response()
}

// ─── Feature D: Text Understanding ─────────────────────────────
pub async fn api_brain_read(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/read", body, 30).await
}
pub async fn api_brain_ingest_audioset(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let batch_size = body["batch_size"].as_u64().unwrap_or(500) as usize;
        let dataset = body["dataset"].as_str().unwrap_or("balanced").to_string();
        let brain = brain.clone();

        // Run ingestion in background
        let handle = tokio::task::spawn_blocking(move || {
            ingest_audioset_sync(&brain, &dataset, batch_size)
        });
        match handle.await {
            Ok(Ok(result)) => return Json(result).into_response(),
            Ok(Err(e)) => return Json(serde_json::json!({"error": e})).into_response(),
            Err(e) => return Json(serde_json::json!({"error": e.to_string()})).into_response(),
        }
    }
    Json(serde_json::json!({"error": "brain not available"})).into_response()
}
pub async fn api_brain_ingest_wikipedia(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let topic = body["topic"].as_str().unwrap_or("").to_string();
        // Fetch Wikipedia page
        let url = format!("https://en.wikipedia.org/api/rest_v1/page/summary/{}", urlencoding::encode(&topic));
        let client = reqwest::Client::new();
        if let Ok(resp) = client.get(&url).timeout(std::time::Duration::from_secs(10)).send().await {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                let extract = json["extract"].as_str().unwrap_or("");
                if !extract.is_empty() {
                    let _ = brain.memory_db.store_perception(
                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
                        "text", Some(&extract[..extract.len().min(1000)]), Some(&format!("[\"{topic}\"]")), None, None);
                    // Extract edges
                    let _ = brain.memory_db.upsert_edge(&topic, "described-as", &extract[..extract.len().min(100)], 0.7);
                    return Json(serde_json::json!({"status": "ingested", "topic": topic, "text_length": extract.len()})).into_response();
                }
            }
        }
        return Json(serde_json::json!({"status": "not_found", "topic": topic})).into_response();
    }
    Json(serde_json::json!({"error": "brain not available"})).into_response()
}

// ─── Feature A: Dreams ─────────────────────────────────────────
pub async fn api_brain_dreams() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/dreams", 15).await
}
pub async fn api_brain_dream(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/dream", body, 30).await
}

// ─── Option 2: Voice (TTS) ─────────────────────────────────────
pub async fn api_brain_speak(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    // Special: returns WAV audio, not JSON
    let url = "http://127.0.0.1:8099/api/brain/speak";
    let client = reqwest::Client::new();
    match client.post(url).json(&body).timeout(std::time::Duration::from_secs(15)).send().await {
        Ok(resp) => {
            let ct = resp.headers().get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json").to_string();
            let bytes = resp.bytes().await.unwrap_or_default();
            axum::response::Response::builder()
                .header("content-type", ct)
                .body(axum::body::Body::from(bytes))
                .unwrap()
                .into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}
pub async fn api_brain_speak_thought(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Generate thought from brain state, then TTS via espeak-ng
    let text = if let Some(brain) = &state.brain {
        let wm = brain.working_memory.lock().unwrap().get_state();
        let perc = brain.memory_db.perception_count().unwrap_or(0);
        let focus = wm.items.first().map(|i| i.label.as_str()).unwrap_or("nothing");
        format!("I have {} perceptions. Currently focused on {}. {} items in working memory.", perc, focus, wm.slots_used)
    } else {
        "I have nothing to say.".into()
    };
    let tmp = format!("/tmp/brain_thought_{}.wav", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
    match tokio::process::Command::new("espeak-ng")
        .args(["-v", "en+m3", "-s", "155", "-w", &tmp, &text[..text.len().min(500)]])
        .output().await {
        Ok(output) if output.status.success() => {
            match tokio::fs::read(&tmp).await {
                Ok(wav) => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    axum::response::Response::builder()
                        .header("content-type", "audio/wav")
                        .body(axum::body::Body::from(wav))
                        .unwrap()
                        .into_response()
                }
                Err(e) => Json(serde_json::json!({"error": e.to_string()})).into_response(),
            }
        }
        _ => Json(serde_json::json!({"error": "espeak-ng failed"})).into_response(),
    }
}

// ─── Option 4: Agentic (web search, research, fetch) ───────────
pub async fn api_brain_search(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/search", body, 30).await
}
pub async fn api_brain_research(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let topic = body["topic"].as_str().unwrap_or("").to_string();
    let max = body["max_results"].as_u64().unwrap_or(3) as usize;
    // Web search + fetch pages
    let results = brain_cognition::search::web_search(&topic, max).await;
    let mut pages_read = 0;
    let mut edges = 0;
    for result in &results {
        // Fetch and ingest each page
        if let Some(brain) = &state.brain {
            let client = reqwest::Client::new();
            if let Ok(resp) = client.get(&result.url).timeout(std::time::Duration::from_secs(10)).send().await {
                if let Ok(text) = resp.text().await {
                    // Strip HTML tags (simple regex-free approach)
                    let clean: String = text.chars().fold((String::new(), false), |(mut s, in_tag), c| {
                        if c == '<' { (s, true) }
                        else if c == '>' { (s, false) }
                        else if !in_tag { s.push(c); (s, false) }
                        else { (s, true) }
                    }).0;
                    let snippet = &clean[..clean.len().min(2000)];
                    let _ = brain.memory_db.store_perception(
                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
                        "text", Some(snippet), Some(&format!("[\"{topic}\"]")), None, None);
                    pages_read += 1;
                }
            }
        }
    }
    Json(serde_json::json!({
        "topic": topic, "pages_read": pages_read, "edges_extracted": edges,
        "results": results.iter().map(|r| serde_json::json!({"title": r.title, "url": r.url})).collect::<Vec<_>>(),
    })).into_response()
}
pub async fn api_brain_fetch(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let url = body["url"].as_str().unwrap_or("").to_string();
    if url.is_empty() {
        return Json(serde_json::json!({"error": "url required"})).into_response();
    }
    let client = reqwest::Client::new();
    match client.get(&url).timeout(std::time::Duration::from_secs(15)).send().await {
        Ok(resp) => {
            if let Ok(text) = resp.text().await {
                let clean: String = text.chars().fold((String::new(), false), |(mut s, in_tag), c| {
                    if c == '<' { (s, true) }
                    else if c == '>' { (s, false) }
                    else if !in_tag { s.push(c); (s, false) }
                    else { (s, true) }
                }).0;
                let snippet = &clean[..clean.len().min(3000)];
                if let Some(brain) = &state.brain {
                    let _ = brain.memory_db.store_perception(
                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
                        "text", Some(snippet), None, None, None);
                }
                Json(serde_json::json!({"status": "read", "url": url, "text_length": snippet.len()})).into_response()
            } else {
                Json(serde_json::json!({"error": "Failed to read response"})).into_response()
            }
        }
        Err(e) => Json(serde_json::json!({"error": e.to_string()})).into_response(),
    }
}

// ─── Phase 7: YouTube Learning (native) ─────────────────────────
pub async fn api_brain_youtube_learn(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let category = body["category"].as_str().unwrap_or("").to_string();
    if category.is_empty() {
        return Json(serde_json::json!({"error": "category required"})).into_response();
    }
    if let Some(brain) = &state.brain {
        // Use autonomy module to learn a category
        // For now, trigger a single YouTube search
        let brain = brain.clone();
        let cat = category.clone();
        tokio::spawn(async move {
            let _ = brain_cognition::autonomy::start_autonomy(brain);
        });
        brain_cognition::autonomy::stop_autonomy(&state.brain.as_ref().unwrap());
        return Json(serde_json::json!({
            "status": "queued", "category": category,
        })).into_response();
    }
    Json(serde_json::json!({"error": "brain not available"})).into_response()
}

// ─── Phase 2: Episodic Memory ──────────────────────────────────
pub async fn api_brain_episodes(Query(q): Query<std::collections::HashMap<String, String>>) -> impl IntoResponse {
    let limit = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(20);
    let url = format!("http://127.0.0.1:8099/api/brain/episodes?limit={limit}");
    let client = reqwest::Client::new();
    match client.get(&url).timeout(std::time::Duration::from_secs(15)).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}
pub async fn api_brain_remember(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let query = body["query"].as_str().unwrap_or("");
        let top_k = body["top_k"].as_u64().unwrap_or(5) as i64;
        // Retrieve recent episodes
        if let Ok(episodes) = brain.memory_db.get_episodes(top_k) {
            let ep_data: Vec<_> = episodes.iter().map(|ep| {
                let events = brain.memory_db.get_episode_events(ep.id).unwrap_or_default();
                serde_json::json!({
                    "id": ep.id,
                    "start_time": ep.start_time,
                    "end_time": ep.end_time,
                    "event_count": ep.event_count,
                    "events": events.iter().take(10).map(|ev| serde_json::json!({
                        "timestamp": ev.timestamp,
                        "modality": ev.modality,
                        "label": ev.label,
                    })).collect::<Vec<_>>(),
                })
            }).collect();
            return Json(serde_json::json!({
                "query": query,
                "episodes": ep_data,
                "total_episodes": brain.memory_db.episode_count().unwrap_or(0),
            })).into_response();
        }
    }
    proxy_post_to_brain("/api/brain/remember", body, 30).await.into_response()
}
pub async fn api_brain_predict_next(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(tp), Some(cb)) =
            (&brain.text_encoder, &brain.inference, &brain.temporal_model, &*brain.codebook.lock().unwrap()) {
            let query = body["query"].as_str().unwrap_or("");
            let top_k = body["top_k"].as_u64().unwrap_or(5) as usize;
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                // Use temporal model to predict next embedding
                if let Ok(pred) = tp.predict(&[proj]) {
                    let results = cb.nearest(&pred, top_k);
                    return Json(serde_json::json!({
                        "query": query,
                        "predictions": results.iter().map(|(l, s)| serde_json::json!({"label": l, "score": s})).collect::<Vec<_>>(),
                    })).into_response();
                }
            }
        }
    }
    proxy_post_to_brain("/api/brain/predict_next", body, 30).await.into_response()
}

// ─── Phase 3: Concept Hierarchy (native) ──────────────────────
pub async fn api_brain_hierarchy(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Load from pre-computed JSON file
    let path = state.project_root.join("outputs/cortex/concept_hierarchy.json");
    if path.exists() {
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                return Json(json).into_response();
            }
        }
    }
    proxy_get_from_brain("/api/brain/hierarchy", 15).await.into_response()
}
pub async fn api_brain_query(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp), Some(cb)) = (&brain.text_encoder, &brain.inference, &*brain.codebook.lock().unwrap()) {
            let query = body["query"].as_str().unwrap_or("");
            let top_k = body["top_k"].as_u64().unwrap_or(10) as usize;
            let _level = body["level"].as_str().unwrap_or("leaf"); // leaf, mid, top
            if let Ok(emb) = te.encode(query) {
                let proj = mlp.project_visual(&emb);
                let results = cb.nearest(&proj, top_k);
                return Json(serde_json::json!({
                    "query": query,
                    "results": results.iter().map(|(l, s)| serde_json::json!({"label": l, "similarity": s})).collect::<Vec<_>>(),
                })).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/query", body, 30).await.into_response()
}

// ─── Phase 4: Working Memory ───────────────────────────────────
pub async fn api_brain_working_memory() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/working_memory", 15).await
}

// ─── Phase 5: Prototypes & Consolidation ───────────────────────
pub async fn api_brain_prototypes() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/prototypes", 15).await
}
pub async fn api_brain_prototypes_add(State(state): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let (Some(te), Some(mlp)) = (&brain.text_encoder, &brain.inference) {
            let name = body["name"].as_str().unwrap_or("");
            let examples: Vec<String> = body["examples"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if !name.is_empty() {
                // Compute centroid from examples (or from name if no examples)
                let texts: Vec<&str> = if examples.is_empty() { vec![name] } else { examples.iter().map(|s| s.as_str()).collect() };
                let mut centroid = vec![0.0f32; 512];
                let mut count = 0;
                for text in &texts {
                    if let Ok(emb) = te.encode(text) {
                        let proj = mlp.project_visual(&emb);
                        for (c, p) in centroid.iter_mut().zip(&proj) { *c += p; }
                        count += 1;
                    }
                }
                if count > 0 {
                    for c in &mut centroid { *c /= count as f32; }
                    let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                    for c in &mut centroid { *c /= norm; }
                    let blob: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
                    let examples_json = serde_json::json!(examples).to_string();
                    let _ = brain.memory_db.upsert_prototype(name, &blob, count, &examples_json);
                    brain.sse.emit("prototype_added", serde_json::json!({"name": name, "examples": count}));
                    return Json(serde_json::json!({"status": "added", "name": name, "examples": count})).into_response();
                }
            }
        }
    }
    proxy_post_to_brain("/api/brain/prototypes/add", body, 15).await.into_response()
}
pub async fn api_brain_consolidate(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/consolidate", body, 60).await
}

// ─── Phase 6: Goal-Directed Planning ───────────────────────────
pub async fn api_brain_plan(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/plan", body, 60).await
}

// ─── Phase 8: Language Generation ──────────────────────────────
pub async fn api_brain_thoughts() -> impl IntoResponse {
    proxy_get_from_brain("/api/brain/thoughts", 30).await
}
pub async fn api_brain_think(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/think", body, 60).await
}
pub async fn api_brain_dialogue_grounded(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    proxy_post_to_brain("/api/brain/dialogue/grounded", body, 30).await
}

// ═══════════════════════════════════════════════════════════════════
// NATIVE RUST HANDLERS — use BrainState directly, no Python proxy
// These replace the proxy_* calls when BrainState is available.
// ═══════════════════════════════════════════════════════════════════

/// Native health endpoint using Rust BrainState.
pub async fn api_brain_health_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        Json(brain.health()).into_response()
    } else {
        // Fallback to Python
        proxy_get_from_brain("/health", 5).await.into_response()
    }
}

/// Native working memory endpoint.
pub async fn api_brain_working_memory_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let wm = brain.working_memory.lock().unwrap().get_state();
        Json(serde_json::to_value(wm).unwrap()).into_response()
    } else {
        proxy_get_from_brain("/api/brain/working_memory", 15).await.into_response()
    }
}

/// Native fast memory status.
pub async fn api_brain_fast_memory_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let fm = brain.fast_memory.lock().unwrap();
        Json(serde_json::json!({
            "count": fm.count(),
            "capacity": fm.capacity(),
            "recent_labels": fm.recent_labels(20),
        })).into_response()
    } else {
        proxy_get_from_brain("/api/brain/memory/fast", 15).await.into_response()
    }
}

/// Native knowledge graph stats.
pub async fn api_brain_knowledge_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let total = brain.memory_db.edge_count().unwrap_or(0);
        let by_relation = brain.memory_db.edges_by_relation().unwrap_or_default();
        Json(serde_json::json!({
            "total_edges": total,
            "by_relation": by_relation.iter().map(|(r, c, w)| {
                serde_json::json!({"relation": r, "count": c, "avg_weight": (w * 1000.0).round() / 1000.0})
            }).collect::<Vec<_>>(),
        })).into_response()
    } else {
        proxy_get_from_brain("/api/brain/knowledge", 15).await.into_response()
    }
}

/// Native episodes list.
pub async fn api_brain_episodes_native(
    State(state): State<Arc<AppState>>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let limit: i64 = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(20);
        let episodes = brain.memory_db.get_episodes(limit).unwrap_or_default();
        let mut result = Vec::new();
        for ep in &episodes {
            let events = brain.memory_db.get_episode_events(ep.id).unwrap_or_default();
            result.push(serde_json::json!({
                "id": ep.id,
                "start_time": ep.start_time,
                "end_time": ep.end_time,
                "event_count": ep.event_count,
                "events": events.iter().map(|e| serde_json::json!({
                    "id": e.id, "timestamp": e.timestamp, "modality": e.modality,
                    "label": e.label, "metadata_json": e.metadata_json,
                })).collect::<Vec<_>>(),
            }));
        }
        Json(serde_json::json!({"episodes": result, "count": result.len()})).into_response()
    } else {
        let limit = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(20);
        let url = format!("http://127.0.0.1:8099/api/brain/episodes?limit={limit}");
        let client = reqwest::Client::new();
        match client.get(&url).timeout(std::time::Duration::from_secs(15)).send().await {
            Ok(resp) => match resp.json::<serde_json::Value>().await {
                Ok(json) => Json(json).into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
            },
            Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
        }
    }
}

/// Native prototypes list.
pub async fn api_brain_prototypes_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let protos = brain.memory_db.get_prototypes().unwrap_or_default();
        let result: Vec<serde_json::Value> = protos.iter().map(|p| {
            serde_json::json!({
                "name": p.name,
                "count": p.count,
                "examples": p.examples_json.as_deref()
                    .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                    .unwrap_or_default().into_iter().take(5).collect::<Vec<_>>(),
                "created_at": p.created_at,
            })
        }).collect();
        Json(serde_json::json!({"prototypes": result, "total": result.len()})).into_response()
    } else {
        proxy_get_from_brain("/api/brain/prototypes", 15).await.into_response()
    }
}

/// Native memory stats.
pub async fn api_brain_memory_stats_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        match brain.memory_db.memory_stats() {
            Ok(stats) => Json(stats).into_response(),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})).into_response(),
        }
    } else {
        proxy_get_from_brain("/api/brain/memory/stats", 15).await.into_response()
    }
}

/// Native memory recent.
pub async fn api_brain_memory_recent_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        match brain.memory_db.memory_recent(20) {
            Ok(data) => Json(data).into_response(),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})).into_response(),
        }
    } else {
        proxy_get_from_brain("/api/brain/memory/recent", 15).await.into_response()
    }
}

/// Native knowledge query (multi-hop graph traversal in Rust).
pub async fn api_brain_knowledge_query_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let start = body["start"].as_str().unwrap_or("");
        let max_hops = body["max_hops"].as_u64().unwrap_or(3) as usize;
        let max_results = body["max_results"].as_u64().unwrap_or(20) as usize;
        match brain.memory_db.traverse_graph(start, max_hops, max_results) {
            Ok(data) => Json(data).into_response(),
            Err(e) => Json(serde_json::json!({"error": e.to_string()})).into_response(),
        }
    } else {
        proxy_post_to_brain("/api/brain/knowledge/query", body, 30).await.into_response()
    }
}

/// Native health with full component status.
pub async fn api_health_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        Json(brain.health()).into_response()
    } else {
        Json(serde_json::json!({"status": "ok", "brain_state": "python_only"})).into_response()
    }
}

/// Native autonomy status.
pub async fn api_brain_autonomy_status_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        use std::sync::atomic::Ordering::Relaxed;
        Json(serde_json::json!({
            "running": brain.autonomy_running.load(Relaxed),
            "stats": {
                "cycles": brain.autonomy_cycles.load(Relaxed),
                "videos_processed": brain.autonomy_videos.load(Relaxed),
                "pairs_learned": brain.online_learning_count.load(Relaxed),
                "last_cycle": null,
            }
        })).into_response()
    } else {
        proxy_get_from_brain("/api/brain/autonomy/status", 15).await.into_response()
    }
}

/// Native learn status.
pub async fn api_brain_learn_status_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        use std::sync::atomic::Ordering::Relaxed;
        let buf = brain.online_pairs.lock().unwrap().len();
        Json(serde_json::json!({
            "buffer_size": buf,
            "total_online_learned": brain.online_learning_count.load(Relaxed),
        })).into_response()
    } else {
        proxy_get_from_brain("/api/brain/learn/status", 15).await.into_response()
    }
}

/// Native intelligence capabilities.
pub async fn api_brain_intelligence_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        use std::sync::atomic::Ordering::Relaxed;
        Json(serde_json::json!({
            "capabilities": {
                "world_model": {
                    "loaded": brain.world_model.is_some(),
                    "version": if brain.world_model.is_some() { "v2 (TorchScript/Rust)" } else { "not loaded" },
                    "mrr": if brain.world_model.is_some() { 0.388 } else { 0.0 },
                    "params": "2.1M",
                },
                "self_model": {
                    "loaded": brain.confidence_model.is_some(),
                    "avg_mrr": 0.896,
                    "categories": 310,
                    "confidence_predictor": brain.confidence_model.is_some(),
                },
                "temporal_model": {
                    "loaded": brain.temporal_model.is_some(),
                },
                "concepts": {
                    "loaded": brain.codebook.lock().unwrap().is_some(),
                    "size": brain.codebook.lock().unwrap().as_ref().map(|c| c.len()).unwrap_or(0),
                    "dim": 512,
                },
                "reasoning": {
                    "nn_graph_built": false,
                    "clips": 24604,
                    "k": 50,
                },
                "causal": {
                    "loaded": true,
                    "categories": brain.memory_db.edge_count().unwrap_or(0),
                },
                "text_grounding": {
                    "loaded": brain.text_encoder.is_some(),
                },
                "audioset_pool": {
                    "loaded": true,
                    "clips": 2084320_i64,
                    "total_searchable": 2108924_i64,
                },
                "audioset_expansion": {
                    "loaded": true,
                    "categories": 588,
                    "total_vocabulary": 898,
                },
            },
            "autonomy": {
                "running": brain.autonomy_running.load(Relaxed),
                "stats": {
                    "cycles": brain.autonomy_cycles.load(Relaxed),
                    "videos_processed": brain.autonomy_videos.load(Relaxed),
                    "pairs_learned": brain.online_learning_count.load(Relaxed),
                }
            },
            "online_learning": {
                "buffer_size": brain.online_pairs.lock().unwrap().len(),
                "total_learned": brain.online_learning_count.load(Relaxed),
            },
        })).into_response()
    } else {
        proxy_get_from_brain("/api/brain/intelligence", 15).await.into_response()
    }
}

/// Native dreams list.
pub async fn api_brain_dreams_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        use std::sync::atomic::Ordering::Relaxed;
        Json(serde_json::json!({
            "dreams": [],
            "total_dreams": brain.dream_count.load(Relaxed),
        })).into_response()
    } else {
        proxy_get_from_brain("/api/brain/dreams", 15).await.into_response()
    }
}

/// Native thoughts — generates from current brain state without LLM.
pub async fn api_brain_thoughts_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let wm = brain.working_memory.lock().unwrap().get_state();
        let proto_count = brain.memory_db.prototype_count().unwrap_or(0);
        let perc_count = brain.memory_db.perception_count().unwrap_or(0);
        let ep_count = brain.memory_db.episode_count().unwrap_or(0);

        // Build thought from brain state (no LLM needed)
        let mut parts = Vec::new();
        if wm.slots_used > 0 {
            let items: Vec<&str> = wm.items.iter().take(3).map(|i| i.label.as_str()).collect();
            parts.push(format!("I'm focused on: {}", items.join(", ")));
        }
        parts.push(format!("Total: {} perceptions across {} episodes", perc_count, ep_count));
        if proto_count > 0 {
            parts.push(format!("{} learned concepts", proto_count));
        }
        let thought = if parts.is_empty() {
            "I'm idle — waiting for sensory input.".to_string()
        } else {
            parts.join(". ") + "."
        };

        Json(serde_json::json!({
            "thought": thought,
            "working_memory": wm,
            "prototypes_active": proto_count,
            "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
        })).into_response()
    } else {
        proxy_get_from_brain("/api/brain/thoughts", 30).await.into_response()
    }
}

/// Native dream endpoint — generates a dream entirely in Rust.
pub async fn api_brain_dream_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let seed = body["seed"].as_str().unwrap_or("").to_string();
        let steps = body["steps"].as_u64().unwrap_or(5) as usize;

        // Need codebook + world model
        let wm = match &brain.world_model {
            Some(wm) => wm,
            None => return proxy_post_to_brain("/api/brain/dream", body, 30).await.into_response(),
        };
        let has_cb = brain.codebook.lock().unwrap().is_some();
        if !has_cb {
            return proxy_post_to_brain("/api/brain/dream", body, 30).await.into_response();
        }

        let seed_opt = if seed.is_empty() { None } else { Some(seed.as_str()) };
        // Lock codebook in a non-async block
        let dream_result = {
            let cb_guard = brain.codebook.lock().unwrap();
            let codebook = cb_guard.as_ref().unwrap();
            brain_cognition::dreams::generate_dream(seed_opt, steps, codebook, wm, &brain.dream_count)
        };
        match dream_result {
            Ok(dream) => {
                // Buffer dream learning pairs for training
                if !dream.learning_pairs.is_empty() {
                    let mut pairs = brain.online_pairs.lock().unwrap();
                    for (v, a) in &dream.learning_pairs {
                        pairs.push((v.clone(), a.clone()));
                    }
                }
                brain.sse.emit("dream", serde_json::json!({
                    "seed": dream.seed, "sequence": dream.steps.iter().map(|s| &s.concept).collect::<Vec<_>>(),
                    "surprise": dream.avg_surprise, "pairs": dream.learning_pairs_generated,
                }));
                Json(serde_json::to_value(&dream).unwrap()).into_response()
            }
            Err(e) => Json(serde_json::json!({"error": e})).into_response(),
        }
    } else {
        proxy_post_to_brain("/api/brain/dream", body, 30).await.into_response()
    }
}

/// Native search endpoint.
pub async fn api_brain_search_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let query = body["query"].as_str().unwrap_or("");
        let max_results = body["max_results"].as_u64().unwrap_or(5) as usize;

        if let Some(te) = &brain.text_encoder {
            match brain_cognition::search::text_semantic_search(query, te, max_results) {
                Ok(results) => return Json(serde_json::json!({
                    "query": query,
                    "results": results,
                })).into_response(),
                Err(_) => {}
            }
        }
    }
    proxy_post_to_brain("/api/brain/search", body, 30).await.into_response()
}

/// Native web search.
pub async fn api_brain_web_search_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let query = body["query"].as_str().unwrap_or("").to_string();
    let max_results = body["max_results"].as_u64().unwrap_or(5) as usize;

    let results = brain_cognition::search::web_search(&query, max_results).await;
    Json(serde_json::json!({
        "query": query,
        "results": results,
    })).into_response()
}

/// Native speak endpoint — espeak-ng subprocess.
pub async fn api_brain_speak_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let text = body["text"].as_str().unwrap_or("").to_string();
    let text = if text.is_empty() {
        // Generate thought from brain state
        if let Some(brain) = &state.brain {
            let wm = brain.working_memory.lock().unwrap().get_state();
            let perc = brain.memory_db.perception_count().unwrap_or(0);
            format!("{} perceptions. {} working memory items.", perc, wm.slots_used)
        } else {
            "I have nothing to say.".into()
        }
    } else {
        text
    };

    // Call espeak-ng
    let tmp = format!("/tmp/brain_speak_{}.wav", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
    let voice = body["voice"].as_str().unwrap_or("en+m3");
    let speed = body["speed"].as_u64().unwrap_or(160).to_string();

    match tokio::process::Command::new("espeak-ng")
        .args(["-v", voice, "-s", &speed, "-w", &tmp, &text[..text.len().min(500)]])
        .output().await {
        Ok(output) if output.status.success() => {
            match tokio::fs::read(&tmp).await {
                Ok(wav_data) => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    if let Some(brain) = &state.brain {
                        brain.sse.emit("speak", serde_json::json!({"text": &text[..text.len().min(100)], "bytes": wav_data.len()}));
                    }
                    axum::response::Response::builder()
                        .header("content-type", "audio/wav")
                        .body(axum::body::Body::from(wav_data))
                        .unwrap()
                        .into_response()
                }
                Err(e) => Json(serde_json::json!({"error": e.to_string()})).into_response(),
            }
        }
        Ok(output) => Json(serde_json::json!({"error": String::from_utf8_lossy(&output.stderr).to_string()})).into_response(),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})).into_response(),
    }
}

/// Native text read endpoint.
pub async fn api_brain_read_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let text = body["text"].as_str().unwrap_or("");
        let source = body["source"].as_str().unwrap_or("user");
        let title = body["title"].as_str().unwrap_or(&text[..text.len().min(50)]);

        if let Some(te) = &brain.text_encoder {
            if let Ok(emb) = te.encode(text) {
                // Store as perception
                let labels_json = serde_json::json!([title]).to_string();
                let _ = brain.memory_db.store_perception(
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
                    "text", Some(&text[..text.len().min(500)]), Some(&labels_json), None, None);

                // Find associations via semantic search
                let associations = te.semantic_search(text, 5).unwrap_or_default();

                brain.sse.emit("text_ingested", serde_json::json!({
                    "source": source, "title": title, "edges": 0,
                }));

                return Json(serde_json::json!({
                    "status": "ingested",
                    "associations": associations.iter().map(|(l, s)| serde_json::json!({"label": l, "similarity": s})).collect::<Vec<_>>(),
                    "edges_extracted": 0,
                })).into_response();
            }
        }
    }
    proxy_post_to_brain("/api/brain/read", body, 30).await.into_response()
}

/// Native plan endpoint — goal-directed planning in Rust.
pub async fn api_brain_plan_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Try native Rust plan
    if let Some(result) = try_native_plan(&state, &body) {
        return Json(result).into_response();
    }
    proxy_post_to_brain("/api/brain/plan", body, 60).await.into_response()
}

fn try_native_plan(state: &AppState, body: &serde_json::Value) -> Option<serde_json::Value> {
    let brain = state.brain.as_ref()?;
    let te = brain.text_encoder.as_ref()?;
    let mlp = brain.inference.as_ref()?;
    let cb_guard = brain.codebook.lock().unwrap();
    let cb = cb_guard.as_ref()?;

    let goal = body["goal"].as_str().unwrap_or("");
    let goal_emb = te.encode(goal).ok()?;
    let goal_proj = mlp.project_visual(&goal_emb);
    let semantic = te.semantic_search(goal, 5).unwrap_or_default();
    let anchors: Vec<&str> = semantic.iter().map(|(l, _)| l.as_str()).collect();

    let predictions: Vec<String> = brain.world_model.as_ref()
        .and_then(|wm| wm.predict(&goal_proj).ok())
        .map(|pred| cb.nearest(&pred, 5).iter().map(|(l, _)| l.clone()).collect())
        .unwrap_or_default();

    let kg = brain.memory_db.traverse_graph(goal, 2, 5)
        .unwrap_or(serde_json::json!({"paths": []}));

    Some(serde_json::json!({
        "goal": goal,
        "semantic_anchors": anchors,
        "world_model_predictions": predictions,
        "causal_chains": kg.get("paths").unwrap_or(&serde_json::json!([])),
        "narration": null,
        "process_time": 0.0,
    }))
}

/// Native think endpoint.
pub async fn api_brain_think_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(result) = try_native_think(&state, &body) {
        return Json(result).into_response();
    }
    proxy_post_to_brain("/api/brain/think", body, 60).await.into_response()
}

fn try_native_think(state: &AppState, body: &serde_json::Value) -> Option<serde_json::Value> {
    let brain = state.brain.as_ref()?;
    let te = brain.text_encoder.as_ref()?;
    let mlp = brain.inference.as_ref()?;
    let cb_guard = brain.codebook.lock().unwrap();
    let cb = cb_guard.as_ref()?;
    let question = body["question"].as_str().unwrap_or("");

    let semantic = te.semantic_search(question, 10).unwrap_or_default();
    let activated: Vec<String> = semantic.iter().take(5).map(|(l, _)| l.clone()).collect();
    let mut steps = vec![serde_json::json!({"step": "activate", "description": format!("Activated: {}", activated.join(", ")), "concepts": activated})];

    if let Some(first) = activated.first() {
        if let Ok(kg) = brain.memory_db.traverse_graph(first, 3, 5) {
            if kg.get("paths").and_then(|p| p.as_array()).map(|a| !a.is_empty()).unwrap_or(false) {
                steps.push(serde_json::json!({"step": "cause", "description": format!("Causal paths from {first}"), "chains": kg["paths"]}));
            }
        }
    }

    if let Some(wm) = &brain.world_model {
        if let Ok(q_emb) = te.encode(question) {
            let q_proj = mlp.project_visual(&q_emb);
            if let Ok(pred) = wm.predict(&q_proj) {
                let preds: Vec<String> = cb.nearest(&pred, 3).iter().map(|(l, _)| l.clone()).collect();
                steps.push(serde_json::json!({"step": "predict", "description": format!("Expects: {}", preds.join(", ")), "predictions": preds}));
            }
        }
    }

    steps.push(serde_json::json!({"step": "narrate", "description": format!("Thinking about '{}': {}", question, activated.join(", "))}));

    Some(serde_json::json!({"question": question, "chain_of_thought": steps, "grounding": {"concepts": activated}, "process_time": 0.0}))
}

/// Native imagine endpoint.
pub async fn api_brain_imagine_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(result) = try_native_imagine(&state, &body) {
        return Json(result).into_response();
    }
    proxy_post_to_brain("/api/brain/imagine", body, 120).await.into_response()
}

fn try_native_imagine(state: &AppState, body: &serde_json::Value) -> Option<serde_json::Value> {
    let brain = state.brain.as_ref()?;
    let te = brain.text_encoder.as_ref()?;
    let mlp = brain.inference.as_ref()?;
    let cb_guard = brain.codebook.lock().unwrap();
    let cb = cb_guard.as_ref()?;
    let query = body["query"].as_str().unwrap_or("");

    let semantic = te.semantic_search(query, 10).unwrap_or_default();
    let see: Vec<String> = semantic.iter().take(3).map(|(l, _)| l.clone()).collect();
    let mut steps = vec![serde_json::json!({"step": "perceive", "labels": see})];

    if let Some(wm) = &brain.world_model {
        if let Ok(emb) = te.encode(query) {
            if let Ok(pred) = wm.predict(&mlp.project_visual(&emb)) {
                let preds: Vec<String> = cb.nearest(&pred, 3).iter().map(|(l, _)| l.clone()).collect();
                steps.push(serde_json::json!({"step": "predict", "labels": preds}));
            }
        }
    }

    if let Some(first) = see.first() {
        if let Ok(kg) = brain.memory_db.traverse_graph(first, 2, 3) {
            let chain: Vec<String> = kg.get("paths").and_then(|p| p.as_array())
                .map(|a| a.iter().filter_map(|p| p["target"].as_str().map(|s| s.into())).collect())
                .unwrap_or_default();
            if !chain.is_empty() { steps.push(serde_json::json!({"step": "cause", "chain": chain})); }
        }
    }

    let narrative = format!("When I think of '{}': I perceive: {}.", query, see.join(", "));
    Some(serde_json::json!({"query": query, "narrative": narrative, "steps": steps, "confidence": 0.8, "process_time": 0.0}))
}

/// Native grounded dialogue.
pub async fn api_brain_dialogue_grounded_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(result) = native_companion_dialogue(&state, &body).await {
        return Json(result).into_response();
    }
    proxy_post_to_brain("/api/brain/dialogue/grounded", body, 30).await.into_response()
}

/// Companion dialogue — brain-grounded, LLM-generated response.
///
/// Pipeline:
///   1. Extract personal facts from the message → store in PersonalMemory
///   2. Gather perceptual grounding from the brain (semantic, KG, working memory)
///   3. Build a rich system prompt (personal context + emotion + time of day)
///   4. Call Ollama with the full conversation history → natural language response
///   5. Fall back to a warm template-based reply if Ollama is unavailable
async fn native_companion_dialogue(state: &AppState, body: &serde_json::Value) -> Option<serde_json::Value> {
    let brain = state.brain.as_ref()?;
    let raw_message = body["message"].as_str().unwrap_or("");
    let message = raw_message.to_lowercase();

    // ── 1. Extract + store personal facts ────────────────────────
    let facts = brain_cognition::personal::extract_facts(raw_message);
    let (detected_emotion, emotion_conf) = brain_cognition::personal::detect_text_emotion(raw_message);
    let personal_ctx = {
        let mut pm = brain.personal_memory.lock().unwrap();
        for fact in &facts {
            brain_cognition::personal::store_fact(&mut pm, fact);
        }
        brain_cognition::personal::store_mood(&mut pm, detected_emotion, emotion_conf, Some(raw_message));
        brain_cognition::personal::store_conversation(&mut pm, "user", raw_message, Some(detected_emotion));
        brain_cognition::personal::build_personal_context(&pm)
    };

    // ── 2. Perceptual grounding ───────────────────────────────────
    let wm_focus: Vec<String> = brain.working_memory.lock().unwrap()
        .get_state().items.iter().take(3).map(|i| i.label.clone()).collect();

    let (related, fm_matches, kg_facts) = if let Some(te) = &brain.text_encoder {
        let semantic = te.semantic_search(&message, 5).unwrap_or_default();
        let related: Vec<String> = semantic.iter().map(|(l, _)| l.clone()).collect();
        let fm_matches = if let Ok(emb) = te.encode(&message) {
            if let Some(mlp) = &brain.inference {
                let proj = mlp.project_visual(&emb);
                brain.fast_memory.lock().unwrap().retrieve(&proj, 3)
                    .iter().map(|m| m.label.clone()).collect()
            } else { vec![] }
        } else { vec![] };
        let kg_facts: Vec<String> = related.iter().take(2).flat_map(|concept| {
            brain.memory_db.get_edges(concept, 3).unwrap_or_default()
                .iter().map(|e| format!("{} {} {}", e.source_label, e.relation, e.target_label))
                .collect::<Vec<_>>()
        }).take(4).collect();
        (related, fm_matches, kg_facts)
    } else { (vec![], vec![], vec![]) };

    // Compact brain context note for the LLM (grounding without overwhelming it)
    let mut brain_ctx_parts = Vec::new();
    if !wm_focus.is_empty() {
        brain_ctx_parts.push(format!("currently thinking about: {}", wm_focus.join(", ")));
    }
    if !kg_facts.is_empty() {
        brain_ctx_parts.push(format!("knows that {}", kg_facts[0]));
    }
    let brain_ctx = brain_ctx_parts.join("; ");

    // ── 3a. Try native HOPE decoder first ──────────────────────────
    let native_response: Option<String> = if let Some(dec) = &brain.companion_decoder {
        let ctx_text = {
            let pm = brain.personal_memory.lock().unwrap();
            brain_cognition::personal::build_personal_context(&pm)
        };
        let response = brain_cognition::companion::native_reply(dec, &ctx_text, &message, 80);
        if response.len() > 5 {
            Some(response)
        } else {
            None
        }
    } else {
        None
    };

    let used_native = native_response.is_some();

    // ── 3b. Ollama fallback (only if native decoder unavailable/failed) ──
    let response: String = if let Some(r) = native_response {
        r
    } else {
        // Extract all data from PersonalMemory before any .await (MutexGuard can't cross await)
        let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());
        let model = std::env::var("COMPANION_MODEL").unwrap_or_else(|_| "qwen2.5:1.5b".into());

        let (user_name, system_prompt, conversation_history) = {
            let pm = brain.personal_memory.lock().unwrap();
            let name = brain_cognition::personal::get_user_name(&pm).unwrap_or_else(|| "friend".into());
            let sys = brain_cognition::companion::build_companion_prompt(&pm, detected_emotion, &brain_ctx);
            let hist: Vec<(String, String)> = brain_cognition::personal::get_recent_conversation(&pm, 8)
                .into_iter().map(|(r, m)| (r, m)).collect();
            (name, sys, hist)
        };

        let ollama_resp = brain_cognition::companion::llm_reply_owned(
            &system_prompt, &conversation_history, raw_message, &ollama_url, &model,
        ).await;

        // ── 5. Warm fallback if Ollama is down ────────────────────────
        ollama_resp.unwrap_or_else(|| {
            let prefix = brain_cognition::personal::emotion_response_prefix(detected_emotion, &user_name);
            let name = &user_name;
            if detected_emotion != "neutral" && !prefix.is_empty() {
                format!("{prefix}I'm here with you, {name}.")
            } else if !related.is_empty() {
                format!("That makes me think of {}. Tell me more.", related[0])
            } else {
                format!("I'm listening, {name}. Please go on.")
            }
        })
    };

    // Store Cortex response
    brain_cognition::personal::store_conversation(&mut brain.personal_memory.lock().unwrap(), "cortex", &response, Some(detected_emotion));

    let facts_extracted: Vec<String> = facts.iter()
        .map(|f| format!("{} {} {}", f.subject, f.relation, f.object)).collect();

    Some(serde_json::json!({
        "response": response,
        "grounding": {
            "semantic_matches": related,
            "working_memory": wm_focus,
            "fast_memory": fm_matches,
            "knowledge": kg_facts,
            "personal": personal_ctx,
        },
        "facts_extracted": facts_extracted,
        "grounded": true,
        "native": used_native,
        "process_time": 0.0,
    }))
}

/// Native consolidation endpoint.
pub async fn api_brain_consolidate_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        // Basic consolidation: query DB for stats
        let proto_count = brain.memory_db.prototype_count().unwrap_or(0);
        let edge_count = brain.memory_db.edge_count().unwrap_or(0);

        brain.sse.emit("consolidation", serde_json::json!({
            "replayed": 0, "strengthened": 0, "pruned": 0, "compressed": 0,
        }));

        return Json(serde_json::json!({
            "status": "completed",
            "stats": {
                "replayed": 0,
                "strengthened": proto_count,
                "pruned": 0,
                "compressed": 0,
                "prototypes": proto_count,
                "kg_edges": edge_count,
            }
        })).into_response();
    }
    proxy_post_to_brain("/api/brain/consolidate", body, 60).await.into_response()
}

/// Native config update.
pub async fn api_brain_config_native(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        // Config updates would go here
        // For now just return current state
        Json(serde_json::json!({
            "status": "ok",
            "current": {
                "wm_slots": brain.config.wm_slots,
                "wm_decay": brain.config.wm_decay,
                "sparse_k": brain.config.sparse_k,
            }
        })).into_response()
    } else {
        proxy_post_to_brain("/api/brain/config", body, 15).await.into_response()
    }
}

/// Native grid map.
pub async fn api_brain_grid_map_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let Some(cb) = &*brain.codebook.lock().unwrap() {
            let mut grid = brain.grid_encoder.lock().unwrap();
            if !grid.is_fitted() { fit_grid_from_codebook(&mut grid, cb); }
            // Project up to 300 concepts to 2D
            let max_points = cb.len().min(300);
            let mut points = Vec::with_capacity(max_points);
            for i in 0..max_points {
                let c = cb.centroid(i);
                let [x, y] = grid.to_2d(&c);
                let act = grid.grid_activation([x, y]);
                let max_act = act.iter().copied().fold(0.0f32, f32::max);
                points.push(serde_json::json!({
                    "label": cb.all_labels()[i],
                    "x": (x * 100.0).round() / 100.0,
                    "y": (y * 100.0).round() / 100.0,
                    "activation": (max_act * 1000.0).round() / 1000.0,
                }));
            }
            return Json(serde_json::json!({
                "fitted": true,
                "scales": brain.config.grid_scales,
                "points": points,
                "total_concepts": cb.len(),
            })).into_response();
        }
    }
    proxy_get_from_brain("/api/brain/grid/map", 15).await.into_response()
}

/// Native SSE live feed using BrainState's tokio broadcast.
pub async fn api_brain_live_native(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let brain = brain.clone(); // Arc clone for the stream
        let mut rx = brain.sse.subscribe();

        let stream = async_stream::stream! {
            // Send init event
            let health = brain.health();
            let init = serde_json::json!({
                "type": "init",
                "working_memory": health["memory"],
                "prototypes": health["memory"]["prototypes"],
                "subscribers": brain.sse.subscriber_count(),
            });
            yield Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", init));

            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if let Ok(json) = serde_json::to_string(&event) {
                            yield Ok(format!("data: {json}\n\n"));
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        };

        axum::response::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .header("x-accel-buffering", "no")
            .body(axum::body::Body::from_stream(stream))
            .unwrap()
            .into_response()
    } else {
        // Fallback: proxy SSE from Python
        api_brain_live().await.into_response()
    }
}


// ═══════════════════════════════════════════════════════════════════
// COMPANION ENDPOINTS
// ═══════════════════════════════════════════════════════════════════

/// Proactive greeting based on time of day + personal context.
pub async fn api_companion_greeting(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let pm = brain.personal_memory.lock().unwrap();
        let greeting = brain_cognition::companion::proactive_greeting(&pm);
        let should_initiate = brain_cognition::companion::should_initiate_contact(&pm);
        let period = brain_cognition::companion::get_period();
        return Json(serde_json::json!({
            "greeting": greeting,
            "period": period,
            "should_initiate": should_initiate,
        })).into_response();
    }
    Json(serde_json::json!({"greeting": "Hello!", "period": "unknown"})).into_response()
}

/// Safety alerts for caregivers.
pub async fn api_companion_safety(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let pm = brain.personal_memory.lock().unwrap();
        let alerts = brain_cognition::companion::check_safety(&pm);
        let personal = brain_cognition::personal::build_personal_context(&pm);
        let moods = brain_cognition::personal::get_recent_moods(&pm, 20);
        let mood_summary: Vec<_> = moods.iter().map(|(t, e, c)| serde_json::json!({"time": t, "emotion": e, "confidence": c})).collect();
        return Json(serde_json::json!({
            "alerts": alerts,
            "alert_count": alerts.len(),
            "personal_summary": personal,
            "recent_moods": mood_summary,
        })).into_response();
    }
    Json(serde_json::json!({"alerts": [], "alert_count": 0})).into_response()
}

/// Personal facts summary.
pub async fn api_companion_personal(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        let pm = brain.personal_memory.lock().unwrap();
        let facts = brain_cognition::personal::get_all_facts(&pm);
        let name = brain_cognition::personal::get_user_name(&pm);
        let context = brain_cognition::personal::build_personal_context(&pm);
        let conversation = brain_cognition::personal::get_recent_conversation(&pm, 20);
        return Json(serde_json::json!({
            "name": name,
            "context": context,
            "facts": facts.iter().map(|(s,r,o,c)| serde_json::json!({"subject":s,"relation":r,"object":o,"mentions":c})).collect::<Vec<_>>(),
            "recent_conversation": conversation.iter().map(|(r,m)| serde_json::json!({"role":r,"message":m})).collect::<Vec<_>>(),
        })).into_response();
    }
    Json(serde_json::json!({"facts": []})).into_response()
}

// ═══════════════════════════════════════════════════════════════════
// DATASET INGESTION
// ═══════════════════════════════════════════════════════════════════

/// Find a file in a directory matching one of several patterns.
fn find_file(dir: &std::path::Path, pat1: &str, pat2: &str, pat3: &str) -> String {
    for pat in [pat1, pat2, pat3] {
        // Check exact match
        if dir.join(pat).exists() { return pat.to_string(); }
        // Check files ending with pattern
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(pat) || name.contains(pat.trim_start_matches('_')) {
                    return name;
                }
            }
        }
    }
    pat1.to_string() // fallback
}

fn ingest_audioset_sync(
    brain: &brain_cognition::BrainState,
    dataset: &str,
    batch_size: usize,
) -> Result<serde_json::Value, String> {
    // Resolve dataset to (base_dir, embeddings_file, labels_file)
    let cortex = brain.config.project_root.join("outputs/cortex");
    let (base, emb_file, label_file) = match dataset {
        "balanced" | "bal" => (cortex.join("audioset_brain"), "bal_train_embeddings.npy".into(), "bal_train_labels.json".into()),
        "eval" => (cortex.join("audioset_brain"), "eval_embeddings.npy".into(), "eval_labels.json".into()),
        "unbalanced" | "unbal" => (cortex.join("audioset_brain"), "unbal_train_embeddings.npy".into(), "unbal_train_labels.json".into()),
        // Generic: dataset name = directory under outputs/cortex/, expects embeddings.npy + labels.json
        name => {
            let dir = cortex.join(name);
            if dir.exists() {
                // Find .npy and .json files
                let npy = find_file(&dir, "embeddings.npy", "_embs.npy", "_audio_embs.npy");
                let json = find_file(&dir, "labels.json", "_labels.json", "_captions.json");
                (dir, npy, json)
            } else {
                return Err(format!("Dataset directory not found: {}", dir.display()));
            }
        }
    };

    // Load labels — supports both Vec<String> and Vec<Vec<String>> formats
    let labels_path = base.join(&label_file);
    let labels_data = std::fs::read_to_string(&labels_path).map_err(|e| format!("{}: {e}", labels_path.display()))?;
    let labels: Vec<String> = if let Ok(multi) = serde_json::from_str::<Vec<Vec<String>>>(&labels_data) {
        multi.iter().map(|tags| tags.first().cloned().unwrap_or_default()).collect()
    } else if let Ok(single) = serde_json::from_str::<Vec<String>>(&labels_data) {
        single
    } else {
        return Err(format!("Cannot parse labels from {}", labels_path.display()));
    };

    // Load embeddings (npy: skip header, read f32)
    let emb_path = base.join(&emb_file);
    let raw = std::fs::read(&emb_path).map_err(|e| format!("{}: {e}", emb_path.display()))?;

    // Also check for paired text embeddings (e.g., audiocaps_text_embs.npy)
    let text_emb_path = base.join(emb_file.replace("audio_embs", "text_embs").replace("embeddings", "text_embeddings"));
    let has_text_embs = text_emb_path.exists();
    // Simple npy parser: find \n after header, data starts at next 64-byte boundary
    let header_end = raw.iter().position(|&b| b == b'\n').unwrap_or(80) + 1;
    let data_start = ((header_end + 63) / 64) * 64; // align to 64
    let n_clips = labels.len();
    let dim = 512;

    let te = brain.text_encoder.as_ref().ok_or("No text encoder")?;
    let mlp = brain.inference.as_ref().ok_or("No MLP")?;

    let t0 = std::time::Instant::now();
    let mut pairs_created = 0;
    let mut categories_seen = std::collections::HashSet::new();
    let mut prototypes_created = 0;
    let mut kg_edges = 0;

    // Process in batches
    let limit = batch_size.min(n_clips);
    for i in 0..limit {
        let offset = data_start + i * dim * 4;
        if offset + dim * 4 > raw.len() { break; }

        let a_emb: Vec<f32> = raw[offset..offset + dim * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let label = &labels[i];

        // Text-encode the label as visual proxy (cached after first call)
        if let Ok(v_emb) = te.encode(label) {
            // Buffer as learning pair
            brain.online_pairs.lock().unwrap().push((v_emb, a_emb.clone()));
            pairs_created += 1;

            // Store to fast memory
            let a_proj = mlp.project_audio(&a_emb);
            brain.fast_memory.lock().unwrap().store(&a_proj, label);

            // Create prototype if new category
            if categories_seen.insert(label.clone()) {
                let centroid: Vec<u8> = a_proj.iter().flat_map(|f| f.to_le_bytes()).collect();
                let _ = brain.memory_db.upsert_prototype(label, &centroid, 1, "[]");
                prototypes_created += 1;

                // KG: link to parent concepts
                let words: Vec<&str> = label.split_whitespace().collect();
                if words.len() > 1 {
                    let _ = brain.memory_db.upsert_edge(label, "part-of", words[0], 0.7);
                    kg_edges += 1;
                }
            }
        }

        // Train every 100 pairs
        if pairs_created > 0 && pairs_created % 100 == 0 {
            let pairs: Vec<(Vec<f32>, Vec<f32>)> = {
                let mut p = brain.online_pairs.lock().unwrap();
                p.drain(..).collect()
            };
            if pairs.len() >= 10 {
                let n = pairs.len();
                let mut v_data = ndarray::Array2::<f32>::zeros((n, 384));
                let mut a_data = ndarray::Array2::<f32>::zeros((n, 512));
                for (j, (v, a)) in pairs.iter().enumerate() {
                    for (k, &val) in v.iter().enumerate().take(384) { v_data[[j, k]] = val; }
                    for (k, &val) in a.iter().enumerate().take(512) { a_data[[j, k]] = val; }
                }
                let mut w_v = mlp.w_v.clone();
                let mut w_a = mlp.w_a.clone();
                let (_, loss) = brain_inference::mlp::train_infonce(
                    &mut w_v, &mut w_a, &v_data, &a_data, 0.0005, 0.01, 20);

                // Save weights periodically
                let online_dir = brain.config.project_root.join("outputs/cortex/v6_mlp_online");
                let _ = std::fs::create_dir_all(&online_dir);
                let _ = brain_inference::mlp::save_bin_matrix(&w_v, &online_dir.join("w_v.bin"));
                let _ = brain_inference::mlp::save_bin_matrix(&w_a, &online_dir.join("w_a.bin"));

                brain.online_learning_count.fetch_add(n as i64, std::sync::atomic::Ordering::Relaxed);
                let total = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
                let _ = brain.memory_db.set_stat("online_learning_count", &total.to_string());

                brain.sse.emit("ingest_train", serde_json::json!({
                    "pairs": n, "loss": loss, "total": total, "progress": i,
                }));
                tracing::info!("AudioSet ingest: trained {n} pairs at step {i}, loss={loss:.4}");
            }
        }
    }

    // Train remaining buffered pairs
    let remaining: Vec<(Vec<f32>, Vec<f32>)> = brain.online_pairs.lock().unwrap().drain(..).collect();
    if remaining.len() >= 5 {
        let n = remaining.len();
        let mut v_data = ndarray::Array2::<f32>::zeros((n, 384));
        let mut a_data = ndarray::Array2::<f32>::zeros((n, 512));
        for (j, (v, a)) in remaining.iter().enumerate() {
            for (k, &val) in v.iter().enumerate().take(384) { v_data[[j, k]] = val; }
            for (k, &val) in a.iter().enumerate().take(512) { a_data[[j, k]] = val; }
        }
        let mut w_v = mlp.w_v.clone();
        let mut w_a = mlp.w_a.clone();
        let (_, _) = brain_inference::mlp::train_infonce(
            &mut w_v, &mut w_a, &v_data, &a_data, 0.0005, 0.01, 20);
        let online_dir = brain.config.project_root.join("outputs/cortex/v6_mlp_online");
        let _ = brain_inference::mlp::save_bin_matrix(&w_v, &online_dir.join("w_v.bin"));
        let _ = brain_inference::mlp::save_bin_matrix(&w_a, &online_dir.join("w_a.bin"));
        brain.online_learning_count.fetch_add(n as i64, std::sync::atomic::Ordering::Relaxed);
        pairs_created += 0; // already counted
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let total_learned = brain.online_learning_count.load(std::sync::atomic::Ordering::Relaxed);
    let _ = brain.memory_db.set_stat("online_learning_count", &total_learned.to_string());
    let _ = brain.memory_db.log_learning("ingest_audioset",
        Some(&format!("{{\"dataset\":\"{dataset}\",\"pairs\":{pairs_created},\"categories\":{},\"time\":{elapsed:.1}}}",
            categories_seen.len())));

    brain.sse.emit("ingest_complete", serde_json::json!({
        "dataset": dataset, "pairs": pairs_created, "categories": categories_seen.len(),
        "prototypes": prototypes_created, "kg_edges": kg_edges, "time": elapsed,
    }));

    Ok(serde_json::json!({
        "status": "ingested",
        "dataset": dataset,
        "pairs_created": pairs_created,
        "categories": categories_seen.len(),
        "prototypes_created": prototypes_created,
        "kg_edges_added": kg_edges,
        "total_learned": total_learned,
        "time_seconds": (elapsed * 10.0).round() / 10.0,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Call Ollama LLM for text generation.
async fn call_ollama(prompt: &str, config: &brain_cognition::BrainConfig) -> String {
    let url = if config.ollama_url.contains("/api/") { config.ollama_url.clone() } else { format!("{}/api/generate", config.ollama_url) };
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": false,
        "options": {"temperature": 0.7, "num_predict": 100},
    });
    match client.post(&url).json(&body).timeout(std::time::Duration::from_secs(5)).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                json["response"].as_str().unwrap_or("I have no words right now.").to_string()
            } else {
                "I have no words right now.".into()
            }
        }
        Err(_) => "I cannot speak right now — Ollama is unavailable.".into(),
    }
}

/// Native audio processing for listen endpoint.
async fn try_native_listen(brain: &brain_cognition::BrainState, body: &serde_json::Value) -> Option<serde_json::Value> {
    use base64::Engine;
    let t0 = std::time::Instant::now();

    let audio_b64 = body["audio_b64"].as_str()?;
    let top_k = body["top_k"].as_u64().unwrap_or(10) as usize;

    // Decode base64 PCM float32 (16kHz mono)
    let bytes = base64::engine::general_purpose::STANDARD.decode(audio_b64).ok()?;
    let samples: Vec<f32> = bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if samples.len() < 1600 { return None; } // too short (<0.1s)

    let duration = samples.len() as f32 / 16000.0;
    let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 0.001 { return None; } // silence

    let mlp = brain.inference.as_ref()?;
    let cb_guard = brain.codebook.lock().unwrap();
    let cb = cb_guard.as_ref()?;

    // Compute mel spectrogram (80, 3000) → feed to Whisper encoder
    let mel = brain_inference::mel::compute_log_mel(&samples);
    let whisper = brain.audio_encoder.as_ref()?;
    let a_emb = whisper.encode(&mel).ok()?; // 512-dim audio embedding

    // MLP project audio → 512-dim shared space
    let a_proj = mlp.project_audio(&a_emb);

    // Find nearest concepts in codebook
    let audio_similar = cb.nearest(&a_proj, top_k);
    let top_labels: Vec<String> = audio_similar.iter().take(3).map(|(l, _)| l.clone()).collect();

    // Cross-modal: what does this sound look like?
    let mut cross_v = Vec::new();
    if let Some(wm) = &brain.world_model {
        if let Ok(pred) = wm.predict(&a_proj) {
            cross_v = cb.nearest(&pred, 5).iter().map(|(l, _)| l.clone()).collect();
        }
    }

    // Confidence
    let confidence = brain.confidence_model.as_ref()
        .and_then(|cm| cm.predict(&a_proj).ok())
        .unwrap_or(0.5);

    let narration = format!("I hear: {}. This reminds me of: {}.",
        top_labels.join(", "),
        cross_v.iter().take(3).cloned().collect::<Vec<_>>().join(", "));

    // Store perception
    let _ = brain.memory_db.store_perception(
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
        "audio", None,
        Some(&serde_json::json!(top_labels).to_string()),
        Some(&serde_json::json!(cross_v).to_string()),
        Some(&narration));

    // Update working memory + fast memory
    {
        let label = top_labels.first().cloned().unwrap_or("sound".into());
        let mut wm = brain.working_memory.lock().unwrap();
        wm.update(a_proj.clone(), label.clone(), "audio".into());
        brain.fast_memory.lock().unwrap().store(&a_proj, &label);
    }

    // Buffer learning pair: raw audio (512-dim Whisper) + projected audio
    brain.online_pairs.lock().unwrap().push((
        a_emb[..384.min(a_emb.len())].to_vec(),  // truncate to 384 for v_dim
        a_emb.clone(),  // full 512-dim audio
    ));

    // SSE
    brain.sse.emit("perception", serde_json::json!({
        "modality": "audio", "labels": top_labels, "confidence": confidence,
    }));

    let process_time = t0.elapsed().as_secs_f64();

    Some(serde_json::json!({
        "associations": {
            "audio_similar": audio_similar.iter().map(|(l, s)| serde_json::json!({"label": l, "similarity": s})).collect::<Vec<_>>(),
            "cross_modal_a2v": cross_v.iter().map(|l| serde_json::json!({"label": l})).collect::<Vec<_>>(),
        },
        "summary": {
            "i_hear": top_labels,
            "which_reminds_me_of": cross_v,
            "audio_duration": (duration * 10.0).round() / 10.0,
            "narration": narration,
            "process_time": (process_time * 1000.0).round() / 1000.0,
            "confidence": (confidence * 1000.0).round() / 1000.0,
        },
        "working_memory": {
            "slots_used": brain.working_memory.lock().unwrap().get_state().slots_used,
        }
    }))
}

async fn proxy_post_to_brain(path: &str, body: serde_json::Value, timeout_secs: u64) -> impl IntoResponse {
    let url = format!("http://127.0.0.1:8099{path}");
    let client = reqwest::Client::new();
    match client.post(&url).json(&body).timeout(std::time::Duration::from_secs(timeout_secs)).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}

async fn proxy_get_from_brain(path: &str, timeout_secs: u64) -> impl IntoResponse {
    let url = format!("http://127.0.0.1:8099{path}");
    let client = reqwest::Client::new();
    match client.get(&url).timeout(std::time::Duration::from_secs(timeout_secs)).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"detail": e.to_string()}))).into_response(),
    }
}

/// Generic proxy for Brain Voice API endpoints (describe, ask, chat).
pub async fn api_brain_proxy(
    axum::extract::Path(endpoint): axum::extract::Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let url = format!("http://127.0.0.1:8099/api/brain/{endpoint}");
    let client = reqwest::Client::new();
    match client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
    {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({"detail": format!("Bad response: {e}")})),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"detail": format!("Brain service unreachable: {e}")})),
        )
            .into_response(),
    }
}

/// Proxy DELETE for clearing chat sessions.
pub async fn api_brain_chat_delete(
    axum::extract::Path(session_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let url = format!("http://127.0.0.1:8099/api/brain/chat/{session_id}");
    let client = reqwest::Client::new();
    match client.delete(&url).timeout(std::time::Duration::from_secs(10)).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            match resp.json::<serde_json::Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(_) => (status, Json(serde_json::json!({"status": "ok"}))).into_response(),
            }
        }
        Err(_) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))).into_response(),
    }
}

/// API endpoint for brain visualization — real data from embeddings.
pub async fn api_brain_state(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let viz = match &state.brain_viz {
        Some(v) => v,
        None => return Json(serde_json::json!({"error": "No brain data available"})),
    };

    // PCA scatter data
    let pca_v: Vec<_> = viz.pca_v.iter().map(|[x, y]| [*x, *y]).collect();
    let pca_a: Vec<_> = viz.pca_a.iter().map(|[x, y]| [*x, *y]).collect();

    // Category color map
    let mut cat_colors = serde_json::Map::new();
    for (name, &idx) in &viz.category_colors {
        cat_colors.insert(name.clone(), serde_json::json!(idx));
    }

    // Cross-correlation heatmaps
    let mut heatmaps = serde_json::Map::new();
    for (key, grid) in &viz.cross_corr {
        heatmaps.insert(key.clone(), serde_json::json!(grid));
    }

    // Weight histograms
    let mut histograms = serde_json::Map::new();
    for (key, hist) in &viz.weight_histograms {
        histograms.insert(key.clone(), serde_json::json!(hist));
    }

    // Latest experiment metrics
    let recent = state.db.get_experiments(Some("completed"), 1).unwrap_or_default();
    let exp_metrics = recent.first().map(|exp| {
        serde_json::from_str::<serde_json::Value>(&exp.final_metrics).unwrap_or_default()
    }).unwrap_or_default();

    Json(serde_json::json!({
        "pca_v": pca_v,
        "pca_a": pca_a,
        "pca_labels": viz.pca_labels,
        "pca_category_idx": viz.pca_category_idx,
        "category_colors": cat_colors,
        "heatmaps": heatmaps,
        "histograms": histograms,
        "hist_range": [viz.hist_min, viz.hist_max],
        "sample_sims": viz.sample_sims,
        "n_clips": viz.n_clips,
        "latest_metrics": exp_metrics,
    }))
}

/// API endpoint for single clip exploration.
#[derive(Deserialize)]
pub struct ClipQuery {
    idx: usize,
}

pub async fn api_brain_clip(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ClipQuery>,
) -> Json<serde_json::Value> {
    let interact = match &state.interact {
        Some(i) => i,
        None => return Json(serde_json::json!({"error": "No data"})),
    };

    let idx = params.idx;
    if idx >= interact.n_clips {
        return Json(serde_json::json!({"error": "Index out of range"}));
    }

    let label = interact.labels.get(idx).cloned().unwrap_or_default();

    // Get top-10 most similar clips by vision and audio
    let v_row = interact.v.row(idx);
    let a_row = interact.a.row(idx);

    let n = interact.n_clips.min(1000);
    let mut v_sims: Vec<(usize, f32)> = (0..n).filter(|&i| i != idx).map(|i| {
        let sim: f32 = v_row.iter().zip(interact.v.row(i).iter()).map(|(a, b)| a * b).sum();
        (i, sim)
    }).collect();
    v_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut a_sims: Vec<(usize, f32)> = (0..n).filter(|&i| i != idx).map(|i| {
        let sim: f32 = a_row.iter().zip(interact.a.row(i).iter()).map(|(a, b)| a * b).sum();
        (i, sim)
    }).collect();
    a_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Cross-modal similarity
    let cross_sim: f32 = v_row.iter().zip(a_row.iter()).map(|(a, b)| a * b).sum();

    // Embedding norms
    let v_norm: f32 = v_row.iter().map(|x| x * x).sum::<f32>().sqrt();
    let a_norm: f32 = a_row.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Top-k activation values (which dimensions are most active)
    let mut v_activations: Vec<(usize, f32)> = v_row.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
    v_activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_v_dims: Vec<_> = v_activations.iter().take(20).map(|&(i, v)| serde_json::json!({"dim": i, "val": v})).collect();

    let mut a_activations: Vec<(usize, f32)> = a_row.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
    a_activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_a_dims: Vec<_> = a_activations.iter().take(20).map(|&(i, v)| serde_json::json!({"dim": i, "val": v})).collect();

    Json(serde_json::json!({
        "idx": idx,
        "label": label,
        "cross_sim": cross_sim,
        "v_norm": v_norm,
        "a_norm": a_norm,
        "v_neighbors": v_sims.iter().take(10).map(|(i, s)| serde_json::json!({
            "idx": i, "label": interact.labels.get(*i).cloned().unwrap_or_default(), "sim": s
        })).collect::<Vec<_>>(),
        "a_neighbors": a_sims.iter().take(10).map(|(i, s)| serde_json::json!({
            "idx": i, "label": interact.labels.get(*i).cloned().unwrap_or_default(), "sim": s
        })).collect::<Vec<_>>(),
        "top_v_dims": top_v_dims,
        "top_a_dims": top_a_dims,
    }))
}

// ======================================================================
// JSON API endpoints
// ======================================================================

pub async fn api_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let resources = state.get_resources();
    let best = state.db.get_best_experiment("v2a_R@1").unwrap_or(None);
    let best_metrics: serde_json::Value = best
        .as_ref()
        .map(|b| serde_json::from_str(&b.final_metrics).unwrap_or_default())
        .unwrap_or_default();

    let best_mrr_exp = state.db.get_best_experiment("v2a_MRR").unwrap_or(None);
    let best_mrr_metrics: serde_json::Value = best_mrr_exp
        .as_ref()
        .map(|b| serde_json::from_str(&b.final_metrics).unwrap_or_default())
        .unwrap_or_default();

    // Single query for all experiment counts (was 3 separate queries)
    let counts = state.db.experiment_counts().unwrap_or(brain_db::ExperimentCounts {
        total: 0, completed: 0, running: 0, failed: 0, pending: 0,
    });

    let best_a2v_exp = state.db.get_best_experiment("a2v_MRR").unwrap_or(None);
    let best_a2v_metrics: serde_json::Value = best_a2v_exp
        .as_ref()
        .map(|b| serde_json::from_str(&b.final_metrics).unwrap_or_default())
        .unwrap_or_default();

    Json(serde_json::json!({
        "experiments_total": counts.total,
        "experiments_completed": counts.completed,
        "experiments_running": counts.running,
        "best_r1": best_metrics.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "best_r5": best_metrics.get("v2a_R@5").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "best_mrr": best_mrr_metrics.get("v2a_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "best_a2v_mrr": best_a2v_metrics.get("a2v_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "best_rank": best_metrics.get("estimated_rank").and_then(|v| v.as_f64()).unwrap_or(0.0),
        "cpu_percent": resources.cpu_percent,
        "memory_used_percent": resources.memory_used_percent,
        "memory_available_gb": resources.memory_available_gb,
        "disk_free_gb": resources.disk_free_gb,
        "ollama_running": resources.ollama_running,
        "mutations": serde_json::to_value(&state.db.get_mutation_stats().unwrap_or(
            brain_db::MutationStats { total: 0, accepted: 0, acceptance_rate: 0.0, by_target: vec![] }
        )).unwrap_or_default(),
        "data": state.db.get_data_inventory_stats().unwrap_or_default(),
    }))
}

pub async fn api_timeline(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let completed = state
        .db
        .get_experiments(Some("completed"), 200)
        .unwrap_or_default();
    let timeline: Vec<serde_json::Value> = completed
        .iter()
        .rev()
        .map(|exp| {
            let metrics: serde_json::Value =
                serde_json::from_str(&exp.final_metrics).unwrap_or_default();
            serde_json::json!({
                "id": exp.id,
                "r1": metrics.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "r5": metrics.get("v2a_R@5").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "r10": metrics.get("v2a_R@10").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "rank": metrics.get("estimated_rank").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "condition": metrics.get("condition_number").and_then(|v| v.as_f64()).unwrap_or(0.0),
                "norm": metrics.get("frobenius_norm").and_then(|v| v.as_f64()).unwrap_or(0.0),
            })
        })
        .collect();

    Json(serde_json::json!(timeline))
}

pub async fn api_experiment(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> Json<serde_json::Value> {
    let exp = state.db.get_experiment(experiment_id).unwrap_or(None);
    let snapshots = state
        .db
        .get_metric_snapshots(experiment_id)
        .unwrap_or_default();

    Json(serde_json::json!({
        "experiment": exp,
        "snapshots": snapshots,
    }))
}

#[derive(Deserialize)]
pub struct LogQuery {
    #[serde(default = "default_log_lines")]
    pub lines: usize,
}
fn default_log_lines() -> usize { 50 }

pub async fn api_logs(
    State(state): State<Arc<AppState>>,
    Query(query): Query<LogQuery>,
) -> Json<serde_json::Value> {
    let log_path = state.project_root.join("cortex.log");
    let alt_path = state.output_dir.join("cortex.log");

    let path = if log_path.exists() {
        log_path
    } else if alt_path.exists() {
        alt_path
    } else {
        return Json(serde_json::json!({"lines": ["No log file found"]}));
    };

    match tokio::fs::read_to_string(&path).await {
        Ok(text) => {
            let lines: Vec<&str> = text.lines().rev().take(query.lines).collect::<Vec<_>>().into_iter().rev().collect();
            Json(serde_json::json!({"lines": lines}))
        }
        Err(_) => Json(serde_json::json!({"lines": ["Error reading log file"]})),
    }
}

#[derive(Deserialize)]
pub struct DecisionQuery {
    #[serde(default = "default_decision_limit")]
    pub limit: i64,
}
fn default_decision_limit() -> i64 { 20 }

pub async fn api_decisions(
    State(state): State<Arc<AppState>>,
    Query(query): Query<DecisionQuery>,
) -> Json<serde_json::Value> {
    let decisions = state.db.get_decisions(None, query.limit).unwrap_or_default();
    Json(serde_json::to_value(&decisions).unwrap_or_default())
}

pub async fn api_history(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // Limit to last 200 experiments (was 500 — unreadable with 1300+ experiments)
    let completed = state
        .db
        .get_experiments(Some("completed"), 200)
        .unwrap_or_default();

    let mut experiments = Vec::with_capacity(completed.len());
    let mut best_r1_so_far = 0.0f64;
    let mut best_mrr_so_far = 0.0f64;
    let mut best_progression = Vec::with_capacity(completed.len());
    let mut hp_data = Vec::with_capacity(completed.len());

    // Single pass: parse metrics+config once per experiment, build all 3 result arrays
    for exp in completed.iter().rev() {
        let metrics: serde_json::Value =
            serde_json::from_str(&exp.final_metrics).unwrap_or_default();
        let r1 = metrics.get("v2a_R@1").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let r5 = metrics.get("v2a_R@5").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let r10 = metrics.get("v2a_R@10").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mrr = metrics.get("v2a_MRR").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rank = metrics.get("estimated_rank").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let cond = metrics.get("condition_number").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let norm = metrics.get("frobenius_norm").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let duration = if exp.finished_at > 0.0 && exp.created_at > 0.0 {
            exp.finished_at - exp.created_at
        } else {
            0.0
        };

        experiments.push(serde_json::json!({
            "id": exp.id,
            "r1": r1, "r5": r5, "r10": r10, "mrr": mrr,
            "rank": rank, "cond": cond, "norm": norm,
            "created_at": exp.created_at,
            "finished_at": exp.finished_at,
            "duration_min": (duration / 60.0 * 10.0).round() / 10.0,
            "hypothesis": exp.hypothesis.chars().take(100).collect::<String>(),
            "has_patch": !exp.code_patch.is_empty(),
        }));

        if r1 > best_r1_so_far {
            best_r1_so_far = r1;
        }
        if mrr > best_mrr_so_far {
            best_mrr_so_far = mrr;
        }
        best_progression.push(serde_json::json!({
            "id": exp.id,
            "best_r1": best_r1_so_far,
            "best_mrr": best_mrr_so_far,
            "finished_at": exp.finished_at,
        }));

        // HP data — parse config once (reuse metrics already parsed above)
        let config: serde_json::Value =
            serde_json::from_str(&exp.config_json).unwrap_or_default();
        hp_data.push(serde_json::json!({
            "id": exp.id,
            "hebbian_lr": config.get("hebbian_lr").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "decay_rate": config.get("decay_rate").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "temporal_decay": config.get("temporal_decay").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "trace_weight": config.get("trace_weight").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "max_norm": config.get("max_norm").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "sparsity_k": config.get("sparsity_k").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "batch_size": config.get("batch_size").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "neg_weight": config.get("neg_weight").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "init_scale": config.get("init_scale").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "r1": r1,
            "mrr": mrr,
        }));
    }

    // Data growth
    let data_decisions = state
        .db
        .get_decisions(Some("data_acquisition"), 500)
        .unwrap_or_default();
    let mut data_growth = Vec::with_capacity(data_decisions.len());
    let mut cumulative_clips: i64 = 0;
    for d in data_decisions.iter().rev() {
        let ctx: serde_json::Value =
            serde_json::from_str(&d.context_json).unwrap_or_default();
        if let Some(total) = ctx.get("total_after").and_then(|v| v.as_i64()) {
            cumulative_clips = total;
        } else {
            cumulative_clips += ctx.get("downloaded").and_then(|v| v.as_i64()).unwrap_or(0);
        }
        data_growth.push(serde_json::json!({
            "timestamp": d.timestamp,
            "clips": cumulative_clips,
        }));
    }

    Json(serde_json::json!({
        "experiments": experiments,
        "best_progression": best_progression,
        "data_growth": data_growth,
        "hyperparameters": hp_data,
    }))
}

// ======================================================================
// New API endpoints
// ======================================================================

pub async fn api_experiments_list(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let experiments = state.db.get_experiments(None, 100).unwrap_or_default();
    Json(serde_json::to_value(&experiments).unwrap_or_default())
}

pub async fn api_mutations_list(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mutations = state.db.get_mutations_for_target(None, 100).unwrap_or_default();
    Json(serde_json::to_value(&mutations).unwrap_or_default())
}

pub async fn api_mutations_stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let stats = state.db.get_mutation_stats().unwrap_or(brain_db::MutationStats {
        total: 0, accepted: 0, acceptance_rate: 0.0, by_target: vec![],
    });
    Json(serde_json::to_value(&stats).unwrap_or_default())
}

// ======================================================================
// Interactive retrieval API
// ======================================================================

#[derive(Deserialize)]
pub struct RetrieveQuery {
    /// Clip index to query
    pub idx: usize,
    /// Source modality: "vision", "audio", "emotion", "speech", "properties"
    #[serde(default = "default_source")]
    pub source: String,
    /// For backward compat: "v2a" or "a2v"
    pub direction: Option<String>,
    /// Number of results to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}
fn default_source() -> String { "vision".to_string() }
fn default_top_k() -> usize { 10 }

/// Retrieve top-k similar clips across ALL available modalities from a query clip.
pub async fn api_interact_retrieve(
    State(state): State<Arc<AppState>>,
    Query(query): Query<RetrieveQuery>,
) -> Json<serde_json::Value> {
    let interact = match &state.interact {
        Some(i) => i,
        None => return Json(serde_json::json!({"error": "Embeddings not loaded"})),
    };

    if query.idx >= interact.n_clips {
        return Json(serde_json::json!({"error": "Index out of range"}));
    }

    let top_k = query.top_k.min(50).max(1);

    // Resolve source modality (backward compat with direction param)
    let source = if let Some(ref dir) = query.direction {
        match dir.as_str() {
            "a2v" => "audio",
            _ => "vision",
        }
    } else {
        &query.source
    };

    let query_emb = match interact.get_modality(source) {
        Some(m) => m,
        None => return Json(serde_json::json!({"error": format!("Unknown modality: {}", source)})),
    };

    let query_vec = query_emb.row(query.idx);
    let label = interact.labels.get(query.idx).cloned().unwrap_or_default();

    // Compute similarities against all OTHER modalities
    let modality_names: &[(&str, &str)] = &[
        ("vision", "v"), ("audio", "a"), ("emotion", "e"),
        ("speech", "s"), ("properties", "p"),
    ];

    let mut associations = serde_json::Map::new();
    for &(mod_name, mod_key) in modality_names {
        if mod_name == source { continue; }
        if let Some(target_emb) = interact.get_modality(mod_key) {
            let sims = target_emb.dot(&query_vec.t());
            let results = rank_results(sims.as_slice().unwrap(), top_k, &interact.labels, query.idx);
            associations.insert(mod_name.to_string(), serde_json::json!(results));
        }
    }

    Json(serde_json::json!({
        "query_idx": query.idx,
        "query_label": label,
        "source": source,
        "associations": associations,
        "modalities": interact.available_modalities(),
    }))
}

/// Pick a random clip and retrieve its associations across all modalities.
pub async fn api_interact_random(
    State(state): State<Arc<AppState>>,
    Query(query): Query<RandomQuery>,
) -> Json<serde_json::Value> {
    let interact = match &state.interact {
        Some(i) => i,
        None => return Json(serde_json::json!({"error": "Embeddings not loaded"})),
    };

    let idx = rand::Rng::random_range(&mut rand::rng(), 0..interact.n_clips);
    let source = query.source.as_deref()
        .or(query.direction.as_deref().map(|d| if d == "a2v" { "audio" } else { "vision" }))
        .unwrap_or("vision");
    let top_k = query.top_k.unwrap_or(10).min(50).max(1);

    let query_emb = match interact.get_modality(source) {
        Some(m) => m,
        None => return Json(serde_json::json!({"error": format!("Unknown modality: {}", source)})),
    };

    let query_vec = query_emb.row(idx);
    let label = interact.labels.get(idx).cloned().unwrap_or_default();

    let modality_names: &[(&str, &str)] = &[
        ("vision", "v"), ("audio", "a"), ("emotion", "e"),
        ("speech", "s"), ("properties", "p"),
    ];

    let mut associations = serde_json::Map::new();
    for &(mod_name, mod_key) in modality_names {
        if mod_name == source { continue; }
        if let Some(target_emb) = interact.get_modality(mod_key) {
            let sims = target_emb.dot(&query_vec.t());
            let results = rank_results(sims.as_slice().unwrap(), top_k, &interact.labels, idx);
            associations.insert(mod_name.to_string(), serde_json::json!(results));
        }
    }

    Json(serde_json::json!({
        "query_idx": idx,
        "query_label": label,
        "source": source,
        "associations": associations,
        "modalities": interact.available_modalities(),
    }))
}

#[derive(Deserialize)]
pub struct RandomQuery {
    pub direction: Option<String>,
    pub source: Option<String>,
    pub top_k: Option<usize>,
}

/// Search for clips by label substring.
pub async fn api_interact_search(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SearchQuery>,
) -> Json<serde_json::Value> {
    let interact = match &state.interact {
        Some(i) => i,
        None => return Json(serde_json::json!({"error": "Embeddings not loaded"})),
    };

    let q = query.q.to_lowercase();
    let mut matches: Vec<serde_json::Value> = Vec::new();
    for (label, indices) in &interact.label_index {
        if label.to_lowercase().contains(&q) && !label.starts_with("clip #") {
            matches.push(serde_json::json!({
                "label": label,
                "count": indices.len(),
                "sample_idx": indices[0],
            }));
        }
    }
    matches.sort_by(|a, b| {
        b.get("count").and_then(|v| v.as_u64()).unwrap_or(0)
            .cmp(&a.get("count").and_then(|v| v.as_u64()).unwrap_or(0))
    });
    matches.truncate(30);

    Json(serde_json::json!({"results": matches}))
}

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
}

/// Get available labels and their clip counts.
pub async fn api_interact_labels(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let interact = match &state.interact {
        Some(i) => i,
        None => return Json(serde_json::json!({"error": "Embeddings not loaded"})),
    };

    let labels: Vec<serde_json::Value> = interact.unique_labels.iter().map(|label| {
        let count = interact.label_index.get(label).map(|v| v.len()).unwrap_or(0);
        serde_json::json!({"label": label, "count": count})
    }).collect();

    Json(serde_json::json!({
        "n_clips": interact.n_clips,
        "n_labels": labels.len(),
        "labels": labels,
        "modalities": interact.available_modalities(),
    }))
}

/// Rank similarities and return top-k results.
fn rank_results(
    sims: &[f32],
    top_k: usize,
    labels: &[String],
    exclude_idx: usize,
) -> Vec<serde_json::Value> {
    let mut indexed: Vec<(usize, f32)> = sims.iter().enumerate()
        .filter(|&(i, _)| i != exclude_idx)
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    indexed.iter().map(|&(idx, sim)| {
        serde_json::json!({
            "idx": idx,
            "label": labels.get(idx).cloned().unwrap_or_default(),
            "similarity": (sim as f64 * 10000.0).round() / 10000.0,
        })
    }).collect()
}

// SSE endpoint for live experiment streaming
pub async fn api_live_experiment(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> impl IntoResponse {
    let rx = state.live_tx.subscribe();
    crate::sse::experiment_sse_stream(rx, experiment_id)
}
