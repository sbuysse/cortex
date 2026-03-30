//! One-shot DB maintenance: clean zombies + insert v2 best result.

use brain_db::KnowledgeBase;
use std::path::Path;

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

fn main() {
    let db_path = std::env::var("DB_PATH")
        .unwrap_or_else(|_| "/opt/brain/outputs/cortex/knowledge.db".to_string());
    let db = KnowledgeBase::new(Path::new(&db_path)).expect("Failed to open DB");
    let conn = db.pool().get().expect("Failed to get connection");

    // 1. Clean zombie "running" experiments (except currently active ones from cortex)
    //    The currently active cortex cycle started at ~00:34 UTC Mar 18 (IDs 20101-20116).
    //    Keep those, fail everything older.
    let cutoff_id: i64 = std::env::var("KEEP_ABOVE")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(20100); // keep 20101+ as they may be active

    let now_ts = now();
    let failed_running = conn.execute(
        "UPDATE experiments SET status='failed', error_msg='zombie: process died', finished_at=?1 \
         WHERE status='running' AND id <= ?2",
        rusqlite::params![now_ts, cutoff_id],
    ).expect("Failed to update running zombies");
    eprintln!("Marked {} running experiments as failed (id <= {})", failed_running, cutoff_id);

    // Also clean pending zombies
    let failed_pending = conn.execute(
        "UPDATE experiments SET status='failed', error_msg='zombie: never started', finished_at=?1 \
         WHERE status='pending'",
        rusqlite::params![now_ts],
    ).expect("Failed to update pending zombies");
    eprintln!("Marked {} pending experiments as failed", failed_pending);

    // 2. Insert v4 MLP result if requested
    if std::env::var("INSERT_V4").is_ok() {
        let v4_config = serde_json::json!({
            "model_type": "mlp_dual_encoder",
            "d_hidden": 512,
            "hebbian_lr": 0.005,
            "infonce_temperature": 0.005,
            "temp_start": 0.02,
            "temp_end": 0.005,
            "max_steps": 20000,
            "batch_size": 2048,
            "hard_neg_k": 64,
            "hard_neg_frac": 0.5,
            "skip_projection": true,
            "use_whitening": true,
        });
        let v4_metrics = serde_json::json!({
            "v2a_MRR": 0.730284,
            "v2a_R@1": 0.669769,
            "v2a_R@5": 0.801333,
            "v2a_R@10": 0.868639,
            "v2a_MR": 25.867501,
            "a2v_MRR": 0.637769,
            "a2v_R@1": 0.538165,
            "a2v_R@5": 0.761136,
            "a2v_R@10": 0.807714,
            "a2v_MR": 199.848521,
            "eval_pool": 24604,
        });
        let v4_id = db.create_experiment(
            &v4_config,
            "V4 MLP dual encoder: ReLU(V@W_v)·ReLU(A@W_a), hard neg + temp anneal, full 24K eval",
            None,
            "train-v4 MLP",
        ).expect("Failed to create v4 experiment");
        db.update_experiment_status(v4_id, "completed", Some(&v4_metrics), "")
            .expect("Failed to update v4 status");
        eprintln!("Inserted v4 MLP result as experiment #{} (24K MRR=0.730, R@1=0.670)", v4_id);
    }

    // Insert v2 whitened_rect result (legacy)
    let v2_config = serde_json::json!({
        "embed_dim": 512,
        "hebbian_lr": 0.06742,
        "decay_rate": 0.992,
        "temporal_decay": 0.012504953891038895,
        "trace_weight": 0.11518516391515732,
        "max_norm": 22.165,
        "max_steps": 20000,
        "batch_size": 2048,
        "seed": 42,
        "init_scale": 0.01,
        "neg_weight": 0.5,
        "aug_noise": 0.004249527584761381,
        "warmup_factor": 1.0,
        "sparsity_k": 36,
        "use_infonce": true,
        "infonce_temperature": 0.006394,
        "gradient_infonce": true,
        "symmetric_infonce": true,
        "cross_correlation_init": true,
        "cross_correlation_blend": 0.4335651099681854,
        "curriculum": true,
        "curriculum_start_frac": 0.24089980125427246,
        "eval_pool_size": 1000,
        // V2-specific flags
        "skip_projection": true,
        "use_whitening": true,
        "multi_positive": false,
        "ensemble_count": 1,
        "quadratic_scoring": false,
    });

    let v2_metrics = serde_json::json!({
        "v2a_MRR": 0.758193,
        "v2a_R@1": 0.594,
        "v2a_R@5": 0.975,
        "v2a_R@10": 0.998,
        "v2a_MR": 2.443,
        "a2v_MRR": 0.745013,
        "a2v_R@1": 0.570,
        "a2v_R@5": 0.979,
        "a2v_R@10": 0.997,
        "a2v_MR": 3.251,
    });

    let exp_id = db.create_experiment(
        &v2_config,
        "V2 whitened_rect: ZCA whitening + rectangular M (384x512), best checkpoint at step 10000",
        None,
        "train-v2 MODE=whitened_rect",
    ).expect("Failed to create v2 experiment");

    db.update_experiment_status(
        exp_id,
        "completed",
        Some(&v2_metrics),
        "",
    ).expect("Failed to update v2 experiment status");

    eprintln!("Inserted v2 whitened_rect result as experiment #{} (MRR=0.758, R@1=0.594)", exp_id);

    // Verify: what's the new best?
    let best: (i64, f64, f64) = conn.query_row(
        "SELECT id, \
         CAST(json_extract(final_metrics, '$.v2a_MRR') AS REAL), \
         CAST(COALESCE(json_extract(final_metrics, '$.\"v2a_R@1\"'), '0') AS REAL) \
         FROM experiments \
         WHERE status='completed' AND json_extract(final_metrics, '$.v2a_MRR') IS NOT NULL \
         ORDER BY (CAST(json_extract(final_metrics, '$.v2a_MRR') AS REAL) \
                 + CAST(COALESCE(json_extract(final_metrics, '$.\"v2a_R@1\"'), '0') AS REAL)) DESC \
         LIMIT 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).expect("Failed to query best");

    eprintln!("\nNew best experiment: #{} (MRR={:.4}, R@1={:.4}, composite={:.4})",
        best.0, best.1, best.2, best.1 + best.2);

    // Show counts
    let counts: (i64, i64, i64, i64) = conn.query_row(
        "SELECT COUNT(*), \
         SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END), \
         SUM(CASE WHEN status='running' THEN 1 ELSE 0 END), \
         SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) \
         FROM experiments",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    ).expect("Failed to count");

    eprintln!("DB status: {} total, {} completed, {} running, {} failed",
        counts.0, counts.1, counts.2, counts.3);
}
