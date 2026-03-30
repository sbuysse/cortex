//! KnowledgeBase — SQLite-backed persistent storage.
//!
//! Port of Python knowledge_base.py with identical schema.

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::*;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    hypothesis TEXT NOT NULL DEFAULT '',
    final_metrics TEXT NOT NULL DEFAULT '{}',
    parent_id INTEGER,
    error_msg TEXT NOT NULL DEFAULT '',
    code_patch TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0,
    finished_at REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS metric_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL DEFAULT 0,
    subsystem TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    context_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS data_inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL DEFAULT '',
    clip_id TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS code_mutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_file TEXT NOT NULL DEFAULT '',
    target_name TEXT NOT NULL DEFAULT '',
    original_code TEXT NOT NULL DEFAULT '',
    mutated_code TEXT NOT NULL DEFAULT '',
    diff TEXT NOT NULL DEFAULT '',
    llm_prompt TEXT NOT NULL DEFAULT '',
    llm_response TEXT NOT NULL DEFAULT '',
    experiment_id INTEGER,
    score_delta REAL NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_status_id ON experiments(status, id DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_experiment ON metric_snapshots(experiment_id);
CREATE INDEX IF NOT EXISTS idx_mutations_target ON code_mutations(target_name);
CREATE INDEX IF NOT EXISTS idx_mutations_accepted ON code_mutations(accepted);
CREATE INDEX IF NOT EXISTS idx_data_inventory_status ON data_inventory(status);
CREATE INDEX IF NOT EXISTS idx_decisions_subsystem ON decisions(subsystem);
"#;

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Pool error: {0}")]
    Pool(#[from] r2d2::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type DbResult<T> = Result<T, DbError>;

pub struct KnowledgeBase {
    pool: Pool<SqliteConnectionManager>,
}

impl KnowledgeBase {
    pub fn new(db_path: &Path) -> DbResult<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let manager = SqliteConnectionManager::file(db_path);
        let pool = r2d2::Pool::builder().max_size(8).build(manager)?;

        // Initialize schema
        {
            let conn = pool.get()?;
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
            conn.execute_batch(SCHEMA)?;
        }

        Ok(Self { pool })
    }

    pub fn pool(&self) -> &Pool<SqliteConnectionManager> {
        &self.pool
    }

    // ======================================================================
    // Experiments
    // ======================================================================

    pub fn create_experiment(
        &self,
        config: &serde_json::Value,
        hypothesis: &str,
        parent_id: Option<i64>,
        code_patch: &str,
    ) -> DbResult<i64> {
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO experiments (config_json, status, hypothesis, parent_id, code_patch, created_at)
             VALUES (?1, 'pending', ?2, ?3, ?4, ?5)",
            params![
                serde_json::to_string(config)?,
                hypothesis,
                parent_id,
                code_patch,
                now()
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn update_experiment_status(
        &self,
        experiment_id: i64,
        status: &str,
        final_metrics: Option<&serde_json::Value>,
        error_msg: &str,
    ) -> DbResult<()> {
        let conn = self.pool.get()?;
        let finished = if status == "completed" || status == "failed" {
            now()
        } else {
            0.0
        };
        conn.execute(
            "UPDATE experiments SET status=?1, final_metrics=?2, error_msg=?3, finished_at=?4 WHERE id=?5",
            params![
                status,
                serde_json::to_string(&final_metrics.unwrap_or(&serde_json::json!({})))?,
                error_msg,
                finished,
                experiment_id,
            ],
        )?;
        Ok(())
    }

    pub fn get_experiment(&self, id: i64) -> DbResult<Option<Experiment>> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare("SELECT * FROM experiments WHERE id=?1")?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(Experiment {
                id: Some(row.get(0)?),
                config_json: row.get(1)?,
                status: row.get(2)?,
                hypothesis: row.get(3)?,
                final_metrics: row.get(4)?,
                parent_id: row.get(5)?,
                error_msg: row.get(6)?,
                code_patch: row.get(7)?,
                created_at: row.get(8)?,
                finished_at: row.get(9)?,
            })
        })?;
        Ok(rows.next().transpose()?)
    }

    pub fn get_experiments(
        &self,
        status: Option<&str>,
        limit: i64,
    ) -> DbResult<Vec<Experiment>> {
        let conn = self.pool.get()?;
        let mut experiments = Vec::new();

        if let Some(status) = status {
            let mut stmt = conn.prepare(
                "SELECT * FROM experiments WHERE status=?1 ORDER BY id DESC LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![status, limit], row_to_experiment)?;
            for row in rows {
                experiments.push(row?);
            }
        } else {
            let mut stmt =
                conn.prepare("SELECT * FROM experiments ORDER BY id DESC LIMIT ?1")?;
            let rows = stmt.query_map(params![limit], row_to_experiment)?;
            for row in rows {
                experiments.push(row?);
            }
        }

        Ok(experiments)
    }

    pub fn get_best_experiment(&self, metric_key: &str) -> DbResult<Option<Experiment>> {
        let conn = self.pool.get()?;
        // Use json_extract to find the best experiment in SQL (avoids loading 500 rows)
        let sql = format!(
            "SELECT id, config_json, status, hypothesis, final_metrics, parent_id, \
             error_msg, code_patch, created_at, finished_at \
             FROM experiments \
             WHERE status='completed' AND json_extract(final_metrics, '$.{}') IS NOT NULL \
             ORDER BY CAST(json_extract(final_metrics, '$.{}') AS REAL) DESC \
             LIMIT 1",
            metric_key, metric_key
        );
        let mut stmt = conn.prepare(&sql)?;
        let result = stmt.query_row([], |row| {
            Ok(Experiment {
                id: row.get(0)?,
                config_json: row.get(1)?,
                status: row.get(2)?,
                hypothesis: row.get(3)?,
                final_metrics: row.get(4)?,
                parent_id: row.get(5)?,
                error_msg: row.get(6)?,
                code_patch: row.get(7)?,
                created_at: row.get(8)?,
                finished_at: row.get(9)?,
            })
        });
        match result {
            Ok(exp) => Ok(Some(exp)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn experiment_count(&self, status: Option<&str>) -> DbResult<i64> {
        let conn = self.pool.get()?;
        if let Some(status) = status {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM experiments WHERE status=?1",
                params![status],
                |row| row.get(0),
            )?;
            Ok(count)
        } else {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM experiments",
                [],
                |row| row.get(0),
            )?;
            Ok(count)
        }
    }

    /// Get all experiment counts in a single query (total, completed, running, failed, pending).
    pub fn experiment_counts(&self) -> DbResult<ExperimentCounts> {
        let conn = self.pool.get()?;
        conn.query_row(
            "SELECT COUNT(*) as total, \
             SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END), \
             SUM(CASE WHEN status='running' THEN 1 ELSE 0 END), \
             SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END), \
             SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) \
             FROM experiments",
            [],
            |row| {
                Ok(ExperimentCounts {
                    total: row.get(0)?,
                    completed: row.get(1)?,
                    running: row.get(2)?,
                    failed: row.get(3)?,
                    pending: row.get(4)?,
                })
            },
        ).map_err(Into::into)
    }

    // ======================================================================
    // Metric snapshots
    // ======================================================================

    pub fn add_metric_snapshot(
        &self,
        experiment_id: i64,
        step: i64,
        metrics: &serde_json::Value,
    ) -> DbResult<()> {
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO metric_snapshots (experiment_id, step, metrics_json, timestamp)
             VALUES (?1, ?2, ?3, ?4)",
            params![experiment_id, step, serde_json::to_string(metrics)?, now()],
        )?;
        Ok(())
    }

    pub fn get_metric_snapshots(&self, experiment_id: i64) -> DbResult<Vec<MetricSnapshot>> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM metric_snapshots WHERE experiment_id=?1 ORDER BY step",
        )?;
        let rows = stmt.query_map(params![experiment_id], |row| {
            Ok(MetricSnapshot {
                id: Some(row.get(0)?),
                experiment_id: row.get(1)?,
                step: row.get(2)?,
                metrics_json: row.get(3)?,
                timestamp: row.get(4)?,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ======================================================================
    // Decisions
    // ======================================================================

    pub fn log_decision(
        &self,
        subsystem: &str,
        action: &str,
        reasoning: &str,
        context: Option<&serde_json::Value>,
    ) -> DbResult<()> {
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO decisions (timestamp, subsystem, action, reasoning, context_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                now(),
                subsystem,
                action,
                reasoning,
                serde_json::to_string(&context.unwrap_or(&serde_json::json!({})))?,
            ],
        )?;
        Ok(())
    }

    pub fn get_decisions(
        &self,
        subsystem: Option<&str>,
        limit: i64,
    ) -> DbResult<Vec<Decision>> {
        let conn = self.pool.get()?;
        let mut decisions = Vec::new();

        if let Some(subsystem) = subsystem {
            let mut stmt = conn.prepare(
                "SELECT * FROM decisions WHERE subsystem=?1 ORDER BY timestamp DESC LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![subsystem, limit], row_to_decision)?;
            for row in rows {
                decisions.push(row?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit], row_to_decision)?;
            for row in rows {
                decisions.push(row?);
            }
        }

        Ok(decisions)
    }

    // ======================================================================
    // Data inventory
    // ======================================================================

    pub fn add_data_item(
        &self,
        dataset: &str,
        clip_id: &str,
        path: &str,
        size_bytes: i64,
        status: &str,
    ) -> DbResult<i64> {
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO data_inventory (dataset, clip_id, path, size_bytes, status)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![dataset, clip_id, path, size_bytes, status],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_data_inventory_stats(&self) -> DbResult<std::collections::HashMap<String, i64>> {
        let conn = self.pool.get()?;
        let mut stmt =
            conn.prepare("SELECT status, COUNT(*) as cnt FROM data_inventory GROUP BY status")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        let mut map = std::collections::HashMap::new();
        for row in rows {
            let (status, cnt) = row?;
            map.insert(status, cnt);
        }
        Ok(map)
    }

    /// Get clip_ids from data_inventory ordered by rowid (matches embedding order).
    pub fn get_clip_ids_ordered(&self, limit: i64) -> DbResult<Vec<String>> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT clip_id FROM data_inventory ORDER BY rowid LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit], |row| row.get(0))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }

    // ======================================================================
    // Code mutations
    // ======================================================================

    pub fn add_code_mutation(&self, mutation: &CodeMutation) -> DbResult<i64> {
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO code_mutations (target_file, target_name, original_code, mutated_code,
             diff, llm_prompt, llm_response, experiment_id, score_delta, accepted, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                mutation.target_file,
                mutation.target_name,
                mutation.original_code,
                mutation.mutated_code,
                mutation.diff,
                mutation.llm_prompt,
                mutation.llm_response,
                mutation.experiment_id,
                mutation.score_delta,
                mutation.accepted as i32,
                now(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn update_mutation_result(
        &self,
        mutation_id: i64,
        experiment_id: i64,
        score_delta: f64,
        accepted: bool,
    ) -> DbResult<()> {
        let conn = self.pool.get()?;
        conn.execute(
            "UPDATE code_mutations SET experiment_id=?1, score_delta=?2, accepted=?3 WHERE id=?4",
            params![experiment_id, score_delta, accepted as i32, mutation_id],
        )?;
        Ok(())
    }

    pub fn get_mutations_for_target(
        &self,
        target_name: Option<&str>,
        limit: i64,
    ) -> DbResult<Vec<CodeMutation>> {
        let conn = self.pool.get()?;
        let mut mutations = Vec::new();

        if let Some(target) = target_name {
            let mut stmt = conn.prepare(
                "SELECT * FROM code_mutations WHERE target_name=?1 ORDER BY id DESC LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![target, limit], row_to_mutation)?;
            for row in rows {
                mutations.push(row?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT * FROM code_mutations ORDER BY id DESC LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit], row_to_mutation)?;
            for row in rows {
                mutations.push(row?);
            }
        }

        Ok(mutations)
    }

    pub fn get_accepted_mutations(&self, limit: i64) -> DbResult<Vec<CodeMutation>> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM code_mutations WHERE accepted=1 ORDER BY score_delta DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], row_to_mutation)?;
        let mut mutations = Vec::new();
        for row in rows {
            mutations.push(row?);
        }
        Ok(mutations)
    }

    pub fn get_mutation_count(&self) -> DbResult<i64> {
        let conn = self.pool.get()?;
        Ok(conn.query_row("SELECT COUNT(*) FROM code_mutations", [], |row| row.get(0))?)
    }

    pub fn get_mutation_stats(&self) -> DbResult<MutationStats> {
        let conn = self.pool.get()?;

        // Single query: GROUP BY with ROLLUP-style total via union, or just sum from by_target
        let mut stmt = conn.prepare(
            "SELECT target_name, COUNT(*) as cnt,
             SUM(CASE WHEN accepted=1 THEN 1 ELSE 0 END) as accepted_cnt,
             AVG(score_delta) as avg_delta
             FROM code_mutations GROUP BY target_name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(TargetMutationStats {
                target_name: row.get(0)?,
                cnt: row.get(1)?,
                accepted_cnt: row.get(2)?,
                avg_delta: row.get(3)?,
            })
        })?;

        let mut by_target = Vec::new();
        for row in rows {
            by_target.push(row?);
        }

        // Derive totals from by_target (avoids 2 extra queries)
        let total: i64 = by_target.iter().map(|t| t.cnt).sum();
        let accepted: i64 = by_target.iter().map(|t| t.accepted_cnt).sum();

        Ok(MutationStats {
            total,
            accepted,
            acceptance_rate: accepted as f64 / total.max(1) as f64,
            by_target,
        })
    }

    /// Get the last N completed experiments' metric values for trend analysis.
    /// Returns (experiment_id, metric_value, finished_at) tuples in chronological order.
    pub fn get_metric_trend(
        &self,
        metric_key: &str,
        limit: i64,
    ) -> DbResult<Vec<(i64, f64, f64)>> {
        let conn = self.pool.get()?;
        let sql = format!(
            "SELECT id, CAST(json_extract(final_metrics, '$.{}') AS REAL) as val, finished_at \
             FROM experiments \
             WHERE status='completed' AND json_extract(final_metrics, '$.{}') IS NOT NULL \
             ORDER BY id DESC LIMIT ?1",
            metric_key, metric_key
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?, row.get::<_, f64>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        results.reverse(); // chronological order
        Ok(results)
    }

    /// Get the count of distinct accepted mutation targets.
    pub fn accepted_mutation_targets(&self) -> DbResult<i64> {
        let conn = self.pool.get()?;
        Ok(conn.query_row(
            "SELECT COUNT(DISTINCT target_name) FROM code_mutations WHERE accepted=1",
            [],
            |row| row.get(0),
        )?)
    }

    /// Get the first and last experiment timestamps.
    pub fn experiment_time_span(&self) -> DbResult<(f64, f64)> {
        let conn = self.pool.get()?;
        conn.query_row(
            "SELECT MIN(created_at), MAX(finished_at) FROM experiments WHERE status='completed'",
            [],
            |row| Ok((row.get::<_, f64>(0).unwrap_or(0.0), row.get::<_, f64>(1).unwrap_or(0.0))),
        ).map_err(Into::into)
    }

    pub fn get_best_config(&self, metric_key: &str) -> DbResult<serde_json::Value> {
        let best = self.get_best_experiment(metric_key)?;
        match best {
            Some(exp) => {
                let mut config: serde_json::Value =
                    serde_json::from_str(&exp.config_json).unwrap_or_default();
                let metrics: serde_json::Value =
                    serde_json::from_str(&exp.final_metrics).unwrap_or_default();
                if let (Some(c), Some(m)) = (config.as_object_mut(), metrics.as_object()) {
                    for (k, v) in m {
                        c.insert(k.clone(), v.clone());
                    }
                }
                Ok(config)
            }
            None => Ok(serde_json::json!({})),
        }
    }
}

fn row_to_experiment(row: &rusqlite::Row) -> rusqlite::Result<Experiment> {
    Ok(Experiment {
        id: Some(row.get(0)?),
        config_json: row.get(1)?,
        status: row.get(2)?,
        hypothesis: row.get(3)?,
        final_metrics: row.get(4)?,
        parent_id: row.get(5)?,
        error_msg: row.get(6)?,
        code_patch: row.get(7)?,
        created_at: row.get(8)?,
        finished_at: row.get(9)?,
    })
}

fn row_to_decision(row: &rusqlite::Row) -> rusqlite::Result<Decision> {
    Ok(Decision {
        id: Some(row.get(0)?),
        timestamp: row.get(1)?,
        subsystem: row.get(2)?,
        action: row.get(3)?,
        reasoning: row.get(4)?,
        context_json: row.get(5)?,
    })
}

fn row_to_mutation(row: &rusqlite::Row) -> rusqlite::Result<CodeMutation> {
    Ok(CodeMutation {
        id: Some(row.get(0)?),
        target_file: row.get(1)?,
        target_name: row.get(2)?,
        original_code: row.get(3)?,
        mutated_code: row.get(4)?,
        diff: row.get(5)?,
        llm_prompt: row.get(6)?,
        llm_response: row.get(7)?,
        experiment_id: row.get(8)?,
        score_delta: row.get(9)?,
        accepted: row.get::<_, i32>(10)? != 0,
        created_at: row.get(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_kb() -> KnowledgeBase {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        // Leak dir so it doesn't get cleaned up during test
        std::mem::forget(dir);
        KnowledgeBase::new(&path).unwrap()
    }

    #[test]
    fn test_create_and_get_experiment() {
        let kb = test_kb();
        let config = serde_json::json!({"lr": 0.01});
        let id = kb.create_experiment(&config, "test hypothesis", None, "").unwrap();
        let exp = kb.get_experiment(id).unwrap().unwrap();
        assert_eq!(exp.status, "pending");
        assert_eq!(exp.hypothesis, "test hypothesis");
    }

    #[test]
    fn test_mutation_stats() {
        let kb = test_kb();
        let mutation = CodeMutation {
            id: None,
            target_file: "test.py".to_string(),
            target_name: "HebbianAssociation.update".to_string(),
            original_code: "original".to_string(),
            mutated_code: "mutated".to_string(),
            diff: "diff".to_string(),
            llm_prompt: "".to_string(),
            llm_response: "".to_string(),
            experiment_id: None,
            score_delta: 0.01,
            accepted: true,
            created_at: 0.0,
        };
        kb.add_code_mutation(&mutation).unwrap();
        let stats = kb.get_mutation_stats().unwrap();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.accepted, 1);
    }
}
