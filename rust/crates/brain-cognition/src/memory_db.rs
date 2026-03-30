//! Memory database — SQLite wrapper for brain_memory.db tables.
//!
//! Manages: perceptions, episodes, episode_events, prototypes, goals,
//! knowledge_edges, youtube_learning_log, stats, reflections, learning_log.

use rusqlite::{params, Connection, Result as SqlResult};
use serde::Serialize;
use std::path::Path;
use std::sync::Mutex;

/// Thread-safe wrapper around SQLite connection for brain memory.
pub struct MemoryDb {
    pub(crate) conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Episode {
    pub id: i64,
    pub start_time: f64,
    pub end_time: Option<f64>,
    pub event_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EpisodeEvent {
    pub id: i64,
    pub episode_id: i64,
    pub timestamp: f64,
    pub modality: String,
    pub label: Option<String>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Perception {
    pub id: i64,
    pub timestamp: f64,
    pub modality: String,
    pub transcription: Option<String>,
    pub top_labels: Option<String>,
    pub narration: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KnowledgeEdge {
    pub source_label: String,
    pub relation: String,
    pub target_label: String,
    pub weight: f64,
    pub evidence_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrototypeRow {
    pub name: String,
    pub count: i64,
    pub examples_json: Option<String>,
    pub created_at: f64,
    pub centroid_blob: Vec<u8>,
}

impl MemoryDb {
    /// Open (or create) the brain memory database and run migrations.
    pub fn open(path: &Path) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;

        // Run schema creation
        conn.execute_batch(include_str!("sql/schema.sql"))?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    // ── Episodes ──────────────────────────────────────────────

    pub fn create_episode(&self, start_time: f64) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute("INSERT INTO episodes (start_time) VALUES (?1)", params![start_time])?;
        Ok(conn.last_insert_rowid())
    }

    pub fn close_episode(&self, id: i64, end_time: f64) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("UPDATE episodes SET end_time = ?1 WHERE id = ?2", params![end_time, id])?;
        Ok(())
    }

    pub fn add_episode_event(&self, episode_id: i64, timestamp: f64, modality: &str,
                              embedding: Option<&[u8]>, label: Option<&str>,
                              metadata_json: Option<&str>) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO episode_events (episode_id, timestamp, modality, embedding_blob, label, metadata_json) VALUES (?1,?2,?3,?4,?5,?6)",
            params![episode_id, timestamp, modality, embedding, label, metadata_json])?;
        conn.execute("UPDATE episodes SET event_count = event_count + 1, end_time = ?1 WHERE id = ?2",
                     params![timestamp, episode_id])?;
        Ok(())
    }

    pub fn get_episodes(&self, limit: i64) -> SqlResult<Vec<Episode>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, start_time, end_time, event_count FROM episodes ORDER BY id DESC LIMIT ?1")?;
        let rows = stmt.query_map(params![limit], |row| {
            Ok(Episode {
                id: row.get(0)?,
                start_time: row.get(1)?,
                end_time: row.get(2)?,
                event_count: row.get(3)?,
            })
        })?;
        rows.collect()
    }

    pub fn get_episode_events(&self, episode_id: i64) -> SqlResult<Vec<EpisodeEvent>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, episode_id, timestamp, modality, label, metadata_json FROM episode_events WHERE episode_id = ?1 ORDER BY timestamp")?;
        let rows = stmt.query_map(params![episode_id], |row| {
            Ok(EpisodeEvent {
                id: row.get(0)?,
                episode_id: row.get(1)?,
                timestamp: row.get(2)?,
                modality: row.get(3)?,
                label: row.get(4)?,
                metadata_json: row.get(5)?,
            })
        })?;
        rows.collect()
    }

    pub fn get_episode_embeddings(&self, episode_id: i64) -> SqlResult<Vec<Vec<u8>>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT embedding_blob FROM episode_events WHERE episode_id = ?1 AND embedding_blob IS NOT NULL ORDER BY timestamp")?;
        let rows = stmt.query_map(params![episode_id], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob)
        })?;
        rows.collect()
    }

    // ── Perceptions ──────────────────────────────────────────

    pub fn store_perception(&self, timestamp: f64, modality: &str,
                             transcription: Option<&str>, top_labels: Option<&str>,
                             cross_labels: Option<&str>, narration: Option<&str>) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO perceptions (timestamp, modality, transcription, top_labels, cross_labels, narration) VALUES (?1,?2,?3,?4,?5,?6)",
            params![timestamp, modality, transcription, top_labels, cross_labels, narration])?;
        Ok(conn.last_insert_rowid())
    }

    pub fn perception_count(&self) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT count(*) FROM perceptions", [], |row| row.get(0))
    }

    pub fn episode_count(&self) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT count(*) FROM episodes", [], |row| row.get(0))
    }

    // ── Knowledge Graph ──────────────────────────────────────

    pub fn upsert_edge(&self, source: &str, relation: &str, target: &str, weight: f64) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        conn.execute(
            "INSERT INTO knowledge_edges (source_label, relation, target_label, weight, last_seen) VALUES (?1,?2,?3,?4,?5) \
             ON CONFLICT(source_label, relation, target_label) DO UPDATE SET evidence_count = evidence_count + 1, \
             weight = max(weight, excluded.weight), last_seen = excluded.last_seen",
            params![source, relation, target, weight, now])?;
        Ok(())
    }

    pub fn get_edges(&self, source: &str, limit: i64) -> SqlResult<Vec<KnowledgeEdge>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT source_label, relation, target_label, weight, evidence_count FROM knowledge_edges WHERE source_label = ?1 ORDER BY weight DESC LIMIT ?2")?;
        let rows = stmt.query_map(params![source, limit], |row| {
            Ok(KnowledgeEdge {
                source_label: row.get(0)?,
                relation: row.get(1)?,
                target_label: row.get(2)?,
                weight: row.get(3)?,
                evidence_count: row.get(4)?,
            })
        })?;
        rows.collect()
    }

    pub fn edge_count(&self) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT count(*) FROM knowledge_edges", [], |row| row.get(0))
    }

    pub fn edges_by_relation(&self) -> SqlResult<Vec<(String, i64, f64)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT relation, count(*), avg(weight) FROM knowledge_edges GROUP BY relation ORDER BY count(*) DESC")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?, row.get::<_, f64>(2)?))
        })?;
        rows.collect()
    }

    // ── Prototypes ───────────────────────────────────────────

    pub fn get_prototypes(&self) -> SqlResult<Vec<PrototypeRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT name, count, examples_json, created_at, centroid_blob FROM prototypes ORDER BY count DESC")?;
        let rows = stmt.query_map([], |row| {
            Ok(PrototypeRow {
                name: row.get(0)?,
                count: row.get(1)?,
                examples_json: row.get(2)?,
                created_at: row.get(3)?,
                centroid_blob: row.get::<_, Vec<u8>>(4).unwrap_or_default(),
            })
        })?;
        rows.collect()
    }

    pub fn upsert_prototype(&self, name: &str, centroid: &[u8], count: i64,
                             examples_json: &str) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        conn.execute(
            "INSERT INTO prototypes (name, centroid_blob, count, examples_json, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,?5) \
             ON CONFLICT(name) DO UPDATE SET centroid_blob=excluded.centroid_blob, count=excluded.count, \
             examples_json=excluded.examples_json, updated_at=excluded.updated_at",
            params![name, centroid, count, examples_json, now])?;
        Ok(())
    }

    pub fn prototype_count(&self) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT count(*) FROM prototypes", [], |row| row.get(0))
    }

    // ── Stats ────────────────────────────────────────────────

    pub fn get_stat(&self, key: &str) -> SqlResult<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let result = conn.query_row(
            "SELECT value FROM stats WHERE key = ?1", params![key],
            |row| row.get::<_, String>(0));
        match result {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn set_stat(&self, key: &str, value: &str) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        conn.execute(
            "INSERT INTO stats (key, value, updated_at) VALUES (?1,?2,?3) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            params![key, value, now])?;
        Ok(())
    }

    // ── Learning Log ─────────────────────────────────────────

    pub fn log_learning(&self, event: &str, details: Option<&str>) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        conn.execute(
            "INSERT INTO learning_log (timestamp, event, details) VALUES (?1,?2,?3)",
            params![now, event, details])?;
        Ok(())
    }

    // ── YouTube Learning Log ─────────────────────────────────

    pub fn log_youtube(&self, video_id: &str, category: &str, status: &str, pairs: i64) -> SqlResult<()> {
        let conn = self.conn.lock().unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        conn.execute(
            "INSERT INTO youtube_learning_log (video_id, category, timestamp, status, pairs_generated) VALUES (?1,?2,?3,?4,?5)",
            params![video_id, category, now, status, pairs])?;
        Ok(())
    }

    // ── Reflection ───────────────────────────────────────────

    pub fn reflection_count(&self) -> SqlResult<i64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT count(*) FROM reflections", [], |row| row.get(0))
    }

    // ── Memory Stats ─────────────────────────────────────────

    pub fn memory_stats(&self) -> SqlResult<serde_json::Value> {
        let conn = self.conn.lock().unwrap();
        let total_perceptions: i64 = conn.query_row("SELECT count(*) FROM perceptions", [], |r| r.get(0))?;
        let total_reflections: i64 = conn.query_row("SELECT count(*) FROM reflections", [], |r| r.get(0))?;

        // Perceptions by modality
        let mut stmt = conn.prepare("SELECT modality, count(*) FROM perceptions GROUP BY modality")?;
        let by_mod: Vec<(String, i64)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.filter_map(|r| r.ok()).collect();
        let by_modality: serde_json::Map<String, serde_json::Value> = by_mod.into_iter()
            .map(|(k, v)| (k, serde_json::Value::Number(v.into()))).collect();

        // Last perception timestamp
        let last_ts: Option<f64> = conn.query_row(
            "SELECT max(timestamp) FROM perceptions", [], |r| r.get(0)).ok();

        // Online pairs learned
        let online: String = conn.query_row(
            "SELECT COALESCE(value, '0') FROM stats WHERE key = 'online_learning_count'", [],
            |r| r.get(0)).unwrap_or_else(|_| "0".to_string());

        Ok(serde_json::json!({
            "perceptions_by_modality": by_modality,
            "total_perceptions": total_perceptions,
            "total_reflections": total_reflections,
            "online_pairs_learned": online.parse::<i64>().unwrap_or(0),
            "last_perception_timestamp": last_ts,
        }))
    }

    /// Recent perceptions + reflections.
    pub fn memory_recent(&self, limit: i64) -> SqlResult<serde_json::Value> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, timestamp, modality, transcription, top_labels, narration FROM perceptions ORDER BY id DESC LIMIT ?1")?;
        let perceptions: Vec<serde_json::Value> = stmt.query_map(params![limit], |r| {
            Ok(serde_json::json!({
                "id": r.get::<_, i64>(0)?,
                "timestamp": r.get::<_, f64>(1)?,
                "modality": r.get::<_, String>(2)?,
                "transcription": r.get::<_, Option<String>>(3)?,
                "top_labels": r.get::<_, Option<String>>(4)?,
                "narration": r.get::<_, Option<String>>(5)?,
            }))
        })?.filter_map(|r| r.ok()).collect();

        let mut stmt2 = conn.prepare("SELECT id, timestamp, insight FROM reflections ORDER BY id DESC LIMIT ?1")?;
        let reflections: Vec<serde_json::Value> = stmt2.query_map(params![limit / 2], |r| {
            Ok(serde_json::json!({
                "id": r.get::<_, i64>(0)?,
                "timestamp": r.get::<_, f64>(1)?,
                "insight": r.get::<_, String>(2)?,
            }))
        })?.filter_map(|r| r.ok()).collect();

        Ok(serde_json::json!({
            "perceptions": perceptions,
            "reflections": reflections,
        }))
    }

    /// Knowledge graph multi-hop traversal.
    pub fn traverse_graph(&self, start: &str, max_hops: usize, max_results: usize) -> SqlResult<serde_json::Value> {
        let conn = self.conn.lock().unwrap();
        let mut visited = std::collections::HashSet::new();
        visited.insert(start.to_string());
        let mut frontier: Vec<(String, Vec<String>, f64)> = vec![(start.to_string(), vec![start.to_string()], 1.0)];
        let mut results = Vec::new();

        for _hop in 0..max_hops {
            let mut next_frontier = Vec::new();
            for (node, path, acc_w) in &frontier {
                let mut stmt = conn.prepare(
                    "SELECT target_label, relation, weight FROM knowledge_edges WHERE source_label = ?1 ORDER BY weight DESC LIMIT 20")?;
                let edges: Vec<(String, String, f64)> = stmt.query_map(params![node], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?.filter_map(|r| r.ok()).collect();

                for (target, relation, weight) in edges {
                    if !visited.contains(&target) {
                        visited.insert(target.clone());
                        let new_w = acc_w * weight;
                        let mut new_path = path.clone();
                        new_path.push(format!("--{relation}-->"));
                        new_path.push(target.clone());
                        results.push(serde_json::json!({
                            "target": target,
                            "path": new_path,
                            "hops": _hop + 1,
                            "weight": (new_w * 10000.0).round() / 10000.0,
                            "relation": relation,
                        }));
                        next_frontier.push((target, new_path, new_w));
                    }
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() { break; }
        }

        results.sort_by(|a, b| {
            let wa = a["weight"].as_f64().unwrap_or(0.0);
            let wb = b["weight"].as_f64().unwrap_or(0.0);
            wb.partial_cmp(&wa).unwrap()
        });
        results.truncate(max_results);

        // Also get direct edges
        let mut stmt = conn.prepare(
            "SELECT target_label, relation, weight, evidence_count FROM knowledge_edges WHERE source_label = ?1 ORDER BY weight DESC LIMIT 20")?;
        let direct: Vec<serde_json::Value> = stmt.query_map(params![start], |r| {
            Ok(serde_json::json!({
                "target": r.get::<_, String>(0)?,
                "relation": r.get::<_, String>(1)?,
                "weight": r.get::<_, f64>(2)?,
                "evidence": r.get::<_, i64>(3)?,
            }))
        })?.filter_map(|r| r.ok()).collect();

        Ok(serde_json::json!({
            "start": start,
            "direct_edges": direct,
            "paths": results,
            "total_paths": results.len(),
        }))
    }

    /// YouTube learning log stats.
    pub fn youtube_stats(&self) -> SqlResult<(i64, Vec<(String, i64)>)> {
        let conn = self.conn.lock().unwrap();
        let total: i64 = conn.query_row(
            "SELECT count(*) FROM youtube_learning_log WHERE status = 'success'", [], |r| r.get(0))?;
        let mut stmt = conn.prepare(
            "SELECT category, count(*) FROM youtube_learning_log WHERE status = 'success' GROUP BY category ORDER BY count(*) DESC LIMIT 20")?;
        let cats: Vec<(String, i64)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.filter_map(|r| r.ok()).collect();
        Ok((total, cats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_db() -> MemoryDb {
        MemoryDb::open(&PathBuf::from(":memory:")).unwrap()
    }

    #[test]
    fn test_episode_lifecycle() {
        let db = temp_db();
        let ep_id = db.create_episode(1000.0).unwrap();
        db.add_episode_event(ep_id, 1001.0, "audio", None, Some("thunder"), None).unwrap();
        db.add_episode_event(ep_id, 1002.0, "audio", None, Some("rain"), None).unwrap();
        db.close_episode(ep_id, 1002.0).unwrap();

        let episodes = db.get_episodes(10).unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].event_count, 2);

        let events = db.get_episode_events(ep_id).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].label.as_deref(), Some("thunder"));
    }

    #[test]
    fn test_perception_count() {
        let db = temp_db();
        db.store_perception(1000.0, "audio", None, Some("[\"thunder\"]"), None, None).unwrap();
        db.store_perception(1001.0, "visual", None, Some("[\"rain\"]"), None, None).unwrap();
        assert_eq!(db.perception_count().unwrap(), 2);
    }

    #[test]
    fn test_knowledge_edges() {
        let db = temp_db();
        db.upsert_edge("thunder", "causes", "rain", 0.8).unwrap();
        db.upsert_edge("thunder", "causes", "rain", 0.9).unwrap(); // should update
        let edges = db.get_edges("thunder", 10).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].evidence_count, 2);
        assert!(edges[0].weight >= 0.9); // max(0.8, 0.9)
        assert_eq!(db.edge_count().unwrap(), 1);
    }

    #[test]
    fn test_stats() {
        let db = temp_db();
        db.set_stat("test_key", "hello").unwrap();
        assert_eq!(db.get_stat("test_key").unwrap(), Some("hello".into()));
        assert_eq!(db.get_stat("nonexistent").unwrap(), None);
    }

    #[test]
    fn test_prototypes() {
        let db = temp_db();
        db.upsert_prototype("thunder", &[0u8; 2048], 5, "[\"thunder storm\"]").unwrap();
        let protos = db.get_prototypes().unwrap();
        assert_eq!(protos.len(), 1);
        assert_eq!(protos[0].name, "thunder");
        assert_eq!(protos[0].count, 5);
    }
}
