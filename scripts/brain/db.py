"""Database layer — SQLite connection management and schema migrations."""
import sqlite3
import threading
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Thread-local connections
_db_local = threading.local()


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    conn = getattr(_db_local, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _db_local.conn = conn
    return conn


# Schema version → SQL statements
MIGRATIONS = {
    1: """
        CREATE TABLE IF NOT EXISTS perceptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL, modality TEXT NOT NULL,
            transcription TEXT, top_labels TEXT, cross_labels TEXT,
            imagination TEXT, narration TEXT
        );
        CREATE TABLE IF NOT EXISTS reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL, insight TEXT NOT NULL,
            perception_count INTEGER, online_pairs_learned INTEGER
        );
        CREATE TABLE IF NOT EXISTS learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL, event TEXT NOT NULL, details TEXT
        );
        CREATE TABLE IF NOT EXISTS stats (
            key TEXT PRIMARY KEY, value TEXT, updated_at REAL
        );
    """,
    2: """
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time REAL NOT NULL, end_time REAL,
            context TEXT, event_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS episode_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL, timestamp REAL NOT NULL,
            modality TEXT NOT NULL, embedding_blob BLOB,
            label TEXT, metadata_json TEXT,
            FOREIGN KEY (episode_id) REFERENCES episodes(id)
        );
        CREATE INDEX IF NOT EXISTS idx_episode_events_episode ON episode_events(episode_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_time);
    """,
    3: """
        CREATE TABLE IF NOT EXISTS prototypes (
            name TEXT PRIMARY KEY, centroid_blob BLOB NOT NULL,
            count INTEGER DEFAULT 1, examples_json TEXT,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL, embedding_blob BLOB,
            priority REAL DEFAULT 1.0, status TEXT DEFAULT 'active',
            created_at REAL NOT NULL, completed_at REAL
        );
    """,
    4: """
        CREATE TABLE IF NOT EXISTS knowledge_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_label TEXT NOT NULL, relation TEXT NOT NULL,
            target_label TEXT NOT NULL, weight REAL DEFAULT 1.0,
            evidence_count INTEGER DEFAULT 1, last_seen REAL NOT NULL,
            metadata_json TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_unique
            ON knowledge_edges(source_label, relation, target_label);
        CREATE INDEX IF NOT EXISTS idx_ke_source ON knowledge_edges(source_label);
        CREATE INDEX IF NOT EXISTS idx_ke_target ON knowledge_edges(target_label);
    """,
    5: """
        CREATE TABLE IF NOT EXISTS youtube_learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL, category TEXT NOT NULL,
            timestamp REAL NOT NULL, status TEXT NOT NULL,
            pairs_generated INTEGER DEFAULT 0, error_text TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_yll_category ON youtube_learning_log(category);
    """,
}


def run_migrations(db_path: str | Path):
    """Run pending schema migrations."""
    conn = get_connection(db_path)
    # Get current version
    try:
        row = conn.execute("SELECT value FROM stats WHERE key='schema_version'").fetchone()
        current = int(row["value"]) if row else 0
    except Exception:
        current = 0

    applied = 0
    for version in sorted(MIGRATIONS.keys()):
        if version > current:
            log.info(f"Running migration v{version}...")
            conn.executescript(MIGRATIONS[version])
            conn.execute(
                "INSERT INTO stats (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                ("schema_version", str(version), time.time()))
            conn.commit()
            applied += 1

    if applied:
        log.info(f"Applied {applied} migrations (now at v{max(MIGRATIONS.keys())})")
    return applied
