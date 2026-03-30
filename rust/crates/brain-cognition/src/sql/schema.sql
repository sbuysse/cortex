-- Brain memory database schema (all tables)
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
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_label TEXT NOT NULL, relation TEXT NOT NULL,
    target_label TEXT NOT NULL, weight REAL DEFAULT 1.0,
    evidence_count INTEGER DEFAULT 1, last_seen REAL NOT NULL,
    metadata_json TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ke_unique ON knowledge_edges(source_label, relation, target_label);
CREATE INDEX IF NOT EXISTS idx_ke_source ON knowledge_edges(source_label);
CREATE INDEX IF NOT EXISTS idx_ke_target ON knowledge_edges(target_label);
CREATE TABLE IF NOT EXISTS youtube_learning_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL, category TEXT NOT NULL,
    timestamp REAL NOT NULL, status TEXT NOT NULL,
    pairs_generated INTEGER DEFAULT 0, error_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_yll_category ON youtube_learning_log(category);

