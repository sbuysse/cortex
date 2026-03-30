//! PersonalMemory — brain-native store for companion data.
//!
//! Replaces SQLite tables (personal_facts, conversations, mood_log) with
//! in-memory structures persisted to a JSON file.  The brain hears something,
//! encodes it, and stores it here — no relational database involved.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;

/// A fact the brain learned about the person it is talking with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredFact {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f64,
    pub emotional_valence: f64,
    pub first_mentioned: f64,
    pub last_mentioned: f64,
    pub mention_count: i64,
}

/// One turn in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvTurn {
    pub timestamp: f64,
    pub role: String,
    pub message: String,
    pub emotion: Option<String>,
}

/// A mood observation extracted from speech or context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodEntry {
    pub timestamp: f64,
    pub emotion: String,
    pub confidence: f64,
    pub trigger: Option<String>,
}

/// On-disk snapshot for persistence.
#[derive(Serialize, Deserialize, Default)]
struct Snapshot {
    facts: Vec<StoredFact>,
    conversations: Vec<ConvTurn>,
    moods: Vec<MoodEntry>,
}

/// The brain's companion memory — what it has heard and learned about the person.
pub struct PersonalMemory {
    pub facts: Vec<StoredFact>,
    pub conversations: VecDeque<ConvTurn>,
    pub moods: VecDeque<MoodEntry>,
    path: Option<PathBuf>,
    conv_cap: usize,
    mood_cap: usize,
}

impl PersonalMemory {
    const DEFAULT_CONV_CAP: usize = 500;
    const DEFAULT_MOOD_CAP: usize = 200;

    /// Create an empty in-memory store (no persistence).
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            conversations: VecDeque::new(),
            moods: VecDeque::new(),
            path: None,
            conv_cap: Self::DEFAULT_CONV_CAP,
            mood_cap: Self::DEFAULT_MOOD_CAP,
        }
    }

    /// Load from a JSON file (or create fresh if missing).
    pub fn load(path: PathBuf) -> Self {
        let snapshot: Snapshot = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        Self {
            facts: snapshot.facts,
            conversations: snapshot.conversations.into_iter().collect(),
            moods: snapshot.moods.into_iter().collect(),
            path: Some(path),
            conv_cap: Self::DEFAULT_CONV_CAP,
            mood_cap: Self::DEFAULT_MOOD_CAP,
        }
    }

    /// Persist current state to the JSON file (best-effort).
    pub fn save(&self) {
        let Some(path) = &self.path else { return };
        let snapshot = Snapshot {
            facts: self.facts.clone(),
            conversations: self.conversations.iter().cloned().collect(),
            moods: self.moods.iter().cloned().collect(),
        };
        if let Ok(json) = serde_json::to_string(&snapshot) {
            let _ = std::fs::write(path, json);
        }
    }

    // ── Facts ─────────────────────────────────────────────────

    /// Upsert a fact: increment mention count and update confidence if already known.
    pub fn upsert_fact(&mut self, subject: &str, relation: &str, object: &str,
                       confidence: f64, emotional_valence: f64) {
        let now = now_secs();
        if let Some(f) = self.facts.iter_mut().find(|f| {
            f.subject == subject && f.relation == relation && f.object == object
        }) {
            f.last_mentioned = now;
            f.mention_count += 1;
            f.confidence = (f.confidence + 0.05).min(1.0);
        } else {
            self.facts.push(StoredFact {
                subject: subject.into(),
                relation: relation.into(),
                object: object.into(),
                confidence,
                emotional_valence,
                first_mentioned: now,
                last_mentioned: now,
                mention_count: 1,
            });
        }
        self.save();
    }

    /// All facts sorted by relevance (mention count desc, then recency).
    pub fn all_facts(&self) -> Vec<&StoredFact> {
        let mut v: Vec<&StoredFact> = self.facts.iter().collect();
        v.sort_by(|a, b| b.mention_count.cmp(&a.mention_count)
            .then(b.last_mentioned.partial_cmp(&a.last_mentioned).unwrap_or(std::cmp::Ordering::Equal)));
        v.truncate(50);
        v
    }

    /// Facts where subject or object matches the query.
    pub fn facts_about(&self, query: &str) -> Vec<&StoredFact> {
        let q = query.to_lowercase();
        let mut v: Vec<&StoredFact> = self.facts.iter()
            .filter(|f| f.subject.to_lowercase().contains(&q) || f.object.to_lowercase().contains(&q))
            .collect();
        v.sort_by(|a, b| b.mention_count.cmp(&a.mention_count));
        v.truncate(20);
        v
    }

    /// Get the user's name if known.
    pub fn user_name(&self) -> Option<&str> {
        self.facts.iter()
            .filter(|f| f.subject == "user" && f.relation == "name")
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .map(|f| f.object.as_str())
    }

    // ── Conversations ─────────────────────────────────────────

    /// Store a conversation turn (circular buffer).
    pub fn push_conversation(&mut self, role: &str, message: &str, emotion: Option<&str>) {
        let now = now_secs();
        self.conversations.push_back(ConvTurn {
            timestamp: now,
            role: role.into(),
            message: message[..message.len().min(500)].into(),
            emotion: emotion.map(str::to_string),
        });
        while self.conversations.len() > self.conv_cap {
            self.conversations.pop_front();
        }
        self.save();
    }

    /// Get recent conversation turns (chronological order, newest last).
    pub fn recent_conversation(&self, limit: usize) -> Vec<(&str, &str)> {
        let n = self.conversations.len();
        let start = if n > limit { n - limit } else { 0 };
        self.conversations.iter().skip(start)
            .map(|t| (t.role.as_str(), t.message.as_str()))
            .collect()
    }

    /// Timestamp of the last conversation turn (any role).
    pub fn last_conversation_ts(&self) -> Option<f64> {
        self.conversations.back().map(|t| t.timestamp)
    }

    // ── Moods ─────────────────────────────────────────────────

    /// Record a mood observation (circular buffer).
    pub fn push_mood(&mut self, emotion: &str, confidence: f64, trigger: Option<&str>) {
        let now = now_secs();
        self.moods.push_back(MoodEntry {
            timestamp: now,
            emotion: emotion.into(),
            confidence,
            trigger: trigger.map(str::to_string),
        });
        while self.moods.len() > self.mood_cap {
            self.moods.pop_front();
        }
        self.save();
    }

    /// Recent mood entries (newest last).
    pub fn recent_moods(&self, limit: usize) -> Vec<(f64, &str, f64)> {
        let n = self.moods.len();
        let start = if n > limit { n - limit } else { 0 };
        self.moods.iter().skip(start)
            .map(|m| (m.timestamp, m.emotion.as_str(), m.confidence))
            .collect()
    }

    /// Dominant emotion in the last `hours` hours.
    pub fn mood_trend(&self, hours: f64) -> Option<&str> {
        let cutoff = now_secs() - hours * 3600.0;
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for m in &self.moods {
            if m.timestamp > cutoff {
                *counts.entry(m.emotion.as_str()).or_insert(0) += 1;
            }
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(e, _)| e)
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_fact_increments() {
        let mut mem = PersonalMemory::new();
        mem.upsert_fact("Marie", "is-daughter-of", "user", 0.9, 0.3);
        mem.upsert_fact("Marie", "is-daughter-of", "user", 0.9, 0.3);
        assert_eq!(mem.facts[0].mention_count, 2);
        assert!(mem.facts[0].confidence > 0.9);
    }

    #[test]
    fn test_user_name() {
        let mut mem = PersonalMemory::new();
        mem.upsert_fact("user", "name", "Marguerite", 0.95, 0.2);
        assert_eq!(mem.user_name(), Some("Marguerite"));
    }

    #[test]
    fn test_conversation_cap() {
        let mut mem = PersonalMemory::new();
        mem.conv_cap = 3;
        for i in 0..5 {
            mem.push_conversation("user", &format!("msg {i}"), None);
        }
        assert_eq!(mem.conversations.len(), 3);
        assert_eq!(mem.conversations.front().unwrap().message, "msg 2");
    }

    #[test]
    fn test_mood_trend() {
        let mut mem = PersonalMemory::new();
        mem.push_mood("sad", 0.7, None);
        mem.push_mood("sad", 0.8, None);
        mem.push_mood("happy", 0.5, None);
        assert_eq!(mem.mood_trend(1.0), Some("sad"));
    }

    #[test]
    fn test_facts_about() {
        let mut mem = PersonalMemory::new();
        mem.upsert_fact("Marie", "is-daughter-of", "user", 0.9, 0.3);
        mem.upsert_fact("user", "enjoys", "gardening", 0.9, 0.6);
        let r = mem.facts_about("marie");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].subject, "Marie");
    }

    #[test]
    fn test_json_roundtrip() {
        let dir = std::env::temp_dir().join("test_personal_memory.json");
        {
            let mut mem = PersonalMemory::load(dir.clone());
            mem.upsert_fact("user", "name", "Jean", 0.95, 0.0);
            mem.push_conversation("user", "Hello!", Some("happy"));
            mem.push_mood("happy", 0.8, None);
        }
        let mem2 = PersonalMemory::load(dir.clone());
        assert_eq!(mem2.user_name(), Some("Jean"));
        assert_eq!(mem2.conversations.len(), 1);
        assert_eq!(mem2.moods.len(), 1);
        let _ = std::fs::remove_file(dir);
    }
}
