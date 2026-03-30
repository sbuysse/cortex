//! Personal knowledge — extract facts from conversation and store in brain memory.
//!
//! Parses conversation for people, relationships, health, preferences.
//! All state lives in PersonalMemory (in-memory + JSON), never in SQL.

use regex_lite::Regex;
use crate::personal_memory::PersonalMemory;

/// A fact about the user's personal life (extracted, not yet stored).
#[derive(Debug, Clone)]
pub struct PersonalFact {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f64,
    pub emotional_valence: f64,
}

/// Extract personal facts from a conversation message.
pub fn extract_facts(message: &str) -> Vec<PersonalFact> {
    let msg = message.to_lowercase();
    let mut facts = Vec::new();

    // Family relationships: "my daughter Marie", "my son Pierre"
    let family_re = Regex::new(r"my\s+(daughter|son|wife|husband|brother|sister|mother|father|grandson|granddaughter|niece|nephew|friend|partner)\s+(\w+)").unwrap();
    for cap in family_re.captures_iter(&msg) {
        let relation = cap.get(1).unwrap().as_str();
        let name = capitalize(cap.get(2).unwrap().as_str());
        facts.push(PersonalFact {
            subject: name.clone(),
            relation: format!("is-{}-of", relation),
            object: "user".into(),
            confidence: 0.95,
            emotional_valence: 0.3,
        });
    }

    // "X lives in Y", "X is from Y"
    let location_re = Regex::new(r"(\w+)\s+(?:lives?|living|is from|comes? from|moved? to)\s+(?:in\s+)?(\w[\w\s]{1,20})").unwrap();
    for cap in location_re.captures_iter(&msg) {
        let who = capitalize(cap.get(1).unwrap().as_str());
        let where_ = cap.get(2).unwrap().as_str().trim().to_string();
        if who.len() > 1 && where_.len() > 1 {
            facts.push(PersonalFact {
                subject: who, relation: "lives-in".into(), object: where_,
                confidence: 0.8, emotional_valence: 0.0,
            });
        }
    }

    // Health: "my knee hurts", "I have pain in my back"
    let health_re = Regex::new(r"(?:my\s+(\w+)\s+(?:hurts?|aches?|is\s+sore|pain)|(?:pain|ache|problem)\s+(?:in|with)\s+(?:my\s+)?(\w+)|I\s+(?:have|got)\s+(?:a\s+)?(\w+\s+\w+)\s+(?:problem|issue|condition))").unwrap();
    for cap in health_re.captures_iter(&msg) {
        let body_part = cap.get(1).or(cap.get(2)).or(cap.get(3))
            .map(|m| m.as_str().to_string()).unwrap_or_default();
        if !body_part.is_empty() {
            facts.push(PersonalFact {
                subject: "user".into(), relation: "has-pain-in".into(), object: body_part,
                confidence: 0.85, emotional_valence: -0.5,
            });
        }
    }

    // Preferences: "I like/love/enjoy X", "I hate/dislike X"
    let like_re = Regex::new(r"(?i)I\s+(?:really\s+)?(?:like|love|enjoy|adore)\s+(\w[\w\s]{1,30})").unwrap();
    for cap in like_re.captures_iter(&msg) {
        let thing = cap.get(1).unwrap().as_str().trim().to_string();
        if thing.len() > 1 {
            facts.push(PersonalFact {
                subject: "user".into(), relation: "enjoys".into(), object: thing,
                confidence: 0.9, emotional_valence: 0.6,
            });
        }
    }
    let dislike_re = Regex::new(r"(?i)I\s+(?:really\s+)?(?:hate|dislike|can't stand|don't like)\s+(\w[\w\s]{1,30})").unwrap();
    for cap in dislike_re.captures_iter(&msg) {
        let thing = cap.get(1).unwrap().as_str().trim().to_string();
        if thing.len() > 1 {
            facts.push(PersonalFact {
                subject: "user".into(), relation: "dislikes".into(), object: thing,
                confidence: 0.9, emotional_valence: -0.3,
            });
        }
    }

    // Age: "I am 82", "I'm 82 years old"
    let age_re = Regex::new(r"(?i)I(?:'m|\s+am)\s+(\d{2,3})(?:\s+years?\s+old)?").unwrap();
    if let Some(cap) = age_re.captures(&msg) {
        let age = cap.get(1).unwrap().as_str();
        facts.push(PersonalFact {
            subject: "user".into(), relation: "age".into(), object: age.into(),
            confidence: 0.95, emotional_valence: 0.0,
        });
    }

    // Name: "My name is X", "Call me X"
    let name_re = Regex::new(r"(?i)(?:my name is|call me)\s+([A-Z][a-z]{2,15})").unwrap();
    if let Some(cap) = name_re.captures(message) {
        let name = cap.get(1).unwrap().as_str();
        let not_names = ["the","and","but","not","very","really","just","only","also","here","there","that","this","what","when","how"];
        if name.len() > 2 && name.len() < 16 && !not_names.contains(&name.to_lowercase().as_str()) {
            facts.push(PersonalFact {
                subject: "user".into(), relation: "name".into(), object: name.into(),
                confidence: 0.95, emotional_valence: 0.2,
            });
        }
    }

    // Deceased: "X passed away", "X died", "I lost X"
    let deceased_re = Regex::new(r"(\w+)\s+(?:passed away|died|is gone|we lost)\s*(?:in\s+(\d{4}))?").unwrap();
    for cap in deceased_re.captures_iter(&msg) {
        let who = capitalize(cap.get(1).unwrap().as_str());
        let year = cap.get(2).map(|y| y.as_str().to_string());
        let obj = if let Some(y) = year { format!("deceased-{y}") } else { "deceased".into() };
        facts.push(PersonalFact {
            subject: who, relation: "status".into(), object: obj,
            confidence: 0.9, emotional_valence: -0.8,
        });
    }

    // Appointments: "appointment/doctor on Thursday"
    let appt_re = Regex::new(r"(?:appointment|doctor|meeting|visit)\s+(?:on|this|next)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)").unwrap();
    for cap in appt_re.captures_iter(&msg) {
        let day = capitalize(cap.get(1).unwrap().as_str());
        facts.push(PersonalFact {
            subject: "user".into(), relation: "has-appointment".into(), object: day,
            confidence: 0.8, emotional_valence: -0.1,
        });
    }

    facts
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Store a personal fact in brain memory.
pub fn store_fact(mem: &mut PersonalMemory, fact: &PersonalFact) {
    mem.upsert_fact(&fact.subject, &fact.relation, &fact.object,
                    fact.confidence, fact.emotional_valence);
}

/// Query personal facts about a subject.
pub fn get_facts_about(mem: &PersonalMemory, subject: &str) -> Vec<(String, String, String, i64)> {
    mem.facts_about(subject).into_iter()
        .map(|f| (f.subject.clone(), f.relation.clone(), f.object.clone(), f.mention_count))
        .collect()
}

/// Get all personal facts (for context building).
pub fn get_all_facts(mem: &PersonalMemory) -> Vec<(String, String, String, i64)> {
    mem.all_facts().into_iter()
        .map(|f| (f.subject.clone(), f.relation.clone(), f.object.clone(), f.mention_count))
        .collect()
}

/// Get the user's name if known.
pub fn get_user_name(mem: &PersonalMemory) -> Option<String> {
    mem.user_name().map(str::to_string)
}

/// Store a conversation turn.
pub fn store_conversation(mem: &mut PersonalMemory, role: &str, message: &str, emotion: Option<&str>) {
    mem.push_conversation(role, message, emotion);
}

/// Get recent conversation turns.
pub fn get_recent_conversation(mem: &PersonalMemory, limit: usize) -> Vec<(String, String)> {
    mem.recent_conversation(limit).into_iter()
        .map(|(r, m)| (r.to_string(), m.to_string()))
        .collect()
}

/// Build a compact personal context for the HOPE byte-level decoder.
///
/// Kept short (≤60 chars) so the user message dominates the input window.
/// Format: "Name: Albert, 82. Daughter: Marie."
pub fn build_hope_context(mem: &PersonalMemory) -> String {
    let name = get_user_name(mem).unwrap_or_else(|| "friend".into());
    let facts = get_all_facts(mem);

    let mut ctx = format!("Name: {name}");
    if let Some((_, _, age, _)) = facts.iter().find(|(s, r, _, _)| s == "user" && r == "age") {
        ctx += &format!(", {age}");
    }
    ctx += ".";

    // Add at most one close family member
    let family_rels = ["is-daughter-of", "is-son-of", "is-wife-of", "is-husband-of"];
    if let Some((subj, rel, _, _)) = facts.iter().find(|(_, r, _, _)| family_rels.iter().any(|fr| r == fr)) {
        let role = rel.trim_start_matches("is-").trim_end_matches("-of");
        ctx += &format!(" {}: {}.", capitalize(role), subj);
    }

    ctx
}

/// Build a personal context summary for dialogue.
pub fn build_personal_context(mem: &PersonalMemory) -> String {
    let name = get_user_name(mem).unwrap_or_else(|| "friend".into());
    let facts = get_all_facts(mem);

    let mut family = Vec::new();
    let mut health = Vec::new();
    let mut preferences = Vec::new();
    let mut other = Vec::new();

    for (subj, rel, obj, _count) in &facts {
        let fact_str = format!("{} {} {}", subj, rel.replace('-', " "), obj);
        if rel.contains("daughter") || rel.contains("son") || rel.contains("husband")
            || rel.contains("wife") || rel.contains("brother") || rel.contains("sister")
            || rel.contains("friend") || rel.contains("father") || rel.contains("mother") {
            family.push(fact_str);
        } else if rel.contains("pain") || rel.contains("condition") || rel.contains("appointment") {
            health.push(fact_str);
        } else if rel.contains("enjoy") || rel.contains("like") || rel.contains("dislike") {
            preferences.push(fact_str);
        } else if rel != "name" && rel != "age" {
            other.push(fact_str);
        }
    }

    let mut ctx = format!("Name: {name}");
    if let Some((_, _, age, _)) = facts.iter().find(|(s, r, _, _)| s == "user" && r == "age") {
        ctx += &format!(", Age: {age}");
    }
    if !family.is_empty() { ctx += &format!(". Family: {}", family.join("; ")); }
    if !health.is_empty() { ctx += &format!(". Health: {}", health.join("; ")); }
    if !preferences.is_empty() { ctx += &format!(". Likes: {}", preferences.join("; ")); }
    if !other.is_empty() { ctx += &format!(". Also: {}", other.join("; ")); }

    ctx
}

/// Store a mood observation.
pub fn store_mood(mem: &mut PersonalMemory, emotion: &str, confidence: f64, trigger: Option<&str>) {
    mem.push_mood(emotion, confidence, trigger);
}

/// Get recent mood entries.
pub fn get_recent_moods(mem: &PersonalMemory, limit: usize) -> Vec<(f64, String, f64)> {
    mem.recent_moods(limit).into_iter()
        .map(|(t, e, c)| (t, e.to_string(), c))
        .collect()
}

/// Get dominant mood over last N hours.
pub fn get_mood_trend(mem: &PersonalMemory, hours: f64) -> Option<String> {
    mem.mood_trend(hours).map(str::to_string)
}

/// Detect emotion from conversation text (keyword-based).
pub fn detect_text_emotion(text: &str) -> (&'static str, f64) {
    let t = text.to_lowercase();
    if t.contains("miss") || t.contains("sad") || t.contains("lonely") || t.contains("cry")
        || t.contains("passed away") || t.contains("died") || t.contains("alone") {
        return ("sad", 0.7);
    }
    if t.contains("hurt") || t.contains("pain") || t.contains("ache") || t.contains("sick") {
        return ("pain", 0.7);
    }
    if t.contains("happy") || t.contains("wonderful") || t.contains("great") || t.contains("love")
        || t.contains("laugh") || t.contains("smile") || t.contains("joy") {
        return ("happy", 0.6);
    }
    if t.contains("afraid") || t.contains("scared") || t.contains("worried") || t.contains("anxious") {
        return ("fearful", 0.6);
    }
    if t.contains("angry") || t.contains("frustrated") || t.contains("annoyed") || t.contains("furious") {
        return ("angry", 0.6);
    }
    if t.contains("confused") || t.contains("don't understand") || t.contains("what day") {
        return ("confused", 0.6);
    }
    if t.contains("tired") || t.contains("exhausted") || t.contains("sleepy") || t.contains("didn't sleep") {
        return ("tired", 0.6);
    }
    ("neutral", 0.4)
}

/// Build emotion-adaptive response prefix.
pub fn emotion_response_prefix(_emotion: &str, _user_name: &str) -> &'static str {
    match _emotion {
        "sad" => "I'm here with you. ",
        "pain" => "I'm sorry you're uncomfortable. ",
        "happy" => "It's lovely to see you in good spirits! ",
        "fearful" => "It's okay, you're safe. ",
        "angry" => "I understand that's frustrating. ",
        "confused" => "Let me help orient you. ",
        "tired" => "You sound tired — take it easy. ",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_family() {
        let facts = extract_facts("My daughter Marie lives in Lyon");
        assert!(facts.iter().any(|f| f.subject == "Marie" && f.relation.contains("daughter")));
        assert!(facts.iter().any(|f| f.relation == "lives-in" && f.object.contains("lyon")));
    }

    #[test]
    fn test_extract_health() {
        let facts = extract_facts("my knee hurts today");
        assert!(facts.iter().any(|f| f.relation == "has-pain-in" && f.object == "knee"));
    }

    #[test]
    fn test_extract_preference() {
        let facts = extract_facts("I really love gardening");
        assert!(facts.iter().any(|f| f.relation == "enjoys" && f.object.contains("gardening")));
    }

    #[test]
    fn test_extract_deceased() {
        let facts = extract_facts("Jean passed away in 2019");
        assert!(facts.iter().any(|f| f.subject == "Jean" && f.object == "deceased-2019"));
    }

    #[test]
    fn test_extract_name() {
        let facts = extract_facts("My name is Marguerite");
        assert!(facts.iter().any(|f| f.relation == "name" && f.object == "Marguerite"));
    }

    #[test]
    fn test_extract_age() {
        let facts = extract_facts("I'm 82 years old");
        assert!(facts.iter().any(|f| f.relation == "age" && f.object == "82"));
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = PersonalMemory::new();
        let facts = extract_facts("My name is Jean");
        for f in &facts { store_fact(&mut mem, f); }
        assert_eq!(get_user_name(&mem), Some("Jean".into()));
    }
}
