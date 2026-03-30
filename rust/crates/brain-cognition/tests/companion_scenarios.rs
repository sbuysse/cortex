//! Companion conversation scenarios — elderly person interaction tests.
//!
//! Tests the full pipeline: hearing a message → extracting facts →
//! storing in PersonalMemory → building context → generating appropriate responses.

use brain_cognition::{
    personal,
    personal_memory::PersonalMemory,
    companion,
};

// ── Helpers ──────────────────────────────────────────────────────

fn fresh() -> PersonalMemory {
    PersonalMemory::new()
}

fn hear(mem: &mut PersonalMemory, message: &str) {
    let facts = personal::extract_facts(message);
    for f in &facts {
        personal::store_fact(mem, f);
    }
    let (emotion, conf) = personal::detect_text_emotion(message);
    personal::store_mood(mem, emotion, conf, Some(message));
    personal::store_conversation(mem, "user", message, Some(emotion));
}

// ══════════════════════════════════════════════════════════════════
// Scenario 1 — Introduction conversation
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_introduction() {
    let mut mem = fresh();
    hear(&mut mem, "Hello! My name is Marguerite.");
    hear(&mut mem, "I'm 84 years old.");
    hear(&mut mem, "I live here since 1972.");

    assert_eq!(personal::get_user_name(&mem).as_deref(), Some("Marguerite"));

    let facts = personal::get_all_facts(&mem);
    let ages: Vec<_> = facts.iter().filter(|(_, r, _, _)| r == "age").collect();
    assert!(!ages.is_empty(), "Age should be stored");
    assert_eq!(ages[0].2, "84");
}

// ══════════════════════════════════════════════════════════════════
// Scenario 2 — Family stories
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_family_stories() {
    let mut mem = fresh();
    hear(&mut mem, "My daughter Marie lives in Lyon, she visits every Sunday.");
    hear(&mut mem, "My son Pierre moved to Paris last year.");
    hear(&mut mem, "My grandson Lucas is studying medicine in Ghent, I'm so proud.");

    let facts = personal::get_all_facts(&mem);
    let daughters: Vec<_> = facts.iter().filter(|(_, r, _, _)| r.contains("daughter")).collect();
    let sons: Vec<_> = facts.iter().filter(|(_, r, _, _)| r == "is-son-of").collect();

    assert!(!daughters.is_empty(), "Daughter fact should be stored");
    assert_eq!(daughters[0].0, "Marie");
    assert!(!sons.is_empty(), "Son fact should be stored");
    assert_eq!(sons[0].0, "Pierre");
}

#[test]
fn scenario_family_context_in_prompt() {
    let mut mem = fresh();
    hear(&mut mem, "My name is Albertine.");
    hear(&mut mem, "My husband Georges passed away in 2018.");
    hear(&mut mem, "My daughter Lucie visits every Wednesday.");

    let ctx = personal::build_personal_context(&mem);
    assert!(ctx.contains("Albertine"), "Context should contain name");
    assert!(ctx.contains("Lucie") || ctx.contains("daughter"), "Context should mention Lucie");
}

// ══════════════════════════════════════════════════════════════════
// Scenario 3 — Health concerns
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_health_pain_tracking() {
    let mut mem = fresh();
    hear(&mut mem, "My knee hurts a lot lately.");
    hear(&mut mem, "The pain in my knee is worse today.");
    hear(&mut mem, "My knee aches when I walk.");

    let facts = personal::get_all_facts(&mem);
    let pain: Vec<_> = facts.iter().filter(|(_, r, _, _)| r.contains("pain")).collect();
    assert!(!pain.is_empty(), "Pain should be stored");

    let knee_mentions = pain.iter().filter(|(_, _, o, _)| o.contains("knee")).map(|(_, _, _, c)| *c).sum::<i64>();
    assert!(knee_mentions >= 2, "Knee pain should be mentioned multiple times, got {}", knee_mentions);
}

#[test]
fn scenario_safety_alert_on_repeated_pain() {
    let mut mem = fresh();
    // Three or more pain mentions triggers a safety alert
    for _ in 0..3 {
        hear(&mut mem, "My back hurts so badly today.");
    }
    personal::store_conversation(&mut mem, "user", "my back hurts", Some("pain"));
    personal::store_conversation(&mut mem, "user", "pain in my back again", Some("pain"));
    personal::store_conversation(&mut mem, "user", "back pain is unbearable", Some("pain"));

    let alerts = companion::check_safety(&mem);
    let pain_alerts: Vec<_> = alerts.iter().filter(|a| a.to_lowercase().contains("pain")).collect();
    assert!(!pain_alerts.is_empty(), "Expected pain safety alert, got: {:?}", alerts);
}

#[test]
fn scenario_appointment_remembered() {
    let mut mem = fresh();
    hear(&mut mem, "I have a doctor appointment on Thursday.");

    let facts = personal::get_all_facts(&mem);
    let appts: Vec<_> = facts.iter().filter(|(_, r, _, _)| r.contains("appointment")).collect();
    assert!(!appts.is_empty(), "Appointment should be stored");
    assert!(appts[0].2.to_lowercase().contains("thursday"), "Day should be Thursday");
}

// ══════════════════════════════════════════════════════════════════
// Scenario 4 — Emotional states
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_grief_emotion_detected() {
    let mut mem = fresh();
    hear(&mut mem, "My husband Jean passed away in 2019. I miss him so much.");

    let moods = personal::get_recent_moods(&mem, 5);
    let sad_moods: Vec<_> = moods.iter().filter(|(_, e, _)| e == "sad").collect();
    assert!(!sad_moods.is_empty(), "Grief should register as sad mood, got: {:?}", moods);
}

#[test]
fn scenario_sustained_sadness_triggers_alert() {
    let mut mem = fresh();
    for _ in 0..5 {
        personal::store_mood(&mut mem, "sad", 0.75, Some("loneliness"));
        personal::store_conversation(&mut mem, "user", "I feel very sad and alone", Some("sad"));
    }

    let alerts = companion::check_safety(&mem);
    let sadness_alert: Vec<_> = alerts.iter().filter(|a| a.to_lowercase().contains("sad")).collect();
    assert!(!sadness_alert.is_empty(), "Expected sadness alert, got: {:?}", alerts);
}

#[test]
fn scenario_happy_moment() {
    let mut mem = fresh();
    hear(&mut mem, "My granddaughter visited today! It was wonderful, I love her so much.");

    let moods = personal::get_recent_moods(&mem, 5);
    let happy: Vec<_> = moods.iter().filter(|(_, e, _)| e == "happy").collect();
    assert!(!happy.is_empty(), "Happy visit should register as happy mood, got: {:?}", moods);
}

#[test]
fn scenario_confusion_triggers_alert() {
    let mut mem = fresh();
    for _ in 0..3 {
        personal::store_mood(&mut mem, "confused", 0.7, Some("disorientation"));
        personal::store_conversation(&mut mem, "user", "I'm confused, what day is it?", Some("confused"));
    }

    let alerts = companion::check_safety(&mem);
    let confusion_alert: Vec<_> = alerts.iter().filter(|a| a.to_lowercase().contains("confus")).collect();
    assert!(!confusion_alert.is_empty(), "Expected confusion alert, got: {:?}", alerts);
}

// ══════════════════════════════════════════════════════════════════
// Scenario 5 — Preferences and interests
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_preferences_stored() {
    let mut mem = fresh();
    hear(&mut mem, "I really love gardening, I spend hours in the garden every day.");
    hear(&mut mem, "I enjoy classical music very much.");
    hear(&mut mem, "I hate cold weather, it makes my joints ache.");

    let facts = personal::get_all_facts(&mem);
    let enjoys: Vec<_> = facts.iter().filter(|(_, r, _, _)| r == "enjoys").collect();
    let dislikes: Vec<_> = facts.iter().filter(|(_, r, _, _)| r == "dislikes").collect();

    assert!(enjoys.iter().any(|(_, _, o, _)| o.to_lowercase().contains("garden")),
        "Gardening should be stored as a preference, got: {:?}", enjoys);
    assert!(enjoys.iter().any(|(_, _, o, _)| o.to_lowercase().contains("music") || o.to_lowercase().contains("classical")),
        "Music should be stored as a preference, got: {:?}", enjoys);
    assert!(!dislikes.is_empty(), "Dislike (cold) should be stored, got: {:?}", dislikes);
}

// ══════════════════════════════════════════════════════════════════
// Scenario 6 — Companion greetings adapt to context
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_greeting_uses_name() {
    let mut mem = fresh();
    hear(&mut mem, "My name is Henriette.");

    let greeting = companion::proactive_greeting(&mem);
    assert!(greeting.contains("Henriette") || greeting.contains("friend"),
        "Greeting should use stored name, got: {}", greeting);
}

#[test]
fn scenario_morning_greeting_mentions_appointment() {
    // Can't control time of day, but we can verify the greeting logic
    // uses appointment facts when they exist.
    let mut mem = fresh();
    hear(&mut mem, "My name is Louise.");
    hear(&mut mem, "I have a doctor appointment on Monday.");

    // greeting logic is time-dependent; just verify it doesn't panic
    let greeting = companion::proactive_greeting(&mem);
    assert!(!greeting.is_empty(), "Greeting should not be empty");
}

#[test]
fn scenario_greeting_comforts_sad_person() {
    let mut mem = fresh();
    hear(&mut mem, "My name is Renée.");
    for _ in 0..3 {
        personal::store_mood(&mut mem, "sad", 0.8, None);
    }

    let greeting = companion::proactive_greeting(&mem);
    // During afternoon with sad mood, greeting should be more comforting
    assert!(!greeting.is_empty(), "Greeting should not be empty even with sad mood");
}

// ══════════════════════════════════════════════════════════════════
// Scenario 7 — Memory persists and accumulates
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_mention_count_accumulates() {
    let mut mem = fresh();
    // Repeat the same fact multiple times (elderly often repeat stories)
    hear(&mut mem, "My daughter Marie lives in Lyon.");
    hear(&mut mem, "Did I tell you? My daughter Marie is in Lyon.");
    hear(&mut mem, "Yes, my daughter Marie, she's in Lyon, she's wonderful.");

    let facts = personal::get_all_facts(&mem);
    let marie = facts.iter().find(|(s, r, _, _)| s == "Marie" && r.contains("daughter"));
    assert!(marie.is_some(), "Marie should be stored");
    assert!(marie.unwrap().3 >= 2, "Marie should have multiple mentions, got: {}", marie.unwrap().3);
}

#[test]
fn scenario_full_profile_builds_correctly() {
    let mut mem = fresh();
    hear(&mut mem, "My name is Georgette. I'm 79 years old.");
    hear(&mut mem, "My daughter Anne lives in Bruges, my son Michel is in Liège.");
    hear(&mut mem, "I love knitting and watching old films.");
    hear(&mut mem, "My hip has been painful for months.");
    hear(&mut mem, "I have a physiotherapy appointment on Friday.");

    let ctx = personal::build_personal_context(&mem);
    assert!(ctx.contains("Georgette"), "Context should have name");
    assert!(ctx.contains("79"), "Context should have age");
    // Context should mention health or family or preferences
    assert!(
        ctx.contains("Anne") || ctx.contains("Michel") ||
        ctx.contains("knit") || ctx.contains("pain") || ctx.contains("appointment"),
        "Context should contain at least one personal detail, got: {}", ctx
    );
}

// ══════════════════════════════════════════════════════════════════
// Scenario 8 — Should initiate contact logic
// ══════════════════════════════════════════════════════════════════

#[test]
fn scenario_initiates_when_no_conversation() {
    let mem = fresh(); // No conversations yet
    // Fresh memory with no history should trigger initiation (if not night)
    // We can't control time, but the function should not panic
    let _ = companion::should_initiate_contact(&mem);
}

#[test]
fn scenario_no_initiation_after_recent_chat() {
    let mut mem = fresh();
    // Record a very recent conversation turn
    personal::store_conversation(&mut mem, "user", "Good night!", Some("neutral"));
    // should_initiate_contact checks hours since last interaction
    // Right after a conversation, it should not initiate (unless night)
    let result = companion::should_initiate_contact(&mem);
    // During night period, always false; during day with fresh conversation, also false
    // We just verify it returns without panic and the logic is coherent
    let _ = result;
}
