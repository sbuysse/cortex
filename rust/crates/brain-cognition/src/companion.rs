//! Companion module — daily rhythm, proactive prompts, care protocol.
//!
//! Builds context-aware system prompts from brain memory,
//! manages daily schedule, and handles care signals.
//! All data comes from PersonalMemory — no SQL involved.

use crate::personal;
use crate::personal_memory::PersonalMemory;

/// One message in a conversation (role: "user" | "assistant" | "system").
#[derive(serde::Serialize)]
struct OllamaMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Serialize)]
struct OllamaChatReq<'a> {
    model: &'a str,
    messages: Vec<OllamaMsg<'a>>,
    stream: bool,
    options: OllamaOpts,
}

#[derive(serde::Serialize)]
struct OllamaOpts {
    temperature: f32,
    num_predict: u32,
}

/// Build the system prompt — public so routes can extract it before any await.
pub fn build_companion_prompt(mem: &PersonalMemory, current_emotion: &str, brain_context: &str) -> String {
    build_system_prompt_with_context(mem, current_emotion, brain_context)
}

/// Call Ollama with pre-extracted owned data (no Mutex held across await).
pub async fn llm_reply_owned(
    system_prompt: &str,
    history: &[(String, String)],   // (role, message) pairs — oldest first
    user_message: &str,
    ollama_url: &str,
    model: &str,
) -> Option<String> {
    // Build message list
    let mut messages: Vec<OllamaMsg> = vec![OllamaMsg { role: "system", content: system_prompt }];

    // Recent history excluding the current user message (last entry)
    let ctx = &history[..history.len().saturating_sub(1)];
    for (role, msg) in ctx {
        let ollama_role = if role == "cortex" { "assistant" } else { "user" };
        messages.push(OllamaMsg { role: ollama_role, content: msg });
    }
    messages.push(OllamaMsg { role: "user", content: user_message });

    let req = OllamaChatReq {
        model,
        messages,
        stream: false,
        options: OllamaOpts { temperature: 0.7, num_predict: 300 },
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build().ok()?;

    let resp = client
        .post(format!("{}/api/chat", ollama_url))
        .json(&req)
        .send().await.ok()?;

    let body: serde_json::Value = resp.json().await.ok()?;
    body["message"]["content"].as_str().map(|s| s.trim().to_string())
}

/// Generate a response using the native HOPE decoder.
pub fn native_reply(
    decoder: &brain_inference::CompanionDecoder,
    context_text: &str,
    user_message: &str,
    max_tokens: usize,
) -> String {
    decoder.generate(context_text, user_message, max_tokens)
}

/// Build system prompt enriched with brain perceptual grounding.
fn build_system_prompt_with_context(mem: &PersonalMemory, current_emotion: &str, brain_ctx: &str) -> String {
    let personal = personal::build_personal_context(mem);
    let mood_trend = personal::get_mood_trend(mem, 4.0).unwrap_or_else(|| "neutral".into());
    let period = get_period();

    // Brain knowledge is FIRST — the LLM must use it
    let brain_knowledge = if brain_ctx.is_empty() {
        String::new()
    } else {
        format!("\nIMPORTANT — {brain_ctx}\n")
    };

    format!(
        "You are Cortex, a knowledgeable and caring companion.\n\
         {brain_knowledge}\
         {personal}.\n\
         Current mood: {current_emotion}. Time: {period}.\n\n\
         Rules:\n\
         - If you have learned knowledge above, USE IT to answer accurately\n\
         - Be warm, patient, and caring\n\
         - Keep responses to 2-4 sentences\n\
         - If you don't have learned knowledge on the topic, say so honestly\n\
         - Do NOT mention that you are an AI unless directly asked"
    )
}

/// Time-of-day period for daily rhythm.
pub fn get_period() -> &'static str {
    let hour = chrono_hour();
    match hour {
        6..=7 => "early_morning",
        8..=11 => "morning",
        12..=13 => "midday",
        14..=16 => "afternoon",
        17..=19 => "evening",
        20..=21 => "wind_down",
        _ => "night",
    }
}

/// Get a proactive greeting based on time of day and personal context.
pub fn proactive_greeting(mem: &PersonalMemory) -> String {
    let name = personal::get_user_name(mem).unwrap_or_else(|| "friend".into());
    let period = get_period();
    let mood = personal::get_mood_trend(mem, 4.0);

    match period {
        "early_morning" => format!("Good morning, {}! Did you sleep well?", name),
        "morning" => {
            let facts = personal::get_all_facts(mem);
            let appt = facts.iter().find(|(_, r, _, _)| r.contains("appointment"));
            if let Some((_, _, day, _)) = appt {
                format!("Good morning, {}! I remember you have an appointment on {}.", name, day)
            } else {
                format!("Good morning, {}! What would you like to talk about today?", name)
            }
        }
        "midday" => format!("It's lunchtime, {}. Have you eaten?", name),
        "afternoon" => {
            if mood.as_deref() == Some("sad") {
                format!("{}, I noticed you've been a bit down. Would you like to chat about something nice?", name)
            } else {
                format!("How's your afternoon going, {}?", name)
            }
        }
        "evening" => format!("Good evening, {}. What was the best part of your day?", name),
        "wind_down" => format!("It's getting late, {}. Time to relax. Is there anything on your mind?", name),
        "night" => format!("Sleep well, {}. I'll be here in the morning.", name),
        _ => format!("Hello, {}.", name),
    }
}

/// Build the system prompt for LLM conversation.
pub fn build_system_prompt(mem: &PersonalMemory, current_emotion: &str) -> String {
    let personal = personal::build_personal_context(mem);
    let mood_trend = personal::get_mood_trend(mem, 4.0).unwrap_or_else(|| "neutral".into());
    let recent = personal::get_recent_conversation(mem, 6);
    let period = get_period();

    let conv_history = if recent.is_empty() {
        "No recent conversation.".into()
    } else {
        recent.iter().map(|(role, msg)| format!("{}: {}", role, msg)).collect::<Vec<_>>().join("\n")
    };

    format!(
        "You are Cortex, a warm companion for an elderly person. \
         {personal}. \
         Current mood: {current_emotion} (trend over 4h: {mood_trend}). \
         Time of day: {period}. \
         \n\nRecent conversation:\n{conv_history}\n\n\
         Rules:\n\
         - Be warm, patient, and genuine\n\
         - Handle repetition gracefully — never say \"you already told me\"\n\
         - Use their name and reference their stories\n\
         - If they seem sad, be comforting and recall happy memories\n\
         - If confused, gently orient (day, time, who visited)\n\
         - If in pain, acknowledge and suggest calling the nurse\n\
         - Keep responses to 1-3 sentences\n\
         - Never give medical advice\n\
         - Speak naturally, like a caring friend"
    )
}

/// Check if it's time for a proactive check-in.
pub fn should_initiate_contact(mem: &PersonalMemory) -> bool {
    let period = get_period();
    if period == "night" { return false; }

    let Some(last_ts) = mem.last_conversation_ts() else { return true };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    let hours_since = (now - last_ts) / 3600.0;
    hours_since > 2.0
}

/// Generate safety alerts if concerning patterns are detected.
pub fn check_safety(mem: &PersonalMemory) -> Vec<String> {
    let mut alerts = Vec::new();

    // Check: no interaction for >6 hours during day
    let period = get_period();
    if let Some(last_ts) = mem.last_conversation_ts() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        let hours_since = (now - last_ts) / 3600.0;
        if hours_since > 6.0 && period != "night" {
            alerts.push(format!("No interaction for {:.0} hours during daytime", hours_since));
        }
    }

    // Check: pain mentioned multiple times
    let pain_facts = personal::get_facts_about(mem, "user");
    let pain_count: i64 = pain_facts.iter()
        .filter(|(_, r, _, _)| r.contains("pain"))
        .map(|(_, _, _, c)| *c)
        .sum();
    if pain_count >= 3 {
        alerts.push(format!("Pain mentioned {} times — consider nurse check", pain_count));
    }

    // Check: sustained sad mood
    let moods = personal::get_recent_moods(mem, 10);
    let sad_count = moods.iter().filter(|(_, e, _)| e == "sad").count();
    if sad_count >= 5 {
        alerts.push("Sustained sadness detected — emotional support needed".into());
    }

    // Check: confusion detected
    let confused_count = moods.iter().filter(|(_, e, _)| e == "confused").count();
    if confused_count >= 3 {
        alerts.push("Repeated confusion detected — assessment recommended".into());
    }

    alerts
}

fn chrono_hour() -> u32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    ((secs % 86400) / 3600) as u32
}
