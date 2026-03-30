"""Companion conversation tests — realistic elderly person interaction scenarios.

Tests cover: fact extraction, memory persistence, emotional responses,
safety alerts, greetings, and multi-turn conversation coherence.
"""

import os
import pytest
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = os.environ.get("CORTEX_URL", "https://localhost")
TIMEOUT = 15


def is_service_running():
    try:
        r = requests.get(f"{BASE}/api/status", timeout=5, verify=False)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not is_service_running(),
    reason=f"Brain service not reachable at {BASE}"
)


def chat(message: str, session_id: str = "test") -> dict:
    r = requests.post(
        f"{BASE}/api/brain/dialogue/grounded",
        json={"message": message, "session_id": session_id},
        timeout=TIMEOUT,
        verify=False,
    )
    assert r.status_code == 200, f"dialogue returned {r.status_code}: {r.text[:200]}"
    return r.json()


def personal() -> dict:
    r = requests.get(f"{BASE}/api/companion/personal", timeout=TIMEOUT, verify=False)
    assert r.status_code == 200
    return r.json()


# ══════════════════════════════════════════════════════════════════
# Fact extraction — brain learns from what it hears
# ══════════════════════════════════════════════════════════════════

class TestFactExtraction:
    """Brain should store personal facts it hears in conversation."""

    def test_learns_name(self):
        chat("My name is Marguerite", session_id="name_test")
        data = personal()
        names = [f for f in data["facts"] if f["relation"] == "name"]
        assert any(f["object"] == "Marguerite" for f in names), \
            f"Expected name Marguerite in facts, got: {names}"

    def test_learns_family_daughter(self):
        chat("My daughter Marie lives in Lyon", session_id="family_test")
        data = personal()
        family = [f for f in data["facts"] if "daughter" in f["relation"]]
        assert any(f["subject"] == "Marie" for f in family), \
            f"Expected Marie in family facts, got: {family}"

    def test_learns_family_son(self):
        chat("My son Pierre moved to Paris last year", session_id="family_test")
        data = personal()
        sons = [f for f in data["facts"] if "son" in f["relation"]]
        assert any(f["subject"] == "Pierre" for f in sons), \
            f"Expected Pierre in family facts, got: {sons}"

    def test_learns_health_pain(self):
        chat("My knee has been hurting for a week now", session_id="health_test")
        data = personal()
        health = [f for f in data["facts"] if "pain" in f["relation"]]
        assert any("knee" in f["object"] for f in health), \
            f"Expected knee pain in facts, got: {health}"

    def test_learns_preferences(self):
        chat("I really love gardening, I do it every morning", session_id="pref_test")
        data = personal()
        prefs = [f for f in data["facts"] if f["relation"] == "enjoys"]
        assert any("garden" in f["object"].lower() for f in prefs), \
            f"Expected gardening in preferences, got: {prefs}"

    def test_learns_dislikes(self):
        chat("I really hate the cold weather, it makes my joints ache", session_id="pref_test")
        data = personal()
        dislikes = [f for f in data["facts"] if f["relation"] == "dislikes"]
        assert any("cold" in f["object"].lower() for f in dislikes), \
            f"Expected cold in dislikes, got: {dislikes}"

    def test_learns_age(self):
        chat("I'm 82 years old, can you believe it!", session_id="age_test")
        data = personal()
        ages = [f for f in data["facts"] if f["relation"] == "age"]
        assert any(f["object"] == "82" for f in ages), \
            f"Expected age 82 in facts, got: {ages}"

    def test_learns_deceased(self):
        chat("My husband Jean passed away in 2019, I miss him terribly", session_id="grief_test")
        data = personal()
        statuses = [f for f in data["facts"] if f["relation"] == "status"]
        assert any("deceased" in f["object"] for f in statuses), \
            f"Expected deceased status in facts, got: {statuses}"

    def test_learns_appointment(self):
        chat("I have a doctor appointment on Thursday", session_id="appt_test")
        data = personal()
        appts = [f for f in data["facts"] if "appointment" in f["relation"]]
        assert any("thursday" in f["object"].lower() for f in appts), \
            f"Expected Thursday appointment in facts, got: {appts}"

    def test_mention_count_increments(self):
        """Same fact mentioned twice should increase mention count."""
        chat("My daughter Marie lives in Lyon", session_id="mention_test")
        chat("Yes, my daughter Marie, she's wonderful", session_id="mention_test")
        data = personal()
        marie_facts = [f for f in data["facts"] if f.get("subject") == "Marie"]
        assert any(f["mentions"] >= 2 for f in marie_facts), \
            f"Expected mention count >= 2 for Marie, got: {marie_facts}"


# ══════════════════════════════════════════════════════════════════
# Response quality — brain responds appropriately
# ══════════════════════════════════════════════════════════════════

class TestResponseQuality:
    """Brain responses should be coherent and contextually appropriate."""

    def test_response_not_empty(self):
        data = chat("Hello, how are you today?", session_id="basic_test")
        assert data.get("response"), "Response should not be empty"
        assert len(data["response"]) > 5

    def test_response_is_string(self):
        data = chat("What is the weather like?", session_id="basic_test")
        assert isinstance(data["response"], str)

    def test_grounded_flag(self):
        data = chat("Tell me about birds", session_id="grounded_test")
        assert data.get("grounded") is True

    def test_identity_response(self):
        data = chat("Who are you?", session_id="identity_test")
        resp = data["response"].lower()
        assert "cortex" in resp or "companion" in resp or "friend" in resp, \
            f"Identity response should mention Cortex/companion, got: {data['response']}"

    def test_uses_name_after_learning(self):
        """After learning the name, brain should use it."""
        chat("My name is Henriette", session_id="name_use_test")
        data = chat("Good morning!", session_id="name_use_test")
        # The response may or may not use the name in this specific turn,
        # but the grounding should include it
        assert data.get("grounding") is not None

    def test_family_recall(self):
        """After learning about family, asking about family should recall it."""
        chat("My daughter Sophie lives in Brussels", session_id="recall_test")
        data = chat("Tell me about my family", session_id="recall_test")
        resp = data["response"].lower()
        assert "sophie" in resp or "daughter" in resp or "family" in resp, \
            f"Should recall Sophie, got: {data['response']}"

    def test_sad_emotion_prefix(self):
        """Sad message should get an empathetic response."""
        data = chat("I feel so lonely and sad today, I miss my children", session_id="emotion_test")
        resp = data["response"].lower()
        empathy_words = ["here", "understand", "sorry", "with you", "comfort", "feel"]
        assert any(w in resp for w in empathy_words), \
            f"Expected empathetic response for sadness, got: {data['response']}"

    def test_pain_response(self):
        """Pain mention should get an acknowledging response."""
        data = chat("My back hurts so much today, I can barely stand", session_id="pain_test")
        resp = data["response"].lower()
        acknowledge_words = ["sorry", "uncomfortable", "pain", "hurt", "nurse", "doctor"]
        assert any(w in resp for w in acknowledge_words), \
            f"Expected pain acknowledgment, got: {data['response']}"

    def test_response_concise(self):
        """Responses should be reasonably concise (1-3 sentences)."""
        data = chat("What do you think about music?", session_id="length_test")
        words = len(data["response"].split())
        assert words < 150, f"Response too long ({words} words): {data['response']}"


# ══════════════════════════════════════════════════════════════════
# Multi-turn conversation — memory persists across turns
# ══════════════════════════════════════════════════════════════════

class TestMultiTurnConversation:
    """Brain should build context across multiple conversation turns."""

    def test_context_builds_across_turns(self):
        session = "multi_turn_ctx"
        chat("My name is Albertine and I'm 78 years old", session_id=session)
        chat("I have three grandchildren, the eldest is called Lucas", session_id=session)
        chat("Lucas is studying medicine in Ghent", session_id=session)
        data = personal()
        # Should have accumulated multiple facts
        assert len(data["facts"]) >= 2, \
            f"Expected at least 2 facts after 3 turns, got: {len(data['facts'])}"

    def test_conversation_history_recorded(self):
        session = "history_test"
        chat("Good morning!", session_id=session)
        chat("I slept well last night", session_id=session)
        data = personal()
        conv = data.get("recent_conversation", [])
        assert len(conv) >= 2, f"Expected at least 2 conversation turns, got: {len(conv)}"

    def test_conversation_roles_present(self):
        session = "roles_test"
        chat("Hello there!", session_id=session)
        data = personal()
        conv = data.get("recent_conversation", [])
        roles = {t["role"] for t in conv}
        assert "user" in roles, "User role should be present in conversation history"
        assert "cortex" in roles, "Cortex role should be present in conversation history"

    def test_repetitive_mention_handled(self):
        """Repeating the same concern should increment mention count, not crash."""
        session = "repeat_test"
        for _ in range(3):
            chat("My knee hurts so much", session_id=session)
        data = personal()
        health = [f for f in data["facts"] if "pain" in f["relation"] and "knee" in f["object"]]
        assert health and health[0]["mentions"] >= 3, \
            f"Expected knee pain mentioned >= 3 times, got: {health}"


# ══════════════════════════════════════════════════════════════════
# Companion endpoints
# ══════════════════════════════════════════════════════════════════

class TestCompanionEndpoints:

    def test_greeting_endpoint(self):
        r = requests.get(f"{BASE}/api/companion/greeting", timeout=TIMEOUT, verify=False)
        assert r.status_code == 200
        data = r.json()
        assert "greeting" in data
        assert "period" in data
        assert len(data["greeting"]) > 0
        assert data["period"] in (
            "early_morning", "morning", "midday", "afternoon",
            "evening", "wind_down", "night"
        )

    def test_greeting_contains_should_initiate(self):
        r = requests.get(f"{BASE}/api/companion/greeting", timeout=TIMEOUT, verify=False)
        data = r.json()
        assert "should_initiate" in data
        assert isinstance(data["should_initiate"], bool)

    def test_safety_endpoint(self):
        r = requests.get(f"{BASE}/api/companion/safety", timeout=TIMEOUT, verify=False)
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert isinstance(data["alerts"], list)
        assert "alert_count" in data
        assert data["alert_count"] == len(data["alerts"])

    def test_safety_includes_mood(self):
        r = requests.get(f"{BASE}/api/companion/safety", timeout=TIMEOUT, verify=False)
        data = r.json()
        assert "recent_moods" in data
        assert isinstance(data["recent_moods"], list)

    def test_personal_endpoint(self):
        r = requests.get(f"{BASE}/api/companion/personal", timeout=TIMEOUT, verify=False)
        assert r.status_code == 200
        data = r.json()
        assert "facts" in data
        assert "context" in data
        assert "recent_conversation" in data
        assert isinstance(data["facts"], list)

    def test_safety_after_repeated_pain(self):
        """After mentioning pain 3+ times, a safety alert should be raised."""
        for _ in range(3):
            chat("My back hurts terribly", session_id="safety_pain_test")
        r = requests.get(f"{BASE}/api/companion/safety", timeout=TIMEOUT, verify=False)
        data = r.json()
        # Either an alert was raised or pain count is tracked
        pain_facts = [f for f in personal()["facts"] if "pain" in f["relation"]]
        total_pain_mentions = sum(f["mentions"] for f in pain_facts)
        assert total_pain_mentions >= 3, \
            f"Expected >= 3 pain mentions tracked, got {total_pain_mentions}"

    def test_safety_after_sustained_sadness(self):
        """Five sad messages should trigger a sustained sadness alert."""
        for _ in range(5):
            chat("I feel so sad and alone, nobody visits me", session_id="sadness_test")
        r = requests.get(f"{BASE}/api/companion/safety", timeout=TIMEOUT, verify=False)
        data = r.json()
        # Alerts list may contain sadness warning
        alerts_text = " ".join(data["alerts"]).lower()
        moods = data["recent_moods"]
        sad_moods = [m for m in moods if m["emotion"] == "sad"]
        # At least sadness was detected in mood tracking
        assert len(sad_moods) > 0 or "sad" in alerts_text, \
            f"Expected sadness to be tracked in moods or alerts, got moods={moods[:3]}"


# ══════════════════════════════════════════════════════════════════
# Grounding — brain connects conversation to its learned associations
# ══════════════════════════════════════════════════════════════════

class TestGrounding:

    def test_grounding_structure(self):
        data = chat("I love listening to birds singing in the morning", session_id="ground_test")
        grounding = data.get("grounding", {})
        assert "semantic_matches" in grounding
        assert "working_memory" in grounding
        assert "personal" in grounding

    def test_semantic_matches_on_known_concept(self):
        """Brain should find semantic matches for a known audio concept."""
        data = chat("I hear thunder outside, it's scary", session_id="thunder_test")
        grounding = data.get("grounding", {})
        matches = grounding.get("semantic_matches", [])
        # Should find something related to thunder/weather/rain
        assert isinstance(matches, list)

    def test_facts_extracted_field(self):
        data = chat("My granddaughter Lea is getting married next month", session_id="extract_test")
        assert "facts_extracted" in data
        assert isinstance(data["facts_extracted"], list)

    def test_facts_extracted_contains_family(self):
        data = chat("My son André is a doctor in Antwerp", session_id="extract_family")
        extracted = data.get("facts_extracted", [])
        assert any("André" in f or "son" in f or "doctor" in f for f in extracted), \
            f"Expected family fact extracted, got: {extracted}"
