import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from generate_companion_data import build_scenario, format_context_text

def test_build_scenario_has_required_keys():
    s = build_scenario()
    assert "name" in s
    assert "age" in s
    assert "family" in s
    assert "emotion" in s
    assert "period" in s

def test_format_context_text_not_empty():
    scenario = {
        "name": "Marguerite", "age": 84, "family": ["Marie (daughter)"],
        "health": ["knee pain"], "preferences": ["gardening"],
        "emotion": "sad", "period": "afternoon",
        "user_message": "I miss my daughter.",
    }
    ctx = format_context_text(scenario)
    assert "Marguerite" in ctx
    assert len(ctx) > 50
