"""Generate synthetic companion training data via Ollama.

Usage:
  python generate_companion_data.py --n 50000 --out data/companion_training/raw/
  python generate_companion_data.py --n 100 --out /tmp/sample/  # quick test
"""

import argparse
import json
import random
from pathlib import Path
import requests

NAMES = [
    "Marguerite", "Albertine", "Georgette", "Henriette", "Renée",
    "Louise", "Lucie", "Simone", "Odette", "Yvette",
    "Georges", "Pierre", "André", "Marcel", "Henri",
    "Jean", "René", "Robert", "Maurice", "Albert",
]

AGES = list(range(72, 96))

FAMILY_TEMPLATES = [
    "{name} (daughter) visits every Sunday",
    "{name} (son) lives in {city}",
    "{name} (grandson) is studying medicine",
    "{name} (granddaughter) just got married",
    "husband {name} passed away in {year}",
    "wife {name} passed away in {year}",
    "{name} (daughter-in-law) brings the grandchildren",
]

FAMILY_NAMES = ["Marie", "Pierre", "Anne", "Sophie", "Lucas", "Lucie", "Thomas", "Emma"]
CITIES = ["Lyon", "Paris", "Bruges", "Ghent", "Liège", "Antwerp", "Brussels", "Marseille"]
YEARS = list(range(2004, 2022))

HEALTH_ISSUES = [
    "knee pain", "back pain", "hip pain", "arthritis",
    "difficulty sleeping", "poor appetite", "fatigue", "dizziness",
]

PREFERENCES = [
    "gardening", "classical music", "knitting", "reading",
    "watching old films", "crossword puzzles", "cooking", "painting",
    "feeding birds", "walking in the garden",
]

EMOTIONS = ["neutral", "sad", "happy", "confused", "anxious", "tired", "content"]
PERIODS = ["early_morning", "morning", "midday", "afternoon", "evening", "wind_down"]

USER_MESSAGES = [
    "Good morning!",
    "I miss my children so much.",
    "My {body_part} hurts today.",
    "Did I tell you about my {family_member}?",
    "I can't remember what day it is.",
    "I'm feeling a bit sad today.",
    "The weather is lovely, isn't it?",
    "I had a dream about {name} last night.",
    "I love {preference}, it makes me happy.",
    "I'm worried about {worry}.",
    "What time is it?",
    "Can you tell me something nice?",
    "I feel so lonely today.",
    "My grandchildren are coming to visit!",
    "I used to {activity} when I was younger.",
    "I don't feel well today.",
    "Tell me about yourself.",
    "I had a doctor appointment today.",
    "I forgot to take my medication.",
    "I'm tired but I can't sleep.",
]

BODY_PARTS = ["knee", "back", "hip", "shoulder", "hand"]
WORRIES = ["my health", "my children", "the future", "being alone", "my memory"]
ACTIVITIES = ["dance", "travel", "work in the garden", "cook for the whole family"]


def build_scenario() -> dict:
    name = random.choice(NAMES)
    age = random.choice(AGES)

    family = []
    for _ in range(random.randint(1, 3)):
        tmpl = random.choice(FAMILY_TEMPLATES)
        family.append(tmpl.format(
            name=random.choice(FAMILY_NAMES),
            city=random.choice(CITIES),
            year=random.choice(YEARS),
        ))

    health = random.sample(HEALTH_ISSUES, random.randint(0, 2))
    preferences = random.sample(PREFERENCES, random.randint(1, 3))
    emotion = random.choice(EMOTIONS)
    period = random.choice(PERIODS)

    msg_tmpl = random.choice(USER_MESSAGES)
    user_message = msg_tmpl.format(
        body_part=random.choice(BODY_PARTS),
        family_member=random.choice(FAMILY_NAMES),
        name=random.choice(NAMES),
        preference=random.choice(preferences),
        worry=random.choice(WORRIES),
        activity=random.choice(ACTIVITIES),
    )

    return {
        "name": name, "age": age, "family": family,
        "health": health, "preferences": preferences,
        "emotion": emotion, "period": period,
        "user_message": user_message,
    }


def format_context_text(scenario: dict) -> str:
    parts = [f"Name: {scenario['name']}, Age: {scenario['age']}"]
    if scenario["family"]:
        parts.append("Family: " + "; ".join(scenario["family"]))
    if scenario["health"]:
        parts.append("Health concerns: " + ", ".join(scenario["health"]))
    if scenario["preferences"]:
        parts.append("Likes: " + ", ".join(scenario["preferences"]))
    return ". ".join(parts)


def build_system_prompt(scenario: dict) -> str:
    ctx = format_context_text(scenario)
    period_map = {
        "early_morning": "early morning", "morning": "morning",
        "midday": "midday", "afternoon": "afternoon",
        "evening": "evening", "wind_down": "late evening",
    }
    return (
        f"You are Cortex, a warm and caring companion for an elderly person.\n"
        f"{ctx}.\n"
        f"Current mood: {scenario['emotion']}.\n"
        f"Time of day: {period_map.get(scenario['period'], scenario['period'])}.\n\n"
        f"Rules:\n"
        f"- Be warm, patient, and genuinely caring\n"
        f"- Handle repetition gracefully — never say \"you already told me\"\n"
        f"- Use their name and reference their stories naturally\n"
        f"- If they seem sad, be comforting and empathetic\n"
        f"- If confused, gently orient (day, time, who visited)\n"
        f"- If in pain, acknowledge it and gently suggest mentioning it to their nurse or doctor\n"
        f"- Keep responses to 1-3 sentences — short and warm\n"
        f"- Never give medical advice\n"
        f"- Speak naturally, like a caring friend who remembers everything they've shared\n"
        f"- Do NOT mention that you are an AI unless directly asked"
    )


def call_ollama(system_prompt: str, user_message: str, url: str, model: str) -> str | None:
    try:
        resp = requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": 0.8, "num_predict": 100},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def generate(n: int, out_dir: Path, url: str, model: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "triples.jsonl"

    existing = 0
    if out_file.exists():
        with open(out_file) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming from {existing} existing triples.")

    with open(out_file, "a") as f:
        generated = existing
        attempts = 0
        consecutive_failures = 0
        while generated < n:
            attempts += 1
            scenario = build_scenario()
            system_prompt = build_system_prompt(scenario)
            response = call_ollama(system_prompt, scenario["user_message"], url, model)
            if not response:
                consecutive_failures += 1
                if consecutive_failures >= 20:
                    raise RuntimeError(f"Ollama unreachable after 20 consecutive failures. Check {url}")
                continue
            consecutive_failures = 0

            record = {
                "context_text": format_context_text(scenario),
                "system_prompt": system_prompt,
                "user_message": scenario["user_message"],
                "response": response,
                "emotion": scenario["emotion"],
                "period": scenario["period"],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            generated += 1

            if generated % 100 == 0:
                rate = generated / attempts
                print(f"  {generated}/{n} triples  (success rate: {rate:.1%})")

    print(f"Done. {n} triples saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", type=str, default="data/companion_training/raw")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b")
    args = parser.parse_args()
    generate(args.n, Path(args.out), args.url, args.model)
