#!/bin/bash
# Cortex Learning Benchmark — Proves the spiking brain learns from YouTube videos.
#
# The LLM fabricates answers for topics it doesn't know.
# Cortex watches a video, then answers with ACCURATE information from the video.
# The proof is accuracy, not "I don't know" vs "I know."
#
# Usage: CORTEX_URL=https://localhost:8443 OLLAMA_URL=http://localhost:11434 \
#        COMPANION_MODEL=qwen2.5:32b ./benchmark_learning.sh

set -euo pipefail

CORTEX="${CORTEX_URL:-https://localhost:8443}"
OLLAMA="${OLLAMA_URL:-http://localhost:11434}"
MODEL="${COMPANION_MODEL:-qwen2.5:32b}"
TICK_WAIT=40  # seconds to wait for brain tick between queries

ask_llm() {
    local q="$1"
    curl -s --max-time 15 "$OLLAMA/api/chat" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$q Answer in 2-3 sentences.\"}],\"stream\":false,\"options\":{\"num_predict\":200}}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message',{}).get('content','TIMEOUT'))" 2>/dev/null || echo "TIMEOUT"
}

ask_cortex() {
    local q="$1"
    curl -sk --max-time 15 -X POST "$CORTEX/api/brain/dialogue/grounded" \
        -H "Content-Type: application/json" \
        -d "{\"message\":\"$q\"}" 2>/dev/null || echo "{}"
}

learn_video() {
    local url="$1"
    curl -sk --max-time 60 -X POST "$CORTEX/api/brain/learn/academic" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"$url\"}" 2>/dev/null || echo "{}"
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Cortex Learning Benchmark                       ║"
echo "║  Watch YouTube → Spiking Brain → Accurate Answers            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Config: Cortex=$CORTEX Model=$MODEL Tick_wait=${TICK_WAIT}s"
echo ""

# ── Topics: things the LLM fabricates answers about ──────────────
TOPICS=("TurboQuant" "COCONUT reasoning" "RWKV Eagle" "Evolutionary Model Merge" "Mamba SSM")
VIDEOS=(
    "https://www.youtube.com/watch?v=7YVrb3-ABYE"
    "https://www.youtube.com/watch?v=mhKC3Avqy2E"
    "https://www.youtube.com/watch?v=f2voC6K2JDk"
    "https://www.youtube.com/watch?v=4ADxymEyH90"
    "https://www.youtube.com/watch?v=9c_bEQ7J68c"
)
QUESTIONS=(
    "How does TurboQuant reduce memory usage for AI models? What specific technique does it use?"
    "What is COCONUT and how does it change LLM reasoning? What does continuous latent space mean?"
    "What is the RWKV Eagle architecture? How does it use matrix-valued states?"
    "How does Sakana AI evolutionary model merge work? What optimization does it use?"
    "What is the Mamba state space model? How does it compare to Transformers in speed?"
)
N=${#TOPICS[@]}

# ── PHASE 1: Raw LLM baseline ────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 1: Raw LLM (no Cortex) — likely fabricated answers"
echo "═══════════════════════════════════════════════════════════════"
echo ""

RAW_ANSWERS=()
for ((i=0; i<N; i++)); do
    echo "[$i] ${TOPICS[$i]}"
    echo "  Q: ${QUESTIONS[$i]}"
    ANS=$(ask_llm "${QUESTIONS[$i]}")
    RAW_ANSWERS+=("$ANS")
    echo "  A: $ANS"
    echo ""
done

# ── PHASE 2: Teach Cortex ────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 2: Teach Cortex (watch YouTube videos)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for ((i=0; i<N; i++)); do
    echo "[$i] ${TOPICS[$i]}: ${VIDEOS[$i]}"
    RESULT=$(learn_video "${VIDEOS[$i]}")
    CONCEPTS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('concepts_learned','ERROR'))" 2>/dev/null || echo "ERROR")
    echo "  → $CONCEPTS concepts"
done

# ── PHASE 3: Ask Cortex with associative recall ──────────────────
# Each question: prime → wait for tick → ask (reads previous tick's associations)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 3: Ask Cortex (with spiking brain associations)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

CORTEX_ANSWERS=()
BRAIN_ASSOCS=()

for ((i=0; i<N; i++)); do
    # Prime
    echo "[$i] Priming: ${TOPICS[$i]}"
    ask_cortex "${TOPICS[$i]}" > /dev/null
    echo "  Waiting ${TICK_WAIT}s for brain tick..."
    sleep $TICK_WAIT

    # Ask (reads associations from the prime's tick)
    echo "  Asking: ${QUESTIONS[$i]}"
    RESP=$(ask_cortex "${QUESTIONS[$i]}")
    ANS=$(echo "$RESP" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('response','TIMEOUT'))" 2>/dev/null || echo "TIMEOUT")
    ASSOC=$(echo "$RESP" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); a=d.get('brain_associations',[]); print(', '.join(a[:3]) if a else 'none')" 2>/dev/null || echo "none")
    CORTEX_ANSWERS+=("$ANS")
    BRAIN_ASSOCS+=("$ASSOC")
    echo "  Response: $ANS"
    echo "  Brain: $ASSOC"
    echo ""
    sleep $TICK_WAIT
done

# ── RESULTS ──────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "RESULTS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for ((i=0; i<N; i++)); do
    echo "┌─ ${TOPICS[$i]} ─────────────────────────────────────"
    echo "│ RAW LLM:  ${RAW_ANSWERS[$i]:0:120}..."
    echo "│ CORTEX:   ${CORTEX_ANSWERS[$i]:0:120}..."
    echo "│ BRAIN:    ${BRAIN_ASSOCS[$i]}"
    echo "└────────────────────────────────────────────────────────"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "Benchmark complete. $(date)"
echo "═══════════════════════════════════════════════════════════════"
