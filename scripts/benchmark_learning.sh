#!/bin/bash
# Cortex Learning Benchmark
# Proves the spiking brain learns from YouTube videos — knowledge the LLM doesn't have.
#
# Usage: CORTEX_URL=https://localhost:8443 OLLAMA_URL=http://localhost:11434 ./benchmark_learning.sh
#
# What it does:
# 1. Asks 5 questions to the raw LLM (no Cortex) — records "I don't know" answers
# 2. Teaches Cortex by watching YouTube videos about each topic
# 3. Asks Cortex the same questions — records informed answers with brain associations
# 4. Outputs a comparison table

set -e

CORTEX_URL="${CORTEX_URL:-https://localhost:8443}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MODEL="${COMPANION_MODEL:-qwen2.5:1.5b}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Cortex Learning Benchmark                       ║"
echo "║  Spiking brain + foundation encoders + associative recall    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Cortex: $CORTEX_URL"
echo "Ollama: $OLLAMA_URL ($MODEL)"
echo ""

# 5 topics with YouTube videos the LLM likely doesn't know well
declare -A TOPICS
declare -A VIDEOS
declare -A QUESTIONS

TOPICS[0]="TurboQuant"
VIDEOS[0]="https://www.youtube.com/watch?v=7YVrb3-ABYE"
QUESTIONS[0]="How does TurboQuant reduce memory usage for AI models?"

TOPICS[1]="Transformer Squared"
VIDEOS[1]="https://www.youtube.com/watch?v=rx0wP9k4wGM"
QUESTIONS[1]="What is Transformer Squared by Sakana AI and how does it work?"

TOPICS[2]="Liquid Neural Networks"
VIDEOS[2]="https://www.youtube.com/watch?v=RI35E5ewBuI"
QUESTIONS[2]="How do Liquid Neural Networks from MIT work?"

TOPICS[3]="KAN networks"
VIDEOS[3]="https://www.youtube.com/watch?v=7zpz_AlFW2w"
QUESTIONS[3]="What are Kolmogorov-Arnold Networks and how do they differ from MLPs?"

TOPICS[4]="Mamba architecture"
VIDEOS[4]="https://www.youtube.com/watch?v=9c_bEQ7J68c"
QUESTIONS[4]="What is the Mamba state space model architecture?"

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 1: Ask raw LLM (no Cortex, no brain)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

declare -A RAW_ANSWERS

for i in 0 1 2 3 4; do
    echo "[$i] ${TOPICS[$i]}: ${QUESTIONS[$i]}"
    RAW_ANSWERS[$i]=$(curl -s --max-time 30 "$OLLAMA_URL/api/chat" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"${QUESTIONS[$i]}\"}],\"stream\":false,\"options\":{\"num_predict\":200}}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message',{}).get('content','ERROR')[:200])" 2>/dev/null || echo "TIMEOUT")
    echo "  → ${RAW_ANSWERS[$i]}"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 2: Teach Cortex (watch YouTube videos)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for i in 0 1 2 3 4; do
    echo "[$i] Learning: ${TOPICS[$i]} from ${VIDEOS[$i]}"
    RESULT=$(curl -sk --max-time 60 -X POST "$CORTEX_URL/api/brain/learn/academic" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"${VIDEOS[$i]}\"}" 2>/dev/null || echo "{}")
    CONCEPTS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('concepts_learned','ERROR'))" 2>/dev/null || echo "ERROR")
    echo "  → $CONCEPTS concepts learned"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 3: Prime brain (enqueue queries for associative recall)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for i in 0 1 2 3 4; do
    echo "[$i] Priming: ${TOPICS[$i]}"
    curl -sk --max-time 15 -X POST "$CORTEX_URL/api/brain/dialogue/grounded" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"${TOPICS[$i]}\"}" > /dev/null 2>&1
    sleep 35  # Wait for brain tick to process
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 4: Ask Cortex (with brain associations)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

declare -A CORTEX_ANSWERS
declare -A BRAIN_ASSOC

for i in 0 1 2 3 4; do
    echo "[$i] ${TOPICS[$i]}: ${QUESTIONS[$i]}"
    RESPONSE=$(curl -sk --max-time 30 -X POST "$CORTEX_URL/api/brain/dialogue/grounded" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"${QUESTIONS[$i]}\"}" 2>/dev/null || echo "{}")
    CORTEX_ANSWERS[$i]=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('response','ERROR')[:200])" 2>/dev/null || echo "ERROR")
    BRAIN_ASSOC[$i]=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(', '.join(d.get('brain_associations',[])[:3]) or 'none')" 2>/dev/null || echo "ERROR")
    echo "  Response: ${CORTEX_ANSWERS[$i]}"
    echo "  Brain associations: ${BRAIN_ASSOC[$i]}"
    echo ""
    sleep 35  # Wait for next tick
done

echo "═══════════════════════════════════════════════════════════════"
echo "RESULTS COMPARISON"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "| Topic | Raw LLM | Cortex | Brain Associations |"
echo "|-------|---------|--------|-------------------|"

for i in 0 1 2 3 4; do
    RAW_SHORT="${RAW_ANSWERS[$i]:0:60}..."
    CTX_SHORT="${CORTEX_ANSWERS[$i]:0:60}..."
    echo "| ${TOPICS[$i]} | $RAW_SHORT | $CTX_SHORT | ${BRAIN_ASSOC[$i]} |"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Benchmark complete."
echo "═══════════════════════════════════════════════════════════════"
