#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Container entrypoint for the ArXiv Research Assistant app service.
#
# Steps:
#   1. Wait for Ollama to become ready.
#   2. Pull the configured LLM model into Ollama.
#   3. Run data ingestion (no-op if ChromaDB is already populated).
#   4. Launch the Streamlit frontend.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
INGEST_LIMIT="${INGEST_LIMIT:-100000}"
CHROMA_PATH="${CHROMA_PATH:-/app/data/chroma}"

# ── 1. Wait for Ollama ────────────────────────────────────────────────────────
echo "Waiting for Ollama at ${OLLAMA_HOST} …"
MAX_WAIT=120
WAITED=0
until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: Ollama did not become ready within ${MAX_WAIT}s. Aborting."
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done
echo "Ollama is ready."

# ── 2. Pull the LLM model ─────────────────────────────────────────────────────
echo "Pulling model '${OLLAMA_MODEL}' (skipped if already cached) …"
curl -sf "${OLLAMA_HOST}/api/pull" \
     -H "Content-Type: application/json" \
     -d "{\"name\": \"${OLLAMA_MODEL}\"}" \
     | tail -1   # print only the final status line
echo ""

# ── 3. Warm up Ollama (loads model weights into RAM before first user query) ──
echo "Warming up ${OLLAMA_MODEL} (first inference loads weights, ~60s on CPU) …"
curl -sf "${OLLAMA_HOST}/api/generate" \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"${OLLAMA_MODEL}\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
     --max-time 300 > /dev/null 2>&1 && echo "Model warm." || echo "Warmup timed out — model will load on first query."

# ── 4. Data ingestion (background) ────────────────────────────────────────────
echo "Starting data ingestion in background (limit=${INGEST_LIMIT}) …"
python src/ingest.py --limit "${INGEST_LIMIT}" --chroma-path "${CHROMA_PATH}" &

# ── 4. Launch Streamlit immediately ───────────────────────────────────────────
echo "Starting Streamlit on port 8501 …"
exec streamlit run src/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
