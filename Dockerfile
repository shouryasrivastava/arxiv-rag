# ─────────────────────────────────────────────────────────────────────────────
# ArXiv Research Assistant — Application image
#
# This image contains:
#   • Python 3.11 (slim)
#   • All Python dependencies (CPU-only PyTorch)
#   • The src/ application code and startup script
#
# The Ollama LLM server runs as a SEPARATE container (see docker-compose.yml).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System utilities needed for HuggingFace downloads and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Install CPU-only PyTorch first to keep image lean (~900 MB vs 3 GB for CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY src/     ./src/
COPY scripts/ ./scripts/
RUN chmod +x scripts/entrypoint.sh

# ── Streamlit port ────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check (Streamlit responds on /_stcore/health) ─────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["scripts/entrypoint.sh"]
