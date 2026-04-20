# ArXiv Research Assistant — CS 6120 Final Project

A fully local **Retrieval-Augmented Generation (RAG)** system that answers research questions grounded in 100k+ ArXiv scientific papers, with clickable citations back to the original papers.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        User (Browser)                            │
└─────────────────────────────┬────────────────────────────────────┘
                              │  HTTP :8501
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Streamlit Frontend  (src/app.py)                │
│   • Query input & example prompts                                │
│   • Streaming token display                                       │
│   • Clickable citation cards  →  arxiv.org/abs/<id>              │
└──────────────┬───────────────────────────┬───────────────────────┘
               │  retrieve()               │  query_stream()
               ▼                           ▼
┌──────────────────────────┐   ┌───────────────────────────────────┐
│  ChromaDB (persistent)   │   │  RAG Pipeline  (src/rag.py)       │
│  • Cosine-similarity HNSW│◄──│  • Embed query (MiniLM-L6-v2)    │
│  • 100k+ paper abstracts │   │  • Top-k retrieval               │
│  • Metadata: id, url,    │   │  • Prompt construction           │
│    title, authors, cats  │   │  • Stream to Ollama              │
└──────────────────────────┘   └──────────────────┬────────────────┘
                                                  │  REST :11434
                                                  ▼
                                ┌─────────────────────────────────┐
                                │  Ollama  (llama3.2:3b default)  │
                                │  • 100 % local inference        │
                                │  • Grounded prompt template     │
                                └─────────────────────────────────┘
                 Ingestion (one-time, runs on first boot)
                 ─────────────────────────────────────────
                 HuggingFace datasets API
                   └─► Cornell-University/arxiv (streaming)
                         └─► sentence-transformers embed
                               └─► ChromaDB upsert
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker | ≥ 24 | |
| Docker Compose | ≥ 2.20 | bundled with Docker Desktop |
| (Optional) NVIDIA GPU | — | for faster LLM inference |

---

## Quickstart (Docker — recommended)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Final_Project

# 2. Build and start all services
docker compose up --build
```

On **first run** the container will automatically:
1. Pull `llama3.2:3b` into Ollama (~2 GB).
2. Download and embed 100,000 ArXiv papers into ChromaDB (~20–40 min, one-time).
3. Launch the Streamlit UI on **http://localhost:8501**.

Subsequent starts skip steps 1–2 because both the model and the vector store are persisted in Docker volumes.

> **GCP / remote server:** replace `localhost` with the VM's external IP.
> Open firewall port **8501** in your GCP VPC network rules.

### GPU support (optional)

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Uncomment the `deploy.resources` block in `docker-compose.yml` under the `ollama` service.
3. Restart: `docker compose up --build`.

---

## Running locally (without Docker)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies (CPU PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Install and start Ollama  https://ollama.com/download
ollama serve &
ollama pull llama3.2:3b

# 4. Ingest data (downloads ~4 GB, one-time)
python src/ingest.py --limit 100000

# 5. Launch the Streamlit app
streamlit run src/app.py
```

---

## Configuration

All settings are controlled by environment variables (or the `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PATH` | `./data/chroma` | Path to ChromaDB persistent directory |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model for embeddings |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama REST endpoint |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model tag to use for generation |
| `INGEST_LIMIT` | `100000` | Number of ArXiv papers to index |

### Changing the LLM model

Edit `OLLAMA_MODEL` in `docker-compose.yml`:

```yaml
OLLAMA_MODEL: mistral:7b   # higher quality, needs more RAM/VRAM
```

---

## Dataset

**Cornell-University/arxiv** (via HuggingFace Datasets)

- **Source:** https://huggingface.co/datasets/Cornell-University/arxiv
- **Scale:** 1.7 M+ papers; this system indexes the first 100 k by default.
- **Fields used:** `id`, `title`, `abstract`, `authors`, `categories`, `update_date`
- **Citation link format:** `https://arxiv.org/abs/<arxiv_id>`

If HuggingFace is unavailable the ingestion script automatically falls back to the official ArXiv OAI export API.

---

## Project structure

```
Final_Project/
├── src/
│   ├── ingest.py       # Download + embed + store in ChromaDB
│   ├── rag.py          # RAG pipeline: retrieve → generate
│   └── app.py          # Streamlit frontend
├── scripts/
│   └── entrypoint.sh   # Container startup script
├── data/               # ChromaDB lives here (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Team contributions

*(Fill in before submission)*

| Member | Contribution |
|--------|-------------|
| | |
| | |
