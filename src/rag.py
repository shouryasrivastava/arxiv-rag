"""
RAG pipeline for the ArXiv Research Assistant.

Exposes a single `RAGPipeline` class that:
  1. Embeds the user query with sentence-transformers.
  2. Retrieves the top-k most relevant papers from ChromaDB.
  3. Calls the local Ollama LLM (via REST) to generate a grounded answer.

Both blocking and streaming generation are supported.
"""

from __future__ import annotations

import json
import os
from typing import Generator, Optional

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ── Configuration (overridable via environment) ──────────────────────────────
CHROMA_PATH  = os.environ.get("CHROMA_PATH",   "./data/chroma")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",   "all-MiniLM-L6-v2")
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL",  "llama3.2:3b")
COLLECTION   = "arxiv_papers"
DEFAULT_TOP_K = 5

# ── Prompt template ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a knowledgeable research assistant grounded in peer-reviewed academic literature.
Answer the user's question using ONLY the information provided in the retrieved paper abstracts below.
- Cite papers by their bracketed index number, e.g. [1], [2].
- If the context does not contain enough information to answer the question, say so explicitly.
- Do NOT speculate beyond the provided context.
- Be concise but thorough. Use bullet points or numbered lists when helpful.\
"""

_CONTEXT_BLOCK = """\
--- Retrieved Papers ---
{papers}
-----------------------\
"""

_PAPER_ENTRY = """\
[{idx}] Title: {title}
    Authors: {authors}
    Categories: {categories}
    Published: {date}
    ArXiv URL: {url}
    Abstract: {abstract}
"""

_USER_PROMPT = """\
{context}

Question: {question}

Answer:\
"""


class RAGPipeline:
    """End-to-end RAG pipeline: embed → retrieve → generate."""

    def __init__(self) -> None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        self._embedder = SentenceTransformer(EMBED_MODEL)

        print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
        self._client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection = self._client.get_collection(COLLECTION)
        print(f"RAG pipeline ready  ({self._collection.count():,} documents indexed).")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> tuple[str, list[dict]]:
        """
        Blocking RAG query.

        Returns
        -------
        answer  : str  – LLM-generated answer
        papers  : list[dict] – retrieved papers with metadata + relevance score
        """
        papers = self.retrieve(question, top_k=top_k, category_filter=category_filter)
        answer = self._generate(question, papers, stream=False)
        return answer, papers

    def query_stream(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> tuple[list[dict], Generator[str, None, None]]:
        """
        Streaming RAG query.

        Returns
        -------
        papers  : list[dict]       – retrieved papers (available immediately)
        stream  : Generator[str]   – token-by-token LLM output
        """
        papers = self.retrieve(question, top_k=top_k, category_filter=category_filter)
        return papers, self._generate(question, papers, stream=True)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embed `query` and return the top-k nearest papers from ChromaDB.

        Each returned dict has keys:
            id, document, metadata (arxiv_id, title, authors, categories,
            date, url), distance, score (cosine similarity 0-1).
        """
        q_embedding = self._embedder.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()

        where: Optional[dict] = None
        if category_filter and category_filter.strip():
            # ChromaDB $contains operator for string metadata fields
            where = {"categories": {"$contains": category_filter.strip()}}

        results = self._collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        papers = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            papers.append({
                "id":       results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": dist,
                "score":    max(0.0, 1.0 - dist),   # cosine similarity
            })

        return papers

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, question: str, papers: list[dict]) -> str:
        paper_texts = ""
        for i, p in enumerate(papers, start=1):
            m = p["metadata"]
            # Abstract is everything after the first "\n\n" in the stored document
            doc_parts = p["document"].split("\n\n", 1)
            abstract  = doc_parts[1].strip() if len(doc_parts) > 1 else p["document"]

            paper_texts += _PAPER_ENTRY.format(
                idx        = i,
                title      = m.get("title",      "N/A"),
                authors    = m.get("authors",    "N/A"),
                categories = m.get("categories", "N/A"),
                date       = m.get("date",       "N/A"),
                url        = m.get("url",        "N/A"),
                abstract   = abstract,
            )

        context = _CONTEXT_BLOCK.format(papers=paper_texts.strip())
        return _USER_PROMPT.format(context=context, question=question)

    def _generate(
        self,
        question: str,
        papers: list[dict],
        stream: bool,
    ):
        prompt  = self._build_prompt(question, papers)
        payload = {
            "model":  OLLAMA_MODEL,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.1,
                "top_p":       0.9,
                "num_predict": 768,
            },
        }

        url = f"{OLLAMA_HOST}/api/generate"

        if stream:
            return self._stream_response(url, payload)
        else:
            return self._blocking_response(url, payload)

    def _blocking_response(self, url: str, payload: dict) -> str:
        try:
            resp = requests.post(url, json=payload, timeout=(10, 600))
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ConnectionError:
            return (
                f"**Error:** Cannot reach Ollama at `{OLLAMA_HOST}`. "
                "Make sure the Ollama service is running and the model is pulled."
            )
        except Exception as exc:
            return f"**Error generating response:** {exc}"

    def _stream_response(self, url: str, payload: dict) -> Generator[str, None, None]:
        try:
            with requests.post(url, json=payload, stream=True, timeout=(10, 600)) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    chunk = json.loads(raw_line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            yield (
                f"\n\n**Error:** Cannot reach Ollama at `{OLLAMA_HOST}`. "
                "Make sure the Ollama service is running and the model is pulled."
            )
        except Exception as exc:
            yield f"\n\n**Error generating response:** {exc}"

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def health_check(self) -> dict:
        """Return a dict with system health information."""
        ollama_ok = False
        ollama_models: list[str] = []
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.ok:
                ollama_ok    = True
                ollama_models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass

        return {
            "chroma_documents": self.document_count,
            "embed_model":      EMBED_MODEL,
            "ollama_host":      OLLAMA_HOST,
            "ollama_model":     OLLAMA_MODEL,
            "ollama_reachable": ollama_ok,
            "ollama_models":    ollama_models,
        }
