"""
ArXiv Research Assistant — Streamlit frontend.

Run locally:
    streamlit run src/app.py

In Docker the entrypoint script handles startup.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# ── Page configuration (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS tweaks ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Slightly wider main column */
        .block-container { max-width: 1200px; padding-top: 1.5rem; }
        /* Citation card styling */
        .citation-card {
            border-left: 4px solid #4a90d9;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.5rem;
            background: #f8f9fa;
            border-radius: 0 6px 6px 0;
        }
        .citation-card a { color: #1a73e8; text-decoration: none; font-weight: 600; }
        .citation-card a:hover { text-decoration: underline; }
        .score-badge {
            display: inline-block;
            background: #e8f0fe;
            color: #1967d2;
            border-radius: 12px;
            padding: 1px 8px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        /* Hide Streamlit's default menu in demo mode */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Lazy-load RAG pipeline (cached for the session) ───────────────────────────

@st.cache_resource(show_spinner="Loading RAG pipeline…")
def get_pipeline():
    """Initialise and cache the RAGPipeline singleton."""
    # Make sure src/ is importable when app.py is launched from project root
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from rag import RAGPipeline  # noqa: PLC0415
    return RAGPipeline()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    top_k = st.slider(
        "Papers to retrieve", min_value=1, max_value=10, value=5,
        help="How many ArXiv papers to retrieve for each query.",
    )

    category_filter = st.text_input(
        "Category filter",
        value="",
        placeholder="e.g. cs.AI  or  stat.ML",
        help="Restrict retrieval to papers whose category field contains this string.",
    )

    st.markdown("---")

    st.markdown("### System status")
    if st.button("Check health", use_container_width=True):
        try:
            pipeline = get_pipeline()
            health   = pipeline.health_check()
            st.success(f"ChromaDB: **{health['chroma_documents']:,}** documents")
            if health["ollama_reachable"]:
                st.success(f"Ollama: reachable at `{health['ollama_host']}`")
                if health["ollama_models"]:
                    st.info("Models: " + ", ".join(f"`{m}`" for m in health["ollama_models"]))
            else:
                st.error(f"Ollama: unreachable at `{health['ollama_host']}`")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")

    st.markdown("---")
    st.markdown(
        """
        **ArXiv Research Assistant** uses a fully local RAG stack:
        - **Embeddings:** `all-MiniLM-L6-v2`
        - **Vector DB:** ChromaDB (cosine similarity)
        - **LLM:** Ollama (local inference)
        - **Data:** Cornell-University/arxiv (100k+ papers)

        All inference runs on your own hardware — no data leaves your server.
        """,
    )


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("📚 ArXiv Research Assistant")
st.caption(
    "Ask a research question and receive an answer grounded in peer-reviewed papers — "
    "with clickable citations you can verify."
)

# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLES = [
    "What are the key ideas behind attention mechanisms in transformers?",
    "How does reinforcement learning from human feedback (RLHF) work?",
    "What are recent advances in diffusion models for image generation?",
    "Explain the main challenges in low-resource neural machine translation.",
    "What methods exist for making large language models more efficient?",
    "How is graph neural networks applied to drug discovery?",
]

with st.expander("💡 Example queries — click one to use it"):
    cols = st.columns(2)
    for idx, ex in enumerate(EXAMPLES):
        col = cols[idx % 2]
        if col.button(ex, key=f"ex_{idx}", use_container_width=True):
            st.session_state["prefill_query"] = ex

# ── Query input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")

query = st.text_area(
    "Your research question:",
    value=prefill,
    height=110,
    placeholder=(
        "e.g. What are the main approaches to semi-supervised learning in NLP?"
    ),
    key="query_input",
)

col_btn, col_spacer = st.columns([1, 6])
search_clicked = col_btn.button("🔍 Ask", type="primary", use_container_width=True)

# ── Answer + Citations ────────────────────────────────────────────────────────
if search_clicked and query.strip():

    # Load pipeline (shows spinner on first load)
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(f"Failed to initialise RAG pipeline: {exc}")
        st.stop()

    # ── Retrieve ──────────────────────────────────────────────────────────────
    with st.spinner("Retrieving relevant papers…"):
        papers, token_stream = pipeline.query_stream(
            query,
            top_k=top_k,
            category_filter=category_filter or None,
        )

    if not papers:
        st.warning("No relevant papers found. Try a different query or remove the category filter.")
        st.stop()

    # ── Layout: answer (left) | citations (right) ─────────────────────────────
    col_answer, col_cites = st.columns([3, 2], gap="large")

    # ── Streaming answer ──────────────────────────────────────────────────────
    with col_answer:
        st.subheader("Answer")
        answer_box  = st.empty()
        full_answer = ""
        start_time  = time.time()

        with st.spinner("Generating answer (this can take 30–90s on CPU)…"):
            for token in token_stream:
                full_answer += token
                if full_answer:          # only update once we have something
                    answer_box.markdown(full_answer + "▌")

        # Final render without cursor
        if full_answer.strip():
            answer_box.markdown(full_answer)
        else:
            answer_box.error(
                "Ollama returned an empty response. "
                "The model may still be loading — please wait 30 s and try again."
            )

        elapsed = time.time() - start_time
        st.caption(f"Generated in {elapsed:.1f}s  |  Model: `{os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')}`")

    # ── Citation cards ────────────────────────────────────────────────────────
    with col_cites:
        st.subheader(f"Sources  ({len(papers)} papers)")

        for i, paper in enumerate(papers, start=1):
            m     = paper["metadata"]
            score = paper["score"]
            url   = m.get("url", "#")
            title = m.get("title", "Unknown title")
            cats  = m.get("categories", "")
            date  = m.get("date", "")
            authors_raw = m.get("authors", "")
            # Trim long author lists
            authors = (authors_raw[:80] + "…") if len(authors_raw) > 80 else authors_raw

            # Abstract text (stored as "title\n\nabstract")
            doc_parts = paper["document"].split("\n\n", 1)
            abstract  = doc_parts[1].strip() if len(doc_parts) > 1 else paper["document"]
            abstract_preview = (abstract[:280] + "…") if len(abstract) > 280 else abstract

            with st.expander(f"[{i}] {title[:72]}{'…' if len(title) > 72 else ''}", expanded=i == 1):
                st.markdown(
                    f"<div class='citation-card'>"
                    f"<a href='{url}' target='_blank'>🔗 {title}</a><br>"
                    f"<small>{authors}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                meta_cols = st.columns(2)
                meta_cols[0].caption(f"**Date:** {date}")
                meta_cols[1].markdown(
                    f"<span class='score-badge'>similarity {score:.3f}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"**Categories:** `{cats}`")
                st.markdown("**Abstract:**")
                st.caption(abstract_preview)
                st.markdown(
                    f"[View full paper on ArXiv ↗]({url})",
                    unsafe_allow_html=False,
                )

elif search_clicked and not query.strip():
    st.warning("Please enter a question before searching.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>CS 6120 Final Project · Northeastern University · "
    "All inference is local — powered by Ollama + ChromaDB.</small>",
    unsafe_allow_html=True,
)
