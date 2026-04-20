#!/usr/bin/env python3
"""
Data ingestion script for ArXiv RAG system.

Streams the Cornell-University/arxiv dataset from HuggingFace and ingests
papers into ChromaDB in rolling batches — embedding and inserting as data
arrives rather than buffering everything in memory first.

Usage:
    python src/ingest.py [--limit N] [--categories cs.AI cs.LG ...]
"""

import os
import sys
import argparse
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Force stdout/stderr to be unbuffered so docker logs shows progress
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Configuration (overridable via environment) ──────────────────────────────
CHROMA_PATH  = os.environ.get("CHROMA_PATH",  "./data/chroma")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "all-MiniLM-L6-v2")
COLLECTION   = "arxiv_papers"
INGEST_BATCH = 512          # stream this many papers before embedding+inserting
EMBED_BATCH  = 64           # sentence-transformer encode batch size
API_BATCH    = 200          # results per arxiv API page
API_DELAY    = 3.0          # seconds between API calls (rate limit)

DEFAULT_CATS = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR",
    "stat.ML", "cs.NE", "cs.RO",
]


# ── Record helpers ────────────────────────────────────────────────────────────

def _make_record(arxiv_id, title, abstract, authors, categories, date) -> dict:
    safe_id = arxiv_id.replace("/", "_")
    return {
        "id":         safe_id,
        "arxiv_id":   arxiv_id,
        "title":      title,
        "abstract":   abstract,
        "authors":    authors[:500],
        "categories": categories,
        "date":       date,
        "url":        f"https://arxiv.org/abs/{arxiv_id}",
        "document":   f"{title}\n\n{abstract}",
    }


# ── Data source: HuggingFace streaming generator ─────────────────────────────

def hf_paper_stream(limit: int, categories: list[str] | None) -> Generator[dict, None, None]:
    """Yields paper dicts one-by-one from the HuggingFace arxiv dataset."""
    from datasets import load_dataset

    print("Connecting to HuggingFace (Cornell-University/arxiv, streaming)…", flush=True)
    ds = load_dataset("Cornell-University/arxiv", split="train", streaming=True)

    cat_filter = set(categories) if categories else None
    yielded = 0

    for paper in ds:
        if yielded >= limit:
            break

        abstract = (paper.get("abstract") or "").strip().replace("\n", " ")
        title    = (paper.get("title")    or "").strip().replace("\n", " ")
        if not abstract or not title:
            continue

        if cat_filter:
            paper_cats = set((paper.get("categories") or "").split())
            if not paper_cats.intersection(cat_filter):
                continue

        yield _make_record(
            arxiv_id  = paper["id"].strip(),
            title     = title,
            abstract  = abstract,
            authors   = (paper.get("authors")     or ""),
            categories= (paper.get("categories")  or ""),
            date      = (paper.get("update_date") or ""),
        )
        yielded += 1


# ── Data source: ArXiv API fallback generator ─────────────────────────────────

def api_paper_stream(limit: int, categories: list[str] | None) -> Generator[dict, None, None]:
    """Yields paper dicts from the official ArXiv export API."""
    cats    = categories or DEFAULT_CATS
    yielded = 0
    print(f"Fetching via ArXiv API (categories: {cats})…", flush=True)

    for cat in cats:
        if yielded >= limit:
            break
        start = 0

        while yielded < limit:
            url = (
                "http://export.arxiv.org/api/query"
                f"?search_query=cat:{cat}"
                f"&start={start}&max_results={API_BATCH}"
                "&sortBy=submittedDate&sortOrder=descending"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"  API error ({cat}): {exc}. Moving to next category.", flush=True)
                break

            ns      = {"atom": "http://www.w3.org/2005/Atom"}
            root    = ET.fromstring(resp.content)
            entries = root.findall("atom:entry", ns)
            if not entries:
                break

            for entry in entries:
                if yielded >= limit:
                    break
                raw_id   = entry.find("atom:id",      ns).text or ""
                arxiv_id = raw_id.split("/abs/")[-1].strip()
                title    = (entry.find("atom:title",   ns).text or "").strip().replace("\n", " ")
                abstract = (entry.find("atom:summary", ns).text or "").strip().replace("\n", " ")
                if not title or not abstract:
                    continue
                authors = ", ".join(
                    (a.find("atom:name", ns).text or "").strip()
                    for a in entry.findall("atom:author", ns)
                )
                date_el = entry.find("atom:updated", ns)
                date    = (date_el.text or "")[:10] if date_el is not None else ""
                yield _make_record(arxiv_id, title, abstract, authors, cat, date)
                yielded += 1

            print(f"  [{cat}] {yielded} papers so far…", flush=True)
            start += API_BATCH
            if len(entries) < API_BATCH:
                break
            time.sleep(API_DELAY)


# ── Core: stream → embed → insert ─────────────────────────────────────────────

def stream_and_ingest(
    paper_gen: Generator[dict, None, None],
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    limit: int,
    existing_ids: set[str],
) -> int:
    """
    Consumes `paper_gen`, embeds papers in batches of INGEST_BATCH,
    and upserts into ChromaDB immediately.  Returns total inserted.
    """
    total_inserted = 0
    batch: list[dict] = []

    pbar = tqdm(total=limit, desc="Ingesting", unit="papers", file=sys.stdout)

    for paper in paper_gen:
        if paper["id"] in existing_ids:
            pbar.update(1)
            continue

        batch.append(paper)
        pbar.update(1)

        if len(batch) >= INGEST_BATCH:
            total_inserted += _flush_batch(batch, embed_model, collection)
            batch.clear()

    # flush remainder
    if batch:
        total_inserted += _flush_batch(batch, embed_model, collection)

    pbar.close()
    return total_inserted


def _flush_batch(
    batch: list[dict],
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int:
    texts      = [p["document"] for p in batch]
    embeddings = embed_model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    collection.upsert(
        ids        = [p["id"] for p in batch],
        embeddings = embeddings.tolist(),
        documents  = texts,
        metadatas  = [
            {
                "arxiv_id":   p["arxiv_id"],
                "title":      p["title"],
                "authors":    p["authors"],
                "categories": p["categories"],
                "date":       p["date"],
                "url":        p["url"],
            }
            for p in batch
        ],
    )
    count = collection.count()
    print(f"  → ChromaDB now has {count:,} documents", flush=True)
    return len(batch)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest ArXiv papers into ChromaDB.")
    parser.add_argument("--limit",       type=int, default=int(os.environ.get("INGEST_LIMIT", 50_000)))
    parser.add_argument("--categories",  nargs="+", default=None)
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()

    Path(args.chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=args.chroma_path)

    # ── Short-circuit if already populated ───────────────────────────────────
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    existing_count = collection.count()

    if not args.force and existing_count >= args.limit:
        print(
            f"Collection already has {existing_count:,} docs "
            f"(requested {args.limit:,}). Skipping. Use --force to re-ingest.",
            flush=True,
        )
        return

    existing_ids: set[str] = set()
    if existing_count > 0:
        print(f"Resuming — {existing_count:,} docs already indexed.", flush=True)
        existing_ids = set(collection.get(include=[])["ids"])

    remaining = args.limit - existing_count
    print(f"Target: {args.limit:,} papers  |  Still needed: {remaining:,}", flush=True)

    # ── Load embedding model ─────────────────────────────────────────────────
    print(f"Loading embedding model: {EMBED_MODEL}", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL)

    # ── Stream and ingest ────────────────────────────────────────────────────
    try:
        gen = hf_paper_stream(limit=args.limit, categories=args.categories)
        inserted = stream_and_ingest(gen, embed_model, collection, args.limit, existing_ids)
    except Exception as exc:
        print(f"HuggingFace stream failed: {exc}\nFalling back to ArXiv API…", flush=True)
        gen = api_paper_stream(limit=args.limit, categories=args.categories)
        inserted = stream_and_ingest(gen, embed_model, collection, args.limit, existing_ids)

    print(
        f"\nDone. Inserted {inserted:,} new papers. "
        f"Total in collection: {collection.count():,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
