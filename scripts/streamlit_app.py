#!/usr/bin/env python3
"""Streamlit frontend for PaperRAG."""

import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# OPTIMIZED_BY_CODEX_STEP_1
IN_DOCKER = _as_bool(os.getenv("PAPERRAG_IN_DOCKER"), default=False)
DEFAULT_API_BASE = "http://api:8000" if IN_DOCKER else "http://localhost:8000"
API_BASE = os.getenv("PAPERRAG_API_BASE_URL", DEFAULT_API_BASE)
APP_TITLE = os.getenv("PAPERRAG_STREAMLIT_TITLE", "📚 PaperRAG — AI Paper Knowledge Base")

# OPTIMIZED_BY_CODEX_STEP_5
FEEDBACK_DB = Path(os.getenv("PAPERRAG_FEEDBACK_DB", "data/streamlit_feedback.sqlite3"))


def _init_feedback_db() -> None:
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT,
                rating INTEGER NOT NULL
            )
            """
        )


def _save_feedback(query: str, chunk: dict, rating: int) -> None:
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute(
            """
            INSERT INTO feedback (query, chunk_id, source_type, title, rating)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                query,
                chunk.get("chunk_id", ""),
                chunk.get("source_type", ""),
                chunk.get("title") or chunk.get("file_name") or chunk.get("doc_id", ""),
                rating,
            ),
        )


def _arxiv_url(paper_id: str | None) -> str | None:
    if not paper_id:
        return None
    cleaned = paper_id.strip()
    if not cleaned:
        return None
    return f"https://arxiv.org/abs/{quote(cleaned)}"


def _render_paper_card(index: int, chunk: dict, query_text: str) -> None:
    title = chunk.get("title") or chunk.get("file_name") or chunk.get("doc_id", "N/A")
    source_type = chunk.get("source_type", "metadata")
    score = chunk.get("score", chunk.get("final_score", 0.0))
    excerpt = (chunk.get("text") or "")[:700]
    chunk_id = chunk.get("chunk_id", f"chunk-{index}")

    with st.container(border=True):
        head_cols = st.columns([8, 1, 1])
        with head_cols[0]:
            st.markdown(f"### {index}. {title}")
            st.caption(f"source={source_type} | score={score:.4f} | chunk_id={chunk_id}")

            if source_type == "pdf":
                st.markdown(
                    f"**File:** `{chunk.get('file_name', '')}`  \\n"
                    f"**Page:** `{chunk.get('page_no', '')}`"
                )
            else:
                paper_id = chunk.get("paper_id") or chunk.get("doc_id")
                link = _arxiv_url(paper_id)
                if link:
                    st.markdown(f"[🔗 arXiv 直达链接]({link})")
                st.markdown(
                    f"**Paper ID:** `{paper_id or ''}`  \\n"
                    f"**Categories:** `{', '.join(chunk.get('categories', []))}`"
                )

            st.write(excerpt + ("..." if len(chunk.get("text", "")) > 700 else ""))

        up_key = f"fb_up_{index}_{chunk_id}"
        down_key = f"fb_down_{index}_{chunk_id}"

        with head_cols[1]:
            if st.button("👍", key=up_key):
                _save_feedback(query_text, chunk, rating=1)
                st.success("已记录")

        with head_cols[2]:
            if st.button("👎", key=down_key):
                _save_feedback(query_text, chunk, rating=-1)
                st.warning("已记录")


_init_feedback_db()

st.set_page_config(page_title="PaperRAG", page_icon="📚", layout="wide")
st.title(APP_TITLE)

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Answer Mode", ["concise", "analysis"])
    top_k = st.slider("Top K", 3, 20, 5)

    st.divider()
    st.header("Index Management")
    data_path = st.text_input("Data file path", "")
    limit = st.number_input("Load limit (0 = all)", 0, 1000000, 100)
    rebuild = st.checkbox("Rebuild index", True)

    if st.button("Build Index"):
        with st.spinner("Building index..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest/build",
                    json={"data_path": data_path, "limit": limit, "rebuild": rebuild},
                    timeout=600,
                )
                if resp.ok:
                    result = resp.json()
                    st.success(f"{result['message']} ({result['elapsed_ms']:.0f}ms)")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.metric("Documents", health.get("documents", 0))
        st.metric("Chunks", health.get("chunks", 0))
        st.metric("Index Ready", "✅" if health.get("index_ready") else "❌")
        st.caption(f"Feedback DB: {FEEDBACK_DB}")
    except Exception:
        st.warning("API not reachable")

# Main query area
query = st.text_area("Ask a question about AI papers:", placeholder="e.g. What is LoRA and how does it work?")

if st.button("Search & Answer", type="primary") and query:
    with st.spinner("Searching and generating answer..."):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"query": query, "top_k": top_k, "mode": mode},
                timeout=120,
            )
            if resp.ok:
                result = resp.json()

                st.markdown("### Answer")
                st.markdown(result["answer"])

                st.markdown("### Retrieved Paper Cards")
                for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                    _render_paper_card(i, chunk, query)

                with st.expander("Retrieval Trace"):
                    st.json(result.get("retrieval_trace", {}))
                    st.write(f"Evidence Level: {result.get('evidence_level', 'metadata')}")
                    st.write(f"Total time: {result.get('elapsed_ms', 0):.0f} ms")
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
# STEP_1_SUMMARY: Streamlit now supports docker-aware API endpoint and runtime title overrides.
# STEP_5_SUMMARY: Streamlit now provides paper cards, arXiv jump links, and SQLite-based thumbs feedback.
