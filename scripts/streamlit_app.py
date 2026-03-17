#!/usr/bin/env python3
"""Streamlit frontend for PaperRAG."""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="PaperRAG", page_icon="📚", layout="wide")
st.title("📚 PaperRAG — AI Paper Knowledge Base")

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
    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.metric("Documents", health.get("documents", 0))
        st.metric("Chunks", health.get("chunks", 0))
        st.metric("Index Ready", "✅" if health.get("index_ready") else "❌")
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

                st.markdown("### Retrieved Papers")
                for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                    with st.expander(f"[{i}] {chunk['title']} (score: {chunk['final_score']:.4f})"):
                        st.write(f"**Authors:** {', '.join(chunk.get('authors', [])[:5])}")
                        st.write(f"**Categories:** {', '.join(chunk.get('categories', []))}")
                        st.write(f"**Date:** {chunk.get('update_date', '')}")
                        st.write(f"**Text:** {chunk['text'][:500]}...")

                with st.expander("Retrieval Trace"):
                    st.json(result.get("retrieval_trace", {}))
                    st.write(f"Total time: {result.get('elapsed_ms', 0):.0f} ms")
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
