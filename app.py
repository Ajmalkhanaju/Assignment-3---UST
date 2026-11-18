"""
Part 5 - Streamlit front-end for the runbook RAG assistant.

    streamlit run app.py
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

import rag_core as R

DEFAULT_PDF = Path(__file__).parent / "data" / "troubleshooting_runbook.pdf"
INDEX_DIR = Path(__file__).parent / "index"

st.set_page_config(page_title="Runbook RAG Assistant", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder(name: str) -> R.Embedder:
    return R.Embedder(name)


@st.cache_resource(show_spinner="Processing PDF and building the vector index…")
def get_retriever(pdf_path: str, pdf_hash: str, embed_name: str, size: int, overlap: int, strategy: str):
    # pdf_hash is part of the cache key so an uploaded PDF with the same name is re-indexed
    cache = INDEX_DIR if Path(pdf_path) == DEFAULT_PDF else None
    retriever, pages = R.build_index(pdf_path, get_embedder(embed_name), size, overlap, strategy, cache)
    return retriever, len(pages)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key", type="password",
                            value=R.load_api_key() or "")
    model = st.text_input("Gemini model", value=R.DEFAULT_LLM_MODEL)
    k = st.slider("Top-k chunks", 1, 10, R.DEFAULT_TOP_K)
    with st.expander("Indexing"):
        strategy = st.selectbox("Chunking strategy", ["section", "fixed"])
        size = st.number_input("Chunk size (tokens)", 100, 1000, R.DEFAULT_CHUNK_TOKENS, 50)
        overlap = st.number_input("Overlap (tokens)", 0, 300, R.DEFAULT_OVERLAP_TOKENS, 10)
        embed_name = st.text_input("Embedding model", R.DEFAULT_EMBED_MODEL)
        upload = st.file_uploader("Use a different PDF", type="pdf")
    show_ctx = st.checkbox("Show retrieved passages", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []

if upload is not None:
    tmp = Path(tempfile.gettempdir()) / f"rag_upload_{upload.name}"
    tmp.write_bytes(upload.getvalue())
    pdf_path = tmp
else:
    pdf_path = DEFAULT_PDF

retriever, n_pages = get_retriever(str(pdf_path), R.file_sha1(pdf_path), embed_name,
                                   int(size), int(overlap), strategy)

llm = None
if api_key:
    try:
        llm = R.GeminiLLM(model=model, api_key=api_key)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Gemini init failed: {e}")
rag = R.RAGPipeline(retriever, llm)

# ------------------------------------------------------------------ header
st.title("🛠️ Runbook Knowledge Assistant")
st.caption(f"**{pdf_path.name}**: {n_pages} pages, {len(retriever.store.chunks)} chunks, "
           f"{retriever.store.backend}, {embed_name}, LLM: {model if llm else 'not configured (extractive fallback)'}")

if "messages" not in st.session_state:
    st.session_state.messages = []

examples = ["What is the default access token TTL?",
            "Pods restart with exit code 137, what should I check?",
            "How fast must a Sev1 be acknowledged?",
            "What are the likely causes of ERR-503?"]
cols = st.columns(len(examples))
clicked = next((q for c, q in zip(cols, examples) if c.button(q, use_container_width=True)), None)


def render_answer(msg):
    st.markdown(msg["content"])
    g = msg["grounding"]
    badge = "✅ grounded" if g["grounded"] else "⚠️ check grounding"
    detail = f"cited {', '.join(g['cited_ids']) or 'none'}"
    if g["invalid_citations"]:
        detail += f" · invalid citations {g['invalid_citations']}"
    if g["numbers_not_in_context"]:
        detail += f" · numbers not in context {g['numbers_not_in_context']}"
    if g["abstained"]:
        detail += " · abstained (not in runbook)"
    st.caption(f"{badge}: {detail}")
    if show_ctx:
        with st.expander(f"Retrieved passages (k={len(msg['sources'])})"):
            for s in msg["sources"]:
                cited = "**cited**" if s["id"] in g["cited_ids"] else ""
                st.markdown(f"**[{s['id']}]** {s['citation']} · score `{s['score']:.3f}` {cited}")
                st.code(s["text"], language=None)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_answer(m) if m["role"] == "assistant" else st.markdown(m["content"])

question = st.chat_input("Ask about an error code, escalation, or command…") or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                a = rag.ask(question, k)
                msg = {"role": "assistant", "content": a.answer, "grounding": a.grounding,
                       "sources": [{"id": r.chunk.id, "citation": r.chunk.citation(), "score": r.score,
                                    "text": r.chunk.text} for r in a.results]}
            except Exception as e:  # noqa: BLE001
                msg = {"role": "assistant", "content": f"Error: {e}",
                       "grounding": R.grounding_report("", []), "sources": []}
        render_answer(msg)
    st.session_state.messages.append(msg)
