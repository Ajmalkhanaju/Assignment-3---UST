"""
rag_core.py - Retrieval-Augmented Generation over the Contoso Troubleshooting Runbook.

Pipeline
    PDF --(pdfplumber: text + tables)--> pages --(clean)--> sections --(token chunker)--> chunks
    chunks --(SentenceTransformer bge-small)--> vectors --(FAISS IndexFlatIP)--> vector store
    query --(embed)--> top-k chunks --(grounded prompt)--> Gemini --> answer with [C#] citations

Used by both the notebook (Parts 1-4) and the Streamlit app (Part 5).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------------------
# Defaults (chosen from the experiments in the notebook)
# --------------------------------------------------------------------------------------
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, 512-token input window
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
DEFAULT_CHUNK_TOKENS = 300
DEFAULT_OVERLAP_TOKENS = 60
DEFAULT_TOP_K = 5
NOT_FOUND_MSG = "The runbook does not contain this information."

# Repeated running header/footer printed on every page of the runbook.
_FOOTER_RE = re.compile(
    r"Contoso Platform Engineering \| Troubleshooting Runbook v[\d.]+ \| Internal - Synthetic demo data"
    r"(\s*Page \d+)?"
)
_PAGE_ONLY_RE = re.compile(r"^\s*Page \d+\s*$")

# Section headings. Error-code entries ("ERR-504 - Gateway timeout"), numbered sections
# ("5. Severity matrix") and appendix sections ("B. Post-incident checklist").
# Numbered *list items* ("2. Identify the error code ... payload.") end with a period, so the
# "no period, short" rule keeps them out.
_ERROR_HEADING_RE = re.compile(r"^((?:ERR|DB|MEM|DISK|CERT)-[A-Z0-9]+) - (.+)$")
_NUM_HEADING_RE = re.compile(r"^(\d{1,2}|[A-H])\. ([A-Z][^.]{2,45})$")


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------
@dataclass
class Page:
    number: int          # 1-based
    text: str            # cleaned text, tables serialised as "Header: value | ..." rows
    raw_text: str        # pypdf-style raw text, kept for the quality comparison
    n_tables: int = 0


@dataclass
class Section:
    title: str
    code: str | None     # error code if this is an error-code section
    page_start: int
    lines: list[tuple[int, str]] = field(default_factory=list)  # (page, line)


@dataclass
class Chunk:
    id: str
    text: str            # text that is embedded (contains a short section header)
    section: str
    code: str | None
    pages: list[int]
    n_tokens: int

    def citation(self) -> str:
        pg = ",".join(str(p) for p in self.pages)
        return f"{self.section} (p.{pg})"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


# --------------------------------------------------------------------------------------
# Part 1 - loading and preprocessing
# --------------------------------------------------------------------------------------
def _serialise_table(rows: list[list[str | None]]) -> list[str]:
    """Turn a table into one self-describing line per row, e.g.
    'Service: payment-gateway | Domain: Payments | Owning team: Team Phoenix | ...'.
    Plain text extraction prints tables one cell per line, which destroys the row structure
    (a chunk containing 'Team Phoenix' would no longer say which service it owns)."""
    clean = [[re.sub(r"\s+", " ", c or "").strip() for c in r] for r in rows if r]
    if len(clean) < 2:
        return [" | ".join(r) for r in clean]
    header, body = clean[0], clean[1:]
    out = []
    for r in body:
        cells = [f"{h}: {v}" for h, v in zip(header, r) if v]
        out.append("- " + " | ".join(cells))
    return out


def clean_text(text: str) -> str:
    text = _FOOTER_RE.sub("", text)
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if not ln or _PAGE_ONLY_RE.match(ln):
            continue
        lines.append(ln)
    return "\n".join(lines)


def load_pdf(path: str | Path) -> list[Page]:
    """Extract every page. Text outside tables comes from pdfplumber in reading order; each
    table is serialised row-by-row and inserted at its vertical position on the page."""
    import pdfplumber
    from pypdf import PdfReader

    raw_pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    pages: list[Page] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, p in enumerate(pdf.pages):
            tables = sorted(p.find_tables(), key=lambda t: t.bbox[1])
            parts, top = [], 0.0
            for t in tables:
                x0, t_top, x1, t_bottom = t.bbox
                if t_top > top:
                    parts.append(p.crop((0, top, p.width, t_top)).extract_text() or "")
                parts.append("\n".join(_serialise_table(t.extract())))
                top = t_bottom
            parts.append(p.crop((0, top, p.width, p.height)).extract_text() or "")
            pages.append(Page(i + 1, clean_text("\n".join(parts)), raw_pages[i], len(tables)))
    return pages


def text_quality_report(pages: list[Page]) -> dict:
    """Simple diagnostics for Part 1 ('explore document structure')."""
    raw = "\n".join(p.raw_text for p in pages)
    clean = "\n".join(p.text for p in pages)
    raw_lines = [l for l in raw.splitlines() if l.strip()]
    return {
        "pages": len(pages),
        "tables_found": sum(p.n_tables for p in pages),
        "raw_words": len(raw.split()),
        "clean_words": len(clean.split()),
        "footer_lines_removed": len(_FOOTER_RE.findall(raw)),
        "raw_short_lines(<=3 words)": sum(1 for l in raw_lines if len(l.split()) <= 3),
        "raw_short_line_ratio": round(sum(1 for l in raw_lines if len(l.split()) <= 3) / max(1, len(raw_lines)), 3),
        "non_ascii_chars": sum(1 for ch in clean if ord(ch) > 127),
        "empty_pages": sum(1 for p in pages if not p.text.strip()),
    }


# --------------------------------------------------------------------------------------
# Tokenisation (used for chunk sizing)
# --------------------------------------------------------------------------------------
_tokenizer_cache: dict[str, Callable[[str], int]] = {}


def get_token_counter(model_name: str = DEFAULT_EMBED_MODEL) -> Callable[[str], int]:
    """Count tokens with the embedding model's own WordPiece tokenizer, so chunk sizes are
    expressed in the unit that actually limits the embedder (512 tokens for bge-small).
    Falls back to a regex word/punctuation count (~ +10-25% vs WordPiece) if unavailable."""
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        fn = lambda s: len(tok.encode(s, add_special_tokens=False))
    except Exception:
        fn = lambda s: len(re.findall(r"\w+|[^\w\s]", s))
    _tokenizer_cache[model_name] = fn
    return fn


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------
def split_sections(pages: list[Page]) -> list[Section]:
    sections: list[Section] = [Section("Document header", None, 1)]
    for page in pages:
        lines = page.text.splitlines()
        # Pages 2+ open with a group banner ("Severity and Escalation", "Database Errors",
        # "Appendix: Quick Reference and Post-Incident / Checklist"). It is not part of the
        # previous section, so drop it instead of letting it leak into that section's chunk.
        if page.number > 1 and lines and "." not in lines[0] and len(lines[0].split()) <= 8:
            lines = lines[1:]
            if lines and len(lines[0].split()) == 1 and not _NUM_HEADING_RE.match(lines[0]):
                lines = lines[1:]
        for line in lines:
            m_err = _ERROR_HEADING_RE.match(line)
            m_num = _NUM_HEADING_RE.match(line)
            if m_err:
                sections.append(Section(line, m_err.group(1), page.number))
            elif m_num:
                sections.append(Section(line, None, page.number))
            else:
                sections[-1].lines.append((page.number, line))
    return [s for s in sections if s.lines]


def _sentences(line: str) -> list[str]:
    # Split long lines into sentences, but keep numbered steps like "1. Check ..." intact.
    parts = re.split(r"(?<=[.;])\s+(?=(?:\d+\.\s)?[A-Z(])", line)
    return [p for p in parts if p.strip()]


def _window(units: list[tuple[int, str, bool]], max_tokens: int, overlap: int,
            count: Callable[[str], int], header: str) -> list[tuple[str, list[int], int]]:
    """Greedy packing of sentence units into windows of <= max_tokens, carrying the last
    `overlap` tokens' worth of sentences into the next window."""
    budget = max_tokens - count(header) - 2
    out, cur, cur_tok = [], [], 0
    sizes = [count(u[1]) for u in units]
    i = 0
    while i < len(units):
        t = sizes[i]
        if cur and cur_tok + t > budget:
            out.append(cur)
            # overlap: keep trailing sentences up to `overlap` tokens
            keep, kt = [], 0
            for j in range(len(cur) - 1, -1, -1):
                if kt + cur[j][2] > overlap:
                    break
                keep.insert(0, cur[j]); kt += cur[j][2]
            cur, cur_tok = keep, kt
        cur.append((units[i][0], units[i][1], t, units[i][2])); cur_tok += t
        i += 1
    if cur:
        out.append(cur)
    res = []
    for w in out:
        # keep original line breaks (commands / SQL stay copy-pasteable); sentences that came
        # from the same line are re-joined with a space
        body = "".join(("\n" if (u[3] and n) else (" " if n else "")) + u[1] for n, u in enumerate(w))
        text = f"{header}\n{body}" if header else body
        res.append((text, sorted({u[0] for u in w}), count(text)))
    return res


def chunk_document(pages: list[Page], chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                   overlap_tokens: int = DEFAULT_OVERLAP_TOKENS, strategy: str = "section",
                   count: Callable[[str], int] | None = None) -> list[Chunk]:
    """strategy='section': split on headings first, then window inside each section; every
    chunk is prefixed with its section title so it is self-describing.
    strategy='fixed': baseline - one token window over the whole document, ignoring structure."""
    count = count or get_token_counter()
    chunks: list[Chunk] = []
    if strategy == "fixed":
        units = [(p.number, s, j == 0) for p in pages for ln in p.text.splitlines()
                 for j, s in enumerate(_sentences(ln))]
        for text, pgs, n in _window(units, chunk_tokens, overlap_tokens, count, header=""):
            chunks.append(Chunk(f"C{len(chunks)+1}", text, "-", None, pgs, n))
        return chunks
    for sec in split_sections(pages):
        units = [(pg, s, j == 0) for pg, ln in sec.lines for j, s in enumerate(_sentences(ln))]
        header = f"[Section: {sec.title}]"
        for text, pgs, n in _window(units, chunk_tokens, overlap_tokens, count, header):
            chunks.append(Chunk(f"C{len(chunks)+1}", text, sec.title, sec.code, pgs, n))
    return chunks


# --------------------------------------------------------------------------------------
# Part 2 - embeddings and vector store
# --------------------------------------------------------------------------------------
class Embedder:
    """backend='st'    : SentenceTransformers (default BAAI/bge-small-en-v1.5)
       backend='tfidf' : offline fallback (no model download) - used only for smoke tests."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, backend: str = "st"):
        self.model_name, self.backend = model_name, backend
        if backend == "st":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                         token_pattern=r"(?u)\b[\w-]+\b")
            self.dim, self.query_prefix, self._fitted = None, "", False
        else:
            raise ValueError(backend)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype="float32")
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            return self._norm(self.model.encode(texts, batch_size=32, show_progress_bar=False))
        v = self.model.fit_transform(texts).toarray(); self._fitted = True
        self.dim = v.shape[1]
        return self._norm(v)

    def embed_query(self, query: str) -> np.ndarray:
        if self.backend == "st":
            return self._norm(self.model.encode([self.query_prefix + query]))
        return self._norm(self.model.transform([query]).toarray())


class VectorStore:
    """Cosine-similarity search. Vectors are L2-normalised, so inner product == cosine.
    Uses FAISS IndexFlatIP (exact search - right for a few hundred chunks); falls back to a
    NumPy matrix product if faiss is not installed. Chunks + metadata are stored alongside."""

    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        try:
            import faiss
            self.index, self._faiss = faiss.IndexFlatIP(dim), True
        except ImportError:
            self.index, self._faiss, self._mat = None, False, np.zeros((0, dim), "float32")

    @property
    def backend(self) -> str:
        return "faiss.IndexFlatIP" if self._faiss else "numpy"

    def add(self, vectors: np.ndarray, chunks: list[Chunk]):
        assert len(vectors) == len(chunks)
        if self._faiss:
            self.index.add(vectors.astype("float32"))
        else:
            self._mat = np.vstack([self._mat, vectors.astype("float32")])
        self.chunks.extend(chunks)

    def search(self, qvec: np.ndarray, k: int) -> list[RetrievedChunk]:
        k = min(k, len(self.chunks))
        if self._faiss:
            scores, idx = self.index.search(qvec.astype("float32"), k)
            pairs = zip(idx[0], scores[0])
        else:
            s = (self._mat @ qvec[0])
            top = np.argsort(-s)[:k]
            pairs = zip(top, s[top])
        return [RetrievedChunk(self.chunks[i], float(sc), r + 1) for r, (i, sc) in enumerate(pairs) if i >= 0]

    # persistence --------------------------------------------------------------------
    def save(self, folder: str | Path, meta: dict):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        if self._faiss:
            import faiss
            faiss.write_index(self.index, str(folder / "index.faiss"))
        else:
            np.save(folder / "vectors.npy", self._mat)
        (folder / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], indent=1))
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, folder: str | Path) -> tuple["VectorStore", dict]:
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        store = cls(meta["dim"])
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text())]
        if store._faiss and (folder / "index.faiss").exists():
            import faiss
            store.index = faiss.read_index(str(folder / "index.faiss"))
        elif (folder / "vectors.npy").exists():
            vec = np.load(folder / "vectors.npy")
            if store._faiss:
                store.index.add(vec)
            else:
                store._mat = vec
        else:
            raise FileNotFoundError("no saved vectors")
        store.chunks = chunks
        return store, meta


def file_sha1(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# Part 3 - retrieval
# --------------------------------------------------------------------------------------
class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder, self.store = embedder, store

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        return self.store.search(self.embedder.embed_query(query), k)


def build_index(pdf_path: str | Path, embedder: Embedder, chunk_tokens=DEFAULT_CHUNK_TOKENS,
                overlap_tokens=DEFAULT_OVERLAP_TOKENS, strategy="section",
                cache_dir: str | Path | None = None) -> tuple[Retriever, list[Page]]:
    """Load -> chunk -> embed -> store. Re-uses a saved index when the PDF hash, embedding
    model and chunking parameters all match."""
    pages = load_pdf(pdf_path)
    if embedder.backend == "tfidf":          # TF-IDF is fitted per corpus: never share one
        embedder = Embedder(backend="tfidf")
    meta_key = {"pdf_sha1": file_sha1(pdf_path), "embed_model": embedder.model_name,
                "backend": embedder.backend, "chunk_tokens": chunk_tokens,
                "overlap_tokens": overlap_tokens, "strategy": strategy}
    if cache_dir and embedder.backend == "st":
        try:
            store, meta = VectorStore.load(cache_dir)
            if all(meta.get(k) == v for k, v in meta_key.items()):
                return Retriever(embedder, store), pages
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
    count = get_token_counter(embedder.model_name)
    chunks = chunk_document(pages, chunk_tokens, overlap_tokens, strategy, count)
    vecs = embedder.embed_documents([c.text for c in chunks])
    store = VectorStore(vecs.shape[1]); store.add(vecs, chunks)
    if cache_dir and embedder.backend == "st":
        store.save(cache_dir, {**meta_key, "dim": int(vecs.shape[1]), "n_chunks": len(chunks)})
    return Retriever(embedder, store), pages


# --------------------------------------------------------------------------------------
# Part 4 - grounded generation
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are the Contoso on-call assistant. You answer questions about the Contoso
Platform Troubleshooting Runbook using ONLY the context passages supplied with each question.

Rules:
1. Use only facts stated in the CONTEXT. Do not use outside knowledge, even if you are confident
   it is true (e.g. general Kubernetes or PostgreSQL advice that the runbook does not state).
2. Cite every factual sentence with the passage id(s) it comes from, like [C3] or [C3][C7].
3. If the context does not contain the answer, reply exactly: "{NOT_FOUND_MSG}"
   If it answers only part of the question, answer that part and say which part is not covered.
4. Copy commands, thresholds, numbers and error strings exactly as written; put commands in
   code blocks. Never invent flags, values or steps.
5. If passages contradict each other, or the question assumes something the context contradicts,
   say so and quote both sides.
6. Be concise and operational: lead with the direct answer, then steps if relevant."""


def format_context(results: list[RetrievedChunk]) -> str:
    blocks = []
    for r in results:
        c = r.chunk
        blocks.append(f"[{c.id}] (section: {c.section}; page {','.join(map(str, c.pages))})\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[RetrievedChunk]) -> str:
    return (f"CONTEXT:\n{format_context(results)}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer using only the CONTEXT above, with [C#] citations.")


def load_api_key() -> str | None:
    """Gemini key lookup order: GEMINI_API_KEY / GOOGLE_API_KEY env vars -> Colab Secrets
    (name: GEMINI_API_KEY) -> a .env file next to this module (GEMINI_API_KEY=...)."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key.strip()
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key.strip()
    except Exception:  # not in Colab, secret missing, or notebook access not granted
        pass
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


class GeminiLLM:
    """Thin wrapper over the google-genai SDK. Reads GEMINI_API_KEY / GOOGLE_API_KEY."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL, api_key: str | None = None,
                 temperature: float = 0.1):
        from google import genai
        key = api_key or load_api_key()
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)")
        self.client, self.model, self.temperature = genai.Client(api_key=key), model, temperature

    def generate(self, system: str, user: str) -> str:
        from google.genai import types
        cfg = dict(system_instruction=system, max_output_tokens=2048)
        try:
            # Low temperature for deterministic, extractive answers. Gemini 3.6+ models no
            # longer accept sampling params, so retry without it if the API rejects it.
            resp = self.client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(temperature=self.temperature, **cfg))
        except Exception as e:  # noqa: BLE001
            if "temperature" not in str(e).lower():
                raise
            resp = self.client.models.generate_content(
                model=self.model, contents=user, config=types.GenerateContentConfig(**cfg))
        return (resp.text or "").strip()


@dataclass
class RAGAnswer:
    question: str
    answer: str
    results: list[RetrievedChunk]
    grounding: dict


def grounding_report(answer: str, results: list[RetrievedChunk]) -> dict:
    """Cheap, automatic evidence of grounding:
    - citations: which [C#] ids the answer cites and whether each was actually retrieved
    - numbers: every number / threshold in the answer must appear in the retrieved context
    - abstained: did the model use the not-found message"""
    ctx = " ".join(r.chunk.text for r in results)
    retrieved_ids = {r.chunk.id for r in results}
    cited = sorted(set(re.findall(r"\[(C\d+)\]", answer)), key=lambda s: int(s[1:]))
    ans_wo_cites = re.sub(r"\[C\d+\]", "", answer)
    # numbers keep an attached % so "3%" is not excused by an unrelated "3" in the context
    num_re = r"\d+(?:[.,]\d+)?%?"
    nums = set(re.findall(num_re, ans_wo_cites))
    ctx_nums = set(re.findall(num_re, ctx))
    unsupported = sorted(nums - ctx_nums)
    return {
        "cited_ids": cited,
        "invalid_citations": [c for c in cited if c not in retrieved_ids],
        "numbers_in_answer": len(nums),
        "numbers_not_in_context": unsupported,
        "abstained": NOT_FOUND_MSG.lower().rstrip(".") in answer.lower(),
        "grounded": (not unsupported) and all(c in retrieved_ids for c in cited)
                    and (bool(cited) or NOT_FOUND_MSG.lower().rstrip(".") in answer.lower()),
    }


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: GeminiLLM | None):
        self.retriever, self.llm = retriever, llm

    def ask(self, question: str, k: int = DEFAULT_TOP_K) -> RAGAnswer:
        results = self.retriever.retrieve(question, k)
        if self.llm is None:
            # No API key: return the best passage verbatim (extractive fallback).
            top = results[0].chunk if results else None
            answer = (f"(LLM not configured - top passage shown)\n\n{top.text} [{top.id}]"
                      if top else NOT_FOUND_MSG)
        else:
            answer = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, results))
        return RAGAnswer(question, answer, results, grounding_report(answer, results))


# --------------------------------------------------------------------------------------
# Retrieval evaluation (used for the chunk-size and k experiments)
# --------------------------------------------------------------------------------------
# Each query is labelled with a substring that must appear in the section title of a relevant
# chunk. Queries are paraphrased so they do not simply repeat the heading words.
EVAL_SET = [
    ("Users are getting token_expired 401 errors right after an auth deploy", "ERR-401"),
    ("How should a client back off when it gets HTTP 429?", "ERR-429"),
    ("A new release is throwing stack traces with 500s, what do I do?", "ERR-500"),
    ("The bank API is down and payment-gateway returns upstream connect errors", "ERR-503"),
    ("Requests time out after 30 seconds and p95 latency is above 3 seconds", "ERR-504"),
    ("Writes fail with 'try restarting transaction'", "DB-1205"),
    ("timeout waiting for connection from pool on order-api", "DB-POOL"),
    ("Pods keep restarting with exit code 137", "MEM-OOM"),
    ("No space left on device and OpenSearch indices went read-only", "DISK-FULL"),
    ("SSL handshake failed when calling auth.contoso.internal", "CERT-EXP"),
    ("How quickly must a Sev2 be acknowledged and who do we escalate to?", "Severity matrix"),
    ("Which team owns inventory-service and what datastore does it use?", "Service map"),
    ("When must an incident commander be assigned?", "Escalation procedure"),
    ("What is the MTTR target for Sev2 this quarter?", "Metrics we track"),
    ("What should I do in the first five minutes after being paged?", "Standard first five minutes"),
]


def evaluate_retrieval(retriever: Retriever, ks=(1, 3, 5, 7), eval_set=EVAL_SET) -> dict:
    """hit@k = share of queries with a relevant chunk in the top k; MRR = mean 1/rank of the
    first relevant chunk (searching to depth max(ks)). Also reports avg context tokens at each k
    - the cost side of choosing a larger k."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}; rr = 0.0; tok = {k: 0 for k in ks}; misses = []
    for q, label in eval_set:
        res = retriever.retrieve(q, max_k)
        rank = next((r.rank for r in res if label.lower() in r.chunk.section.lower()
                     or label.lower() in r.chunk.text.lower()), None)
        rr += 1 / rank if rank else 0
        for k in ks:
            hits[k] += int(bool(rank and rank <= k))
            tok[k] += sum(r.chunk.n_tokens for r in res[:k])
        if not rank or rank > 1:
            misses.append((q, label, rank))
    n = len(eval_set)
    out = {f"hit@{k}": round(hits[k] / n, 3) for k in ks}
    out["MRR"] = round(rr / n, 3)
    out.update({f"ctx_tokens@{k}": round(tok[k] / n) for k in ks})
    out["not_rank1"] = misses
    return out
