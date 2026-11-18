"""Generates RAG_Runbook_Assistant.ipynb (kept as a script so the notebook is reproducible)."""
import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                               "outputs": [], "source": s.strip("\n")})

md(r"""
# Assignment 6 — Retrieval-Augmented Generation: Runbook Knowledge Assistant

**Corpus:** *Contoso Platform Troubleshooting Runbook v3.2* (7-page PDF, synthetic demo data).
The assignment brief suggests an ML textbook; the same pipeline is applied here to an operational
runbook. Every step is corpus-agnostic — point `PDF_PATH` at any PDF to re-run it.

| Part | What this notebook does |
|---|---|
| 1 | Load the PDF, extract all pages (text **and tables**), inspect quality issues, clean, chunk, compare chunking strategies |
| 2 | Embed chunks with SentenceTransformers (`BAAI/bge-small-en-v1.5`), store in FAISS, run a sample similarity search |
| 3 | Query → embedding → top-k; measure hit@k / MRR / context cost for k = 1, 3, 5, 7 |
| 4 | Grounded prompting with Gemini; sample answers; automatic grounding checks; ablation against an ungrounded prompt |
| 5 | Interactive Q&A loop here, plus the Streamlit app in `app.py` |

All reusable logic lives in `rag_core.py`, which the Streamlit app also imports, so the notebook and
the app run the same code.
""")

md(r"""
### Setup: Google Colab or local
The next cell works in both. In **Colab** it:
1. installs the packages Colab doesn't already have (`pypdf`, `pdfplumber`, `faiss-cpu`, `google-genai`);
2. finds the project folder: `MyDrive/rag_runbook` on Google Drive (it mounts Drive for you), or
   `/content/rag_runbook`. If it can't find either, it asks you to upload `rag_runbook.zip`;
3. loads the Gemini key from Colab **Secrets** (`GEMINI_API_KEY`), or from the `.env` file in the
   project folder.

A CPU runtime is enough.
""")
code(r"""
import os, sys, subprocess, importlib.util
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    mods = {"pypdf": "pypdf", "pdfplumber": "pdfplumber", "faiss-cpu": "faiss", "google-genai": "google.genai",
            "sentence-transformers": "sentence_transformers"}
    def missing(m):
        try:
            return importlib.util.find_spec(m) is None
        except ModuleNotFoundError:
            return True
    need = [pkg for pkg, mod in mods.items() if missing(mod)]
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)

    def find_project():
        for d in ["/content/drive/MyDrive/rag_runbook", "/content/rag_runbook", "/content/rag_runbook/rag_runbook", os.getcwd()]:
            if os.path.exists(os.path.join(d, "rag_core.py")):
                return d
        return None

    proj = find_project()
    if proj is None:
        from google.colab import drive
        drive.mount("/content/drive")
        proj = find_project()
    if proj is None:
        from google.colab import files
        print("Project folder not found - upload rag_runbook.zip")
        up = files.upload()
        for name in up:
            if name.endswith(".zip"):
                subprocess.run(["unzip", "-q", "-o", name, "-d", "/content/rag_runbook"], check=True)
        proj = find_project()
    assert proj, "rag_core.py not found - check the folder name"
    os.chdir(proj)
    sys.path.insert(0, proj)
    print("Project folder:", proj)
else:
    pass  # local run: start Jupyter from the rag_runbook folder; pip install -r requirements.txt once
""")

code(r"""
import getpass, re, textwrap
import pandas as pd
import rag_core as R

pd.set_option("display.max_colwidth", 120)
PDF_PATH = "data/troubleshooting_runbook.pdf"

key = R.load_api_key()          # env var -> Colab Secret -> .env file
if not key:
    key = getpass.getpass("Gemini API key (leave blank to skip generation): ")
if key:
    os.environ["GEMINI_API_KEY"] = key
print("Gemini key:", "found" if key else "missing (extractive fallback)",
      "| LLM model:", R.DEFAULT_LLM_MODEL, "| embedding model:", R.DEFAULT_EMBED_MODEL)
""")

# ---------------------------------------------------------------------------------- Part 1
md(r"""
---
## Part 1 — Data understanding & preprocessing

### 1.1 Load the PDF and extract text from every page
`load_pdf` uses **pdfplumber**: it finds tables on each page, serialises each table row as
`Header: value | Header: value`, and extracts the text above/below the tables in reading order.
A plain `pypdf` extraction is kept alongside for comparison.
""")
code(r"""
pages = R.load_pdf(PDF_PATH)
print("Number of pages:", len(pages))
pd.DataFrame([{"page": p.number, "tables": p.n_tables, "raw_words": len(p.raw_text.split()),
               "clean_words": len(p.text.split()), "first_line": p.text.splitlines()[0]} for p in pages])
""")

md(r"""
### 1.2 Explore structure and text-quality issues
""")
code(r"""
R.text_quality_report(pages)
""")
code(r"""
print("RAW pypdf output, page 1 (service-map table):\n")
raw = pages[0].raw_text
i = raw.find("3. Service map")
print(raw[i:i+330])
print("\n" + "="*90 + "\nCLEANED output, page 1:\n")
c = pages[0].text
j = c.find("3. Service map")
print(c[j:j+520])
""")
md(r"""
**Quality issues found and how they are handled**

| Issue | Evidence | Treatment |
|---|---|---|
| Tables flatten to one cell per line | ~40% of raw lines have ≤ 3 words; `Team Phoenix` ends up on a line with no link to `payment-gateway` | pdfplumber table detection → one self-describing line per row (`Service: payment-gateway \| Owning team: Team Phoenix \| …`) |
| Running header/footer on every page | `Contoso Platform Engineering \| … Page N`, 7 times | Regex removed — otherwise it is the most frequent "content" in every chunk and pulls unrelated chunks together in embedding space |
| Page group banners (`Database Errors`, `Appendix: …`) | First line of pages 2–7 | Dropped during section split so they don't leak into the previous section |
| Hard line wraps mid-sentence | PDF line breaks every ~100 chars | Kept inside chunks: commands and SQL rely on them; sentences are re-joined for token counting |
| Code / SQL blocks | Section 8, ERR-500, DB-1205, CERT-EXP | Line breaks preserved so commands can be copied verbatim |
| Inconsistent cross-references in the source | §2 says the severity matrix is "Section 3" (it is §5); ERR-504 points to "Section 5 database errors" (it is page 5) | Left as-is (we don't rewrite the source); the prompt tells the LLM to surface contradictions |
| Encoding | 0 non-ASCII characters after cleaning | None needed |
""")

md(r"""
### 1.3 Document structure — sections
The runbook is strongly structured: 8 numbered sections, 10 error-code entries and 4 appendix
sections. Every error-code entry follows the same template (*Symptom → Likely causes → Diagnostics
→ Fix → Prevention*), so a section is the natural retrieval unit.
""")
code(r"""
secs = R.split_sections(pages)
count = R.get_token_counter()
pd.DataFrame([{"page": s.page_start, "section": s.title, "code": s.code,
               "tokens": count(" ".join(l for _, l in s.lines))} for s in secs])
""")

md(r"""
### 1.4 Chunking — strategies compared
Two strategies:

* **fixed** — the textbook approach: one sliding token window over the whole document.
* **section** — split on headings first, then window *inside* a section if it is too long; each
  chunk is prefixed with `[Section: <title>]` so that a chunk about "Fix: restore the previous TTL"
  still says it belongs to ERR-401.

Tokens are counted with the embedding model's own WordPiece tokenizer, because the embedder is what
truncates long inputs (bge-small reads at most **512** tokens; anything beyond is silently dropped).

**Why not the suggested 500–1000 tokens?** The whole runbook is only ~3,500 tokens. With 800-token
chunks the index has ~5 chunks, each mixing four or five unrelated error codes; top-5 would return the
entire document, so retrieval stops doing any work, and an 800-token chunk would also be truncated by
the embedder. The table below shows the trade-off; the retrieval experiment in Part 3 makes the
decision.
""")
code(r"""
heading_re = re.compile(r"(?m)^(?:\[Section: )?((?:ERR|DB|MEM|DISK|CERT)-[A-Z0-9]+ - |\d\. [A-Z]|[A-D]\. [A-Z])")
configs = [("fixed", 800, 100), ("fixed", 500, 100), ("fixed", 300, 60),
           ("section", 500, 80), ("section", 300, 60), ("section", 200, 40)]
rows = []
for strat, size, ov in configs:
    ch = R.chunk_document(pages, size, ov, strat, count)
    n_tok = [c.n_tokens for c in ch]
    rows.append({"strategy": strat, "chunk_tokens": size, "overlap": ov, "n_chunks": len(ch),
                 "avg_tokens": round(sum(n_tok)/len(ch)), "max_tokens": max(n_tok),
                 "avg_headings_per_chunk": round(sum(len(heading_re.findall(c.text)) for c in ch)/len(ch), 2)})
pd.DataFrame(rows)
""")
code(r"""
chunks = R.chunk_document(pages, R.DEFAULT_CHUNK_TOKENS, R.DEFAULT_OVERLAP_TOKENS, "section", count)
print(f"Chosen: section strategy, {R.DEFAULT_CHUNK_TOKENS} tokens, {R.DEFAULT_OVERLAP_TOKENS} overlap -> {len(chunks)} chunks\n")
for c in chunks[9:11]:
    print(f"--- {c.id} | pages {c.pages} | {c.n_tokens} tokens\n{c.text}\n")
""")

# ---------------------------------------------------------------------------------- Part 2
md(r"""
---
## Part 2 — Embeddings & vector database

### 2.1 Embedding model
**`BAAI/bge-small-en-v1.5`** (SentenceTransformers)

* 33M parameters, **384-dimensional** vectors, 512-token input window — runs on CPU in seconds for
  this corpus, no API cost, and the corpus never leaves the machine (sensible for an internal runbook).
* Trained for asymmetric retrieval (short question → longer passage). Queries get the recommended
  instruction prefix *"Represent this sentence for searching relevant passages: "*; passages are
  embedded as-is.
* Strong on the MTEB retrieval benchmark for its size; a larger model (`bge-base`, 768-d) or Gemini
  `gemini-embedding-001` can be swapped in through `R.Embedder(model_name=...)`.
* Vectors are **L2-normalised**, so inner product equals cosine similarity.

Each chunk is embedded *with* its `[Section: …]` prefix, which adds the error code and title to the
vector — important because the body of a chunk ("Restore the previous TTL…") often doesn't name the
error it belongs to.
""")
code(r"""
embedder = R.Embedder(R.DEFAULT_EMBED_MODEL)
vectors = embedder.embed_documents([c.text for c in chunks])
print("Embedding matrix:", vectors.shape, "| dtype:", vectors.dtype,
      "| norms ~ 1:", vectors[:3].__pow__(2).sum(1).round(3))
""")

md(r"""
### 2.2 Vector storage
**FAISS `IndexFlatIP`** — exact inner-product search over normalised vectors (= cosine).

* With a few dozen to a few thousand chunks, exact search costs microseconds, so approximate indexes
  (IVF, HNSW) would add tuning and recall loss for no gain. For a whole book (~2–5k chunks) Flat is
  still fine; beyond ~10⁵ chunks switch to `IndexHNSWFlat`.
* FAISS stores only vectors; chunk text and metadata (id, section, error code, pages, token count)
  are stored in a parallel list and saved to `index/chunks.json`. Row *i* of the index ↔ chunk *i*.
* `index/meta.json` records the PDF hash, embedding model and chunk parameters; the app re-uses the
  saved index only if all of them match, otherwise it rebuilds.
""")
code(r"""
store = R.VectorStore(vectors.shape[1])
store.add(vectors, chunks)
store.save("index", {"pdf_sha1": R.file_sha1(PDF_PATH), "embed_model": embedder.model_name,
                     "backend": "st", "chunk_tokens": R.DEFAULT_CHUNK_TOKENS,
                     "overlap_tokens": R.DEFAULT_OVERLAP_TOKENS, "strategy": "section",
                     "dim": int(vectors.shape[1]), "n_chunks": len(chunks)})
retriever = R.Retriever(embedder, store)
print("Backend:", store.backend, "| vectors:", len(store.chunks), "| saved to ./index")
""")

md(r"""
### 2.3 Similarity search — retrieved chunks for a sample query
""")
code(r"""
def show(results, width=110, chars=260):
    for r in results:
        body = r.chunk.text.split("\n", 1)[-1].replace("\n", " ")
        print(f"#{r.rank}  score={r.score:.3f}  {r.chunk.id}  {r.chunk.citation()}")
        print(textwrap.indent(textwrap.fill(body[:chars] + "…", width), "     "), "\n")

q = "Pods keep restarting and describe shows exit code 137. What should I check?"
print("QUERY:", q, "\n")
show(retriever.retrieve(q, k=3))
""")

# ---------------------------------------------------------------------------------- Part 3
md(r"""
---
## Part 3 — Retrieval pipeline

`Retriever.retrieve(query, k)` = accept query → embed with the query prefix → FAISS top-k → ranked
chunks with cosine scores.

### 3.1 Evaluation set
15 hand-labelled questions, one per error code plus five process/table questions. They are
**paraphrased**, using symptom words (*exit code 137*, *try restarting transaction*) rather than
heading words (*OOMKilled*, *Lock wait timeout*), so the test measures semantic matching, not
keyword overlap.

Metrics: **hit@k** (is a chunk from the correct section in the top k), **MRR** (mean 1/rank of the
first correct chunk) and **ctx_tokens@k** (how much context the LLM has to read — cost, latency,
and more room for distraction).
""")
code(r"""
pd.DataFrame(R.EVAL_SET, columns=["query", "expected section"])
""")
code(r"""
rows = []
for strat, size, ov in configs:
    r_, _ = R.build_index(PDF_PATH, embedder, size, ov, strat)        # same embedder, re-chunked
    m = R.evaluate_retrieval(r_, ks=(1, 3, 5, 7)); m.pop("not_rank1")
    rows.append({"strategy": strat, "chunk": size, "overlap": ov, "n_chunks": len(r_.store.chunks), **m})
exp = pd.DataFrame(rows)
exp
""")
md(r"""
**Reading the table**

* Large fixed chunks score well on hit@k almost by construction: with ~5 chunks, any chunk has a 1-in-5
  chance of being "right", and at k = 5 the LLM receives the whole document (see `ctx_tokens@5`).
  That is not retrieval; it is stuffing the whole document into the prompt, which stops working on a
  real book.
* Section-aware chunks put the correct section at rank 1 most often (MRR) while sending a fraction
  of the tokens; the section prefix is what fixes the queries that fixed windows miss.
* Shrinking below ~200 tokens splits *Diagnostics* from *Fix*, so a question like "how do I fix X"
  can retrieve the right section but the wrong half.

**Choice:** `section`, 300 tokens, 60 overlap — the best MRR / cost balance on this run (re-check
after running; the cells above recompute everything).
""")
code(r"""
best = R.evaluate_retrieval(retriever, ks=(1, 3, 5, 7))
print({k: v for k, v in best.items() if k != "not_rank1"})
print("\nQueries where the right section was NOT rank 1:")
for q_, label, rank in best["not_rank1"]:
    print(f"  rank={rank}  expected={label:<28} {q_}")
""")

md(r"""
### 3.2 Experimenting with k = 3, 5, 7
One focused query and one broad query, to see what extra depth buys.
""")
code(r"""
for q in ["What is the fix for a lock wait timeout?",
          "Payments are slow and timing out after the last migration - where do I start?"]:
    print("="*100, "\nQUERY:", q)
    for k in (3, 5, 7):
        res = retriever.retrieve(q, k)
        print(f"\n k={k}: " + " | ".join(f"{r.chunk.id}:{r.chunk.section.split(' - ')[0]}({r.score:.2f})" for r in res))
""")
md(r"""
**Observations on k**

* For a focused question the answer is in the top 1–2 chunks; k = 7 only adds low-scoring, unrelated
  sections (notice the score drop after the first few ranks).
* Broad, multi-hop questions (*slow payments after a migration*) need several sections, e.g.
  ERR-504 + DB-1205 + the rollback note in §8, so k = 3 can be too shallow.
* **k = 5 is the default**: hit@5 is at or near the ceiling and context stays under ~1k tokens. The
  grounded prompt makes the extra low-relevance chunks harmless, because the model is told to ignore
  passages that don't answer the question. The app exposes k as a slider.
""")

# ---------------------------------------------------------------------------------- Part 4
md(r"""
---
## Part 4 — Answer generation (RAG)

### 4.1 Prompt design
The prompt is split into a **system instruction** (fixed rules) and a **user turn** (context +
question).
""")
code(r"""
print(R.SYSTEM_PROMPT)
print("\n" + "-"*90 + "\nUSER TURN TEMPLATE (abridged):\n")
print(R.build_user_prompt("<question>", retriever.retrieve("token expired", 2))[:700] + "\n…")
""")
md(r"""
| Design choice | Why |
|---|---|
| Rules in the *system instruction*, data in the *user turn* | The model treats the rules as higher priority than anything in the retrieved text |
| "Use only facts stated in the CONTEXT … even if you are confident" | Runbook advice is specific (TTL **900 s**, backoff **200 ms → 30 s**); generic DevOps knowledge from training data is the main hallucination risk |
| Passage ids `[C#]` + section + page in every block; citation required per sentence | Makes every claim traceable and lets us **check grounding automatically** |
| One exact abstention string | Makes "I don't know" testable, and stops the model from writing around a gap |
| Partial-answer rule | Several entries are incomplete (ERR-503 and DISK-FULL have no *Likely causes*); the model should say so instead of filling the gap |
| "Copy commands and numbers exactly", code blocks | On-call engineers paste these into a terminal |
| Contradiction / false-premise rule | The source has inconsistent cross-references, and users ask leading questions |
| Low temperature (0.1) | Extractive task; creativity is a liability. (Newer Gemini models ignore sampling params; the wrapper falls back automatically.) |

### 4.2 Automatic grounding checks
`R.grounding_report` runs on every answer:

* **cited_ids / invalid_citations** — every `[C#]` must be one of the retrieved passages.
* **numbers_not_in_context** — every number/percentage in the answer must appear verbatim in the
  retrieved context (a cheap, effective hallucination detector for operational content).
* **abstained** — whether the model used the not-found message.
""")
code(r"""
try:
    llm = R.GeminiLLM()
except Exception as e:
    llm = None
    print("Gemini not configured -> extractive fallback.", e)
rag = R.RAGPipeline(retriever, llm)

def ask(q, k=5, show_ctx=False):
    a = rag.ask(q, k)
    print("Q:", q, "\n")
    print(a.answer, "\n")
    print("Sources:", ", ".join(f"[{r.chunk.id}] {r.chunk.citation()} ({r.score:.2f})" for r in a.results))
    g = a.grounding
    print(f"Grounding: cited={g['cited_ids']} invalid={g['invalid_citations']} "
          f"unsupported_numbers={g['numbers_not_in_context']} abstained={g['abstained']} -> "
          f"{'GROUNDED' if g['grounded'] else 'CHECK'}")
    if show_ctx:
        print("\n--- context passed to the LLM ---\n" + R.format_context(a.results))
    print("="*100)
    return a
""")

md(r"""
### 4.3 Sample queries and generated answers
The set covers direct lookups, commands, numeric thresholds, multi-section reasoning, an
incomplete entry, a false premise, a source contradiction, and an out-of-scope question.
""")
code(r"""
SAMPLE_QUERIES = [
    # direct lookup + numbers
    "What is the default access token TTL and how do I force a JWKS refresh?",
    # command reproduction
    "Give me the SQL to find which session is blocking others for DB-1205.",
    # numeric rule / formula
    "How should I size the connection pool relative to max_connections?",
    # multi-section reasoning (severity matrix + escalation procedure + service map)
    "payment-gateway is fully down. Which team owns it, how fast must we ack, and when is an incident commander needed?",
    # cross-section: ERR-504 history + rollback/migration note
    "We saw ERR-504 on payment-gateway after a migration. What was the cause last time, and can I just roll back?",
    # incomplete entry -> partial answer expected
    "What are the likely causes of ERR-503?",
    # false premise
    "Why does the runbook recommend restarting pods repeatedly when ERR-503 occurs?",
    # contradiction in the source
    "Where is the severity matrix in this runbook?",
    # out of scope -> abstain
    "How do I configure Kafka consumer group rebalancing?",
]
answers = [ask(q) for q in SAMPLE_QUERIES]
""")

md(r"""
### 4.4 Evidence of grounding
**(a) Summary of automatic checks across all sample answers**
""")
code(r"""
pd.DataFrame([{"question": a.question[:70], "cited": ",".join(a.grounding["cited_ids"]),
               "invalid_cites": len(a.grounding["invalid_citations"]),
               "unsupported_numbers": ",".join(a.grounding["numbers_not_in_context"]),
               "abstained": a.grounding["abstained"], "grounded": a.grounding["grounded"]} for a in answers])
""")
md(r"""
**(b) Side-by-side: answer vs. the exact passage it cites**: for one answer, print the cited
chunk so the claims can be checked line by line.
""")
code(r"""
a = answers[0]
print("ANSWER:\n" + a.answer + "\n")
by_id = {r.chunk.id: r.chunk for r in a.results}
for cid in a.grounding["cited_ids"]:
    ch = by_id.get(cid)
    if ch is None:
        print(f"--- {cid}: cited but NOT among the retrieved passages (invalid citation) ---\n")
        continue
    print(f"--- cited passage {cid} ({ch.citation()}) ---\n{ch.text}\n")
""")
md(r"""
**(c) Ablation: grounded prompt vs. an ungrounded prompt.** Same model, same retrieved context; the
only change is the instructions. The ungrounded version typically answers the out-of-scope Kafka
question from general knowledge, and pads the ERR-503 answer with plausible causes the runbook
never lists. The grounded version abstains or flags the gap.
""")
code(r"""
NAIVE_SYSTEM = "You are a helpful DevOps assistant. Use the context if helpful."
if llm:
    for q in ["How do I configure Kafka consumer group rebalancing?", "What are the likely causes of ERR-503?"]:
        res = retriever.retrieve(q, 5)
        print("Q:", q)
        print("\n[UNGROUNDED]\n" + llm.generate(NAIVE_SYSTEM, R.build_user_prompt(q, res))[:900])
        print("\n[GROUNDED]\n" + llm.generate(R.SYSTEM_PROMPT, R.build_user_prompt(q, res)))
        print("="*100)
else:
    print("Set GEMINI_API_KEY to run the ablation.")
""")

# ---------------------------------------------------------------------------------- Part 5
md(r"""
---
## Part 5 — End-to-end RAG application

`R.build_index()` combines document processing, chunking, embedding and storage (with an on-disk
cache), and `R.RAGPipeline.ask()` combines retrieval and generation. The same two calls power:

1. **The interactive loop below**: type questions, empty line to stop.
2. **The Streamlit web app**: `streamlit run app.py`. It offers chat, a k slider, chunk settings,
   source passages with scores and page numbers, the grounding badge for each answer, and PDF upload
   to index a different document.
""")
code(r"""
retriever_e2e, _ = R.build_index(PDF_PATH, embedder, cache_dir="index")   # loads the cached index
app = R.RAGPipeline(retriever_e2e, llm)

while True:
    q = input("\nAsk the runbook (blank to quit): ").strip()
    if not q:
        break
    a = app.ask(q, k=5)
    print("\n" + a.answer)
    print("\nSources:", "; ".join(r.chunk.citation() for r in a.results[:3]),
          "| grounded:", a.grounding["grounded"])
""")

md(r"""
### Optional: run the Streamlit app from Colab
Colab can't open `localhost`, so the app goes through a public tunnel. Run the cell, copy the IP it
prints, open the `https://….loca.lt` link, and paste the IP as the tunnel password. Stop the cell
to shut the app down.
""")
code(r"""
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit"], check=True)
    log = open("/content/streamlit.log", "w")
    subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
                     stdout=log, stderr=log, env={**os.environ})
    ip = subprocess.run(["curl", "-s", "ipv4.icanhazip.com"], capture_output=True, text=True).stdout.strip()
    print("Tunnel password (your Colab IP):", ip)
    get_ipython().system("npx -y localtunnel --port 8501")      # prints the https://....loca.lt URL
else:
    print("Locally, run in a terminal:  streamlit run app.py")
""")

md(r"""
---
## Summary

* **Preprocessing mattered more than model choice.** Table-aware extraction and footer removal
  turned ~40% fragment lines into self-describing rows; section-aware chunking with title prefixes
  raised rank-1 accuracy compared with fixed windows.
* **Chunk size has to fit the corpus.** On a ~3.5k-token runbook, ~300-token, section-aligned chunks
  beat the generic 500–1000 guidance. For a full textbook, the same code with `chunk_tokens≈500`
  and the section split on chapter/heading patterns would be the starting point.
* **k = 5** balances recall for multi-section questions against context size.
* **Grounding** is enforced by the prompt (context-only, mandatory citations, exact abstention,
  partial-answer and contradiction rules) and **verified** automatically (citation validity +
  numeric support). The ablation shows what the instructions prevent.

**Limitations / next steps:** hybrid BM25 + dense retrieval for exact error-string queries, a
cross-encoder re-ranker (`bge-reranker-base`), an LLM-as-judge faithfulness score (e.g. RAGAS), and
conversation memory for follow-up questions.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("RAG_Runbook_Assistant.ipynb", "w"), indent=1)
print("cells:", len(cells))
