# Assignment 6: RAG Knowledge Assistant (Contoso Troubleshooting Runbook)

An end-to-end Retrieval-Augmented Generation system over `data/troubleshooting_runbook.pdf`.

| File | Purpose |
|---|---|
| `RAG_Runbook_Assistant.ipynb` | Parts 1–5 with explanations, experiments, sample answers and grounding evidence |
| `rag_core.py` | All pipeline logic (loading, cleaning, chunking, embeddings, FAISS store, retrieval, prompt, Gemini, grounding checks, evaluation) |
| `app.py` | Part 5 Streamlit app (chat UI, k slider, source passages, grounding badge, PDF upload) |
| `build_notebook.py` | Regenerates the notebook from source (optional) |
| `data/troubleshooting_runbook.pdf` | Corpus |
| `.env` | Your Gemini API key (`GEMINI_API_KEY=...`); don't share or submit it |

## Stack
* **Extraction:** pdfplumber (table-aware) + pypdf (raw comparison)
* **Chunking:** section-aware, 300 tokens / 60 overlap, measured with the embedder's tokenizer
* **Embeddings:** SentenceTransformers `BAAI/bge-small-en-v1.5` (384-d, normalised)
* **Vector DB:** FAISS `IndexFlatIP` (cosine), persisted to `index/`
* **LLM:** Gemini via `google-genai` (default `gemini-3.5-flash`, override with `GEMINI_MODEL`)

## Run in Google Colab
1. Upload the **`rag_runbook` folder** to Google Drive as `MyDrive/rag_runbook`
   (or keep `rag_runbook.zip` handy: the notebook asks for it if it can't find the folder).
2. Open `RAG_Runbook_Assistant.ipynb` in Colab (Drive → right-click → Open with → Google Colaboratory).
3. **Runtime → Run all.** The first code cell installs missing packages, mounts Drive, and moves into
   the project folder. The Gemini key is read from Colab Secrets (`GEMINI_API_KEY`) if set,
   otherwise from the `.env` file in the folder.
4. Optional: the last code cell in Part 5 starts the Streamlit app through a localtunnel link
   (password = the IP the cell prints).

> `.env` holds your Gemini key. Delete it (or replace it with a Colab Secret) before you submit or
> share the folder.

## Setup (Windows PowerShell)
```powershell
cd C:\Users\midhu\Downloads\ustassignments\rag_runbook
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# the key is read from .env automatically; or override:  $env:GEMINI_API_KEY = "your-key"
```
The first run downloads the bge-small model (~130 MB) from Hugging Face.

## Run
```powershell
jupyter notebook RAG_Runbook_Assistant.ipynb     # Parts 1-5; Run All
streamlit run app.py                              # web app at http://localhost:8501
```
Without an API key the system still runs: retrieval works and the answer box shows the top passage
(extractive fallback).

## Using a different PDF (e.g. the ML textbook)
Change `PDF_PATH` in the notebook or upload a PDF in the app sidebar. For a long book, raise the chunk size to
~500 tokens. The section splitter looks for runbook-style headings, so either extend
`_NUM_HEADING_RE` with chapter patterns or use `strategy="fixed"`.
