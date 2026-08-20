# Company Document RAG Project

A local Retrieval-Augmented Generation (RAG) pipeline for question answering over company PDFs. Documents are chunked, embedded, and stored in a FAISS index. At query time the system retrieves the most relevant chunks and answers with Gemini, citing source chunk IDs.

The CLI also prints a **baseline** (no retrieved context) answer next to the **RAG** answer so you can see what grounding the documents adds.

**Author:** Harsh Bajpai

---

## How it works

```
PDF  →  extract text  →  overlapping chunks
                         ↓
              SentenceTransformer embeddings
                         ↓
                    FAISS index
                         ↓
     question → retrieve top-k chunks → Gemini prompt → cited answer
```

1. **Ingest** (`ingest.py`) reads `data/docs/api_file_1.pdf`, splits text into 1,200-character chunks with 200-character overlap, embeds them with `all-MiniLM-L6-v2`, and writes a cosine-similarity FAISS index plus chunk metadata.
2. **Query** (`query.py`) embeds the user question, retrieves the top 4 chunks, and asks Gemini (`gemini-2.5-flash`) to answer **only** from those sources. If the document does not contain the answer, the model is instructed to say it does not know.

A prebuilt index (`index/faiss.index`, `index/meta.json`) is included so you can query immediately after setup. Re-run ingest after you add or replace PDFs.

---

## Repository layout

| Path | Role |
| --- | --- |
| `ingest.py` | Build / rebuild the FAISS index from a PDF |
| `query.py` | Interactive Q&A (baseline vs RAG) |
| `utils.py` | PDF text extraction and chunking |
| `embed_test.py` | Quick embedding smoke test |
| `list_models.py` | List Gemini models available to your API key |
| `data/docs/` | Source PDFs (sample API document included) |
| `index/` | FAISS index and chunk metadata |
| `.env.example` | Template for `GEMINI_API_KEY` |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the env template and add a [Google Gemini API key](https://aistudio.google.com/apikey):

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

The first embedding run downloads `all-MiniLM-L6-v2` from Hugging Face (one-time).

---

## Usage

**Ask questions against the existing index:**

```bash
python query.py
```

Type a question, or `exit` to quit. Each question prints:

- **Baseline (no context)** — Gemini with general knowledge only
- **RAG (with retrieved sources)** — Gemini constrained to retrieved PDF chunks, with `Sources: chunk_id=...`

**Rebuild the index** (after changing the PDF):

```bash
python ingest.py
```

By default ingest reads `data/docs/api_file_1.pdf`. Put additional or replacement PDFs in `data/docs/` and point `PDF_PATH` in `ingest.py` at the file you want indexed.

**Optional helpers:**

```bash
python embed_test.py    # encode two chunks and print vector shape
python list_models.py   # list Gemini model names for your key
```

To print retrieved chunk previews while querying, set `SHOW_SOURCES_TEXT = True` in `query.py`.

---

## Configuration

| Setting | Where | Default |
| --- | --- | --- |
| Gemini model | `query.py` → `MODEL_NAME` | `models/gemini-2.5-flash` |
| Chunk size / overlap | `ingest.py` | 1200 / 200 characters |
| Retrieval depth | `query.py` → `top_k` | 4 |
| Embedding model | ingest + query | `all-MiniLM-L6-v2` (384-d) |
| Similarity | FAISS `IndexFlatIP` after L2-normalization | cosine |

---

## Sample document

The included PDF is API documentation used as a stand-in company knowledge base. After ingest it produces **14 chunks**. Ask questions that are actually in that document (for example, how the API generates text from a prompt) to see cited RAG answers; questions outside the file should yield “I don’t know based on the provided document.”

---

## Stack

Python, pypdf, NumPy, Sentence Transformers, FAISS, Google Gemini (`google-genai`), python-dotenv.

---

## Security

Do not commit `.env`. The GitHub repo is meant to contain code, the sample PDF, and the index — not API keys.
