import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

SHOW_SOURCES_TEXT = False

INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

def build_prompt(question: str, contexts: list[dict]) -> str:
    # contexts is a list of {"chunk_id": int, "text": str}
    formatted = []
    for c in contexts:
        formatted.append(f"[Source chunk_id={c['chunk_id']}]\n{c['text']}")
    context_block = "\n\n---\n\n".join(formatted)

    return f"""
You are a helpful assistant answering questions using ONLY the provided sources.
If the answer is not in the sources, say: "I don't know based on the provided document."

Rules:
- Use ONLY information from the sources below.
- At the end of your answer, add: Sources: chunk_id=..., chunk_id=...
- Only cite chunk IDs that actually support your answer.

Sources:
{context_block}

Question:
{question}

Answer (concise, grounded):
""".strip()

def main():
    MODEL_NAME = "models/gemini-2.5-flash"

    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")

    # Initialize Gemini client
    client = genai.Client(api_key=gemini_key)

    # Load FAISS index and metadata
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("RAG Q&A (Gemini) ready. Type a question (or 'exit').\n")

    while True:
        question = input("Q: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        # Embed query
        qvec = embedder.encode([question]).astype("float32")
        faiss.normalize_L2(qvec)

        # Retrieve top chunks
        top_k = 4
        scores, ids = index.search(qvec, top_k)

        contexts = []
        for idx in ids[0]:
            contexts.append({"chunk_id": idx, "text": meta[idx]["text"]})

        if SHOW_SOURCES_TEXT:
            print("\n--- Retrieved chunks ---")
            for c in contexts:
                preview = c["text"].replace("\n", " ")[:250]
                print(f"chunk_id={c['chunk_id']}: {preview} ...")
            print("--- End retrieved chunks ---\n")   

        # ---- Build RAG prompt (with context) ----
        prompt_rag = build_prompt(question, contexts)

        # ---- Build baseline prompt (no context) ----
        prompt_baseline = f"""
        You are a helpful assistant. Answer the question concisely using your general knowledge.
        If you do not know the answer, say: "I don't know."

        Question:
        {question}

        Answer (concise):
        """.strip()


        # ---- Baseline Answer ----
        baseline_resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_baseline
        )
        baseline_answer = baseline_resp.text.strip()

        # ---- RAG Answer ----
        rag_resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_rag
        )
        rag_answer = rag_resp.text.strip()

        print("\n=== Baseline (no context) ===\n")
        print(baseline_answer)

        print("\n=== RAG (with retrieved sources) ===\n")
        print(rag_answer)
        print("\n")
if __name__ == "__main__":
    main()