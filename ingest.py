import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils import extract_text_from_pdf, chunk_text

PDF_PATH = "data/docs/api_file_1.pdf"
INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 1) Load + chunk
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text, chunk_size=1200, overlap=200)

    print("Total chunks:", len(chunks))

    # 2) Embed all chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)  # shape: (num_chunks, dim)
    vectors = np.array(vectors, dtype="float32")

    dim = vectors.shape[1]
    print("Embedding dimension:", dim)

    # 3) Build FAISS index (cosine similarity)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, FAISS_PATH)

    # 4) Save metadata so we can map IDs -> chunk text later
    meta = []
    for i, ch in enumerate(chunks):
        meta.append({"chunk_id": i, "text": ch})

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", FAISS_PATH)
    print("Saved:", META_PATH)

if __name__ == "__main__":
    main()