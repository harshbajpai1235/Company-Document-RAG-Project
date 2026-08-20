from sentence_transformers import SentenceTransformer
from utils import extract_text_from_pdf, chunk_text

# Small, fast, good-quality embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

text = extract_text_from_pdf("data/docs/api_file_1.pdf")
chunks = chunk_text(text, chunk_size=1200, overlap=200)

test_chunks = chunks[:2]
vectors = model.encode(test_chunks)

print("Number of test chunks:", len(test_chunks))
print("Embedding dimension:", vectors.shape[1])
print("First 5 numbers of embedding 1:", vectors[0][:5])
print("First 5 numbers of embedding 2:", vectors[1][:5])