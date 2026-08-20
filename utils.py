from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF file and returns extracted text as a single string.
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Splits text into overlapping chunks.
    chunk_size and overlap are in characters for simplicity.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks