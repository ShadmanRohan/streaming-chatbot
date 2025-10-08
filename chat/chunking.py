import textwrap

def chunk_text(text: str, max_len: int = 500):
    """Split text into roughly equal chunks."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        parts = textwrap.wrap(para, width=max_len)
        chunks.extend(parts)
    return chunks

