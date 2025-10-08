from chat.models import DocumentChunk
from chat.embedding_utils import embed_text, cosine_similarity

def search_similar_chunks(query: str, top_k: int = 3):
    query_emb = embed_text(query)
    scored = []
    for ch in DocumentChunk.objects.all():
        if not ch.embedding:
            continue
        score = cosine_similarity(query_emb, ch.embedding)
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

