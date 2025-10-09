import numpy as np
from typing import List, Tuple, Optional
from chat.models import DocumentChunk
from chat.embedding_utils import embed_text, cosine_similarity


def maximal_marginal_relevance(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    lambda_param: float = 0.5,
    top_k: int = 3
) -> List[int]:
    """
    Maximal Marginal Relevance algorithm to select diverse results.
    
    Args:
        query_embedding: The query embedding vector
        candidate_embeddings: List of candidate chunk embeddings
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0). Default 0.5.
        top_k: Number of results to return
        
    Returns:
        List of indices of selected candidates
    """
    if not candidate_embeddings:
        return []
    
    selected_indices = []
    candidate_indices = list(range(len(candidate_embeddings)))
    
    # Calculate relevance scores for all candidates
    relevance_scores = [
        cosine_similarity(query_embedding, emb) 
        for emb in candidate_embeddings
    ]
    
    # First selection: most relevant
    first_idx = max(candidate_indices, key=lambda i: relevance_scores[i])
    selected_indices.append(first_idx)
    candidate_indices.remove(first_idx)
    
    # Iteratively select remaining items
    while len(selected_indices) < top_k and candidate_indices:
        mmr_scores = []
        
        for idx in candidate_indices:
            # Relevance to query
            relevance = relevance_scores[idx]
            
            # Maximum similarity to already selected items
            max_sim_to_selected = max([
                cosine_similarity(
                    candidate_embeddings[idx],
                    candidate_embeddings[selected_idx]
                )
                for selected_idx in selected_indices
            ])
            
            # MMR score: balance relevance and diversity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            mmr_scores.append((mmr_score, idx))
        
        # Select item with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
    
    return selected_indices


def search(
    query: str,
    top_k: int = 3,
    use_mmr: bool = True,
    lambda_param: float = 0.5,
    document_ids: Optional[List[str]] = None,
    session_id: Optional[str] = None
) -> List[Tuple[float, DocumentChunk]]:
    """
    Advanced search with MMR and filtering options.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        use_mmr: Whether to apply MMR for diversity (default True)
        lambda_param: MMR trade-off parameter (0=diversity, 1=relevance)
        document_ids: Optional list of document IDs to filter by
        session_id: Optional session ID to filter documents by session
        
    Returns:
        List of (score, chunk) tuples ordered by relevance/MMR
    """
    query_emb = embed_text(query)
    
    # Get all chunks with embeddings
    chunks_query = DocumentChunk.objects.filter(embedding__isnull=False)
    
    # Filter by session if provided
    if session_id:
        chunks_query = chunks_query.filter(document__session_id=session_id)
    
    # Filter by document IDs if provided
    if document_ids:
        chunks_query = chunks_query.filter(document_id__in=document_ids)
    
    chunks = list(chunks_query)
    
    if not chunks:
        return []
    
    # Calculate initial relevance scores
    scored = [
        (cosine_similarity(query_emb, chunk.embedding), chunk)
        for chunk in chunks
    ]
    
    if not use_mmr:
        # Simple top-k by relevance
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
    
    # Apply MMR
    embeddings = [chunk.embedding for chunk in chunks]
    selected_indices = maximal_marginal_relevance(
        query_embedding=query_emb,
        candidate_embeddings=embeddings,
        lambda_param=lambda_param,
        top_k=min(top_k, len(chunks))
    )
    
    # Return selected chunks with their scores
    results = [(scored[idx][0], chunks[idx]) for idx in selected_indices]
    
    return results
