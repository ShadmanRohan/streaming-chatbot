"""
Retrieve Node - Performs RAG retrieval if needed.
"""
import logging
from typing import Dict, Any
from chat.retrieval import search

logger = logging.getLogger(__name__)


def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant document chunks using semantic search with MMR.
    Only runs if need_retrieval is True.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with retrieved_chunks
    """
    # Skip if retrieval not needed
    if not state.get('need_retrieval', False):
        logger.info("Skipping retrieval (not needed)")
        return {
            **state,
            'retrieved_chunks': []
        }
    
    try:
        # Use existing search function with MMR
        results = search(
            query=state['last_user_msg'],
            top_k=state.get('top_k', 3),
            use_mmr=state.get('use_mmr', True),
            lambda_param=state.get('lambda_param', 0.5)
        )
        
        # Format results
        retrieved_chunks = [
            {
                'text': chunk.text,
                'score': float(score),
                'chunk_id': str(chunk.id),
                'document': chunk.document.filename
            }
            for score, chunk in results
        ]
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        return {
            **state,
            'retrieved_chunks': retrieved_chunks
        }
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        # Continue without retrieval on error
        return {
            **state,
            'retrieved_chunks': [],
            'error': f"Retrieval error: {str(e)}"
        }

