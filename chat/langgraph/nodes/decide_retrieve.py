"""
DecideRetrieve Node - Decides whether retrieval is needed for the query.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def decide_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide if retrieval is needed based on user message.
    
    Uses simple heuristics:
    - Message contains question words
    - Message is long enough to need context
    - Message references factual information
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with need_retrieval flag
    """
    message = state['last_user_msg'].lower()
    
    # Heuristic 1: Contains question indicators
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'explain', 'tell', 'describe']
    has_question = any(word in message for word in question_words) or '?' in message
    
    # Heuristic 2: References documents or sources
    reference_words = ['document', 'file', 'source', 'according to', 'based on']
    references_docs = any(word in message for word in reference_words)
    
    # Heuristic 3: Long message (likely needs detailed answer)
    is_long = len(message.split()) > 10
    
    # Heuristic 4: Not a simple greeting or acknowledgment
    simple_responses = ['hi', 'hello', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no']
    is_simple = message.strip() in simple_responses
    
    # Decision: retrieve if question, references, or long (but not if simple)
    need_retrieval = (has_question or references_docs or is_long) and not is_simple
    
    logger.info(f"DecideRetrieve: need_retrieval={need_retrieval} for message: {message[:50]}...")
    
    return {
        **state,
        'need_retrieval': need_retrieval
    }

