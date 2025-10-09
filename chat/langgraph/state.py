"""
LangGraph state schema for chat orchestration.
"""
from typing import TypedDict, List, Optional, Dict, Any


class GraphState(TypedDict):
    """
    State passed between LangGraph nodes.
    """
    # Session & input
    session_id: str
    last_user_msg: str
    model: str
    
    # Context loading
    history: List[Dict[str, str]]  # [{role: str, content: str}, ...]
    summary: Optional[str]
    
    # Retrieval decision & results
    need_retrieval: bool
    retrieved_chunks: List[Dict[str, Any]]  # [{text, score, chunk_id, document}, ...]
    
    # LLM synthesis
    draft: str  # Final response text
    
    # Metadata
    metadata: Dict[str, Any]  # {tokens_used, retrieval_count, etc.}
    error: Optional[str]
    
    # Configuration
    top_k: int
    use_mmr: bool
    lambda_param: float

