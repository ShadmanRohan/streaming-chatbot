"""
LangGraph orchestration for chat flow.

This module defines the chat orchestration graph using LangGraph.
The graph coordinates: history loading, retrieval decision, RAG search,
LLM synthesis, and session summarization.
"""
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .state import GraphState
from .nodes import (
    load_history,
    decide_retrieve,
    retrieve,
    synthesize,
    summarize
)

logger = logging.getLogger(__name__)


def create_chat_graph() -> StateGraph:
    """
    Create and compile the chat orchestration graph.
    
    Flow:
        START → LoadHistory → DecideRetrieve → [Retrieve?] → Synthesize → Summarize → END
    
    Returns:
        Compiled LangGraph application
    """
    # Create graph
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("load_history", load_history)
    graph.add_node("decide_retrieve", decide_retrieve)
    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.add_node("summarize", summarize)
    
    # Define edges
    graph.set_entry_point("load_history")
    graph.add_edge("load_history", "decide_retrieve")
    
    # Conditional edge: retrieve only if needed
    def should_retrieve(state: Dict[str, Any]) -> str:
        """Route to retrieve node or skip to synthesize."""
        if state.get('need_retrieval', False):
            return "retrieve"
        return "synthesize"
    
    graph.add_conditional_edges(
        "decide_retrieve",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "synthesize": "synthesize"
        }
    )
    
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "summarize")
    graph.add_edge("summarize", END)
    
    # Compile
    return graph.compile()


# Create singleton graph instance
chat_graph = create_chat_graph()


def run_graph(
    session_id: str,
    user_message: str,
    model: str = 'gpt-4o-mini',
    top_k: int = 3,
    use_mmr: bool = True,
    lambda_param: float = 0.5
) -> Dict[str, Any]:
    """
    Run the chat graph orchestration.
    
    Args:
        session_id: Chat session UUID
        user_message: User's message
        model: LLM model to use
        top_k: Number of chunks to retrieve
        use_mmr: Use MMR for diversity
        lambda_param: MMR trade-off parameter
        
    Returns:
        Dict with response content, chunks, and metadata
    """
    # Initialize state
    initial_state = {
        'session_id': session_id,
        'last_user_msg': user_message,
        'model': model,
        'history': [],
        'summary': None,
        'need_retrieval': True,  # Will be decided by node
        'retrieved_chunks': [],
        'draft': '',
        'metadata': {},
        'error': None,
        'top_k': top_k,
        'use_mmr': use_mmr,
        'lambda_param': lambda_param
    }
    
    try:
        # Execute graph
        final_state = chat_graph.invoke(initial_state)
        
        # Check for errors
        if final_state.get('error'):
            raise Exception(final_state['error'])
        
        return {
            'content': final_state.get('draft', ''),
            'retrieved_chunks': final_state.get('retrieved_chunks', []),
            'metadata': final_state.get('metadata', {})
        }
        
    except Exception as e:
        logger.error(f"Graph execution error: {e}")
        raise

