"""
SynthesizeStream Node - Builds prompt and streams LLM response.
"""
import logging
from typing import Dict, Any, Generator
from chat.llm import stream_llm
from chat.prompts import SYSTEM_PROMPT, format_retrieved_chunks, sanitize_user_input

logger = logging.getLogger(__name__)


def synthesize_stream(state: Dict[str, Any]) -> Generator[str, None, Dict[str, Any]]:
    """
    Build the chat prompt and stream LLM response.
    
    Args:
        state: Current graph state with history, summary, retrieved_chunks, last_user_msg
        
    Yields:
        Text deltas from LLM
        
    Returns:
        Final state with accumulated draft response
    """
    try:
        # Build messages manually (same logic as build_simple_prompt but with history)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add summary if exists
        if state.get('summary'):
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {state['summary']}"
            })
        
        # Add retrieved chunks if any
        retrieved_chunks = state.get('retrieved_chunks', [])
        if retrieved_chunks:
            context = format_retrieved_chunks(retrieved_chunks)
            messages.append({
                "role": "system",
                "content": f"Relevant information:\n{context}"
            })
        
        # Add history
        for msg in state.get('history', []):
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": sanitize_user_input(state['last_user_msg'])
        })
        
        # Stream LLM response
        accumulated = ""
        for delta in stream_llm(
            messages=messages,
            model=state.get('model', 'gpt-4o-mini'),
            temperature=state.get('temperature', 0.7),
            max_tokens=state.get('max_tokens', 2000)
        ):
            accumulated += delta
            yield delta
        
        # Yield final state with accumulated response
        yield {
            **state,
            'draft': accumulated,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Synthesis streaming error: {e}")
        # On error, yield state with error
        yield {
            **state,
            'draft': '',
            'error': f"Synthesis error: {e}"
        }

