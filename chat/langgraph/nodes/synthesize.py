"""
Synthesize Node - Calls LLM to generate response.
"""
import logging
from typing import Dict, Any
from openai import OpenAIError, RateLimitError, AuthenticationError

from chat import llm as llm_module
from chat.prompts import build_chat_prompt
from chat.models import ChatMessage

logger = logging.getLogger(__name__)


def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build prompt and call LLM to generate response.
    
    Args:
        state: Current graph state with history and retrieved chunks
        
    Returns:
        Updated state with draft (LLM response) and metadata
    """
    try:
        # Convert history to ChatMessage-like objects for build_chat_prompt
        # build_chat_prompt expects ChatMessage objects, but we have dicts
        # So we'll build the messages array directly here
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant with access to a knowledge base. Answer questions accurately based on the provided context. If the context doesn't contain relevant information, say so clearly. Be concise but comprehensive."}
        ]
        
        # Add summary if exists
        if state.get('summary'):
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {state['summary']}"
            })
        
        # Add retrieved context
        if state.get('retrieved_chunks'):
            chunks_text = "\n\n".join([
                f"[Source {i+1}: {chunk.get('document', 'Unknown')} (relevance: {chunk.get('score', 0):.2f})]\n{chunk['text']}"
                for i, chunk in enumerate(state['retrieved_chunks'])
            ])
            messages.append({
                "role": "system",
                "content": f"Relevant information from knowledge base:\n\n{chunks_text}"
            })
        
        # Add conversation history (exclude last user message if it's already there)
        for msg in state.get('history', []):
            # Skip if this is the current user message
            if msg.get('content') != state['last_user_msg']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": state['last_user_msg']
        })
        
        # Call LLM
        llm_response = llm_module.call_llm(
            messages=messages,
            model=state.get('model', 'gpt-4o-mini'),
            temperature=0.7,
            max_tokens=2000
        )
        
        # Update state
        return {
            **state,
            'draft': llm_response['content'],
            'metadata': {
                **state.get('metadata', {}),
                'tokens_used': llm_response['tokens_used'],
                'model': llm_response['model'],
                'finish_reason': llm_response.get('finish_reason', 'stop'),
                'retrieval_count': len(state.get('retrieved_chunks', [])),
                'context_messages': len(state.get('history', []))
            },
            'error': None
        }
        
    except (AuthenticationError, RateLimitError, OpenAIError) as e:
        logger.error(f"LLM error: {e}")
        return {
            **state,
            'draft': '',
            'error': f"LLM error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return {
            **state,
            'draft': '',
            'error': f"Synthesis error: {str(e)}"
        }

