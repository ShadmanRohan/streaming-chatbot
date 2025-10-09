"""
LoadHistory Node - Loads conversation context from database with token budget.
"""
import logging
from typing import Dict, Any
from django.conf import settings
from chat.models import ChatSession, ChatMessage
from chat.llm import count_tokens

logger = logging.getLogger(__name__)


def load_history(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load conversation history and session summary from database.
    Applies token budget to limit context size while preserving minimum turns.
    
    Args:
        state: Current graph state with session_id
        
    Returns:
        Updated state with history and summary (token-bounded)
    """
    session_id = state['session_id']
    
    # Get memory config
    memory_config = getattr(settings, 'MEMORY_CONFIG', {})
    max_tokens = memory_config.get('max_tokens_context', 3000)
    min_turns = memory_config.get('history_min_turns', 6)
    
    try:
        # Get session
        session = ChatSession.objects.get(id=session_id)
        
        # Load more messages than we need (for trimming)
        messages = list(ChatMessage.objects.filter(
            session=session
        ).order_by('-created_at')[:20])
        
        # Token-bounded trimming (newest first, then reverse)
        history = []
        token_count = 0
        
        for msg in messages:
            msg_tokens = count_tokens(msg.content, state.get('model', 'gpt-4o-mini'))
            
            # Keep message if:
            # 1. Still under budget, OR
            # 2. Haven't reached minimum turns yet
            if token_count + msg_tokens <= max_tokens or len(history) < min_turns:
                history.append({
                    'role': msg.role,
                    'content': msg.content
                })
                token_count += msg_tokens
            else:
                # Budget exceeded and have minimum turns
                break
        
        # Reverse to chronological order (oldest first)
        history.reverse()
        
        logger.info(f"Loaded {len(history)} messages ({token_count} tokens) for session {session_id}")
        
        # Update state
        return {
            **state,
            'history': history,
            'summary': session.long_term_summary,
            'error': None
        }
        
    except ChatSession.DoesNotExist:
        logger.error(f"Session {session_id} not found")
        return {
            **state,
            'history': [],
            'summary': None,
            'error': 'Session not found'
        }
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return {
            **state,
            'history': [],
            'summary': None,
            'error': str(e)
        }

