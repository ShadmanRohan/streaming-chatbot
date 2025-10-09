"""
LoadHistory Node - Loads conversation context from database.
"""
import logging
from typing import Dict, Any
from chat.models import ChatSession, ChatMessage

logger = logging.getLogger(__name__)


def load_history(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load conversation history and session summary from database.
    
    Args:
        state: Current graph state with session_id
        
    Returns:
        Updated state with history and summary
    """
    session_id = state['session_id']
    
    try:
        # Get session
        session = ChatSession.objects.get(id=session_id)
        
        # Load last N messages (excluding current user message if already created)
        messages = ChatMessage.objects.filter(
            session=session
        ).order_by('-created_at')[:10]
        
        # Convert to history format (chronological order)
        history = [
            {
                'role': msg.role,
                'content': msg.content
            }
            for msg in reversed(messages)
        ]
        
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

