"""
Summarize Node - Updates session summary every 5 assistant turns.
"""
import logging
from typing import Dict, Any
from chat.models import ChatSession, ChatMessage
from chat.llm import call_llm

logger = logging.getLogger(__name__)


def summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update long-term session summary every 5 assistant messages.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state (summary updated in database if needed)
    """
    session_id = state['session_id']
    
    try:
        session = ChatSession.objects.get(id=session_id)
        
        # Count assistant messages
        assistant_count = ChatMessage.objects.filter(
            session=session,
            role='assistant'
        ).count()
        
        # Update summary every 5 turns
        if assistant_count > 0 and assistant_count % 5 == 0:
            logger.info(f"Generating summary for session {session_id} (turn {assistant_count})")
            
            # Get recent messages for summary
            recent_messages = ChatMessage.objects.filter(
                session=session
            ).order_by('-created_at')[:20]
            
            # Build conversation text
            conversation_text = "\n".join([
                f"{msg.role.capitalize()}: {msg.content}"
                for msg in reversed(recent_messages)
            ])
            
            # Generate summary using LLM
            summary_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise conversation summaries."
                },
                {
                    "role": "user",
                    "content": f"Create a brief summary of this conversation (2-3 sentences):\n\n{conversation_text}"
                }
            ]
            
            try:
                summary_response = call_llm(
                    messages=summary_prompt,
                    model=state.get('model', 'gpt-4o-mini'),
                    temperature=0.5,
                    max_tokens=200
                )
                
                summary = summary_response['content']
                
                # Save to database
                session.long_term_summary = summary
                session.save()
                
                logger.info(f"Summary updated: {summary[:100]}...")
                
                return {
                    **state,
                    'summary': summary
                }
                
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                # Don't fail the whole request if summary fails
                return state
        
        # No summary needed this turn
        return state
        
    except ChatSession.DoesNotExist:
        logger.error(f"Session {session_id} not found for summarization")
        return state
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return state

