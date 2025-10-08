"""
Prompt templates and builders for chat functionality.
"""
from typing import List, Dict, Optional
from chat.models import ChatMessage


SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base. 
Answer questions accurately based on the provided context. If the context doesn't 
contain relevant information, say so clearly. Be concise but comprehensive."""


def sanitize_user_input(message: str) -> str:
    """
    Sanitize user input to prevent prompt injection attacks.
    
    Args:
        message: User's input message
        
    Returns:
        Sanitized message
    """
    # Remove potential prompt injection patterns
    dangerous_patterns = [
        'ignore previous instructions',
        'ignore all previous',
        'system:',
        'assistant:',
        '<|im_start|>',
        '<|im_end|>',
        '<|endoftext|>',
    ]
    
    cleaned = message
    for pattern in dangerous_patterns:
        cleaned = cleaned.replace(pattern, '')
    
    return cleaned.strip()


def format_retrieved_chunks(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks for inclusion in prompt.
    
    Args:
        chunks: List of dicts with 'text', 'document', 'score'
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return ""
    
    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        formatted_chunks.append(
            f"[Source {i}: {chunk.get('document', 'Unknown')} (relevance: {chunk.get('score', 0):.2f})]\n"
            f"{chunk['text']}"
        )
    
    return "\n\n".join(formatted_chunks)


def truncate_history(
    messages: List[ChatMessage], 
    max_messages: int = 10
) -> List[ChatMessage]:
    """
    Truncate conversation history to most recent messages.
    
    Args:
        messages: List of ChatMessage objects
        max_messages: Maximum number of messages to keep
        
    Returns:
        Truncated list of messages
    """
    if len(messages) <= max_messages:
        return messages
    
    # Keep the most recent max_messages
    return messages[-max_messages:]


def build_chat_prompt(
    user_message: str,
    retrieved_chunks: List[Dict],
    context_messages: List[ChatMessage],
    summary: Optional[str] = None,
    max_context_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Build complete prompt for OpenAI chat completion.
    
    Args:
        user_message: Current user message
        retrieved_chunks: Retrieved document chunks from RAG
        context_messages: Previous conversation messages
        summary: Optional long-term conversation summary
        max_context_messages: Maximum context messages to include
        
    Returns:
        List of message dicts for OpenAI API
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Add long-term summary if exists
    if summary:
        messages.append({
            "role": "system",
            "content": f"Previous conversation summary: {summary}"
        })
    
    # Add retrieved context from RAG
    if retrieved_chunks:
        context_text = format_retrieved_chunks(retrieved_chunks)
        messages.append({
            "role": "system",
            "content": f"Relevant information from knowledge base:\n\n{context_text}"
        })
    
    # Add recent conversation history
    truncated_history = truncate_history(context_messages, max_context_messages)
    for msg in truncated_history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Add current user message (sanitized)
    sanitized_message = sanitize_user_input(user_message)
    messages.append({
        "role": "user",
        "content": sanitized_message
    })
    
    return messages


def build_simple_prompt(user_message: str) -> List[Dict[str, str]]:
    """
    Build a simple prompt without RAG or context.
    Useful for testing or simple Q&A.
    
    Args:
        user_message: User's question
        
    Returns:
        Simple messages array
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sanitize_user_input(user_message)}
    ]

