"""
LLM integration module for OpenAI API.
API key is loaded from environment variables, never hardcoded.
"""
import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI, OpenAIError, RateLimitError, AuthenticationError
import tiktoken

logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client with API key from environment.
    Raises ValueError if API key not found.
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )
    
    return OpenAI(api_key=api_key)


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text for a given model.
    
    Args:
        text: The text to count tokens for
        model: The model name (default: gpt-4o-mini)
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in a messages array (for chat completions).
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: The model name
        
    Returns:
        Approximate number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = 0
    for message in messages:
        # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        num_tokens += 4  # Base tokens per message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    
    num_tokens += 2  # Every reply is primed with <|start|>assistant
    return num_tokens


def call_llm(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
) -> Dict:
    """
    Call OpenAI LLM with error handling.
    
    Args:
        messages: Chat messages array
        model: Model name (loaded from settings if not provided)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional OpenAI parameters
        
    Returns:
        Dict with 'content', 'tokens_used', and 'model'
        
    Raises:
        AuthenticationError: If API key is invalid
        RateLimitError: If rate limit exceeded
        OpenAIError: For other OpenAI errors
    """
    client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            'content': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens,
            'model': response.model,
            'finish_reason': response.choices[0].finish_reason
        }
        
    except AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise
    except RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        raise
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling LLM: {e}")
        raise


def validate_api_key() -> bool:
    """
    Validate that OpenAI API key is configured and valid.
    
    Returns:
        True if API key is valid, False otherwise
    """
    try:
        client = get_openai_client()
        # Make a minimal API call to test the key
        client.models.list()
        return True
    except AuthenticationError:
        return False
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return False

