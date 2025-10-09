from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.db import transaction
from django.utils import timezone
from openai import OpenAIError, RateLimitError, AuthenticationError
import logging

from .models import Document, DocumentChunk, ChatSession, ChatMessage
from .serializers import (
    DocumentSerializer, 
    DocumentUploadSerializer,
    DocumentChunkSerializer,
    ChatSessionSerializer,
    ChatMessageSerializer,
    ChatRequestSerializer,
    ChatResponseSerializer
)
from .chunking import chunk_text
from .embedding_utils import embed_text
from .retrieval import search
from .llm import count_tokens
from .langgraph import run_graph

logger = logging.getLogger(__name__)


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def get_serializer_class(self):
        if self.action == 'upload':
            return DocumentUploadSerializer
        return DocumentSerializer
    
    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload(self, request):
        """
        Upload a document and optionally auto-process (chunk + embed)
        Binds document to session if session_id provided
        """
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        uploaded_file = serializer.validated_data['file']
        auto_process = serializer.validated_data.get('auto_process', True)
        session_id = serializer.validated_data.get('session_id')
        
        # Validate session if provided
        session = None
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id)
            except ChatSession.DoesNotExist:
                return Response(
                    {'error': 'Session not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
        
        # Read file content
        try:
            content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return Response(
                {'error': 'File must be UTF-8 encoded text'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create document with session binding
        with transaction.atomic():
            # Delete previous documents AND chat history for this session (one file per session)
            if session:
                old_docs = Document.objects.filter(session=session)
                old_count = old_docs.count()
                old_docs.delete()  # Cascades to DocumentChunk due to ON DELETE CASCADE
                
                # Clear chat history too
                old_messages = ChatMessage.objects.filter(session=session)
                msg_count = old_messages.count()
                old_messages.delete()
                
                # Reset session summary
                session.session_summary = ""
                session.save()
                
                if old_count > 0 or msg_count > 0:
                    logger.info(f"Cleared session {session_id}: {old_count} document(s), {msg_count} message(s)")
            
            document = Document.objects.create(
                filename=uploaded_file.name,
                raw_text=content,
                file_size=uploaded_file.size,
                session=session
            )
            
            chunk_count = 0
            if auto_process:
                # Chunk and embed
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    embedding = embed_text(chunk)
                    DocumentChunk.objects.create(
                        document=document,
                        chunk_index=i,
                        text=chunk,
                        embedding=embedding
                    )
                    chunk_count += 1
        
        return Response({
            'id': str(document.id),
            'filename': document.filename,
            'file_size': uploaded_file.size,
            'chunk_count': chunk_count,
            'auto_processed': auto_process,
            'session_id': str(session.id) if session else None,
            'created_at': document.created_at
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['get'])
    def chunks(self, request, pk=None):
        """
        Get all chunks for a specific document
        """
        document = self.get_object()
        chunks = document.chunks.all().order_by('chunk_index')
        serializer = DocumentChunkSerializer(chunks, many=True)
        return Response(serializer.data)


class ChatSessionViewSet(viewsets.ModelViewSet):
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer


class ChatViewSet(viewsets.ViewSet):
    """
    Chat endpoint for conversational AI with RAG.
    """
    
    @action(detail=False, methods=['post'])
    def send(self, request):
        """
        POST /api/chat/send/
        
        Send a message and get AI response with LangGraph orchestration.
        """
        # 1. Validate request
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        session_id = serializer.validated_data['session_id']
        message = serializer.validated_data['message']
        top_k = serializer.validated_data.get('top_k', 3)
        use_mmr = serializer.validated_data.get('use_mmr', True)
        lambda_param = serializer.validated_data.get('lambda_param', 0.5)
        model = serializer.validated_data.get('model', 'gpt-4o-mini')
        
        # 2. Validate session exists
        try:
            session = ChatSession.objects.get(id=session_id)
        except ChatSession.DoesNotExist:
            return Response(
                {'error': 'Session not found', 'code': 'SESSION_NOT_FOUND'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # 3. Create user message
        with transaction.atomic():
            user_msg = ChatMessage.objects.create(
                session=session,
                role='user',
                content=message,
                token_count=count_tokens(message, model)
            )
            
            # 4. Run LangGraph orchestration
            try:
                result = run_graph(
                    session_id=str(session_id),
                    user_message=message,
                    model=model,
                    top_k=top_k,
                    use_mmr=use_mmr,
                    lambda_param=lambda_param
                )
                
                assistant_content = result['content']
                retrieved_chunks = result['retrieved_chunks']
                metadata = result['metadata']
                tokens_used = metadata.get('tokens_used', 0)
                
            except AuthenticationError as e:
                logger.error(f"OpenAI authentication failed: {e}")
                return Response(
                    {'error': 'LLM authentication failed. Please check API key configuration.',
                     'code': 'LLM_AUTH_ERROR'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            except RateLimitError as e:
                logger.error(f"OpenAI rate limit: {e}")
                return Response(
                    {'error': 'Rate limit exceeded. Please try again later.',
                     'code': 'RATE_LIMIT_EXCEEDED'},
                    status=status.HTTP_429_TOO_MANY_REQUESTS
                )
            except OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                return Response(
                    {'error': f'LLM service error: {str(e)}',
                     'code': 'LLM_SERVICE_ERROR'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            except ValueError as e:
                logger.error(f"Configuration error: {e}")
                return Response(
                    {'error': str(e), 'code': 'CONFIGURATION_ERROR'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            except Exception as e:
                logger.error(f"Graph execution error: {e}")
                return Response(
                    {'error': 'An unexpected error occurred',
                     'code': 'INTERNAL_ERROR'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # 5. Save assistant message
            assistant_msg = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=assistant_content,
                token_count=tokens_used
            )
            
            # 6. Update session timestamp
            session.updated_at = timezone.now()
            session.save()
        
        # 7. Return response
        response_data = {
            'session_id': str(session.id),
            'message_id': str(assistant_msg.id),
            'content': assistant_content,
            'retrieved_chunks': retrieved_chunks,
            'metadata': metadata,
            'orchestration': 'langgraph'  # Indicate using LangGraph
        }
        
        return Response(response_data, status=status.HTTP_200_OK)


@api_view(['POST'])
def retrieve_chunks(request):
    """
    Debug endpoint to test retrieval with MMR support
    POST /api/retrieve/
    {
        "query": "search text",
        "top_k": 3,
        "use_mmr": true,
        "lambda_param": 0.5,
        "document_ids": ["uuid1", "uuid2"]  // optional
    }
    """
    query = request.data.get('query')
    if not query:
        return Response(
            {'error': 'query parameter is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    top_k = request.data.get('top_k', 3)
    use_mmr = request.data.get('use_mmr', True)
    lambda_param = request.data.get('lambda_param', 0.5)
    document_ids = request.data.get('document_ids', None)
    
    # Use enhanced search with MMR
    results = search(
        query=query,
        top_k=top_k,
        use_mmr=use_mmr,
        lambda_param=lambda_param,
        document_ids=document_ids
    )
    
    response_data = {
        'query': query,
        'top_k': top_k,
        'use_mmr': use_mmr,
        'lambda_param': lambda_param,
        'results': [
            {
                'score': float(score),
                'chunk_id': str(chunk.id),
                'text': chunk.text,
                'document_id': str(chunk.document.id),
                'document_filename': chunk.document.filename,
                'chunk_index': chunk.chunk_index
            }
            for score, chunk in results
        ]
    }
    
    return Response(response_data)


@api_view(['POST'])
def chat_stream(request):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    
    POST /api/chat/stream/
    {
        "session_id": "uuid",
        "message": "user message",
        "model": "gpt-4o-mini"  // optional
    }
    
    Returns SSE stream with events:
    - data: {"type": "delta", "content": "text chunk"}
    - data: {"type": "done", "message_id": "uuid", "chunks": [...]}
    - data: {"type": "error", "error": "error message"}
    """
    from django.http import StreamingHttpResponse
    from chat.langgraph.graph import run_graph_stream
    from chat.llm import count_tokens
    import json
    import uuid
    
    # Parse request
    session_id = request.data.get('session_id')
    message = request.data.get('message')
    model = request.data.get('model', 'gpt-4o-mini')
    
    if not session_id or not message:
        return Response(
            {'error': 'session_id and message are required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    def event_stream():
        """Generate SSE events"""
        try:
            # Create user message
            user_msg = ChatMessage.objects.create(
                session_id=session_id,
                role='user',
                content=message,
                token_count=count_tokens(message, model)
            )
            
            # Stream response
            accumulated = ""
            retrieved_chunks = []
            metadata = {}
            
            for item in run_graph_stream(
                session_id=session_id,
                user_message=message,
                model=model
            ):
                if isinstance(item, str):
                    # Delta from LLM
                    accumulated += item
                    yield f"data: {json.dumps({'type': 'delta', 'content': item})}\n\n"
                else:
                    # Final result dict
                    accumulated = item.get('content', accumulated)
                    retrieved_chunks = item.get('retrieved_chunks', [])
                    metadata = item.get('metadata', {})
            
            # Save assistant message
            assistant_msg = ChatMessage.objects.create(
                session_id=session_id,
                role='assistant',
                content=accumulated,
                token_count=count_tokens(accumulated, model)
            )
            
            # Update session timestamp
            session = ChatSession.objects.get(id=session_id)
            session.save(update_fields=['updated_at'])
            
            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'message_id': str(assistant_msg.id), 'chunks': len(retrieved_chunks)})}\n\n"
            
        except ChatSession.DoesNotExist:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Session not found'})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    
    return response
