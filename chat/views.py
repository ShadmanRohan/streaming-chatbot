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
from .retrieval import search_similar_chunks, search
from .llm import call_llm, count_tokens
from .prompts import build_chat_prompt

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
        """
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        uploaded_file = serializer.validated_data['file']
        auto_process = serializer.validated_data.get('auto_process', True)
        
        # Read file content
        try:
            content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return Response(
                {'error': 'File must be UTF-8 encoded text'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create document
        with transaction.atomic():
            document = Document.objects.create(
                filename=uploaded_file.name,
                raw_text=content
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
            'chunk_count': chunk_count,
            'auto_processed': auto_process,
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
        
        Send a message and get AI response with RAG.
        """
        # 1. Validate request
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        session_id = serializer.validated_data['session_id']
        message = serializer.validated_data['message']
        retrieve = serializer.validated_data.get('retrieve', True)
        top_k = serializer.validated_data.get('top_k', 3)
        use_mmr = serializer.validated_data.get('use_mmr', True)
        lambda_param = serializer.validated_data.get('lambda_param', 0.5)
        model = serializer.validated_data.get('model', 'gpt-4o-mini')
        
        # 2. Get session
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
            
            # 4. Load conversation context
            context_messages = list(
                ChatMessage.objects.filter(session=session)
                .order_by('-created_at')[:10]
            )
            context_messages.reverse()  # Chronological order
            
            # Exclude the just-created user message from context
            # (it will be added separately in the prompt)
            context_messages = context_messages[:-1] if len(context_messages) > 1 else []
            
            summary = session.long_term_summary
            
            # 5. Retrieve relevant chunks (RAG)
            retrieved_chunks = []
            if retrieve:
                try:
                    results = search(
                        query=message,
                        top_k=top_k,
                        use_mmr=use_mmr,
                        lambda_param=lambda_param
                    )
                    retrieved_chunks = [
                        {
                            'text': chunk.text,
                            'score': float(score),
                            'document': chunk.document.filename,
                            'chunk_id': str(chunk.id)
                        }
                        for score, chunk in results
                    ]
                except Exception as e:
                    logger.error(f"Retrieval error: {e}")
                    # Continue without retrieval if it fails
            
            # 6. Build prompt
            messages = build_chat_prompt(
                user_message=message,
                retrieved_chunks=retrieved_chunks,
                context_messages=context_messages,
                summary=summary
            )
            
            # 7. Call LLM
            try:
                llm_response = call_llm(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                assistant_content = llm_response['content']
                tokens_used = llm_response['tokens_used']
                
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
                # Catch API key not configured error
                logger.error(f"Configuration error: {e}")
                return Response(
                    {'error': str(e), 'code': 'CONFIGURATION_ERROR'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return Response(
                    {'error': 'An unexpected error occurred',
                     'code': 'INTERNAL_ERROR'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # 8. Save assistant message
            assistant_msg = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=assistant_content,
                token_count=tokens_used
            )
            
            # 9. Update session timestamp
            session.updated_at = timezone.now()
            session.save()
        
        # 10. Return response
        response_data = {
            'session_id': str(session.id),
            'message_id': str(assistant_msg.id),
            'content': assistant_content,
            'retrieved_chunks': retrieved_chunks,
            'metadata': {
                'tokens_used': tokens_used,
                'retrieval_count': len(retrieved_chunks),
                'context_messages': len(context_messages),
                'model': model
            }
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
