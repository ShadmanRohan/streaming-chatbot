from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.db import transaction

from .models import Document, DocumentChunk, ChatSession, ChatMessage
from .serializers import (
    DocumentSerializer, 
    DocumentUploadSerializer,
    DocumentChunkSerializer,
    ChatSessionSerializer,
    ChatMessageSerializer
)
from .chunking import chunk_text
from .embedding_utils import embed_text
from .retrieval import search_similar_chunks, search


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
