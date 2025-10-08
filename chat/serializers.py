from rest_framework import serializers
from .models import Document, DocumentChunk, ChatSession, ChatMessage


class DocumentChunkSerializer(serializers.ModelSerializer):
    has_embedding = serializers.SerializerMethodField()
    
    class Meta:
        model = DocumentChunk
        fields = ['id', 'chunk_index', 'text', 'has_embedding', 'created_at']
    
    def get_has_embedding(self, obj):
        return obj.embedding is not None


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()
    chunks = DocumentChunkSerializer(many=True, read_only=True)
    
    class Meta:
        model = Document
        fields = ['id', 'filename', 'raw_text', 'chunk_count', 'chunks', 'created_at']
        read_only_fields = ['id', 'created_at']
    
    def get_chunk_count(self, obj):
        return obj.chunks.count()


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    auto_process = serializers.BooleanField(default=True)
    
    def validate_file(self, value):
        # Limit file size to 10MB
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 10MB")
        return value


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'session', 'role', 'content', 'token_count', 'created_at']
        read_only_fields = ['id', 'created_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = ['id', 'title', 'message_count', 'messages', 'long_term_summary', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        return obj.messages.count()

