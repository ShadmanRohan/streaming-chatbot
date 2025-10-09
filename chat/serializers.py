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
        fields = ['id', 'filename', 'raw_text', 'file_size', 'session', 'chunk_count', 'chunks', 'created_at']
        read_only_fields = ['id', 'created_at']
    
    def get_chunk_count(self, obj):
        return obj.chunks.count()


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    auto_process = serializers.BooleanField(default=True)
    session_id = serializers.UUIDField(required=False, allow_null=True)
    
    def validate_file(self, value):
        # Limit file size to 500KB
        if value.size > 500 * 1024:
            raise serializers.ValidationError(f"File size ({value.size / 1024:.1f}KB) exceeds 500KB limit")
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


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chat request"""
    session_id = serializers.UUIDField(required=True)
    message = serializers.CharField(required=True, max_length=4000)
    retrieve = serializers.BooleanField(default=True)
    top_k = serializers.IntegerField(default=3, min_value=1, max_value=10)
    use_mmr = serializers.BooleanField(default=True)
    lambda_param = serializers.FloatField(default=0.5, min_value=0.0, max_value=1.0)
    model = serializers.CharField(default='gpt-4o-mini', max_length=50)
    
    def validate_message(self, value):
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()


class RetrievedChunkSerializer(serializers.Serializer):
    """Serializer for retrieved chunk in response"""
    text = serializers.CharField()
    score = serializers.FloatField()
    document = serializers.CharField()
    chunk_id = serializers.UUIDField()


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response"""
    session_id = serializers.UUIDField()
    message_id = serializers.UUIDField()
    content = serializers.CharField()
    retrieved_chunks = RetrievedChunkSerializer(many=True)
    metadata = serializers.DictField()

