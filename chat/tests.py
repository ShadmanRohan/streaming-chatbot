from django.test import TestCase
from unittest.mock import patch, MagicMock
import numpy as np

from .models import Document, DocumentChunk
from .retrieval import maximal_marginal_relevance, search
from .chunking import chunk_text


class ChunkingTests(TestCase):
    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        text = "This is sentence one.\nThis is sentence two.\nThis is sentence three."
        chunks = chunk_text(text, max_len=50)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(c) <= 50 for c in chunks))
    
    def test_chunk_text_empty(self):
        """Test chunking empty text"""
        chunks = chunk_text("")
        self.assertEqual(len(chunks), 0)
    
    def test_chunk_text_long_paragraph(self):
        """Test chunking long paragraph"""
        text = "word " * 100
        chunks = chunk_text(text, max_len=100)
        self.assertGreater(len(chunks), 1)


class MMRTests(TestCase):
    def test_mmr_basic(self):
        """Test MMR with fake embeddings"""
        # Query embedding
        query_emb = [1.0, 0.0, 0.0]
        
        # Candidates: first is most relevant, second is similar to first, third is diverse
        candidates = [
            [0.9, 0.1, 0.0],  # Very similar to query
            [0.8, 0.2, 0.0],  # Also similar to query AND first candidate
            [0.0, 0.0, 1.0],  # Different from query but diverse
        ]
        
        # With high lambda (relevance), should prefer first two
        indices = maximal_marginal_relevance(query_emb, candidates, lambda_param=1.0, top_k=2)
        self.assertEqual(len(indices), 2)
        self.assertIn(0, indices)  # Most relevant should be selected
        
        # With low lambda (diversity), should prefer diverse items
        indices = maximal_marginal_relevance(query_emb, candidates, lambda_param=0.3, top_k=2)
        self.assertEqual(len(indices), 2)
        self.assertIn(0, indices)  # First is always selected (most relevant)
    
    def test_mmr_empty_candidates(self):
        """Test MMR with no candidates"""
        query_emb = [1.0, 0.0]
        candidates = []
        indices = maximal_marginal_relevance(query_emb, candidates, top_k=3)
        self.assertEqual(indices, [])
    
    def test_mmr_single_candidate(self):
        """Test MMR with single candidate"""
        query_emb = [1.0, 0.0]
        candidates = [[0.9, 0.1]]
        indices = maximal_marginal_relevance(query_emb, candidates, top_k=3)
        self.assertEqual(indices, [0])
    
    def test_mmr_top_k_larger_than_candidates(self):
        """Test MMR when top_k > number of candidates"""
        query_emb = [1.0, 0.0]
        candidates = [[0.9, 0.1], [0.8, 0.2]]
        indices = maximal_marginal_relevance(query_emb, candidates, top_k=5)
        self.assertEqual(len(indices), 2)


class RetrievalTests(TestCase):
    def setUp(self):
        """Create test documents and chunks"""
        self.doc1 = Document.objects.create(
            filename="test1.txt",
            raw_text="Machine learning and artificial intelligence"
        )
        self.doc2 = Document.objects.create(
            filename="test2.txt",
            raw_text="Cooking recipes and kitchen tips"
        )
    
    @patch('chat.retrieval.embed_text')
    def test_search_with_mmr(self, mock_embed):
        """Test search with MMR enabled"""
        # Mock embeddings
        mock_embed.return_value = [1.0, 0.0, 0.0]
        
        # Create chunks with embeddings
        DocumentChunk.objects.create(
            document=self.doc1,
            chunk_index=0,
            text="ML is great",
            embedding=[0.9, 0.1, 0.0]
        )
        DocumentChunk.objects.create(
            document=self.doc1,
            chunk_index=1,
            text="AI is amazing",
            embedding=[0.85, 0.15, 0.0]
        )
        DocumentChunk.objects.create(
            document=self.doc2,
            chunk_index=0,
            text="Cooking is fun",
            embedding=[0.0, 0.0, 1.0]
        )
        
        results = search("test query", top_k=2, use_mmr=True)
        self.assertEqual(len(results), 2)
        # Results should be diverse due to MMR
    
    @patch('chat.retrieval.embed_text')
    def test_search_without_mmr(self, mock_embed):
        """Test search without MMR (pure relevance)"""
        mock_embed.return_value = [1.0, 0.0, 0.0]
        
        DocumentChunk.objects.create(
            document=self.doc1,
            chunk_index=0,
            text="Test chunk",
            embedding=[0.9, 0.1, 0.0]
        )
        
        results = search("test query", top_k=3, use_mmr=False)
        self.assertLessEqual(len(results), 3)
    
    @patch('chat.retrieval.embed_text')
    def test_search_with_document_filter(self, mock_embed):
        """Test search filtered by document IDs"""
        mock_embed.return_value = [1.0, 0.0, 0.0]
        
        chunk1 = DocumentChunk.objects.create(
            document=self.doc1,
            chunk_index=0,
            text="ML chunk",
            embedding=[0.9, 0.1, 0.0]
        )
        chunk2 = DocumentChunk.objects.create(
            document=self.doc2,
            chunk_index=0,
            text="Cooking chunk",
            embedding=[0.8, 0.2, 0.0]
        )
        
        # Filter to only doc1
        results = search("test", top_k=5, document_ids=[str(self.doc1.id)])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1].document_id, self.doc1.id)
    
    def test_search_no_embeddings(self):
        """Test search when chunks have no embeddings"""
        DocumentChunk.objects.create(
            document=self.doc1,
            chunk_index=0,
            text="No embedding",
            embedding=None
        )
        
        with patch('chat.retrieval.embed_text') as mock_embed:
            mock_embed.return_value = [1.0, 0.0, 0.0]
            results = search("test", top_k=3)
            self.assertEqual(len(results), 0)


class DocumentModelTests(TestCase):
    def test_create_document(self):
        """Test document creation"""
        doc = Document.objects.create(
            filename="test.txt",
            raw_text="Test content"
        )
        self.assertEqual(doc.filename, "test.txt")
        self.assertEqual(doc.raw_text, "Test content")
        self.assertIsNotNone(doc.id)
    
    def test_document_chunk_relationship(self):
        """Test document-chunk relationship"""
        doc = Document.objects.create(
            filename="test.txt",
            raw_text="Test"
        )
        chunk = DocumentChunk.objects.create(
            document=doc,
            chunk_index=0,
            text="Chunk text",
            embedding=[1.0, 0.0]
        )
        
        self.assertEqual(doc.chunks.count(), 1)
        self.assertEqual(chunk.document, doc)
