"""
Test cases for REXI services.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from rexi.services.neo4j_service import Neo4jService
from rexi.services.qdrant_service import QdrantService
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService


class TestEmbeddingService:
    """Test cases for EmbeddingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rexi.services.embedding_service.get_settings'):
            self.embedding_service = EmbeddingService()
    
    def test_encode_text(self):
        """Test text encoding."""
        text = "This is a test sentence."
        
        embedding = self.embedding_service.encode_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_encode_texts(self):
        """Test batch text encoding."""
        texts = ["First sentence.", "Second sentence."]
        
        embeddings = self.embedding_service.encode_texts(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        text1 = "Similar text"
        text2 = "Similar text"
        text3 = "Different content"
        
        emb1 = self.embedding_service.encode_text(text1)
        emb2 = self.embedding_service.encode_text(text2)
        emb3 = self.embedding_service.encode_text(text3)
        
        sim_12 = self.embedding_service.compute_similarity(emb1, emb2)
        sim_13 = self.embedding_service.compute_similarity(emb1, emb3)
        
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1
        assert sim_12 > sim_13  # Similar texts should have higher similarity
    
    def test_get_embedding_dimension(self):
        """Test embedding dimension retrieval."""
        dimension = self.embedding_service.get_embedding_dimension()
        assert isinstance(dimension, int)
        assert dimension > 0


class TestNeo4jService:
    """Test cases for Neo4jService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rexi.services.neo4j_service.get_settings'):
            self.neo4j_service = Neo4jService()
            self.neo4j_service.driver = Mock()
    
    def test_create_node(self):
        """Test node creation."""
        label = "TestNode"
        properties = {"name": "test", "value": 42}
        
        mock_result = {"n": {"id": 1, **properties}}
        self.neo4j_service.driver.session.return_value.__enter__.return_value.run.return_value.__iter__.return_value = [mock_result]
        
        result = self.neo4j_service.create_node(label, properties)
        
        assert result == mock_result["n"]
    
    def test_find_nodes(self):
        """Test node finding."""
        label = "TestNode"
        properties = {"name": "test"}
        
        mock_result = [{"n": {"id": 1, "name": "test"}}]
        self.neo4j_service.driver.session.return_value.__enter__.return_value.run.return_value.__iter__.return_value = mock_result
        
        result = self.neo4j_service.find_nodes(label, properties)
        
        assert len(result) == 1
        assert result[0]["name"] == "test"
    
    def test_create_relationship(self):
        """Test relationship creation."""
        source_id = "1"
        target_id = "2"
        relationship_type = "CONNECTS_TO"
        properties = {"strength": 0.8}
        
        mock_result = {"r": {"id": 1, **properties}}
        self.neo4j_service.driver.session.return_value.__enter__.return_value.run.return_value.__iter__.return_value = [mock_result]
        
        result = self.neo4j_service.create_relationship(source_id, target_id, relationship_type, properties)
        
        assert result == mock_result["r"]


class TestQdrantService:
    """Test cases for QdrantService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rexi.services.qdrant_service.get_settings'):
            self.qdrant_service = QdrantService()
            self.qdrant_service.client = Mock()
    
    def test_create_collection(self):
        """Test collection creation."""
        collection_name = "test_collection"
        vector_size = 384
        
        self.qdrant_service.client.create_collection.return_value = True
        
        result = self.qdrant_service.create_collection(collection_name, vector_size)
        
        assert result is True
        self.qdrant_service.client.create_collection.assert_called_once()
    
    def test_search(self):
        """Test vector search."""
        collection_name = "test_collection"
        query_vector = [0.1] * 384
        
        mock_hits = [
            Mock(id="1", score=0.9, payload={"text": "result1"}),
            Mock(id="2", score=0.8, payload={"text": "result2"})
        ]
        self.qdrant_service.client.search.return_value = mock_hits
        
        result = self.qdrant_service.search(collection_name, query_vector)
        
        assert len(result) == 2
        assert result[0]["score"] == 0.9
        assert result[0]["payload"]["text"] == "result1"
    
    def test_upsert_points(self):
        """Test point upsertion."""
        collection_name = "test_collection"
        points = [{"id": "1", "vector": [0.1] * 384, "payload": {"text": "test"}}]
        
        self.qdrant_service.client.upsert.return_value = True
        
        result = self.qdrant_service.upsert_points(collection_name, points)
        
        assert result is True


class TestLLMService:
    """Test cases for LLMService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rexi.services.llm_service.get_settings'):
            self.llm_service = LLMService()
    
    @patch('rexi.services.llm_service.openai')
    def test_chat_completion(self, mock_openai):
        """Test chat completion."""
        messages = [{"role": "user", "content": "Hello"}]
        expected_response = "Hello! How can I help you?"
        
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content=expected_response))
        ]
        
        result = self.llm_service.chat_completion(messages)
        
        assert result == expected_response
    
    def test_count_tokens(self):
        """Test token counting."""
        text = "This is a test sentence."
        
        with patch.object(self.llm_service, 'tokenizer') as mock_tokenizer:
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            
            result = self.llm_service.count_tokens(text)
            
            assert result == 5
    
    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a longer test sentence that should be truncated."
        max_tokens = 5
        
        with patch.object(self.llm_service, 'tokenizer') as mock_tokenizer:
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
            mock_tokenizer.decode.return_value = "This is a longer"
            
            result = self.llm_service.truncate_text(text, max_tokens)
            
            assert len(self.llm_service.tokenizer.encode(result)) <= max_tokens


class TestIntegration:
    """Integration tests for services working together."""
    
    @pytest.mark.asyncio
    async def test_embedding_to_qdrant_flow(self):
        """Test embedding generation and storage flow."""
        with patch('rexi.services.embedding_service.get_settings'), \
             patch('rexi.services.qdrant_service.get_settings'):
            
            embedding_service = EmbeddingService()
            qdrant_service = QdrantService()
            
            # Mock Qdrant client
            qdrant_service.client = Mock()
            qdrant_service.client.upsert.return_value = True
            
            # Generate embedding
            text = "Test text for embedding"
            embedding = embedding_service.encode_text(text)
            
            # Store in Qdrant
            point = {
                "id": "test_point",
                "vector": embedding,
                "payload": {"text": text}
            }
            
            result = qdrant_service.upsert_points("test_collection", [point])
            
            assert result is True
            qdrant_service.client.upsert.assert_called_once()
    
    def test_neo4j_qdrant_integration(self):
        """Test Neo4j and Qdrant integration."""
        # This would test the full knowledge graph functionality
        # For now, just test that services can be instantiated together
        with patch('rexi.services.neo4j_service.get_settings'), \
             patch('rexi.services.qdrant_service.get_settings'), \
             patch('rexi.services.embedding_service.get_settings'):
            
            neo4j_service = Neo4jService()
            qdrant_service = QdrantService()
            embedding_service = EmbeddingService()
            
            assert neo4j_service is not None
            assert qdrant_service is not None
            assert embedding_service is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
