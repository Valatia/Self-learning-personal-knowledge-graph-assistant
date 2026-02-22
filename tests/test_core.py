"""
Test cases for REXI core modules.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from rexi.core.knowledge_graph import KnowledgeGraph
from rexi.core.ingestion import IngestionEngine
from rexi.core.reasoning import ReasoningEngine
from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.models.documents import Document, DocumentType


class TestKnowledgeGraph:
    """Test cases for KnowledgeGraph."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rexi.core.knowledge_graph.Neo4jService'), \
             patch('rexi.core.knowledge_graph.QdrantService'), \
             patch('rexi.core.knowledge_graph.EmbeddingService'):
            
            self.knowledge_graph = KnowledgeGraph()
    
    def test_add_entity(self):
        """Test adding an entity."""
        entity = Entity(
            name="Test Entity",
            type=EntityType.CONCEPT,
            description="A test entity",
            confidence=0.9
        )
        
        # Mock the Neo4j service
        mock_node = Mock()
        mock_node.id = 1
        self.knowledge_graph.neo4j_service.create_node.return_value = mock_node
        
        entity_id = self.knowledge_graph.add_entity(entity)
        
        assert entity_id is not None
        assert entity.id == entity_id
        self.knowledge_graph.neo4j_service.create_node.assert_called_once()
    
    def test_add_relationship(self):
        """Test adding a relationship."""
        relationship = Relationship(
            source_entity_id="1",
            target_entity_id="2",
            type=RelationshipType.RELATED_TO,
            confidence=0.8
        )
        
        # Mock the Neo4j service
        mock_edge = Mock()
        mock_edge.id = 1
        self.knowledge_graph.neo4j_service.create_relationship.return_value = mock_edge
        
        relationship_id = self.knowledge_graph.add_relationship(relationship)
        
        assert relationship_id is not None
        assert relationship.id == relationship_id
        self.knowledge_graph.neo4j_service.create_relationship.assert_called_once()
    
    def test_get_entity(self):
        """Test getting an entity."""
        entity_id = "1"
        
        # Mock the Neo4j service
        mock_node_data = {
            "id": 1,
            "name": "Test Entity",
            "type": "concept",
            "description": "A test entity",
            "confidence": 0.9
        }
        self.knowledge_graph.neo4j_service.find_nodes.return_value = [mock_node_data]
        
        entity = self.knowledge_graph.get_entity(entity_id)
        
        assert entity is not None
        assert entity.name == "Test Entity"
        assert entity.type == EntityType.CONCEPT
    
    def test_find_similar_entities(self):
        """Test finding similar entities."""
        query_text = "test concept"
        
        # Mock services
        self.knowledge_graph.embedding_service.encode_text.return_value = [0.1] * 384
        self.knowledge_graph.qdrant_service.search.return_value = [
            {"id": "entity_1", "score": 0.9, "payload": {"entity_id": "1", "name": "Similar Entity"}}
        ]
        
        mock_entity = Entity(
            id="1",
            name="Similar Entity",
            type=EntityType.CONCEPT
        )
        self.knowledge_graph.get_entity = Mock(return_value=mock_entity)
        
        results = self.knowledge_graph.find_similar_entities(query_text)
        
        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.9
        assert results[0]["entity"].name == "Similar Entity"


class TestIngestionEngine:
    """Test cases for IngestionEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rexi.core.ingestion.EmbeddingService'), \
             patch('rexi.core.ingestion.LLMService'), \
             patch('rexi.core.ingestion.KnowledgeGraph'):
            
            self.ingestion_engine = IngestionEngine()
    
    @pytest.mark.asyncio
    async def test_ingest_text(self):
        """Test text ingestion."""
        text = "This is a test document about machine learning."
        title = "Test Document"
        
        # Mock services
        self.ingestion_engine.embedding_service.encode_text.return_value = [0.1] * 384
        self.ingestion_engine.llm_service.is_available.return_value = True
        self.ingestion_engine.llm_service.generate_entities_and_relations.return_value = {
            "entities": [
                {"name": "machine learning", "type": "concept", "confidence": 0.9}
            ],
            "relations": []
        }
        
        document = await self.ingestion_engine.ingest_text(text, title)
        
        assert document is not None
        assert document.title == title
        assert document.content == text
        assert document.type == DocumentType.TEXT
        assert document.word_count == len(text.split())
        assert document.processing_status == "completed"
    
    @pytest.mark.asyncio
    async def test_extract_knowledge(self):
        """Test knowledge extraction from document."""
        document = Document(
            id="1",
            title="Test",
            content="Machine learning enables AI systems to learn from data.",
            type=DocumentType.TEXT
        )
        
        # Mock LLM service
        self.ingestion_engine.llm_service.is_available.return_value = True
        self.ingestion_engine.llm_service.generate_entities_and_relations.return_value = {
            "entities": [
                {"name": "Machine learning", "type": "concept", "confidence": 0.9},
                {"name": "AI systems", "type": "concept", "confidence": 0.8}
            ],
            "relations": [
                {"source": "Machine learning", "target": "AI systems", "type": "enables", "confidence": 0.8}
            ]
        }
        
        # Mock knowledge graph
        self.ingestion_engine.knowledge_graph.add_entity = Mock(return_value="entity_1")
        self.ingestion_engine.knowledge_graph.add_relationship = Mock(return_value="rel_1")
        
        await self.ingestion_engine._extract_knowledge(document)
        
        # Verify entities were added
        assert self.ingestion_engine.knowledge_graph.add_entity.call_count == 2
        
        # Verify relationship was added
        assert self.ingestion_engine.knowledge_graph.add_relationship.call_count == 1
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        max_tokens = 10
        
        # Mock tokenizer
        self.ingestion_engine.llm_service.count_tokens.side_effect = lambda x: len(x.split())
        
        chunks = self.ingestion_engine._chunk_text(text, max_tokens)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= max_tokens for chunk in chunks)


class TestReasoningEngine:
    """Test cases for ReasoningEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('rexi.core.reasoning.KnowledgeGraph'), \
             patch('rexi.core.reasoning.LLMService'), \
             patch('rexi.core.reasoning.EmbeddingService'):
            
            self.reasoning_engine = ReasoningEngine()
    
    def test_answer_query(self):
        """Test query answering."""
        query = "What is machine learning?"
        
        # Mock services
        self.reasoning_engine.knowledge_graph.find_similar_entities.return_value = []
        self.reasoning_engine.embedding_service.encode_text.return_value = [0.1] * 384
        self.reasoning_engine.llm_service.is_available.return_value = True
        self.reasoning_engine.llm_service.answer_question.return_value = "Machine learning is a subset of AI..."
        
        result = self.reasoning_engine.answer_query(query)
        
        assert "answer" in result
        assert result["query"] == query
        assert "confidence" in result
        assert "timestamp" in result
    
    def test_extract_query_entities(self):
        """Test entity extraction from query."""
        query = "Tell me about Machine Learning and AI"
        
        entities = self.reasoning_engine._extract_query_entities(query)
        
        assert "Machine" in entities or "Machine Learning" in entities
        assert "AI" in entities
    
    def test_find_relevant_entities(self):
        """Test finding relevant entities."""
        query = "machine learning"
        query_entities = ["Machine Learning"]
        
        # Mock knowledge graph
        mock_entity = Entity(
            id="1",
            name="Machine Learning",
            type=EntityType.CONCEPT
        )
        
        self.reasoning_engine.knowledge_graph.find_similar_entities.return_value = [
            {"entity": mock_entity, "similarity_score": 0.9}
        ]
        
        relevant_entities = self.reasoning_engine._find_relevant_entities(query, query_entities)
        
        assert len(relevant_entities) == 1
        assert relevant_entities[0].name == "Machine Learning"
    
    def test_explain_relationship(self):
        """Test relationship explanation."""
        source_entity = "1"
        target_entity = "2"
        
        # Mock path finding
        mock_path = Mock()
        mock_path.nodes = ["1", "2"]
        mock_path.length = 1
        mock_path.confidence = 0.8
        
        self.reasoning_engine.knowledge_graph.find_path.return_value = mock_path
        
        # Mock entities
        mock_source = Entity(id="1", name="Source", type=EntityType.CONCEPT)
        mock_target = Entity(id="2", name="Target", type=EntityType.CONCEPT)
        
        self.reasoning_engine.knowledge_graph.get_entity.side_effect = lambda x: {
            "1": mock_source,
            "2": mock_target
        }.get(x)
        
        explanation = self.reasoning_engine.explain_relationship(source_entity, target_entity)
        
        assert "explanation" in explanation
        assert "confidence" in explanation
        assert explanation["confidence"] == 0.8


class TestIntegration:
    """Integration tests for core modules."""
    
    @pytest.mark.asyncio
    async def test_full_ingestion_to_reasoning_flow(self):
        """Test full flow from ingestion to reasoning."""
        with patch('rexi.core.ingestion.EmbeddingService'), \
             patch('rexi.core.ingestion.LLMService'), \
             patch('rexi.core.ingestion.KnowledgeGraph'), \
             patch('rexi.core.reasoning.KnowledgeGraph'), \
             patch('rexi.core.reasoning.LLMService'), \
             patch('rexi.core.reasoning.EmbeddingService'):
            
            # Set up ingestion
            ingestion_engine = IngestionEngine()
            ingestion_engine.llm_service.is_available.return_value = True
            ingestion_engine.llm_service.generate_entities_and_relations.return_value = {
                "entities": [{"name": "Test", "type": "concept", "confidence": 0.9}],
                "relations": []
            }
            
            # Set up reasoning
            reasoning_engine = ReasoningEngine()
            reasoning_engine.llm_service.is_available.return_value = True
            reasoning_engine.llm_service.answer_question.return_value = "Test answer"
            
            # Ingest text
            text = "This is a test about machine learning concepts."
            document = await ingestion_engine.ingest_text(text)
            
            assert document is not None
            assert document.processing_status == "completed"
            
            # Query the knowledge graph
            query = "What is machine learning?"
            result = reasoning_engine.answer_query(query)
            
            assert "answer" in result
            assert result["answer"] == "Test answer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
