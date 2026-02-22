"""
Core knowledge graph module for REXI.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.models.knowledge_graph import KnowledgeGraphNode, KnowledgeGraphEdge, KnowledgeGraphPath
from rexi.services.neo4j_service import Neo4jService
from rexi.services.qdrant_service import QdrantService
from rexi.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Main knowledge graph management class."""
    
    def __init__(self):
        """Initialize knowledge graph."""
        self.neo4j_service = Neo4jService()
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        
        # Ensure vector collection exists
        self._init_vector_collection()
    
    def _init_vector_collection(self):
        """Initialize vector collection for embeddings."""
        collection_name = "knowledge_embeddings"
        if not self.qdrant_service.collection_exists(collection_name):
            self.qdrant_service.create_collection(
                collection_name, 
                self.embedding_service.get_embedding_dimension()
            )
    
    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the knowledge graph."""
        try:
            # Add to Neo4j
            node_properties = {
                "name": entity.name,
                "type": entity.type.value,
                "description": entity.description,
                "confidence": entity.confidence,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "source_count": len(entity.source_references),
                "privacy_level": entity.privacy_level,
                **entity.properties
            }
            
            node_result = self.neo4j_service.create_node("Entity", node_properties)
            
            if not node_result:
                raise ValueError("Failed to create node in Neo4j")
            
            entity.id = str(node_result.id)
            
            # Add to vector database if embedding exists
            if entity.embedding:
                point_id = f"entity_{entity.id}"
                point = {
                    "id": point_id,
                    "vector": entity.embedding,
                    "payload": {
                        "type": "entity",
                        "entity_id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.type.value,
                        "description": entity.description
                    }
                }
                
                from qdrant_client.models import PointStruct
                point_struct = PointStruct(
                    id=point_id,
                    vector=entity.embedding,
                    payload=point["payload"]
                )
                
                self.qdrant_service.upsert_points("knowledge_embeddings", [point_struct])
            
            logger.info(f"Added entity: {entity.name} ({entity.id})")
            return entity.id
            
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            raise
    
    def add_relationship(self, relationship: Relationship) -> str:
        """Add a relationship to the knowledge graph."""
        try:
            # Add to Neo4j
            edge_properties = {
                "type": relationship.type.value,
                "strength_score": relationship.strength_score,
                "confidence": relationship.confidence,
                "created_at": relationship.created_at.isoformat(),
                "updated_at": relationship.updated_at.isoformat(),
                "evidence_count": len(relationship.evidence_references),
                "valid_from": relationship.valid_from.isoformat() if relationship.valid_from else None,
                "valid_to": relationship.valid_to.isoformat() if relationship.valid_to else None,
                **relationship.properties
            }
            
            edge_result = self.neo4j_service.create_relationship(
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.type.value,
                edge_properties
            )
            
            if not edge_result:
                raise ValueError("Failed to create relationship in Neo4j")
            
            relationship.id = str(edge_result.id)
            
            logger.info(f"Added relationship: {relationship.type} ({relationship.id})")
            return relationship.id
            
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            raise
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        try:
            nodes = self.neo4j_service.find_nodes("Entity", {"id": entity_id})
            if not nodes:
                return None
            
            node_data = nodes[0]
            return Entity(
                id=entity_id,
                name=node_data.get("name"),
                type=EntityType(node_data.get("type")),
                description=node_data.get("description"),
                confidence=node_data.get("confidence", 1.0),
                created_at=datetime.fromisoformat(node_data.get("created_at")),
                updated_at=datetime.fromisoformat(node_data.get("updated_at")),
                properties={k: v for k, v in node_data.items() 
                          if k not in ["name", "type", "description", "confidence", "created_at", "updated_at"]},
                privacy_level=node_data.get("privacy_level", "private")
            )
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    def get_relationships(self, entity_id: str = None, relationship_type: str = None) -> List[Relationship]:
        """Get relationships matching criteria."""
        try:
            relationships = []
            results = self.neo4j_service.find_relationships(
                source_id=entity_id, 
                relationship_type=relationship_type
            )
            
            for result in results:
                rel_data = result.get("r", {})
                relationships.append(Relationship(
                    id=str(rel_data.get("id")),
                    source_entity_id=str(result.get("a", {}).get("id")),
                    target_entity_id=str(result.get("b", {}).get("id")),
                    type=RelationshipType(rel_data.get("type")),
                    strength_score=rel_data.get("strength_score", 1.0),
                    confidence=rel_data.get("confidence", 1.0),
                    created_at=datetime.fromisoformat(rel_data.get("created_at")),
                    updated_at=datetime.fromisoformat(rel_data.get("updated_at")),
                    properties={k: v for k, v in rel_data.items() 
                              if k not in ["id", "type", "strength_score", "confidence", "created_at", "updated_at"]}
                ))
            
            return relationships
        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return []
    
    def find_similar_entities(
        self, 
        query_text: str, 
        entity_type: Optional[EntityType] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Find entities similar to query text."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode_text(query_text)
            
            # Search in vector database
            filter_conditions = None
            if entity_type:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                filter_conditions = Filter(
                    must=[
                        FieldCondition(
                            key="entity_type",
                            match=MatchValue(value=entity_type.value)
                        )
                    ]
                )
            
            search_results = self.qdrant_service.search(
                "knowledge_embeddings",
                query_embedding,
                limit=limit,
                filter_conditions=filter_conditions
            )
            
            # Get full entity details
            similar_entities = []
            for result in search_results:
                entity_id = result["payload"].get("entity_id")
                if entity_id:
                    entity = self.get_entity(entity_id)
                    if entity:
                        similar_entities.append({
                            "entity": entity,
                            "similarity_score": result["score"]
                        })
            
            return similar_entities
        except Exception as e:
            logger.error(f"Failed to find similar entities: {e}")
            return []
    
    def find_path(
        self, 
        source_id: str, 
        target_id: str, 
        max_depth: int = 5
    ) -> Optional[KnowledgeGraphPath]:
        """Find path between two entities."""
        try:
            path_results = self.neo4j_service.find_path(source_id, target_id, max_depth)
            
            if not path_results:
                return None
            
            path_data = path_results[0].get("path")
            if not path_data:
                return None
            
            # Extract nodes and edges from path
            nodes = []
            edges = []
            
            for node in path_data.nodes:
                nodes.append(str(node.id))
            
            for relationship in path_data.relationships:
                edges.append(str(relationship.id))
            
            return KnowledgeGraphPath(
                nodes=nodes,
                edges=edges,
                length=len(nodes) - 1,
                confidence=1.0  # Could be calculated based on edge confidences
            )
        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return None
    
    def get_entity_neighbors(self, entity_id: str, depth: int = 1) -> List[Entity]:
        """Get neighboring entities."""
        try:
            neighbor_results = self.neo4j_service.get_node_neighbors(entity_id, depth)
            
            neighbors = []
            for result in neighbor_results:
                neighbor_data = result.get("neighbor")
                if neighbor_data:
                    entity = Entity(
                        id=str(neighbor_data.id),
                        name=neighbor_data.get("name"),
                        type=EntityType(neighbor_data.get("type")),
                        description=neighbor_data.get("description"),
                        confidence=neighbor_data.get("confidence", 1.0)
                    )
                    neighbors.append(entity)
            
            return neighbors
        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []
    
    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update an entity."""
        try:
            success = self.neo4j_service.update_node(entity_id, updates)
            if success:
                logger.info(f"Updated entity {entity_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to update entity: {e}")
            return False
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        try:
            success = self.neo4j_service.delete_node(entity_id)
            if success:
                # Also delete from vector database
                point_id = f"entity_{entity_id}"
                self.qdrant_service.delete_points("knowledge_embeddings", [point_id])
                logger.info(f"Deleted entity {entity_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete entity: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        try:
            # Count entities
            entity_count = len(self.neo4j_service.find_nodes("Entity"))
            
            # Count relationships
            relationship_count = len(self.neo4j_service.find_relationships())
            
            # Get vector collection info
            vector_info = self.qdrant_service.get_collection_info("knowledge_embeddings")
            
            return {
                "entity_count": entity_count,
                "relationship_count": relationship_count,
                "vector_count": vector_info.get("points_count", 0) if vector_info else 0,
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close all connections."""
        try:
            self.neo4j_service.close()
            logger.info("Knowledge graph connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
