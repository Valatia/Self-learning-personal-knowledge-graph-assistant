"""
Neo4j service for knowledge graph operations.
"""

from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase, Driver
import logging

from rexi.config.settings import get_settings

logger = logging.getLogger(__name__)

class Neo4jService:
    """Service for Neo4j graph database operations."""
    
    def __init__(self):
        """Initialize Neo4j service."""
        self.settings = get_settings()
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password)
            )
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Execute a Cypher query."""
        if not self.driver:
            self._connect()
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict:
        """Create a node with given label and properties."""
        query = f"""
        CREATE (n:{label} $properties)
        RETURN n
        """
        result = self.execute_query(query, {"properties": properties})
        return result[0]["n"] if result else None
    
    def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str, 
        properties: Dict[str, Any] = None
    ) -> Dict:
        """Create a relationship between two nodes."""
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $source_id AND id(b) = $target_id
        CREATE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        RETURN r
        """
        result = self.execute_query(query, {
            "source_id": int(source_id),
            "target_id": int(target_id),
            "properties": properties or {}
        })
        return result[0]["r"] if result else None
    
    def find_nodes(self, label: str, properties: Dict[str, Any] = None) -> List[Dict]:
        """Find nodes with given label and properties."""
        query = f"""
        MATCH (n:{label})
        WHERE $properties IS NULL OR all(key IN keys($properties) WHERE n[key] = $properties[key])
        RETURN n
        """
        result = self.execute_query(query, {"properties": properties})
        return [record["n"] for record in result]
    
    def find_relationships(
        self, 
        source_id: str = None, 
        target_id: str = None, 
        relationship_type: str = None
    ) -> List[Dict]:
        """Find relationships matching criteria."""
        conditions = []
        if source_id:
            conditions.append(f"id(a) = {int(source_id)}")
        if target_id:
            conditions.append(f"id(b) = {int(target_id)}")
        if relationship_type:
            conditions.append(f"type(r) = '{relationship_type}'")
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        query = f"""
        MATCH (a)-[r]->(b)
        WHERE {where_clause}
        RETURN r, a, b
        """
        result = self.execute_query(query)
        return result
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> Dict:
        """Update node properties."""
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        SET n += $properties
        RETURN n
        """
        result = self.execute_query(query, {
            "node_id": int(node_id),
            "properties": properties
        })
        return result[0]["n"] if result else None
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships."""
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        result = self.execute_query(query, {"node_id": int(node_id)})
        return result[0]["deleted_count"] > 0 if result else False
    
    def get_node_neighbors(self, node_id: str, depth: int = 1) -> List[Dict]:
        """Get neighboring nodes within specified depth."""
        query = f"""
        MATCH (n)-[r*1..{depth}]-(neighbor)
        WHERE id(n) = $node_id
        RETURN DISTINCT neighbor, r
        """
        result = self.execute_query(query, {"node_id": int(node_id)})
        return result
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> List[Dict]:
        """Find shortest path between two nodes."""
        query = f"""
        MATCH path = shortestPath((a)-[*1..{max_depth}]-(b))
        WHERE id(a) = $source_id AND id(b) = $target_id
        RETURN path
        """
        result = self.execute_query(query, {
            "source_id": int(source_id),
            "target_id": int(target_id)
        })
        return result
