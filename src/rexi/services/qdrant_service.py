"""
Qdrant service for vector database operations.
"""

from typing import List, Dict, Optional, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import logging

from rexi.config.settings import get_settings

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for Qdrant vector database operations."""
    
    def __init__(self):
        """Initialize Qdrant service."""
        self.settings = get_settings()
        self.client: Optional[QdrantClient] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant."""
        try:
            self.client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port
            )
            logger.info("Connected to Qdrant database")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(self, collection_name: str, vector_size: int = 384) -> bool:
        """Create a new collection."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def upsert_points(
        self, 
        collection_name: str, 
        points: List[PointStruct]
    ) -> bool:
        """Insert or update points in collection."""
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            return False
    
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Filter] = None
    ) -> List[Dict]:
        """Search for similar vectors."""
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_point(self, collection_name: str, point_id: str) -> Optional[Dict]:
        """Get a specific point by ID."""
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id]
            )
            
            if points:
                return {
                    "id": points[0].id,
                    "payload": points[0].payload,
                    "vector": points[0].vector if hasattr(points[0], 'vector') else None
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get point {point_id}: {e}")
            return None
    
    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete points by IDs."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return False
    
    def update_point_payload(
        self, 
        collection_name: str, 
        point_id: str, 
        payload: Dict[str, Any]
    ) -> bool:
        """Update point payload."""
        try:
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point_id]
            )
            logger.info(f"Updated payload for point {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update payload: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def scroll_collection(
        self, 
        collection_name: str, 
        limit: int = 100,
        offset: Optional[int] = None
    ) -> List[Dict]:
        """Scroll through collection points."""
        try:
            points, next_page_offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset
            )
            
            results = []
            for point in points:
                results.append({
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector if hasattr(point, 'vector') else None
                })
            
            return results, next_page_offset
        except Exception as e:
            logger.error(f"Failed to scroll collection: {e}")
            return [], None
