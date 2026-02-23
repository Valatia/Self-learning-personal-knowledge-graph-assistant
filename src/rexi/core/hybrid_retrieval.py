"""
Hybrid retrieval engine for REXI - combines graph, vector, and keyword search.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.models.knowledge_graph import KnowledgeGraphNode, KnowledgeGraphEdge, KnowledgeGraphPath
from rexi.services.neo4j_service import Neo4jService
from rexi.services.qdrant_service import QdrantService
from rexi.services.embedding_service import EmbeddingService
from rexi.core.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class HybridRetrievalEngine:
    """Hybrid retrieval engine combining multiple search methods."""
    
    def __init__(self):
        """Initialize hybrid retrieval engine."""
        self.knowledge_graph = KnowledgeGraph()
        self.neo4j_service = Neo4jService()
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        
        # Retrieval weights
        self.weights = {
            "semantic_similarity": 0.35,
            "graph_relevance": 0.30,
            "temporal_relevance": 0.15,
            "confidence": 0.10,
            "popularity": 0.10
        }
        
        # Search parameters
        self.max_results = 50
        self.similarity_threshold = 0.3
        self.max_graph_depth = 3
        self.temporal_window_days = 365
        
        # Caching
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def hybrid_search(self, query: str, filters: Dict = None, temporal_context: Dict = None) -> Dict:
        """Perform hybrid search combining multiple methods."""
        try:
            # Check cache
            cache_key = self._generate_cache_key(query, filters, temporal_context)
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return cached_result["result"]
            
            search_start = datetime.utcnow()
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_service.encode_text(query)
            
            # Step 2: Vector similarity search
            vector_results = self._vector_similarity_search(query_embedding, filters)
            
            # Step 3: Graph traversal search
            graph_results = self._graph_traversal_search(query, filters)
            
            # Step 4: Keyword search (if applicable)
            keyword_results = self._keyword_search(query, filters)
            
            # Step 5: Temporal filtering
            if temporal_context:
                vector_results = self._apply_temporal_filter(vector_results, temporal_context)
                graph_results = self._apply_temporal_filter(graph_results, temporal_context)
                keyword_results = self._apply_temporal_filter(keyword_results, temporal_context)
            
            # Step 6: Combine and rank results
            combined_results = self._combine_search_results(
                vector_results, graph_results, keyword_results, query_embedding
            )
            
            # Step 7: Apply final ranking
            ranked_results = self._rank_results(combined_results, query, query_embedding)
            
            # Step 8: Generate explanations
            explained_results = self._generate_explanations(ranked_results, query)
            
            search_result = {
                "query": query,
                "results": explained_results[:self.max_results],
                "total_found": len(explained_results),
                "search_method": "hybrid",
                "components": {
                    "vector_search": len(vector_results),
                    "graph_search": len(graph_results),
                    "keyword_search": len(keyword_results)
                },
                "search_time": (datetime.utcnow() - search_start).total_seconds(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache result
            self.query_cache[cache_key] = {
                "result": search_result,
                "timestamp": datetime.utcnow()
            }
            
            logger.info(f"Hybrid search completed: {len(search_result['results'])} results in {search_result['search_time']:.2f}s")
            
            return search_result
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _vector_similarity_search(self, query_embedding: List[float], filters: Dict = None) -> List[Dict]:
        """Perform vector similarity search."""
        try:
            # Search in Qdrant
            search_params = {
                "collection_name": "knowledge_embeddings",
                "query_vector": query_embedding,
                "limit": self.max_results * 2,  # Get more for filtering
                "score_threshold": self.similarity_threshold
            }
            
            if filters:
                search_params["filter"] = self._build_qdrant_filter(filters)
            
            search_results = self.qdrant_service.search_points(**search_params)
            
            # Convert to standard format
            results = []
            for point in search_results:
                result = {
                    "id": point.id,
                    "type": "entity",
                    "score": point.score,
                    "source": "vector_search",
                    "payload": point.payload or {}
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []
    
    def _graph_traversal_search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Perform graph traversal search."""
        try:
            # Extract entities from query
            query_entities = self._extract_query_entities(query)
            
            results = []
            
            for entity_name in query_entities:
                # Find entity in graph
                entity_nodes = self.neo4j_service.find_nodes("Entity", {"name": entity_name})
                
                for node in entity_nodes:
                    # Traverse graph from this node
                    traversal_results = self._traverse_from_node(node["id"], filters)
                    results.extend(traversal_results)
            
            # If no entities found, do general graph search
            if not query_entities:
                general_results = self._general_graph_search(query, filters)
                results.extend(general_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Graph traversal search failed: {e}")
            return []
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query."""
        # Simple extraction - in production, use NER
        import re
        
        # Look for capitalized terms (potential entities)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Filter common words
        common_words = {"The", "This", "That", "What", "When", "Where", "Why", "How"}
        entities = [term for term in capitalized_terms if term not in common_words]
        
        return entities
    
    def _traverse_from_node(self, node_id: str, filters: Dict = None) -> List[Dict]:
        """Traverse graph from a specific node."""
        results = []
        
        try:
            # Get neighbors
            neighbors = self.neo4j_service.get_neighbors(node_id, max_depth=self.max_graph_depth)
            
            for neighbor in neighbors:
                result = {
                    "id": neighbor["id"],
                    "type": neighbor.get("type", "entity"),
                    "score": self._calculate_graph_relevance_score(neighbor),
                    "source": "graph_traversal",
                    "path": neighbor.get("path", []),
                    "properties": neighbor.get("properties", {})
                }
                
                # Apply filters
                if self._passes_filters(result, filters):
                    results.append(result)
            
        except Exception as e:
            logger.error(f"Graph traversal from node {node_id} failed: {e}")
        
        return results
    
    def _general_graph_search(self, query: str, filters: Dict = None) -> List[Dict]:
        """General graph search when no specific entities found."""
        results = []
        
        try:
            # Search for nodes with properties matching query terms
            query_terms = query.lower().split()
            
            for term in query_terms:
                # Search in entity names and descriptions
                cypher_query = """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS $term OR toLower(n.description) CONTAINS $term
                RETURN n, 1.0 as relevance_score
                LIMIT $limit
                """
                
                params = {
                    "term": term,
                    "limit": self.max_results // len(query_terms)
                }
                
                search_results = self.neo4j_service.execute_query(cypher_query, params)
                
                for record in search_results:
                    node = record["n"]
                    result = {
                        "id": node.get("id"),
                        "type": node.get("type", "entity"),
                        "score": record["relevance_score"],
                        "source": "graph_keyword_search",
                        "properties": dict(node)
                    }
                    
                    if self._passes_filters(result, filters):
                        results.append(result)
            
        except Exception as e:
            logger.error(f"General graph search failed: {e}")
        
        return results
    
    def _keyword_search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Perform keyword-based search."""
        results = []
        
        try:
            # This would integrate with a search engine like Elasticsearch
            # For now, implement simple keyword matching in Neo4j
            
            query_terms = [term.lower() for term in query.split() if len(term) > 2]
            
            for term in query_terms:
                # Search in entity names, descriptions, and properties
                cypher_query = """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS $term 
                   OR toLower(n.description) CONTAINS $term
                   OR any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS $term)
                RETURN n, 
                       CASE WHEN toLower(n.name) CONTAINS $term THEN 1.0 
                            WHEN toLower(n.description) CONTAINS $term THEN 0.8 
                            ELSE 0.6 END as keyword_score
                LIMIT $limit
                """
                
                params = {
                    "term": term,
                    "limit": self.max_results // len(query_terms)
                }
                
                search_results = self.neo4j_service.execute_query(cypher_query, params)
                
                for record in search_results:
                    node = record["n"]
                    result = {
                        "id": node.get("id"),
                        "type": node.get("type", "entity"),
                        "score": record["keyword_score"],
                        "source": "keyword_search",
                        "properties": dict(node)
                    }
                    
                    if self._passes_filters(result, filters):
                        results.append(result)
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
        
        return results
    
    def _apply_temporal_filter(self, results: List[Dict], temporal_context: Dict) -> List[Dict]:
        """Apply temporal filtering to results."""
        if not temporal_context:
            return results
        
        filtered_results = []
        
        for result in results:
            if self._passes_temporal_filter(result, temporal_context):
                filtered_results.append(result)
        
        return filtered_results
    
    def _passes_temporal_filter(self, result: Dict, temporal_context: Dict) -> bool:
        """Check if result passes temporal filter."""
        properties = result.get("properties", {})
        
        # Check valid_from/valid_to
        if "valid_from" in temporal_context:
            valid_from = properties.get("valid_from")
            if valid_from:
                if isinstance(valid_from, str):
                    valid_from = datetime.fromisoformat(valid_from.replace('Z', '+00:00'))
                if valid_from > temporal_context["valid_from"]:
                    return False
        
        if "valid_to" in temporal_context:
            valid_to = properties.get("valid_to")
            if valid_to:
                if isinstance(valid_to, str):
                    valid_to = datetime.fromisoformat(valid_to.replace('Z', '+00:00'))
                if valid_to < temporal_context["valid_to"]:
                    return False
        
        # Check time window
        if "time_window" in temporal_context:
            window_start = temporal_context["time_window"]["start"]
            window_end = temporal_context["time_window"]["end"]
            
            created_at = properties.get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                if not (window_start <= created_at <= window_end):
                    return False
        
        return True
    
    def _combine_search_results(self, vector_results: List[Dict], graph_results: List[Dict], 
                              keyword_results: List[Dict], query_embedding: List[float]) -> List[Dict]:
        """Combine results from different search methods."""
        combined = {}
        
        # Add vector results
        for result in vector_results:
            result_id = result["id"]
            combined[result_id] = result
            combined[result_id]["vector_score"] = result["score"]
            combined[result_id]["graph_score"] = 0.0
            combined[result_id]["keyword_score"] = 0.0
        
        # Add graph results
        for result in graph_results:
            result_id = result["id"]
            if result_id in combined:
                combined[result_id]["graph_score"] = result["score"]
                combined[result_id]["sources"] = combined[result_id].get("sources", []) + [result["source"]]
            else:
                result["vector_score"] = 0.0
                result["graph_score"] = result["score"]
                result["keyword_score"] = 0.0
                combined[result_id] = result
        
        # Add keyword results
        for result in keyword_results:
            result_id = result["id"]
            if result_id in combined:
                combined[result_id]["keyword_score"] = result["score"]
                combined[result_id]["sources"] = combined[result_id].get("sources", []) + [result["source"]]
            else:
                result["vector_score"] = 0.0
                result["graph_score"] = 0.0
                result["keyword_score"] = result["score"]
                combined[result_id] = result
        
        return list(combined.values())
    
    def _rank_results(self, results: List[Dict], query: str, query_embedding: List[float]) -> List[Dict]:
        """Rank combined results using hybrid scoring."""
        for result in results:
            # Calculate component scores
            semantic_score = result.get("vector_score", 0.0)
            graph_score = result.get("graph_score", 0.0)
            keyword_score = result.get("keyword_score", 0.0)
            
            # Calculate temporal relevance
            temporal_score = self._calculate_temporal_relevance(result)
            
            # Get confidence
            confidence = result.get("properties", {}).get("confidence", 0.5)
            
            # Calculate popularity (based on connections, etc.)
            popularity = self._calculate_popularity(result)
            
            # Calculate final hybrid score
            final_score = (
                semantic_score * self.weights["semantic_similarity"] +
                graph_score * self.weights["graph_relevance"] +
                temporal_score * self.weights["temporal_relevance"] +
                confidence * self.weights["confidence"] +
                popularity * self.weights["popularity"]
            )
            
            result["final_score"] = final_score
            result["component_scores"] = {
                "semantic": semantic_score,
                "graph": graph_score,
                "keyword": keyword_score,
                "temporal": temporal_score,
                "confidence": confidence,
                "popularity": popularity
            }
        
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return results
    
    def _calculate_temporal_relevance(self, result: Dict) -> float:
        """Calculate temporal relevance score."""
        properties = result.get("properties", {})
        
        # More recent items get higher scores
        created_at = properties.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            days_old = (datetime.utcnow() - created_at).days
            # Exponential decay
            temporal_score = np.exp(-days_old / 365)  # Half-life of 1 year
            return min(temporal_score, 1.0)
        
        return 0.5  # Default score for items without timestamp
    
    def _calculate_popularity(self, result: Dict) -> float:
        """Calculate popularity score based on connections and usage."""
        properties = result.get("properties", {})
        
        # Number of connections
        connection_count = properties.get("connection_count", 0)
        
        # Usage frequency
        usage_count = properties.get("usage_count", 0)
        
        # Normalize and combine
        connection_score = min(connection_count / 10.0, 1.0)  # Normalize to 0-1
        usage_score = min(usage_count / 100.0, 1.0)  # Normalize to 0-1
        
        return (connection_score + usage_score) / 2.0
    
    def _calculate_graph_relevance_score(self, node: Dict) -> float:
        """Calculate graph relevance score for a node."""
        # Base score based on node properties
        base_score = 0.5
        
        # Boost based on node type
        node_type = node.get("type", "entity")
        type_boosts = {
            "person": 0.2,
            "concept": 0.15,
            "project": 0.1,
            "tool": 0.1,
            "organization": 0.15
        }
        base_score += type_boosts.get(node_type, 0.0)
        
        # Boost based on confidence
        confidence = node.get("confidence", 0.5)
        base_score *= confidence
        
        return min(base_score, 1.0)
    
    def _generate_explanations(self, results: List[Dict], query: str) -> List[Dict]:
        """Generate explanations for search results."""
        for result in results:
            explanation_parts = []
            
            # Explain why it was found
            sources = result.get("sources", [result.get("source", "unknown")])
            explanation_parts.append(f"Found via {', '.join(sources)}")
            
            # Explain component scores
            component_scores = result.get("component_scores", {})
            if component_scores:
                high_components = [comp for comp, score in component_scores.items() if score > 0.5]
                if high_components:
                    explanation_parts.append(f"High {', '.join(high_components)} scores")
            
            # Explain relevance
            if result.get("final_score", 0) > 0.7:
                explanation_parts.append("Highly relevant match")
            elif result.get("final_score", 0) > 0.4:
                explanation_parts.append("Moderately relevant")
            
            result["explanation"] = " | ".join(explanation_parts)
        
        return results
    
    def _build_qdrant_filter(self, filters: Dict) -> Dict:
        """Build Qdrant filter from filter dictionary."""
        qdrant_filter = {"must": []}
        
        for key, value in filters.items():
            if key == "entity_type":
                qdrant_filter["must"].append({
                    "key": "type",
                    "match": {"value": value}
                })
            elif key == "confidence_min":
                qdrant_filter["must"].append({
                    "key": "confidence",
                    "range": {"gte": value}
                })
            elif key == "created_after":
                qdrant_filter["must"].append({
                    "key": "created_at",
                    "range": {"gte": value.isoformat()}
                })
        
        return qdrant_filter
    
    def _passes_filters(self, result: Dict, filters: Dict) -> bool:
        """Check if result passes all filters."""
        if not filters:
            return True
        
        properties = result.get("properties", {})
        
        for key, value in filters.items():
            if key == "entity_type" and properties.get("type") != value:
                return False
            elif key == "confidence_min" and properties.get("confidence", 0) < value:
                return False
            elif key == "created_after":
                created_at = properties.get("created_at")
                if created_at:
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if created_at < value:
                        return False
        
        return True
    
    def _generate_cache_key(self, query: str, filters: Dict, temporal_context: Dict) -> str:
        """Generate cache key for query."""
        import hashlib
        
        key_data = f"{query}:{str(filters)}:{str(temporal_context)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: Dict) -> bool:
        """Check if cached result is still valid."""
        cache_time = cached_result.get("timestamp")
        if not cache_time:
            return False
        
        if isinstance(cache_time, str):
            cache_time = datetime.fromisoformat(cache_time.replace('Z', '+00:00'))
        
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl
    
    def get_retrieval_statistics(self) -> Dict:
        """Get statistics about retrieval performance."""
        return {
            "weights": self.weights,
            "search_parameters": {
                "max_results": self.max_results,
                "similarity_threshold": self.similarity_threshold,
                "max_graph_depth": self.max_graph_depth,
                "temporal_window_days": self.temporal_window_days
            },
            "cache_size": len(self.query_cache),
            "cache_ttl": self.cache_ttl,
            "search_methods": ["vector_similarity", "graph_traversal", "keyword_search"]
        }
