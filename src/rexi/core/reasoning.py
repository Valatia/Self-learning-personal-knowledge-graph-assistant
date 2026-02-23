"""
Reasoning engine for REXI knowledge graph.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.models.knowledge_graph import KnowledgeGraphPath
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService
from rexi.core.knowledge_graph import KnowledgeGraph
from rexi.core.hybrid_retrieval import HybridRetrievalEngine
from rexi.agents.advanced_reasoning import AdvancedReasoningEngine

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """Engine for reasoning over knowledge graph."""
    
    def __init__(self):
        """Initialize reasoning engine."""
        self.knowledge_graph = KnowledgeGraph()
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self.hybrid_retrieval = HybridRetrievalEngine()
        self.advanced_reasoning = AdvancedReasoningEngine()
    
    def answer_query(
        self, 
        query: str, 
        max_hops: int = 3,
        temperature: float = 0.3,
        reasoning_type: str = "auto"
    ) -> Dict[str, Any]:
        """Answer a query using reasoning over the knowledge graph."""
        try:
            # Step 1: Determine reasoning type
            if reasoning_type == "auto":
                reasoning_type = self._determine_reasoning_type(query)
            
            # Step 2: Use appropriate reasoning method
            if reasoning_type in ["multi_hop", "causal", "analogical", "counterfactual"]:
                return self._advanced_reasoning_answer(query, reasoning_type, max_hops, temperature)
            else:
                return self._standard_reasoning_answer(query, max_hops, temperature)
                
        except Exception as e:
            logger.error(f"Query answering failed: {e}")
            return {
                "query": query,
                "answer": "I'm sorry, I couldn't process your query.",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _determine_reasoning_type(self, query: str) -> str:
        """Determine the best reasoning type for the query."""
        query_lower = query.lower()
        
        # Check for complex reasoning patterns
        if any(word in query_lower for word in ["why", "cause", "because", "due to", "leads to"]):
            return "causal"
        elif any(word in query_lower for word in ["what if", "if", "would", "could", "imagine"]):
            return "counterfactual"
        elif any(word in query_lower for word in ["like", "similar", "compare", "analogy"]):
            return "analogical"
        elif any(word in query_lower for word in ["how", "process", "steps", "chain", "sequence"]):
            return "multi_hop"
        else:
            return "standard"
    
    def _advanced_reasoning_answer(self, query: str, reasoning_type: str, max_hops: int, temperature: float) -> Dict:
        """Use advanced reasoning engine for complex queries."""
        try:
            if reasoning_type == "causal":
                result = self.advanced_reasoning.causal_reasoning(query)
            elif reasoning_type == "counterfactual":
                # Extract conditions for counterfactual
                changed_conditions = self._extract_counterfactual_conditions(query)
                result = self.advanced_reasoning.counterfactual_reasoning(query, changed_conditions)
            elif reasoning_type == "analogical":
                # Extract source entity and target domain
                source_entity, target_domain = self._extract_analogical_components(query)
                result = self.advanced_reasoning.analogical_reasoning(source_entity, target_domain, query)
            else:  # multi_hop
                result = self.advanced_reasoning.multi_hop_reasoning(query)
            
            # Format result for consistency
            return {
                "query": query,
                "answer": result.get("final_answer", {}).get("answer", result.get("answer", "No answer generated")),
                "confidence": result.get("confidence", 0.0),
                "reasoning_type": reasoning_type,
                "reasoning_paths": result.get("reasoning_paths", []),
                "explanation": result.get("explanation", ""),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Advanced reasoning failed: {e}")
            return self._fallback_reasoning_answer(query)
    
    def _standard_reasoning_answer(self, query: str, max_hops: int, temperature: float) -> Dict:
        """Use standard reasoning for simple queries."""
        try:
            # Use hybrid retrieval to find relevant information
            retrieval_result = self.hybrid_retrieval.hybrid_search(query)
            
            # Extract relevant entities and relationships
            relevant_entities = []
            relevant_relationships = []
            
            for result in retrieval_result.get("results", []):
                if result.get("type") == "entity":
                    relevant_entities.append(result)
                elif result.get("type") == "relationship":
                    relevant_relationships.append(result)
            
            # Generate answer using LLM
            if relevant_entities or relevant_relationships:
                answer = self._generate_answer_from_context(query, relevant_entities, relevant_relationships, temperature)
            else:
                answer = "I don't have enough information to answer that question."
            
            return {
                "query": query,
                "answer": answer,
                "confidence": self._calculate_answer_confidence(relevant_entities, relevant_relationships),
                "reasoning_type": "standard",
                "relevant_entities": len(relevant_entities),
                "relevant_relationships": len(relevant_relationships),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Standard reasoning failed: {e}")
            return self._fallback_reasoning_answer(query)
    
    def _generate_answer_from_context(self, query: str, entities: List[Dict], relationships: List[Dict], temperature: float) -> str:
        """Generate answer from retrieved context using LLM."""
        try:
            # Build context string
            context_parts = []
            
            if entities:
                context_parts.append("Relevant entities:")
                for entity in entities[:5]:  # Limit to top 5
                    props = entity.get("properties", {})
                    context_parts.append(f"- {props.get('name', 'Unknown')}: {props.get('description', 'No description')}")
            
            if relationships:
                context_parts.append("\nRelevant relationships:")
                for rel in relationships[:5]:  # Limit to top 5
                    props = rel.get("properties", {})
                    context_parts.append(f"- {props.get('source', 'Unknown')} {props.get('type', 'related to')} {props.get('target', 'Unknown')}")
            
            context = "\n".join(context_parts)
            
            # Generate answer
            system_prompt = """
            You are a knowledgeable assistant. Answer the user's question based on the provided context.
            Be concise, accurate, and helpful. If the context doesn't contain enough information, say so.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            response = self.llm_service.chat_completion(
                messages,
                temperature=temperature,
                max_tokens=500
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I couldn't generate an answer based on the available information."
    
    def _calculate_answer_confidence(self, entities: List[Dict], relationships: List[Dict]) -> float:
        """Calculate confidence in the answer based on retrieved information."""
        if not entities and not relationships:
            return 0.0
        
        # Base confidence from number of relevant items
        base_confidence = min((len(entities) + len(relationships)) / 10.0, 1.0)
        
        # Adjust based on average scores
        scores = []
        for entity in entities:
            scores.append(entity.get("score", 0.5))
        for rel in relationships:
            scores.append(rel.get("score", 0.5))
        
        if scores:
            avg_score = sum(scores) / len(scores)
            return base_confidence * avg_score
        
        return base_confidence
    
    def _extract_counterfactual_conditions(self, query: str) -> List[Dict]:
        """Extract conditions for counterfactual reasoning."""
        # Simple extraction - in production, use more sophisticated NLP
        conditions = []
        
        # Look for "if X then Y" patterns
        import re
        
        patterns = [
            r"if (.+?) then (.+)",
            r"if (.+?), (.+)",
            r"what if (.+?) (.+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    conditions.append({
                        "condition": match[0],
                        "consequence": match[1],
                        "type": "conditional"
                    })
        
        return conditions
    
    def _extract_analogical_components(self, query: str) -> Tuple[str, str]:
        """Extract source entity and target domain for analogical reasoning."""
        # Simple extraction - in production, use more sophisticated NLP
        import re
        
        # Look for "X is like Y" or "compare X to Y" patterns
        patterns = [
            r"(.+?) is like (.+)",
            r"compare (.+?) to (.+)",
            r"(.+?) similar to (.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        # Fallback: return first two capitalized terms
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        if len(capitalized_terms) >= 2:
            return capitalized_terms[0], capitalized_terms[1]
        
        return "", ""
    
    def _fallback_reasoning_answer(self, query: str) -> Dict:
        """Fallback reasoning answer when advanced methods fail."""
        return {
            "query": query,
            "answer": "I'm sorry, I couldn't process your query. Please try rephrasing it.",
            "confidence": 0.1,
            "reasoning_type": "fallback",
            "explanation": "Advanced reasoning failed, using fallback response",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def find_insights(self, entity_types: List[EntityType] = None) -> List[Dict[str, Any]]:
        """Find insights in the knowledge graph."""
        try:
            insights = []
            
            # Insight 1: Find highly connected entities
            insights.extend(self._find_hub_entities(entity_types))
            
            # Insight 2: Find missing connections
            insights.extend(self._find_missing_connections(entity_types))
            
            # Insight 3: Find temporal patterns
            insights.extend(self._find_temporal_patterns(entity_types))
            
            # Insight 4: Find concept clusters
            insights.extend(self._find_concept_clusters(entity_types))
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight discovery failed: {e}")
            return []
    
    def explain_relationship(
        self, 
        source_entity: str, 
        target_entity: str, 
        relationship_type: str = None
    ) -> Dict[str, Any]:
        """Explain the relationship between two entities."""
        try:
            # Find path between entities
            path = self.knowledge_graph.find_path(source_entity, target_entity)
            
            if not path:
                return {
                    "explanation": f"No direct or indirect relationship found between the entities.",
                    "confidence": 0.0,
                    "path": None
                }
            
            # Get detailed path information
            path_details = []
            for i in range(len(path.nodes) - 1):
                node_id = path.nodes[i]
                next_node_id = path.nodes[i + 1]
                
                # Get relationship details
                relationships = self.knowledge_graph.get_relationships(
                    entity_id=node_id
                )
                
                rel = None
                for r in relationships:
                    if r.target_entity_id == next_node_id:
                        rel = r
                        break
                
                if rel:
                    path_details.append({
                        "from": self.knowledge_graph.get_entity(node_id),
                        "relationship": rel,
                        "to": self.knowledge_graph.get_entity(next_node_id)
                    })
            
            # Generate explanation
            explanation = self._generate_path_explanation(path_details)
            
            return {
                "explanation": explanation,
                "confidence": path.confidence,
                "path": path_details,
                "path_length": path.length
            }
            
        except Exception as e:
            logger.error(f"Relationship explanation failed: {e}")
            return {
                "explanation": "Failed to explain relationship due to an error.",
                "error": str(e)
            }
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities from query."""
        try:
            # Simple entity extraction - could be enhanced with NER
            words = query.split()
            entities = []
            
            # Look for capitalized words (potential entities)
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    entities.append(word)
            
            return entities
        except Exception as e:
            logger.error(f"Query entity extraction failed: {e}")
            return []
    
    def _find_relevant_entities(self, query: str, query_entities: List[str]) -> List[Entity]:
        """Find entities relevant to the query."""
        try:
            relevant_entities = []
            
            # Method 1: Direct entity matching
            for entity_name in query_entities:
                entities = self.knowledge_graph.find_similar_entities(entity_name, limit=5)
                for result in entities:
                    if result["similarity_score"] > 0.8:
                        relevant_entities.append(result["entity"])
            
            # Method 2: Semantic similarity search
            if not relevant_entities:
                similar_entities = self.knowledge_graph.find_similar_entities(query, limit=10)
                for result in similar_entities:
                    if result["similarity_score"] > 0.6:
                        relevant_entities.append(result["entity"])
            
            return relevant_entities[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Relevant entity finding failed: {e}")
            return []
    
    def _build_reasoning_subgraph(
        self, 
        entities: List[Entity], 
        max_hops: int
    ) -> Dict[str, Any]:
        """Build a subgraph for reasoning."""
        try:
            subgraph = {
                "nodes": {},
                "edges": [],
                "entity_ids": [entity.id for entity in entities]
            }
            
            # Add initial entities
            for entity in entities:
                subgraph["nodes"][entity.id] = entity
            
            # Expand graph by hops
            for entity in entities:
                neighbors = self.knowledge_graph.get_entity_neighbors(entity.id, max_hops)
                
                for neighbor in neighbors:
                    if neighbor.id not in subgraph["nodes"]:
                        subgraph["nodes"][neighbor.id] = neighbor
                
                # Get relationships
                relationships = self.knowledge_graph.get_relationships(entity.id)
                for rel in relationships:
                    if (rel.source_entity_id in subgraph["entity_ids"] or 
                        rel.target_entity_id in subgraph["entity_ids"] or
                        rel.source_entity_id in subgraph["nodes"] or
                        rel.target_entity_id in subgraph["nodes"]):
                        subgraph["edges"].append(rel)
            
            return subgraph
            
        except Exception as e:
            logger.error(f"Subgraph building failed: {e}")
            return {"nodes": {}, "edges": []}
    
    def _perform_reasoning(self, query: str, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reasoning on the subgraph."""
        try:
            reasoning_result = {
                "path": [],
                "confidence": 0.5,
                "evidence": [],
                "entities_used": []
            }
            
            # Simple reasoning: find most relevant path
            if not subgraph["edges"]:
                return reasoning_result
            
            # Score entities based on query similarity
            query_embedding = self.embedding_service.encode_text(query)
            
            scored_entities = []
            for entity_id, entity in subgraph["nodes"].items():
                if entity.embedding:
                    similarity = self.embedding_service.compute_similarity(
                        query_embedding, entity.embedding
                    )
                    scored_entities.append((entity, similarity))
            
            # Sort by similarity
            scored_entities.sort(key=lambda x: x[1], reverse=True)
            
            # Select top entities for reasoning
            top_entities = scored_entities[:5]
            reasoning_result["entities_used"] = [e[0] for e in top_entities]
            reasoning_result["confidence"] = top_entities[0][1] if top_entities else 0.5
            
            # Find reasoning path
            if len(top_entities) >= 2:
                path = self.knowledge_graph.find_path(
                    top_entities[0][0].id,
                    top_entities[1][0].id
                )
                if path:
                    reasoning_result["path"] = path.nodes
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {"confidence": 0.0, "error": str(e)}
    
    def _generate_answer(
        self, 
        query: str, 
        reasoning_result: Dict[str, Any], 
        temperature: float
    ) -> str:
        """Generate answer based on reasoning result."""
        try:
            if not self.llm_service.is_available():
                return "LLM service not available for answer generation."
            
            # Prepare context
            entities_used = reasoning_result.get("entities_used", [])
            context_parts = []
            
            for entity in entities_used:
                context_parts.append(f"Entity: {entity.name}")
                if entity.description:
                    context_parts.append(f"Description: {entity.description}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Generate answer
            answer = self.llm_service.answer_question(query, context, temperature)
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Failed to generate answer due to an error."
    
    def _find_hub_entities(self, entity_types: List[EntityType]) -> List[Dict[str, Any]]:
        """Find highly connected entities (hubs)."""
        try:
            hubs = []
            
            # This would require degree calculation in Neo4j
            # For now, return empty list
            return hubs
            
        except Exception as e:
            logger.error(f"Hub entity finding failed: {e}")
            return []
    
    def _find_missing_connections(self, entity_types: List[EntityType]) -> List[Dict[str, Any]]:
        """Find potentially missing connections."""
        try:
            missing_connections = []
            
            # This would require more sophisticated analysis
            # For now, return empty list
            return missing_connections
            
        except Exception as e:
            logger.error(f"Missing connection finding failed: {e}")
            return []
    
    def _find_temporal_patterns(self, entity_types: List[EntityType]) -> List[Dict[str, Any]]:
        """Find temporal patterns in the knowledge graph."""
        try:
            patterns = []
            
            # This would require temporal analysis
            # For now, return empty list
            return patterns
            
        except Exception as e:
            logger.error(f"Temporal pattern finding failed: {e}")
            return []
    
    def _find_concept_clusters(self, entity_types: List[EntityType]) -> List[Dict[str, Any]]:
        """Find concept clusters in the knowledge graph."""
        try:
            clusters = []
            
            # This would require clustering algorithms
            # For now, return empty list
            return clusters
            
        except Exception as e:
            logger.error(f"Concept clustering failed: {e}")
            return []
    
    def _generate_path_explanation(self, path_details: List[Dict[str, Any]]) -> str:
        """Generate explanation for a path."""
        try:
            if not path_details:
                return "No path found."
            
            explanation_parts = []
            
            for i, step in enumerate(path_details):
                from_entity = step["from"]
                relationship = step["relationship"]
                to_entity = step["to"]
                
                if i == 0:
                    explanation_parts.append(f"Starting with {from_entity.name}")
                
                explanation_parts.append(
                    f"{from_entity.name} {relationship.type.value.replace('_', ' ')} {to_entity.name}"
                )
            
            return " → ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Path explanation generation failed: {e}")
            return "Failed to generate path explanation."
    
    def close(self):
        """Close reasoning engine."""
        try:
            self.knowledge_graph.close()
            logger.info("Reasoning engine closed")
        except Exception as e:
            logger.error(f"Error closing reasoning engine: {e}")
