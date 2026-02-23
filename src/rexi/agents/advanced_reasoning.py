"""
Advanced reasoning agent for REXI - implements multi-hop and complex reasoning.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.models.knowledge_graph import KnowledgeGraphNode, KnowledgeGraphEdge, KnowledgeGraphPath
from rexi.services.neo4j_service import Neo4jService
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService
from rexi.core.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class AdvancedReasoningEngine:
    """Advanced reasoning engine with multi-hop, causal, and analogical reasoning."""
    
    def __init__(self):
        """Initialize advanced reasoning engine."""
        self.knowledge_graph = KnowledgeGraph()
        self.neo4j_service = Neo4jService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
        # Reasoning parameters
        self.max_hops = 5
        self.confidence_threshold = 0.6
        self.max_paths = 10
        self.similarity_threshold = 0.7
        
        # Reasoning patterns
        self.reasoning_patterns = {
            "causal": ["causes", "leads_to", "results_in", "triggers"],
            "temporal": ["precedes", "follows", "before", "after"],
            "spatial": ["contains", "within", "outside", "adjacent"],
            "hierarchical": ["part_of", "contains", "subordinate", "superordinate"],
            "functional": ["enables", "requires", "used_for", "function_of"],
            "comparative": ["similar_to", "different_from", "better_than", "worse_than"]
        }
        
        # Reasoning cache
        self.reasoning_cache = {}
        self.cache_ttl = 1800  # 30 minutes
    
    def multi_hop_reasoning(self, query: str, start_entities: List[str] = None) -> Dict:
        """Perform multi-hop reasoning to answer complex queries."""
        try:
            reasoning_start = datetime.utcnow()
            
            # Step 1: Parse query and identify reasoning type
            parsed_query = self._parse_reasoning_query(query)
            reasoning_type = parsed_query["type"]
            
            # Step 2: Identify starting entities
            if not start_entities:
                start_entities = self._extract_query_entities(query)
            
            # Step 3: Build reasoning graph
            reasoning_graph = self._build_reasoning_graph(start_entities, reasoning_type, parsed_query)
            
            # Step 4: Find reasoning paths
            reasoning_paths = self._find_reasoning_paths(reasoning_graph, reasoning_type, parsed_query)
            
            # Step 5: Score and rank paths
            scored_paths = self._score_reasoning_paths(reasoning_paths, query)
            
            # Step 6: Generate explanations
            explained_paths = self._generate_path_explanations(scored_paths, query)
            
            # Step 7: Synthesize final answer
            final_answer = self._synthesize_reasoning_answer(explained_paths, query, reasoning_type)
            
            reasoning_result = {
                "query": query,
                "reasoning_type": reasoning_type,
                "start_entities": start_entities,
                "reasoning_paths": explained_paths[:self.max_paths],
                "final_answer": final_answer,
                "confidence": final_answer.get("confidence", 0.0),
                "reasoning_time": (datetime.utcnow() - reasoning_start).total_seconds(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache result
            cache_key = self._generate_reasoning_cache_key(query, start_entities)
            self.reasoning_cache[cache_key] = {
                "result": reasoning_result,
                "timestamp": datetime.utcnow()
            }
            
            logger.info(f"Multi-hop reasoning completed: {reasoning_type} in {reasoning_result['reasoning_time']:.2f}s")
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Multi-hop reasoning failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def causal_reasoning(self, query: str, cause_entity: str = None, effect_entity: str = None) -> Dict:
        """Perform causal reasoning to understand cause-effect relationships."""
        try:
            # Step 1: Identify causal relationships
            causal_paths = self._find_causal_paths(cause_entity, effect_entity)
            
            # Step 2: Analyze causal chains
            causal_chains = self._analyze_causal_chains(causal_paths)
            
            # Step 3: Evaluate causal strength
            evaluated_chains = self._evaluate_causal_strength(causal_chains)
            
            # Step 4: Generate causal explanations
            causal_explanations = self._generate_causal_explanations(evaluated_chains, query)
            
            # Step 5: Synthesize causal answer
            causal_answer = self._synthesize_causal_answer(causal_explanations, query)
            
            return {
                "query": query,
                "reasoning_type": "causal",
                "cause_entity": cause_entity,
                "effect_entity": effect_entity,
                "causal_chains": evaluated_chains,
                "explanations": causal_explanations,
                "answer": causal_answer,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Causal reasoning failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def analogical_reasoning(self, source_entity: str, target_domain: str, query: str) -> Dict:
        """Perform analogical reasoning to transfer knowledge between domains."""
        try:
            # Step 1: Analyze source entity structure
            source_structure = self._analyze_entity_structure(source_entity)
            
            # Step 2: Find similar structures in target domain
            analogical_matches = self._find_analogical_matches(source_structure, target_domain)
            
            # Step 3: Map structural relationships
            structural_mappings = self._map_structural_relationships(source_structure, analogical_matches)
            
            # Step 4: Generate analogical inferences
            analogical_inferences = self._generate_analogical_inferences(structural_mappings, query)
            
            # Step 5: Evaluate analogy quality
            evaluated_analogies = self._evaluate_analogy_quality(analogical_inferences)
            
            return {
                "query": query,
                "reasoning_type": "analogical",
                "source_entity": source_entity,
                "target_domain": target_domain,
                "source_structure": source_structure,
                "analogical_matches": analogical_matches,
                "inferences": evaluated_analogies,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analogical reasoning failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def counterfactual_reasoning(self, query: str, changed_conditions: List[Dict]) -> Dict:
        """Perform counterfactual reasoning to explore alternative scenarios."""
        try:
            # Step 1: Identify current state
            current_state = self._identify_current_state(query)
            
            # Step 2: Apply counterfactual conditions
            counterfactual_states = []
            for condition in changed_conditions:
                counterfactual_state = self._apply_counterfactual_condition(current_state, condition)
                counterfactual_states.append(counterfactual_state)
            
            # Step 3: Reason about counterfactual consequences
            counterfactual_consequences = []
            for state in counterfactual_states:
                consequences = self._reason_counterfactual_consequences(state, query)
                counterfactual_consequences.append(consequences)
            
            # Step 4: Compare with actual outcomes
            comparison = self._compare_counterfactuals(counterfactual_consequences, current_state)
            
            # Step 5: Generate counterfactual explanation
            counterfactual_explanation = self._generate_counterfactual_explanation(comparison, query)
            
            return {
                "query": query,
                "reasoning_type": "counterfactual",
                "current_state": current_state,
                "changed_conditions": changed_conditions,
                "counterfactual_states": counterfactual_states,
                "consequences": counterfactual_consequences,
                "comparison": comparison,
                "explanation": counterfactual_explanation,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Counterfactual reasoning failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def concept_synthesis(self, concepts: List[str], synthesis_type: str = "integration") -> Dict:
        """Synthesize new concepts from existing ones."""
        try:
            # Step 1: Analyze concept relationships
            concept_relationships = self._analyze_concept_relationships(concepts)
            
            # Step 2: Identify synthesis patterns
            synthesis_patterns = self._identify_synthesis_patterns(concept_relationships, synthesis_type)
            
            # Step 3: Generate synthesized concepts
            synthesized_concepts = self._generate_synthesized_concepts(synthesis_patterns)
            
            # Step 4: Evaluate synthesis quality
            evaluated_synthesis = self._evaluate_synthesis_quality(synthesized_concepts)
            
            # Step 5: Create synthesis explanations
            synthesis_explanations = self._create_synthesis_explanations(evaluated_synthesis, concepts)
            
            return {
                "concepts": concepts,
                "synthesis_type": synthesis_type,
                "concept_relationships": concept_relationships,
                "synthesis_patterns": synthesis_patterns,
                "synthesized_concepts": evaluated_synthesis,
                "explanations": synthesis_explanations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Concept synthesis failed: {e}")
            return {
                "concepts": concepts,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def evidence_aggregation(self, claim: str, evidence_sources: List[str] = None) -> Dict:
        """Aggregate and evaluate evidence for a claim."""
        try:
            # Step 1: Find supporting evidence
            supporting_evidence = self._find_supporting_evidence(claim, evidence_sources)
            
            # Step 2: Find contradicting evidence
            contradicting_evidence = self._find_contradicting_evidence(claim, evidence_sources)
            
            # Step 3: Evaluate evidence quality
            evaluated_supporting = self._evaluate_evidence_quality(supporting_evidence)
            evaluated_contradicting = self._evaluate_evidence_quality(contradicting_evidence)
            
            # Step 4: Aggregate evidence weights
            evidence_weights = self._aggregate_evidence_weights(evaluated_supporting, evaluated_contradicting)
            
            # Step 5: Generate evidence assessment
            evidence_assessment = self._generate_evidence_assessment(evidence_weights, claim)
            
            return {
                "claim": claim,
                "supporting_evidence": evaluated_supporting,
                "contradicting_evidence": evaluated_contradicting,
                "evidence_weights": evidence_weights,
                "assessment": evidence_assessment,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evidence aggregation failed: {e}")
            return {
                "claim": claim,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _parse_reasoning_query(self, query: str) -> Dict:
        """Parse query to determine reasoning type and parameters."""
        query_lower = query.lower()
        
        # Determine reasoning type
        reasoning_type = "general"
        if any(word in query_lower for word in ["why", "cause", "because", "due to"]):
            reasoning_type = "causal"
        elif any(word in query_lower for word in ["how", "process", "steps", "procedure"]):
            reasoning_type = "procedural"
        elif any(word in query_lower for word in ["compare", "similar", "different", "like"]):
            reasoning_type = "comparative"
        elif any(word in query_lower for word in ["what if", "if", "would", "could"]):
            reasoning_type = "counterfactual"
        elif any(word in query_lower for word in ["relate", "connect", "link", "associate"]):
            reasoning_type = "relational"
        
        return {
            "type": reasoning_type,
            "original_query": query,
            "keywords": self._extract_keywords(query)
        }
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entity names from query."""
        # Simple extraction - in production, use NER
        import re
        
        # Look for capitalized terms
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Filter common words
        common_words = {"The", "This", "That", "What", "When", "Where", "Why", "How", "It"}
        entities = [term for term in capitalized_terms if term not in common_words and len(term) > 2]
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction
        import re
        
        # Remove stop words and extract meaningful terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "what", "when", "where", "why", "how"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _build_reasoning_graph(self, start_entities: List[str], reasoning_type: str, parsed_query: Dict) -> Dict:
        """Build a reasoning graph from starting entities."""
        reasoning_graph = {
            "nodes": {},
            "edges": [],
            "type": reasoning_type
        }
        
        for entity_name in start_entities:
            # Find entity in knowledge graph
            entity_nodes = self.neo4j_service.find_nodes("Entity", {"name": entity_name})
            
            for node in entity_nodes:
                reasoning_graph["nodes"][node["id"]] = node
                
                # Get neighbors based on reasoning type
                neighbors = self._get_reasoning_neighbors(node["id"], reasoning_type)
                
                for neighbor in neighbors:
                    reasoning_graph["nodes"][neighbor["id"]] = neighbor
                    
                    # Add edge
                    edge = {
                        "source": node["id"],
                        "target": neighbor["id"],
                        "relationship": neighbor.get("relationship", "related_to"),
                        "confidence": neighbor.get("confidence", 0.5)
                    }
                    reasoning_graph["edges"].append(edge)
        
        return reasoning_graph
    
    def _get_reasoning_neighbors(self, entity_id: str, reasoning_type: str) -> List[Dict]:
        """Get neighbors relevant to reasoning type."""
        if reasoning_type == "causal":
            # Focus on causal relationships
            causal_rels = ["causes", "leads_to", "results_in", "triggers"]
            return self.neo4j_service.get_neighbors_by_relationships(entity_id, causal_rels, max_depth=2)
        elif reasoning_type == "procedural":
            # Focus on sequential relationships
            sequential_rels = ["precedes", "follows", "step", "phase"]
            return self.neo4j_service.get_neighbors_by_relationships(entity_id, sequential_rels, max_depth=3)
        elif reasoning_type == "comparative":
            # Focus on similarity relationships
            similarity_rels = ["similar_to", "different_from", "related_to"]
            return self.neo4j_service.get_neighbors_by_relationships(entity_id, similarity_rels, max_depth=2)
        else:
            # General reasoning - get all neighbors
            return self.neo4j_service.get_neighbors(entity_id, max_depth=2)
    
    def _find_reasoning_paths(self, reasoning_graph: Dict, reasoning_type: str, parsed_query: Dict) -> List[Dict]:
        """Find reasoning paths in the graph."""
        paths = []
        
        # Find paths between entities
        nodes = list(reasoning_graph["nodes"].keys())
        
        for i, start_node in enumerate(nodes):
            for end_node in nodes[i+1:]:
                # Find paths between start and end nodes
                graph_paths = self._find_graph_paths(start_node, end_node, reasoning_graph)
                
                for path in graph_paths:
                    if self._is_reasoning_path_valid(path, reasoning_type, parsed_query):
                        paths.append(path)
        
        return paths
    
    def _find_graph_paths(self, start_id: str, end_id: str, graph: Dict, max_length: int = None) -> List[List[str]]:
        """Find all paths between two nodes in the graph."""
        if max_length is None:
            max_length = self.max_hops
        
        paths = []
        visited = set()
        current_path = [start_id]
        
        def dfs(current_id, depth):
            if depth > max_length:
                return
            
            if current_id == end_id and depth > 0:
                paths.append(current_path.copy())
                return
            
            visited.add(current_id)
            
            # Find neighbors
            neighbors = []
            for edge in graph["edges"]:
                if edge["source"] == current_id and edge["target"] not in visited:
                    neighbors.append(edge["target"])
                elif edge["target"] == current_id and edge["source"] not in visited:
                    neighbors.append(edge["source"])
            
            for neighbor in neighbors:
                current_path.append(neighbor)
                dfs(neighbor, depth + 1)
                current_path.pop()
            
            visited.remove(current_id)
        
        dfs(start_id, 0)
        return paths
    
    def _is_reasoning_path_valid(self, path: List[str], reasoning_type: str, parsed_query: Dict) -> bool:
        """Check if a reasoning path is valid for the given type."""
        if len(path) < 2:
            return False
        
        # Check if path contains relevant relationships
        path_relationships = self._get_path_relationships(path)
        
        if reasoning_type == "causal":
            return any(rel in self.reasoning_patterns["causal"] for rel in path_relationships)
        elif reasoning_type == "procedural":
            return any(rel in self.reasoning_patterns["temporal"] for rel in path_relationships)
        elif reasoning_type == "comparative":
            return any(rel in self.reasoning_patterns["comparative"] for rel in path_relationships)
        
        return True
    
    def _get_path_relationships(self, path: List[str]) -> List[str]:
        """Get relationships along a path."""
        relationships = []
        
        for i in range(len(path) - 1):
            # Find edge between consecutive nodes
            edge_found = False
            for edge in self.neo4j_service.get_relationships_between(path[i], path[i+1]):
                relationships.append(edge.get("type", "related_to"))
                edge_found = True
                break
            
            if not edge_found:
                relationships.append("related_to")
        
        return relationships
    
    def _score_reasoning_paths(self, paths: List[List[str]], query: str) -> List[Dict]:
        """Score reasoning paths based on relevance and quality."""
        scored_paths = []
        
        for path in paths:
            # Calculate path score
            path_score = self._calculate_path_score(path, query)
            
            scored_path = {
                "path": path,
                "score": path_score,
                "length": len(path),
                "relationships": self._get_path_relationships(path)
            }
            
            scored_paths.append(scored_path)
        
        # Sort by score
        scored_paths.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_paths
    
    def _calculate_path_score(self, path: List[str], query: str) -> float:
        """Calculate score for a reasoning path."""
        if not path:
            return 0.0
        
        # Base score
        base_score = 1.0
        
        # Length penalty (shorter paths are better)
        length_penalty = max(0.5, 1.0 - (len(path) - 2) * 0.1)
        
        # Relationship relevance
        relationships = self._get_path_relationships(path)
        relationship_score = sum(0.8 if rel in ["causes", "enables", "similar_to"] else 0.5 for rel in relationships) / len(relationships)
        
        # Node relevance to query
        node_relevance = self._calculate_node_relevance(path, query)
        
        # Confidence
        confidence = self._calculate_path_confidence(path)
        
        final_score = base_score * length_penalty * relationship_score * node_relevance * confidence
        
        return min(final_score, 1.0)
    
    def _calculate_node_relevance(self, path: List[str], query: str) -> float:
        """Calculate relevance of path nodes to query."""
        query_keywords = set(self._extract_keywords(query))
        
        relevant_nodes = 0
        for node_id in path:
            node = self.neo4j_service.get_node(node_id)
            if node:
                node_text = f"{node.get('name', '')} {node.get('description', '')}".lower()
                node_keywords = set(self._extract_keywords(node_text))
                
                if query_keywords.intersection(node_keywords):
                    relevant_nodes += 1
        
        return relevant_nodes / len(path) if path else 0.0
    
    def _calculate_path_confidence(self, path: List[str]) -> float:
        """Calculate average confidence along path."""
        confidences = []
        
        for i in range(len(path) - 1):
            # Get relationship confidence
            relationships = self.neo4j_service.get_relationships_between(path[i], path[i+1])
            if relationships:
                rel_confidence = relationships[0].get("confidence", 0.5)
                confidences.append(rel_confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _generate_path_explanations(self, scored_paths: List[Dict], query: str) -> List[Dict]:
        """Generate explanations for reasoning paths."""
        explained_paths = []
        
        for scored_path in scored_paths:
            path = scored_path["path"]
            
            # Generate explanation
            explanation = self._generate_path_explanation(path, query)
            
            explained_path = scored_path.copy()
            explained_path["explanation"] = explanation
            
            explained_paths.append(explained_path)
        
        return explained_paths
    
    def _generate_path_explanation(self, path: List[str], query: str) -> str:
        """Generate explanation for a single reasoning path."""
        if len(path) < 2:
            return "Insufficient path for explanation"
        
        # Get node names
        node_names = []
        for node_id in path:
            node = self.neo4j_service.get_node(node_id)
            if node:
                node_names.append(node.get("name", "Unknown"))
        
        # Get relationships
        relationships = self._get_path_relationships(path)
        
        # Build explanation
        explanation_parts = []
        explanation_parts.append(f"Reasoning path: {' → '.join(node_names)}")
        
        if relationships:
            explanation_parts.append(f"Relationships: {' → '.join(relationships)}")
        
        explanation_parts.append(f"Path length: {len(path)} steps")
        
        return " | ".join(explanation_parts)
    
    def _synthesize_reasoning_answer(self, explained_paths: List[Dict], query: str, reasoning_type: str) -> Dict:
        """Synthesize final answer from reasoning paths."""
        if not explained_paths:
            return {
                "answer": "No reasoning paths found",
                "confidence": 0.0,
                "explanation": "Unable to find relevant reasoning paths"
            }
        
        # Get best path
        best_path = explained_paths[0]
        
        # Generate answer based on reasoning type
        if reasoning_type == "causal":
            answer = self._synthesize_causal_answer(best_path, query)
        elif reasoning_type == "procedural":
            answer = self._synthesize_procedural_answer(best_path, query)
        elif reasoning_type == "comparative":
            answer = self._synthesize_comparative_answer(best_path, query)
        else:
            answer = self._synthesize_general_answer(best_path, query)
        
        return answer
    
    def _synthesize_causal_answer(self, best_path: Dict, query: str) -> Dict:
        """Synthesize causal reasoning answer."""
        path = best_path["path"]
        
        # Extract causal chain
        causal_chain = []
        for i in range(len(path) - 1):
            relationships = self.neo4j_service.get_relationships_between(path[i], path[i+1])
            if relationships and relationships[0].get("type") in self.reasoning_patterns["causal"]:
                causal_chain.append({
                    "cause": path[i],
                    "effect": path[i+1],
                    "relationship": relationships[0].get("type")
                })
        
        answer = f"Based on causal reasoning: "
        if causal_chain:
            answer += f"{' → '.join([step['cause'] + ' causes ' + step['effect'] for step in causal_chain])}"
        else:
            answer += "No clear causal chain identified"
        
        return {
            "answer": answer,
            "confidence": best_path["score"],
            "causal_chain": causal_chain,
            "explanation": best_path["explanation"]
        }
    
    def _synthesize_procedural_answer(self, best_path: Dict, query: str) -> Dict:
        """Synthesize procedural reasoning answer."""
        path = best_path["path"]
        
        answer = f"Based on procedural reasoning: The process involves {len(path)} steps: "
        step_names = []
        
        for node_id in path:
            node = self.neo4j_service.get_node(node_id)
            if node:
                step_names.append(node.get("name", "Unknown step"))
        
        answer += " → ".join(step_names)
        
        return {
            "answer": answer,
            "confidence": best_path["score"],
            "steps": step_names,
            "explanation": best_path["explanation"]
        }
    
    def _synthesize_comparative_answer(self, best_path: Dict, query: str) -> Dict:
        """Synthesize comparative reasoning answer."""
        path = best_path["path"]
        
        if len(path) >= 2:
            node1 = self.neo4j_service.get_node(path[0])
            node2 = self.neo4j_service.get_node(path[1])
            
            if node1 and node2:
                answer = f"Based on comparative reasoning: {node1.get('name', 'Entity 1')} and {node2.get('name', 'Entity 2')} "
                
                relationships = self.neo4j_service.get_relationships_between(path[0], path[1])
                if relationships:
                    rel_type = relationships[0].get("type", "related")
                    answer += f"are {rel_type.replace('_', ' ')}"
                else:
                    answer += "are related"
        
        return {
            "answer": answer,
            "confidence": best_path["score"],
            "comparison": {
                "entity1": node1.get("name") if node1 else "Unknown",
                "entity2": node2.get("name") if node2 else "Unknown"
            },
            "explanation": best_path["explanation"]
        }
    
    def _synthesize_general_answer(self, best_path: Dict, query: str) -> Dict:
        """Synthesize general reasoning answer."""
        path = best_path["path"]
        
        answer = f"Based on reasoning analysis: Found a connection path of {len(path)} entities "
        answer += f"with confidence {best_path['score']:.2f}"
        
        return {
            "answer": answer,
            "confidence": best_path["score"],
            "explanation": best_path["explanation"]
        }
    
    def _generate_reasoning_cache_key(self, query: str, start_entities: List[str]) -> str:
        """Generate cache key for reasoning."""
        import hashlib
        
        key_data = f"{query}:{'|'.join(sorted(start_entities))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_reasoning_statistics(self) -> Dict:
        """Get statistics about reasoning performance."""
        return {
            "max_hops": self.max_hops,
            "confidence_threshold": self.confidence_threshold,
            "max_paths": self.max_paths,
            "similarity_threshold": self.similarity_threshold,
            "reasoning_patterns": {k: len(v) for k, v in self.reasoning_patterns.items()},
            "cache_size": len(self.reasoning_cache),
            "cache_ttl": self.cache_ttl,
            "supported_reasoning_types": ["causal", "procedural", "comparative", "counterfactual", "analogical"]
        }
