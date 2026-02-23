"""
Self-learning module for REXI - implements autonomous knowledge acquisition and evolution.
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import random

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.services.neo4j_service import Neo4jService
from rexi.services.qdrant_service import QdrantService
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService
from rexi.core.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class SelfLearningEngine:
    """Engine for autonomous learning and knowledge acquisition."""
    
    def __init__(self):
        """Initialize self-learning engine."""
        self.neo4j_service = Neo4jService()
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.knowledge_graph = KnowledgeGraph()
        
        # Learning parameters
        self.exploration_rate = 0.3  # Probability of exploration vs exploitation
        self.novelty_threshold = 0.7  # Threshold for considering information novel
        self.decay_rate = 0.05  # Knowledge decay rate
        self.reinforcement_rate = 0.1  # Learning rate for reinforcement
        
        # Learning state
        self.knowledge_gaps = set()
        self.exploration_history = deque(maxlen=1000)
        self.learning_progress = defaultdict(float)
        self.performance_metrics = defaultdict(list)
        
        # Curiosity metrics
        self.curiosity_scores = {}
        self.last_exploration_time = datetime.utcnow()
        
        logger.info("Self-learning engine initialized")
    
    def detect_knowledge_gaps(self) -> List[Dict]:
        """Detect gaps in current knowledge graph."""
        try:
            gaps = []
            
            # Get all entities
            entities = self.neo4j_service.get_all_entities()
            
            # Analyze entity distribution
            entity_types = defaultdict(int)
            entity_connections = defaultdict(int)
            
            for entity_data in entities:
                entity_type = entity_data.get("type", "unknown")
                entity_types[entity_type] += 1
                
                # Count connections (simplified)
                connections = len(entity_data.get("properties", {}).get("connections", []))
                entity_connections[entity_type] += connections
            
            # Identify gaps
            for entity_type, count in entity_types.items():
                if count < 3:  # Less than 3 entities of a type
                    gaps.append({
                        "type": "entity_scarsity",
                        "entity_type": entity_type,
                        "description": f"Very few {entity_type} entities ({count} found)",
                        "severity": "medium",
                        "suggestion": f"Add more {entity_type} entities through targeted learning"
                    })
            
            # Check for isolated entities
            for entity_type, connections in entity_connections.items():
                if connections == 0 and entity_types[entity_type] > 1:
                    gaps.append({
                        "type": "isolated_entities",
                        "entity_type": entity_type,
                        "description": f"Isolated {entity_type} entities with no connections",
                        "severity": "high",
                        "suggestion": f"Explore relationships between {entity_type} entities"
                    })
            
            # Check for missing relationship types
            existing_relationships = self.neo4j_service.get_all_relationships()
            relationship_types = set()
            for rel in existing_relationships:
                rel_data = rel.get("r", {})
                if rel_data:
                    relationship_types.add(rel_data.get("type", "unknown"))
            
            expected_types = {
                RelationshipType.ENABLES, RelationshipType.CAUSES, RelationshipType.IMPROVES,
                RelationshipType.DEPENDS_ON, RelationshipType.LEARNED_FROM,
                RelationshipType.USED_IN, RelationshipType.PART_OF,
                RelationshipType.PRECEDES, RelationshipType.FOLLOWS,
                RelationshipType.INSPIRED_BY, RelationshipType.SUPPORTS,
                RelationshipType.APPLIED_TO, RelationshipType.CONTRADICTS,
                RelationshipType.INSPIRED_BY, RelationshipType.SUPPORTS
            }
            
            missing_types = expected_types - relationship_types
            for rel_type in missing_types:
                gaps.append({
                    "type": "missing_relationships",
                    "relationship_type": rel_type.value,
                    "description": f"No {rel_type.value} relationships found",
                    "severity": "medium",
                    "suggestion": f"Explore {rel_type.value} relationships in documents"
                })
            
            logger.info(f"Detected {len(gaps)} knowledge gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Knowledge gap detection failed: {e}")
            return []
    
    def generate_exploration_suggestions(self, gaps: List[Dict]) -> List[Dict]:
        """Generate suggestions for knowledge exploration based on gaps."""
        suggestions = []
        
        for gap in gaps:
            gap_type = gap["type"]
            
            if gap_type == "entity_scarsity":
                # Suggest specific entity types to explore
                suggestions.append({
                    "type": "entity_exploration",
                    "priority": "high",
                    "description": f"Explore {gap['entity_type']} entities",
                    "method": "targeted_search",
                    "query_suggestions": [
                        f"Find more {gap['entity_type']} in related documents",
                        f"Search for patterns involving {gap['entity_type']}"
                    ]
                })
            
            elif gap_type == "isolated_entities":
                # Suggest relationship exploration
                suggestions.append({
                    "type": "relationship_exploration",
                    "priority": "high", 
                    "description": f"Explore connections between {gap['entity_type']} entities",
                    "method": "graph_traversal",
                    "query_suggestions": [
                        f"How are {gap['entity_type']} entities related?",
                        f"What relationships exist between {gap['entity_type']}?"
                    ]
                })
            
            elif gap_type == "missing_relationships":
                # Suggest relationship discovery
                suggestions.append({
                    "type": "relationship_discovery",
                    "priority": "medium",
                    "description": f"Discover {gap['relationship_type']} relationships",
                    "method": "pattern_mining",
                    "query_suggestions": [
                        f"Find examples of {gap['relationship_type']}",
                        f"Look for {gap['relationship_type']} patterns in documents"
                    ]
                })
        
        return suggestions
    
    def generate_hypotheses(self, context: str, existing_knowledge: List[Dict]) -> List[Dict]:
        """Generate testable hypotheses based on existing knowledge."""
        try:
            hypotheses = []
            
            # Use LLM to generate hypotheses
            if not self.llm_service.is_available():
                logger.warning("LLM service not available for hypothesis generation")
                return []
            
            system_prompt = """
            You are a research assistant. Based on the existing knowledge, generate 3-5 testable hypotheses that could expand understanding.
            
            For each hypothesis, provide:
            - hypothesis statement
            - test method (how to validate)
            - expected outcome
            - confidence score (0-1)
            
            Focus on:
            - Causal relationships
            - Missing connections
            - Concept generalizations
            - Pattern discoveries
            
            Format as JSON array.
            """
            
            # Prepare knowledge summary
            knowledge_summary = self._summarize_knowledge(existing_knowledge)
            
            user_prompt = f"""
            Context: {context}
            
            Existing Knowledge Summary:
            {knowledge_summary}
            
            Generate 3-5 testable hypotheses.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_service.chat_completion(
                messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response
            import json
            try:
                hypothesis_list = json.loads(response)
                if isinstance(hypothesis_list, list):
                    for i, hypothesis in enumerate(hypothesis_list[:5]):  # Limit to 5
                        hypotheses.append({
                            "id": f"hypothesis_{i}",
                            "statement": hypothesis.get("statement", ""),
                            "test_method": hypothesis.get("test_method", ""),
                            "expected_outcome": hypothesis.get("expected_outcome", ""),
                            "confidence": hypothesis.get("confidence", 0.5),
                            "created_at": datetime.utcnow().isoformat(),
                            "status": "pending"
                        })
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM hypothesis response")
            
            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            return []
    
    def _summarize_knowledge(self, knowledge: List[Dict]) -> str:
        """Summarize existing knowledge for hypothesis generation."""
        if not knowledge:
            return "No existing knowledge available."
        
        # Count entities and relationships
        entity_count = len(knowledge)
        relationship_count = sum(len(item.get("relationships", [])) for item in knowledge)
        
        # Identify key concepts
        concepts = []
        for item in knowledge:
            if item.get("type") == EntityType.CONCEPT:
                concepts.append(item.get("name", ""))
        
        return f"""
        Knowledge Base Summary:
        - {entity_count} entities
        - {relationship_count} relationships
        - {len(concepts)} key concepts: {', '.join(concepts[:10])}
        """
    
    def test_hypothesis(self, hypothesis: Dict) -> Dict:
        """Test a hypothesis and update knowledge based on results."""
        try:
            hypothesis_id = hypothesis["id"]
            statement = hypothesis["statement"]
            test_method = hypothesis["test_method"]
            
            logger.info(f"Testing hypothesis: {statement}")
            
            # Simulate testing (in real implementation, this would query external sources)
            # For now, use LLM to evaluate hypothesis
            evaluation_prompt = f"""
            Evaluate the following hypothesis based on existing knowledge:
            
            Hypothesis: {statement}
            Test Method: {test_method}
            
            Provide:
            1. Validity score (0-1): Is this hypothesis consistent with existing knowledge?
            2. Evidence score (0-1): How strong is the evidence for this hypothesis?
            3. Novelty score (0-1): How novel or surprising is this hypothesis?
            4. Actionability score (0-1): How actionable is this hypothesis?
            
            Format as JSON with these four scores.
            """
            
            messages = [
                {"role": "system", "content": evaluation_prompt},
                {"role": "user", "content": f"Evaluate hypothesis: {statement}"}
            ]
            
            response = self.llm_service.chat_completion(
                messages,
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse evaluation
            import json
            try:
                evaluation = json.loads(response)
                if isinstance(evaluation, dict):
                    # Update hypothesis with evaluation results
                    hypothesis.update({
                        "status": "tested",
                        "validity_score": evaluation.get("validity_score", 0.5),
                        "evidence_score": evaluation.get("evidence_score", 0.5),
                        "novelty_score": evaluation.get("novelty_score", 0.5),
                        "actionability_score": evaluation.get("actionability_score", 0.5),
                        "tested_at": datetime.utcnow().isoformat(),
                        "evaluation": evaluation
                    })
                    
                    # Update learning progress
                    self.learning_progress[f"hypothesis_testing"] += 0.1
                    
                    logger.info(f"Hypothesis {hypothesis_id} evaluated: {evaluation}")
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse hypothesis evaluation: {response}")
            
            return hypothesis
            
        except Exception as e:
            logger.error(f"Hypothesis testing failed: {e}")
            return hypothesis
    
    def update_confidence_scores(self, feedback: Dict):
        """Update confidence scores based on feedback."""
        try:
            hypothesis_id = feedback.get("hypothesis_id")
            user_feedback = feedback.get("user_feedback", "")
            outcome = feedback.get("outcome", "")
            
            # Find hypothesis
            # In real implementation, this would query stored hypotheses
            # For now, simulate finding and updating
            logger.info(f"Updating hypothesis {hypothesis_id} based on feedback: {outcome}")
            
            # Update learning progress
            self.learning_progress[f"feedback_learning"] += 0.1
            
            # Adjust exploration parameters based on feedback
            if outcome == "confirmed":
                self.exploration_rate = max(0.1, self.exploration_rate - 0.05)
            elif outcome == "refuted":
                self.exploration_rate = min(0.9, self.exploration_rate + 0.05)
            
            return {"status": "updated", "hypothesis_id": hypothesis_id}
            
        except Exception as e:
            logger.error(f"Confidence update failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def implement_reinforcement_learning(self, query_results: List[Dict]):
        """Implement reinforcement learning from query results."""
        try:
            logger.info("Implementing reinforcement learning from query results")
            
            # Update performance metrics
            for result in query_results:
                query_type = result.get("type", "unknown")
                success = result.get("success", False)
                
                if query_type in self.performance_metrics:
                    self.performance_metrics[query_type].append(success)
                    
                    # Keep only last 100 results for each query type
                    if len(self.performance_metrics[query_type]) > 100:
                        self.performance_metrics[query_type] = self.performance_metrics[query_type][-100:]
                
                    # Update learning rates based on performance
                    if len(self.performance_metrics[query_type]) >= 10:
                        recent_performance = np.mean(self.performance_metrics[query_type][-10:])
                        
                        if recent_performance > 0.7:  # Good performance
                            self.reinforcement_rate = min(0.2, self.reinforcement_rate + 0.01)
                        elif recent_performance < 0.3:  # Poor performance
                            self.reinforcement_rate = max(0.05, self.reinforcement_rate - 0.01)
                
                # Strengthen successful relationships
                if success:
                    self._strengthen_relationships(result)
            
            logger.info(f"Reinforcement learning updated: {self.reinforcement_rate}")
            return {"status": "completed", "updated_metrics": len(self.performance_metrics)}
            
        except Exception as e:
            logger.error(f"Reinforcement learning failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _strengthen_relationships(self, result: Dict):
        """Strengthen relationships based on successful query results."""
        try:
            entities_involved = result.get("entities", [])
            relationships_used = result.get("relationships", [])
            
            # Update relationship strengths
            for rel_info in relationships_used:
                rel_id = rel_info.get("id")
                current_strength = rel_info.get("strength", 0.5)
                
                # Increase strength for successful relationships
                new_strength = min(1.0, current_strength + self.reinforcement_rate)
                
                self.neo4j_service.update_relationship(rel_id, {
                    "strength": new_strength,
                    "updated_at": datetime.utcnow().isoformat()
                })
            
            logger.info(f"Strengthened {len(relationships_used)} relationships")
            
        except Exception as e:
            logger.error(f"Relationship strengthening failed: {e}")
    
    def get_learning_statistics(self) -> Dict:
        """Get comprehensive learning statistics."""
        try:
            stats = {
                "learning_progress": dict(self.learning_progress),
                "performance_metrics": dict(self.performance_metrics),
                "exploration_history": list(self.exploration_history),
                "curiosity_scores": dict(self.curiosity_scores),
                "learning_parameters": {
                    "exploration_rate": self.exploration_rate,
                    "novelty_threshold": self.novelty_threshold,
                    "decay_rate": self.decay_rate,
                    "reinforcement_rate": self.reinforcement_rate
                },
                "last_exploration": self.last_exploration_time.isoformat() if self.last_exploration_time else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get learning statistics: {e}")
            return {}
    
    def autonomous_learning_cycle(self) -> Dict:
        """Execute one complete autonomous learning cycle."""
        try:
            logger.info("Starting autonomous learning cycle")
            
            cycle_results = {
                "cycle_id": f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "started_at": datetime.utcnow().isoformat(),
                "gap_detection": {"status": "completed", "gaps_found": 0},
                "hypothesis_generation": {"status": "pending", "hypotheses_created": 0},
                "hypothesis_testing": {"status": "pending", "hypotheses_tested": 0},
                "reinforcement_learning": {"status": "pending", "performance_updates": 0}
            }
            
            # Step 1: Detect knowledge gaps
            gaps = self.detect_knowledge_gaps()
            cycle_results["gap_detection"] = {
                "status": "completed",
                "gaps_found": len(gaps),
                "gaps": gaps[:5]  # Store first 5 gaps
            }
            
            # Step 2: Generate exploration suggestions
            if gaps:
                suggestions = self.generate_exploration_suggestions(gaps)
                cycle_results["hypothesis_generation"] = {
                    "status": "completed",
                    "suggestions_created": len(suggestions),
                    "suggestions": suggestions[:3]
                }
            
            # Step 3: Generate hypotheses (mock implementation)
            if gaps:
                context = "Knowledge gap analysis for autonomous learning"
                existing_knowledge = []  # Would get from actual graph
                hypotheses = self.generate_hypotheses(context, existing_knowledge)
                cycle_results["hypothesis_generation"] = {
                    "status": "completed",
                    "hypotheses_created": len(hypotheses),
                    "hypotheses": hypotheses[:3]
                }
            
            # Step 4: Test hypotheses (mock implementation)
            if hypotheses:
                for hypothesis in hypotheses[:2]:  # Test first 2 hypotheses
                    tested = self.test_hypothesis(hypothesis)
                    if tested.get("status") == "tested":
                        cycle_results["hypothesis_testing"] = {
                            "status": "completed",
                            "hypotheses_tested": cycle_results["hypothesis_testing"]["hypotheses_tested"] + 1
                        }
            
            # Step 5: Implement reinforcement learning
            mock_query_results = [
                {"type": "entity_search", "success": True, "entities": ["test1", "test2"]},
                {"type": "relationship_query", "success": True, "relationships": ["rel1", "rel2"]}
            ]
            
            reinforcement_results = self.implement_reinforcement_learning(mock_query_results)
            cycle_results["reinforcement_learning"] = reinforcement_results
            
            cycle_results["completed_at"] = datetime.utcnow().isoformat()
            
            # Update exploration history
            self.exploration_history.append({
                "cycle_id": cycle_results["cycle_id"],
                "completed_at": cycle_results["completed_at"],
                "gaps_found": len(gaps),
                "hypotheses_created": len(hypotheses),
                "hypotheses_tested": cycle_results["hypothesis_testing"]["hypotheses_tested"]
            })
            
            logger.info(f"Autonomous learning cycle completed: {cycle_results['cycle_id']}")
            return cycle_results
            
        except Exception as e:
            logger.error(f"Autonomous learning cycle failed: {e}")
            return {"status": "error", "message": str(e)}
