"""
Memory evolution engine for REXI - handles knowledge updates and evolution.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.services.neo4j_service import Neo4jService
from rexi.services.qdrant_service import QdrantService
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService
from rexi.agents.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

class MemoryEvolutionEngine:
    """Engine for evolving and maintaining the knowledge graph over time."""
    
    def __init__(self):
        """Initialize memory evolution engine."""
        self.neo4j_service = Neo4jService()
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.entity_resolver = EntityResolver()
        
        # Evolution parameters
        self.confidence_decay_rate = 0.1  # Confidence decay per month
        self.knowledge_decay_threshold = 0.3  # Minimum confidence before forgetting
        self.conflict_threshold = 0.7  # Confidence threshold for conflict detection
        self.merge_similarity_threshold = 0.85  # Similarity threshold for concept merging
        
        # Evolution history
        self.evolution_history = []
    
    def evolve_knowledge(self, new_entities: List[Entity], new_relationships: List[Relationship]) -> Dict:
        """Main evolution process - integrate new knowledge and evolve existing."""
        evolution_result = {
            "entities_added": 0,
            "entities_merged": 0,
            "entities_updated": 0,
            "entities_forgotten": 0,
            "relationships_added": 0,
            "relationships_updated": 0,
            "conflicts_resolved": 0,
            "concepts_merged": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Step 1: Get existing knowledge
            existing_entities = self._get_existing_entities()
            existing_relationships = self._get_existing_relationships()
            
            # Step 2: Entity resolution and merging
            resolved_entities, merge_stats = self._resolve_and_merge_entities(
                new_entities, existing_entities
            )
            evolution_result.update(merge_stats)
            
            # Step 3: Detect and resolve conflicts
            conflict_resolution_stats = self._detect_and_resolve_conflicts(resolved_entities)
            evolution_result.update(conflict_resolution_stats)
            
            # Step 4: Concept evolution and merging
            concept_evolution_stats = self._evolve_concepts(resolved_entities)
            evolution_result.update(concept_evolution_stats)
            
            # Step 5: Relationship evolution
            relationship_evolution_stats = self._evolve_relationships(
                new_relationships, existing_relationships, resolved_entities
            )
            evolution_result.update(relationship_evolution_stats)
            
            # Step 6: Knowledge decay and forgetting
            forgetting_stats = self._apply_knowledge_decay()
            evolution_result.update(forgetting_stats)
            
            # Step 7: Update temporal information
            self._update_temporal_information(resolved_entities)
            
            # Step 8: Record evolution
            self._record_evolution(evolution_result)
            
            logger.info(f"Knowledge evolution completed: {evolution_result}")
            return evolution_result
            
        except Exception as e:
            logger.error(f"Knowledge evolution failed: {e}")
            evolution_result["error"] = str(e)
            return evolution_result
    
    def _get_existing_entities(self) -> List[Entity]:
        """Get all existing entities from knowledge graph."""
        try:
            neo4j_entities = self.neo4j_service.get_all_entities()
            entities = []
            
            for entity_data in neo4j_entities:
                entity = Entity(
                    id=str(entity_data.get("id", "")),
                    name=entity_data.get("name", ""),
                    type=EntityType(entity_data.get("type", "concept")),
                    description=entity_data.get("description", ""),
                    confidence=entity_data.get("confidence", 0.0),
                    created_at=entity_data.get("created_at", datetime.utcnow()),
                    updated_at=entity_data.get("updated_at", datetime.utcnow()),
                    source_references=entity_data.get("source_references", []),
                    properties=entity_data.get("properties", {}),
                    privacy_level=entity_data.get("privacy_level", "public"),
                    embedding=entity_data.get("embedding", None)
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            logger.error(f"Failed to get existing entities: {e}")
            return []
    
    def _get_existing_relationships(self) -> List[Relationship]:
        """Get all existing relationships from knowledge graph."""
        try:
            neo4j_relationships = self.neo4j_service.get_all_relationships()
            relationships = []
            
            for rel_data in neo4j_relationships:
                rel_info = rel_data.get("r", {})
                source_info = rel_data.get("a", {})
                target_info = rel_data.get("b", {})
                
                relationship = Relationship(
                    id=str(rel_info.get("id", "")),
                    source_entity_id=str(source_info.get("id", "")),
                    target_entity_id=str(target_info.get("id", "")),
                    type=RelationshipType(rel_info.get("type", "related_to")),
                    strength_score=rel_info.get("strength_score", 0.5),
                    confidence=rel_info.get("confidence", 0.5),
                    created_at=rel_info.get("created_at", datetime.utcnow()),
                    updated_at=rel_info.get("updated_at", datetime.utcnow()),
                    evidence_references=rel_info.get("evidence_references", []),
                    properties=rel_info.get("properties", {})
                )
                relationships.append(relationship)
            
            return relationships
        except Exception as e:
            logger.error(f"Failed to get existing relationships: {e}")
            return []
    
    def _resolve_and_merge_entities(self, new_entities: List[Entity], existing_entities: List[Entity]) -> Tuple[List[Entity], Dict]:
        """Resolve and merge entities with existing knowledge."""
        stats = {
            "entities_added": 0,
            "entities_merged": 0,
            "entities_updated": 0
        }
        
        # Combine new and existing entities
        all_entities = new_entities + existing_entities
        
        # Resolve entities
        resolved_entities = self.entity_resolver.resolve_entities(all_entities)
        
        # Determine what was added, merged, or updated
        existing_ids = {e.id for e in existing_entities}
        new_ids = {e.id for e in new_entities}
        resolved_ids = {e.id for e in resolved_entities}
        
        # Count new entities (in resolved but not in existing)
        stats["entities_added"] = len(resolved_ids - existing_ids)
        
        # Count merged entities (multiple original entities merged into one)
        original_count = len(all_entities)
        resolved_count = len(resolved_entities)
        stats["entities_merged"] = original_count - resolved_count
        
        # Count updated entities (existing entities that were modified)
        updated_entities = []
        for resolved in resolved_entities:
            if resolved.id in existing_ids:
                # Check if entity was modified
                original = next(e for e in existing_entities if e.id == resolved.id)
                if self._entity_modified(original, resolved):
                    updated_entities.append(resolved)
        
        stats["entities_updated"] = len(updated_entities)
        
        return resolved_entities, stats
    
    def _entity_modified(self, original: Entity, updated: Entity) -> bool:
        """Check if an entity was significantly modified."""
        return (
            original.name != updated.name or
            original.description != updated.description or
            original.confidence != updated.confidence or
            original.properties != updated.properties
        )
    
    def _detect_and_resolve_conflicts(self, entities: List[Entity]) -> Dict:
        """Detect and resolve conflicts in entity information."""
        stats = {"conflicts_resolved": 0}
        
        # Group entities by name similarity
        conflicts = self._find_entity_conflicts(entities)
        
        for conflict_group in conflicts:
            resolved_conflict = self._resolve_entity_conflict(conflict_group)
            if resolved_conflict:
                stats["conflicts_resolved"] += 1
        
        return stats
    
    def _find_entity_conflicts(self, entities: List[Entity]) -> List[List[Entity]]:
        """Find groups of entities with conflicting information."""
        conflicts = []
        
        # Group by normalized name
        name_groups = defaultdict(list)
        for entity in entities:
            normalized_name = self.entity_resolver._normalize_name(entity.name)
            name_groups[normalized_name].append(entity)
        
        # Find conflicts within groups
        for group in name_groups.values():
            if len(group) > 1:
                # Check for conflicting information
                if self._has_conflicting_information(group):
                    conflicts.append(group)
        
        return conflicts
    
    def _has_conflicting_information(self, entities: List[Entity]) -> bool:
        """Check if entities have conflicting information."""
        if len(entities) < 2:
            return False
        
        # Check for conflicting descriptions
        descriptions = [e.description for e in entities if e.description]
        if len(descriptions) > 1:
            # Simple conflict detection: descriptions with low similarity
            for i, desc1 in enumerate(descriptions):
                for desc2 in descriptions[i+1:]:
                    similarity = self._compute_text_similarity(desc1, desc2)
                    if similarity < 0.3:  # Low similarity indicates potential conflict
                        return True
        
        # Check for conflicting types
        types = [e.type for e in entities]
        if len(set(types)) > 1:
            return True
        
        return False
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Use embedding similarity if available
        try:
            embed1 = self.embedding_service.encode_text(text1)
            embed2 = self.embedding_service.encode_text(text2)
            return self.embedding_service.compute_similarity(embed1, embed2)
        except:
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
    
    def _resolve_entity_conflict(self, conflict_group: List[Entity]) -> Optional[Entity]:
        """Resolve conflict between entities."""
        if not self.llm_service.is_available():
            # Fallback: choose entity with highest confidence
            return max(conflict_group, key=lambda e: e.confidence)
        
        try:
            # Use LLM to resolve conflict
            entity_descriptions = []
            for i, entity in enumerate(conflict_group):
                desc = f"{i+1}. Name: {entity.name}\n"
                desc += f"   Type: {entity.type.value}\n"
                desc += f"   Description: {entity.description or 'N/A'}\n"
                desc += f"   Confidence: {entity.confidence}\n"
                entity_descriptions.append(desc)
            
            system_prompt = """
            You are an expert knowledge resolver. The following entities represent the same concept but have conflicting information.
            Resolve the conflict by creating a single, accurate entity that combines the best information.
            
            Consider:
            - Higher confidence information is more reliable
            - More recent information may be more accurate
            - Detailed descriptions are better than vague ones
            - Resolve type conflicts by choosing the most appropriate type
            
            Respond in JSON format with the resolved entity:
            {
                "name": "resolved name",
                "type": "entity_type",
                "description": "resolved description",
                "confidence": 0.9
            }
            """
            
            user_prompt = "Conflicting entities:\n\n" + "\n".join(entity_descriptions)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_service.chat_completion(
                messages,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse LLM response
            import json
            try:
                result = json.loads(response)
                
                # Create resolved entity
                base_entity = max(conflict_group, key=lambda e: e.confidence)
                
                resolved_entity = Entity(
                    id=base_entity.id,
                    name=result.get("name", base_entity.name),
                    type=EntityType(result.get("type", base_entity.type.value)),
                    description=result.get("description", base_entity.description),
                    confidence=result.get("confidence", max(e.confidence for e in conflict_group)),
                    created_at=min(e.created_at for e in conflict_group),
                    updated_at=datetime.utcnow(),
                    source_references=list(set(sum([e.source_references for e in conflict_group], []))),
                    properties={**base_entity.properties, "conflict_resolved": True}
                )
                
                return resolved_entity
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM conflict resolution response")
        
        except Exception as e:
            logger.error(f"LLM conflict resolution failed: {e}")
        
        # Fallback
        return max(conflict_group, key=lambda e: e.confidence)
    
    def _evolve_concepts(self, entities: List[Entity]) -> Dict:
        """Evolve and merge concepts based on semantic similarity."""
        stats = {"concepts_merged": 0}
        
        # Group entities by type for concept evolution
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)
        
        for entity_type, type_entities in entities_by_type.items():
            # Find concept clusters
            concept_clusters = self._find_concept_clusters(type_entities)
            
            for cluster in concept_clusters:
                if len(cluster) > 1:
                    # Merge concept cluster
                    merged_concept = self._merge_concept_cluster(cluster)
                    if merged_concept:
                        stats["concepts_merged"] += (len(cluster) - 1)
        
        return stats
    
    def _find_concept_clusters(self, entities: List[Entity]) -> List[List[Entity]]:
        """Find clusters of semantically similar concepts."""
        if len(entities) < 2:
            return []
        
        clusters = []
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if entity1 in processed:
                continue
            
            cluster = [entity1]
            processed.add(entity1)
            
            for entity2 in entities[i+1:]:
                if entity2 in processed:
                    continue
                
                # Check semantic similarity
                if entity1.embedding and entity2.embedding:
                    similarity = self.embedding_service.compute_similarity(
                        entity1.embedding, entity2.embedding
                    )
                    
                    if similarity >= self.merge_similarity_threshold:
                        cluster.append(entity2)
                        processed.add(entity2)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _merge_concept_cluster(self, cluster: List[Entity]) -> Optional[Entity]:
        """Merge a cluster of similar concepts."""
        if len(cluster) < 2:
            return None
        
        # Use entity resolver for merging
        return self.entity_resolver._merge_entity_cluster(cluster)
    
    def _evolve_relationships(self, new_relationships: List[Relationship], existing_relationships: List[Relationship], entities: List[Entity]) -> Dict:
        """Evolve relationships based on new knowledge."""
        stats = {
            "relationships_added": 0,
            "relationships_updated": 0
        }
        
        # Create entity lookup
        entity_lookup = {e.name: e for e in entities}
        
        # Process new relationships
        for new_rel in new_relationships:
            # Check if relationship already exists
            existing_rel = self._find_similar_relationship(new_rel, existing_relationships)
            
            if existing_rel:
                # Update existing relationship
                updated_rel = self._update_relationship(existing_rel, new_rel)
                stats["relationships_updated"] += 1
            else:
                # Add new relationship
                stats["relationships_added"] += 1
        
        return stats
    
    def _find_similar_relationship(self, new_rel: Relationship, existing_relationships: List[Relationship]) -> Optional[Relationship]:
        """Find similar existing relationship."""
        for existing_rel in existing_relationships:
            # Check if relationships connect same entities (regardless of direction)
            same_entities = (
                (existing_rel.source_entity_id == new_rel.source_entity_id and 
                 existing_rel.target_entity_id == new_rel.target_entity_id) or
                (existing_rel.source_entity_id == new_rel.target_entity_id and 
                 existing_rel.target_entity_id == new_rel.source_entity_id)
            )
            
            if same_entities and existing_rel.type == new_rel.type:
                return existing_rel
        
        return None
    
    def _update_relationship(self, existing: Relationship, new: Relationship) -> Relationship:
        """Update existing relationship with new information."""
        # Update confidence (take maximum)
        updated_confidence = max(existing.confidence, new.confidence)
        
        # Update strength score (average)
        updated_strength = (existing.strength_score + new.strength_score) / 2
        
        # Merge evidence references
        merged_evidence = list(set(existing.evidence_references + new.evidence_references))
        
        # Create updated relationship
        updated_rel = Relationship(
            id=existing.id,
            source_entity_id=existing.source_entity_id,
            target_entity_id=existing.target_entity_id,
            type=existing.type,
            strength_score=updated_strength,
            confidence=updated_confidence,
            created_at=existing.created_at,
            updated_at=datetime.utcnow(),
            evidence_references=merged_evidence,
            properties={**existing.properties, **new.properties}
        )
        
        return updated_rel
    
    def _apply_knowledge_decay(self) -> Dict:
        """Apply knowledge decay to old, unused knowledge."""
        stats = {"entities_forgotten": 0}
        
        try:
            # Get old entities
            cutoff_date = datetime.utcnow() - timedelta(days=90)  # 3 months ago
            old_entities = self._get_old_entities(cutoff_date)
            
            for entity in old_entities:
                # Apply confidence decay
                decayed_confidence = entity.confidence * (1 - self.confidence_decay_rate)
                
                if decayed_confidence < self.knowledge_decay_threshold:
                    # Forget entity (remove from knowledge graph)
                    self._forget_entity(entity)
                    stats["entities_forgotten"] += 1
                else:
                    # Update confidence
                    self._update_entity_confidence(entity, decayed_confidence)
        
        except Exception as e:
            logger.error(f"Knowledge decay failed: {e}")
        
        return stats
    
    def _get_old_entities(self, cutoff_date: datetime) -> List[Entity]:
        """Get entities older than cutoff date."""
        try:
            neo4j_entities = self.neo4j_service.get_old_entities(cutoff_date)
            entities = []
            
            for entity_data in neo4j_entities:
                entity = Entity(
                    id=str(entity_data.get("id", "")),
                    name=entity_data.get("name", ""),
                    type=EntityType(entity_data.get("type", "concept")),
                    description=entity_data.get("description", ""),
                    confidence=entity_data.get("confidence", 0.0),
                    created_at=entity_data.get("created_at", datetime.utcnow()),
                    updated_at=entity_data.get("updated_at", datetime.utcnow()),
                    source_references=entity_data.get("source_references", []),
                    properties=entity_data.get("properties", {}),
                    privacy_level=entity_data.get("privacy_level", "public"),
                    embedding=entity_data.get("embedding", None)
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            logger.error(f"Failed to get old entities: {e}")
            return []
    
    def _forget_entity(self, entity: Entity):
        """Remove entity from knowledge graph."""
        try:
            # Remove from Neo4j
            self.neo4j_service.delete_node(entity.id)
            
            # Remove from Qdrant
            point_id = f"entity_{entity.id}"
            self.qdrant_service.delete_points("knowledge_embeddings", [point_id])
            
            logger.info(f"Forgotten entity: {entity.name}")
        
        except Exception as e:
            logger.error(f"Failed to forget entity {entity.id}: {e}")
    
    def _update_entity_confidence(self, entity: Entity, new_confidence: float):
        """Update entity confidence."""
        try:
            updates = {"confidence": new_confidence}
            self.neo4j_service.update_node(entity.id, updates)
        except Exception as e:
            logger.error(f"Failed to update entity confidence: {e}")
    
    def _update_temporal_information(self, entities: List[Entity]):
        """Update temporal information for entities."""
        for entity in entities:
            # Update last seen timestamp
            entity.properties["last_seen"] = datetime.utcnow().isoformat()
            
            # Update temporal validity if applicable
            if "valid_from" not in entity.properties:
                entity.properties["valid_from"] = entity.created_at.isoformat()
    
    def _record_evolution(self, evolution_result: Dict):
        """Record evolution in history."""
        evolution_record = {
            "timestamp": evolution_result["timestamp"],
            "statistics": evolution_result,
            "evolution_type": "knowledge_evolution"
        }
        
        self.evolution_history.append(evolution_record)
        
        # Keep only last 100 evolution records
        if len(self.evolution_history) > 100:
            self.evolution_history = self.evolution_history[-100:]
    
    def get_evolution_statistics(self) -> Dict:
        """Get statistics about knowledge evolution."""
        if not self.evolution_history:
            return {"message": "No evolution history available"}
        
        # Calculate aggregate statistics
        total_entities_added = sum(record["statistics"]["entities_added"] for record in self.evolution_history)
        total_entities_merged = sum(record["statistics"]["entities_merged"] for record in self.evolution_history)
        total_conflicts_resolved = sum(record["statistics"]["conflicts_resolved"] for record in self.evolution_history)
        
        return {
            "evolution_count": len(self.evolution_history),
            "total_entities_added": total_entities_added,
            "total_entities_merged": total_entities_merged,
            "total_conflicts_resolved": total_conflicts_resolved,
            "last_evolution": self.evolution_history[-1]["timestamp"],
            "evolution_parameters": {
                "confidence_decay_rate": self.confidence_decay_rate,
                "knowledge_decay_threshold": self.knowledge_decay_threshold,
                "conflict_threshold": self.conflict_threshold,
                "merge_similarity_threshold": self.merge_similarity_threshold
            }
        }
    
    def get_evolution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent evolution history."""
        return self.evolution_history[-limit:] if self.evolution_history else []
