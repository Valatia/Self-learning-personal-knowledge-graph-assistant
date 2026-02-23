"""
Memory evolution core module for REXI - integrates evolution engine with knowledge graph.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.agents.memory_evolution import MemoryEvolutionEngine
from rexi.agents.temporal_reasoning import TemporalReasoningEngine
from rexi.core.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class MemoryEvolutionCore:
    """Core module for memory evolution and temporal reasoning."""
    
    def __init__(self):
        """Initialize memory evolution core."""
        self.knowledge_graph = KnowledgeGraph()
        self.memory_evolution_engine = MemoryEvolutionEngine()
        self.temporal_reasoning_engine = TemporalReasoningEngine()
        
        # Evolution tracking
        self.evolution_sessions = []
        self.last_evolution_time = None
    
    def evolve_memory(self, new_entities: List[Entity] = None, new_relationships: List[Relationship] = None) -> Dict:
        """Main memory evolution process."""
        try:
            evolution_session = {
                "session_id": f"evolution_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "start_time": datetime.utcnow(),
                "new_entities": new_entities or [],
                "new_relationships": new_relationships or []
            }
            
            logger.info(f"Starting memory evolution session: {evolution_session['session_id']}")
            
            # Step 1: Evolve knowledge with new information
            if new_entities or new_relationships:
                evolution_result = self.memory_evolution_engine.evolve_knowledge(
                    new_entities or [], new_relationships or []
                )
                evolution_session["evolution_result"] = evolution_result
            
            # Step 2: Apply temporal reasoning
            temporal_updates = self._apply_temporal_updates(new_entities or [])
            evolution_session["temporal_updates"] = temporal_updates
            
            # Step 3: Create memory snapshots
            snapshot_result = self._create_memory_snapshot()
            evolution_session["snapshot_result"] = snapshot_result
            
            # Step 4: Update evolution statistics
            evolution_session["end_time"] = datetime.utcnow()
            evolution_session["duration"] = (
                evolution_session["end_time"] - evolution_session["start_time"]
            ).total_seconds()
            
            # Record session
            self.evolution_sessions.append(evolution_session)
            self.last_evolution_time = evolution_session["end_time"]
            
            # Keep only last 50 sessions
            if len(self.evolution_sessions) > 50:
                self.evolution_sessions = self.evolution_sessions[-50:]
            
            logger.info(f"Memory evolution completed in {evolution_session['duration']:.2f}s")
            
            return {
                "session_id": evolution_session["session_id"],
                "evolution_result": evolution_session.get("evolution_result", {}),
                "temporal_updates": evolution_session.get("temporal_updates", {}),
                "snapshot_result": evolution_session.get("snapshot_result", {}),
                "duration": evolution_session["duration"],
                "timestamp": evolution_session["end_time"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Memory evolution failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _apply_temporal_updates(self, new_entities: List[Entity]) -> Dict:
        """Apply temporal reasoning to new entities."""
        temporal_updates = {
            "entities_updated": 0,
            "temporal_relationships_created": 0,
            "timeline_events_added": 0
        }
        
        try:
            for entity in new_entities:
                # Add temporal information if available
                if entity.created_at:
                    temporal_data = {
                        "valid_from": entity.created_at,
                        "event_time": entity.created_at,
                        "confidence": entity.confidence
                    }
                    
                    updated_entity = self.temporal_reasoning_engine.add_temporal_information(
                        entity, temporal_data
                    )
                    
                    # Update entity in knowledge graph
                    self.knowledge_graph.update_entity(entity.id, updated_entity.properties)
                    temporal_updates["entities_updated"] += 1
                
                # Create timeline events
                timeline = self.temporal_reasoning_engine.create_memory_timeline(entity.id)
                if "timeline" in timeline:
                    temporal_updates["timeline_events_added"] += len(timeline["timeline"])
            
            # Create temporal relationships between entities
            temporal_relationships = self._create_temporal_relationships(new_entities)
            temporal_updates["temporal_relationships_created"] = len(temporal_relationships)
            
            for rel in temporal_relationships:
                self.knowledge_graph.add_relationship(rel)
            
        except Exception as e:
            logger.error(f"Temporal updates failed: {e}")
            temporal_updates["error"] = str(e)
        
        return temporal_updates
    
    def _create_temporal_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Create temporal relationships between entities."""
        temporal_relationships = []
        
        # Sort entities by creation time
        entities_with_time = [e for e in entities if e.created_at]
        entities_with_time.sort(key=lambda e: e.created_at)
        
        # Create temporal precedence relationships
        for i, entity1 in enumerate(entities_with_time):
            for entity2 in entities_with_time[i+1:]:
                # Check if entities are related (same type or similar context)
                if self._should_create_temporal_relationship(entity1, entity2):
                    temporal_data = {
                        "valid_from": entity1.created_at,
                        "confidence": min(entity1.confidence, entity2.confidence) * 0.8
                    }
                    
                    temporal_rel = self.temporal_reasoning_engine.create_temporal_relationship(
                        entity1.id, entity2.id, "precedes", temporal_data
                    )
                    temporal_relationships.append(temporal_rel)
        
        return temporal_relationships
    
    def _should_create_temporal_relationship(self, entity1: Entity, entity2: Entity) -> bool:
        """Determine if temporal relationship should be created between entities."""
        # Same type entities with close temporal proximity
        if entity1.type == entity2.type:
            time_diff = abs((entity2.created_at - entity1.created_at).days)
            if time_diff <= 30:  # Within 30 days
                return True
        
        # Entities with similar context
        context1 = entity1.properties.get("context", "").lower()
        context2 = entity2.properties.get("context", "").lower()
        
        if context1 and context2:
            # Simple word overlap check
            words1 = set(context1.split())
            words2 = set(context2.split())
            overlap = len(words1.intersection(words2))
            
            if overlap >= 3:  # At least 3 overlapping words
                return True
        
        return False
    
    def _create_memory_snapshot(self) -> Dict:
        """Create a snapshot of the current memory state."""
        try:
            # Get knowledge graph statistics
            stats = self.knowledge_graph.get_statistics()
            
            # Get evolution statistics
            evolution_stats = self.memory_evolution_engine.get_evolution_statistics()
            
            # Get temporal statistics
            temporal_stats = self.temporal_reasoning_engine.get_temporal_statistics()
            
            # Create snapshot
            snapshot = {
                "snapshot_id": f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat(),
                "knowledge_graph_stats": stats,
                "evolution_stats": evolution_stats,
                "temporal_stats": temporal_stats,
                "session_count": len(self.evolution_sessions),
                "last_evolution": self.last_evolution_time.isoformat() if self.last_evolution_time else None
            }
            
            # Store snapshot (in a real implementation, this would be persisted)
            logger.info(f"Created memory snapshot: {snapshot['snapshot_id']}")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Memory snapshot creation failed: {e}")
            return {"error": str(e)}
    
    def get_memory_timeline(self, entity_id: str) -> Dict:
        """Get the timeline for a specific entity."""
        try:
            timeline = self.temporal_reasoning_engine.create_memory_timeline(entity_id)
            return timeline
        except Exception as e:
            logger.error(f"Failed to get memory timeline: {e}")
            return {"error": str(e)}
    
    def reason_temporal_query(self, query: str, time_context: Dict = None) -> Dict:
        """Answer temporal reasoning queries."""
        try:
            return self.temporal_reasoning_engine.reason_temporal_query(query, time_context)
        except Exception as e:
            logger.error(f"Temporal reasoning failed: {e}")
            return {"error": str(e)}
    
    def get_evolution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent evolution history."""
        return self.evolution_sessions[-limit:] if self.evolution_sessions else []
    
    def get_memory_statistics(self) -> Dict:
        """Get comprehensive memory statistics."""
        try:
            # Get statistics from all components
            kg_stats = self.knowledge_graph.get_statistics()
            evolution_stats = self.memory_evolution_engine.get_evolution_statistics()
            temporal_stats = self.temporal_reasoning_engine.get_temporal_statistics()
            
            # Calculate overall statistics
            total_evolution_time = sum(
                session.get("duration", 0) for session in self.evolution_sessions
            )
            
            return {
                "knowledge_graph": kg_stats,
                "evolution": evolution_stats,
                "temporal": temporal_stats,
                "sessions": {
                    "total_sessions": len(self.evolution_sessions),
                    "total_evolution_time": total_evolution_time,
                    "last_evolution": self.last_evolution_time.isoformat() if self.last_evolution_time else None,
                    "average_session_duration": total_evolution_time / len(self.evolution_sessions) if self.evolution_sessions else 0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {"error": str(e)}
    
    def force_evolution_cycle(self) -> Dict:
        """Force a complete evolution cycle."""
        try:
            logger.info("Forcing complete evolution cycle")
            
            # Get current entities and relationships
            # In a real implementation, this would fetch from the knowledge graph
            current_entities = []  # Would fetch from KG
            current_relationships = []  # Would fetch from KG
            
            # Run evolution
            result = self.evolve_memory(current_entities, current_relationships)
            
            return result
            
        except Exception as e:
            logger.error(f"Forced evolution cycle failed: {e}")
            return {"error": str(e)}
    
    def cleanup_old_memories(self, days_threshold: int = 365) -> Dict:
        """Clean up old memories beyond threshold."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # This would need to be implemented in the Neo4j service
            # For now, return placeholder result
            cleanup_result = {
                "cutoff_date": cutoff_date.isoformat(),
                "entities_removed": 0,
                "relationships_removed": 0,
                "space_freed": 0
            }
            
            logger.info(f"Memory cleanup completed for entities older than {days_threshold} days")
            
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close memory evolution core."""
        try:
            self.knowledge_graph.close()
            logger.info("Memory evolution core closed")
        except Exception as e:
            logger.error(f"Error closing memory evolution core: {e}")
