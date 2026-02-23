"""
Temporal reasoning agent for REXI - handles time-based knowledge and reasoning.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.services.neo4j_service import Neo4jService
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class TemporalReasoningEngine:
    """Engine for temporal reasoning and knowledge evolution over time."""
    
    def __init__(self):
        """Initialize temporal reasoning engine."""
        self.neo4j_service = Neo4jService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
        # Temporal edge types
        self.temporal_edge_types = {
            "valid_from": "Temporal validity start",
            "valid_to": "Temporal validity end", 
            "changed_to": "Evolution to new state",
            "deprecated": "No longer valid",
            "revised": "Updated version",
            "precedes": "Happens before",
            "follows": "Happens after",
            "overlaps": "Time overlap",
            "contains": "Time containment"
        }
        
        # Time periods for different reasoning contexts
        self.time_periods = {
            "immediate": timedelta(days=1),
            "short_term": timedelta(days=7),
            "medium_term": timedelta(days=30),
            "long_term": timedelta(days=365),
            "very_long_term": timedelta(days=3650)  # 10 years
        }
    
    def add_temporal_information(self, entity: Entity, temporal_data: Dict) -> Entity:
        """Add temporal information to an entity."""
        temporal_properties = {}
        
        # Add validity periods
        if "valid_from" in temporal_data:
            temporal_properties["valid_from"] = self._parse_datetime(temporal_data["valid_from"])
        
        if "valid_to" in temporal_data:
            temporal_properties["valid_to"] = self._parse_datetime(temporal_data["valid_to"])
        
        # Add event timestamps
        if "event_time" in temporal_data:
            temporal_properties["event_time"] = self._parse_datetime(temporal_data["event_time"])
        
        # Add duration information
        if "duration" in temporal_data:
            temporal_properties["duration"] = temporal_data["duration"]
        
        # Add temporal relationships
        if "temporal_relationships" in temporal_data:
            temporal_properties["temporal_relationships"] = temporal_data["temporal_relationships"]
        
        # Update entity properties
        entity.properties.update(temporal_properties)
        
        # Add temporal embedding if not present
        if not entity.properties.get("temporal_embedding"):
            temporal_text = self._create_temporal_text(entity)
            entity.properties["temporal_embedding"] = self.embedding_service.encode_text(temporal_text)
        
        return entity
    
    def _parse_datetime(self, datetime_input) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if isinstance(datetime_input, datetime):
            return datetime_input
        
        if isinstance(datetime_input, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(datetime_input.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try common formats
                    formats = [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d",
                        "%m/%d/%Y",
                        "%d/%m/%Y"
                    ]
                    for fmt in formats:
                        try:
                            return datetime.strptime(datetime_input, fmt)
                        except ValueError:
                            continue
                except Exception:
                    pass
        
        logger.warning(f"Could not parse datetime: {datetime_input}")
        return None
    
    def _create_temporal_text(self, entity: Entity) -> str:
        """Create temporal text representation for embedding."""
        temporal_parts = []
        
        if entity.created_at:
            temporal_parts.append(f"created {entity.created_at.strftime('%Y-%m-%d')}")
        
        if entity.properties.get("valid_from"):
            valid_from = entity.properties["valid_from"]
            if isinstance(valid_from, datetime):
                temporal_parts.append(f"valid from {valid_from.strftime('%Y-%m-%d')}")
        
        if entity.properties.get("valid_to"):
            valid_to = entity.properties["valid_to"]
            if isinstance(valid_to, datetime):
                temporal_parts.append(f"valid until {valid_to.strftime('%Y-%m-%d')}")
        
        if entity.properties.get("event_time"):
            event_time = entity.properties["event_time"]
            if isinstance(event_time, datetime):
                temporal_parts.append(f"event at {event_time.strftime('%Y-%m-%d')}")
        
        return " ".join(temporal_parts) if temporal_parts else "no temporal information"
    
    def create_temporal_relationship(self, source_entity: str, target_entity: str, 
                                   relationship_type: str, temporal_data: Dict) -> Relationship:
        """Create a temporal relationship between entities."""
        # Validate relationship type
        if relationship_type not in self.temporal_edge_types:
            raise ValueError(f"Invalid temporal relationship type: {relationship_type}")
        
        # Create relationship with temporal properties
        relationship = Relationship(
            source_entity_id=source_entity,
            target_entity_id=target_entity,
            type=RelationshipType.RELATED_TO,  # Base type, temporal info in properties
            confidence=temporal_data.get("confidence", 0.8),
            properties={
                "temporal_type": relationship_type,
                "temporal_description": self.temporal_edge_types[relationship_type],
                **temporal_data
            }
        )
        
        # Add temporal validity
        if "valid_from" in temporal_data:
            relationship.valid_from = self._parse_datetime(temporal_data["valid_from"])
        
        if "valid_to" in temporal_data:
            relationship.valid_to = self._parse_datetime(temporal_data["valid_to"])
        
        return relationship
    
    def reason_temporal_query(self, query: str, time_context: Optional[Dict] = None) -> Dict:
        """Answer temporal reasoning queries."""
        try:
            # Parse temporal aspects of query
            temporal_aspects = self._parse_temporal_query(query)
            
            # Enhance with provided time context
            if time_context:
                temporal_aspects.update(time_context)
            
            # Get relevant entities and relationships
            relevant_entities = self._get_temporally_relevant_entities(temporal_aspects)
            
            # Perform temporal reasoning
            reasoning_result = self._perform_temporal_reasoning(query, temporal_aspects, relevant_entities)
            
            return {
                "query": query,
                "temporal_aspects": temporal_aspects,
                "reasoning_result": reasoning_result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Temporal reasoning failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _parse_temporal_query(self, query: str) -> Dict:
        """Parse temporal aspects from natural language query."""
        temporal_aspects = {}
        
        # Extract time expressions
        time_expressions = self._extract_time_expressions(query)
        if time_expressions:
            temporal_aspects["time_expressions"] = time_expressions
        
        # Extract temporal relationships
        temporal_relations = self._extract_temporal_relations(query)
        if temporal_relations:
            temporal_aspects["temporal_relations"] = temporal_relations
        
        # Determine time period context
        time_period = self._determine_time_period(query)
        if time_period:
            temporal_aspects["time_period"] = time_period
        
        # Extract temporal constraints
        temporal_constraints = self._extract_temporal_constraints(query)
        if temporal_constraints:
            temporal_aspects["constraints"] = temporal_constraints
        
        return temporal_aspects
    
    def _extract_time_expressions(self, text: str) -> List[Dict]:
        """Extract time expressions from text."""
        import re
        
        time_expressions = []
        
        # Date patterns
        date_patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",  # YYYY-MM-DD
            r"\b(\d{1,2}/\d{1,2}/\d{4})\b",  # MM/DD/YYYY or DD/MM/YYYY
            r"\b(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b"  # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                parsed_date = self._parse_datetime(date_str)
                if parsed_date:
                    time_expressions.append({
                        "type": "date",
                        "text": date_str,
                        "datetime": parsed_date,
                        "position": (match.start(), match.end())
                    })
        
        # Relative time expressions
        relative_patterns = [
            r"\b(today|yesterday|tomorrow)\b",
            r"\b(last|next)\s+(week|month|year)\b",
            r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b",
            r"\b(in\s+(\d+)\s+(days?|weeks?|months?|years?))\b"
        ]
        
        for pattern in relative_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                time_expressions.append({
                    "type": "relative",
                    "text": match.group(0),
                    "position": (match.start(), match.end())
                })
        
        return time_expressions
    
    def _extract_temporal_relations(self, text: str) -> List[Dict]:
        """Extract temporal relationships from text."""
        import re
        
        temporal_relations = []
        
        # Temporal relation patterns
        relation_patterns = [
            r"(\w+)\s+(before|after|during|while|when|as\s+soon\s+as)\s+(\w+)",
            r"(\w+)\s+(preceded|followed|preceding|following)\s+(\w+)",
            r"(\w+)\s+(happened|occurred|took\s+place)\s+(before|after|during)\s+(\w+)"
        ]
        
        for pattern in relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 3:
                    source = groups[0]
                    relation_type = groups[1].lower()
                    target = groups[-1]
                    
                    # Map to standard temporal relations
                    standard_relation = self._map_temporal_relation(relation_type)
                    
                    temporal_relations.append({
                        "source": source,
                        "relation": standard_relation,
                        "target": target,
                        "text": match.group(0),
                        "position": (match.start(), match.end())
                    })
        
        return temporal_relations
    
    def _map_temporal_relation(self, relation_text: str) -> str:
        """Map textual temporal relation to standard type."""
        mapping = {
            "before": "precedes",
            "after": "follows",
            "during": "overlaps",
            "while": "overlaps",
            "when": "overlaps",
            "as soon as": "precedes",
            "preceded": "precedes",
            "followed": "follows",
            "preceding": "precedes",
            "following": "follows",
            "happened before": "precedes",
            "happened after": "follows",
            "occurred before": "precedes",
            "occurred after": "follows",
            "took place before": "precedes",
            "took place after": "follows"
        }
        
        return mapping.get(relation_text.lower(), "related_to")
    
    def _determine_time_period(self, query: str) -> Optional[str]:
        """Determine the time period context of the query."""
        query_lower = query.lower()
        
        period_keywords = {
            "immediate": ["today", "now", "current", "right now"],
            "short_term": ["this week", "recent", "last few days", "upcoming"],
            "medium_term": ["this month", "past month", "next month"],
            "long_term": ["this year", "past year", "last year", "next year"],
            "very_long_term": ["decade", "past decade", "last 10 years"]
        }
        
        for period, keywords in period_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return period
        
        return None
    
    def _extract_temporal_constraints(self, query: str) -> Dict:
        """Extract temporal constraints from query."""
        constraints = {}
        
        query_lower = query.lower()
        
        # Time range constraints
        if "between" in query_lower and "and" in query_lower:
            # Extract date range
            dates = self._extract_time_expressions(query)
            if len(dates) >= 2:
                constraints["date_range"] = {
                    "start": dates[0]["datetime"],
                    "end": dates[1]["datetime"]
                }
        
        # Specific time constraints
        if "since" in query_lower:
            constraints["since"] = True
        elif "until" in query_lower:
            constraints["until"] = True
        elif "within" in query_lower:
            constraints["within"] = True
        
        return constraints
    
    def _get_temporally_relevant_entities(self, temporal_aspects: Dict) -> List[Entity]:
        """Get entities relevant to temporal aspects."""
        try:
            # Get entities with temporal validity
            valid_at = None
            if "time_expressions" in temporal_aspects:
                # Use the first time expression as reference
                time_expr = temporal_aspects["time_expressions"][0]
                if "datetime" in time_expr:
                    valid_at = time_expr["datetime"].isoformat()
            
            neo4j_entities = self.neo4j_service.find_entities_with_temporal_validity(valid_at)
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
            logger.error(f"Failed to get temporally relevant entities: {e}")
            return []
    
    def _perform_temporal_reasoning(self, query: str, temporal_aspects: Dict, entities: List[Entity]) -> Dict:
        """Perform temporal reasoning on entities."""
        reasoning_result = {
            "temporal_logic": [],
            "chronological_order": [],
            "time_constraints": [],
            "inferences": []
        }
        
        # Apply temporal logic
        if "temporal_relations" in temporal_aspects:
            for relation in temporal_aspects["temporal_relations"]:
                logic_result = self._apply_temporal_logic(relation, entities)
                reasoning_result["temporal_logic"].append(logic_result)
        
        # Determine chronological order
        chronological_order = self._determine_chronological_order(entities)
        reasoning_result["chronological_order"] = chronological_order
        
        # Apply time constraints
        if "constraints" in temporal_aspects:
            constraint_results = self._apply_time_constraints(
                temporal_aspects["constraints"], entities
            )
            reasoning_result["time_constraints"] = constraint_results
        
        # Generate temporal inferences
        inferences = self._generate_temporal_inferences(entities, temporal_aspects)
        reasoning_result["inferences"] = inferences
        
        return reasoning_result
    
    def _apply_temporal_logic(self, relation: Dict, entities: List[Entity]) -> Dict:
        """Apply temporal logic to a relationship."""
        source_entity = next((e for e in entities if e.name == relation["source"]), None)
        target_entity = next((e for e in entities if e.name == relation["target"]), None)
        
        logic_result = {
            "relation": relation,
            "source_found": source_entity is not None,
            "target_found": target_entity is not None,
            "temporal_order": None,
            "validity": None
        }
        
        if source_entity and target_entity:
            # Determine temporal order
            source_time = self._get_entity_temporal_position(source_entity)
            target_time = self._get_entity_temporal_position(target_entity)
            
            if source_time and target_time:
                if relation["relation"] == "precedes":
                    logic_result["temporal_order"] = source_time < target_time
                elif relation["relation"] == "follows":
                    logic_result["temporal_order"] = source_time > target_time
                elif relation["relation"] == "overlaps":
                    logic_result["temporal_order"] = self._check_time_overlap(source_time, target_time)
                
                # Check validity
                logic_result["validity"] = self._check_temporal_validity(relation, source_entity, target_entity)
        
        return logic_result
    
    def _get_entity_temporal_position(self, entity: Entity) -> Optional[datetime]:
        """Get the temporal position of an entity."""
        # Try different temporal properties
        for prop in ["event_time", "valid_from", "created_at"]:
            if prop in entity.properties:
                value = entity.properties[prop]
                if isinstance(value, datetime):
                    return value
                elif isinstance(value, str):
                    return self._parse_datetime(value)
        
        return None
    
    def _check_time_overlap(self, time1: datetime, time2: datetime) -> bool:
        """Check if two time periods overlap."""
        # For single timestamps, check if they're close (within same day)
        return abs((time1 - time2).days) <= 1
    
    def _check_temporal_validity(self, relation: Dict, source: Entity, target: Entity) -> bool:
        """Check if temporal relation is valid based on entity properties."""
        # Check if entities are temporally valid
        source_valid = self._is_entity_temporally_valid(source)
        target_valid = self._is_entity_temporally_valid(target)
        
        return source_valid and target_valid
    
    def _is_entity_temporally_valid(self, entity: Entity) -> bool:
        """Check if entity is currently temporally valid."""
        now = datetime.utcnow()
        
        # Check validity period
        valid_from = entity.properties.get("valid_from")
        valid_to = entity.properties.get("valid_to")
        
        if valid_from:
            if isinstance(valid_from, str):
                valid_from = self._parse_datetime(valid_from)
            if valid_from and now < valid_from:
                return False
        
        if valid_to:
            if isinstance(valid_to, str):
                valid_to = self._parse_datetime(valid_to)
            if valid_to and now > valid_to:
                return False
        
        return True
    
    def _determine_chronological_order(self, entities: List[Entity]) -> List[Dict]:
        """Determine chronological order of entities."""
        entity_times = []
        
        for entity in entities:
            time_pos = self._get_entity_temporal_position(entity)
            if time_pos:
                entity_times.append({
                    "entity": entity.name,
                    "time": time_pos,
                    "type": entity.type.value
                })
        
        # Sort by time
        entity_times.sort(key=lambda x: x["time"])
        
        return entity_times
    
    def _apply_time_constraints(self, constraints: Dict, entities: List[Entity]) -> List[Dict]:
        """Apply time constraints to filter entities."""
        constraint_results = []
        
        for constraint_type, constraint_value in constraints.items():
            result = {
                "constraint_type": constraint_type,
                "constraint_value": constraint_value,
                "filtered_entities": []
            }
            
            if constraint_type == "date_range" and isinstance(constraint_value, dict):
                start_date = constraint_value.get("start")
                end_date = constraint_value.get("end")
                
                for entity in entities:
                    entity_time = self._get_entity_temporal_position(entity)
                    if entity_time and start_date and end_date:
                        if start_date <= entity_time <= end_date:
                            result["filtered_entities"].append(entity.name)
            
            elif constraint_type == "since":
                # Filter entities from a certain point onwards
                for entity in entities:
                    entity_time = self._get_entity_temporal_position(entity)
                    if entity_time:
                        result["filtered_entities"].append(entity.name)
            
            constraint_results.append(result)
        
        return constraint_results
    
    def _generate_temporal_inferences(self, entities: List[Entity], temporal_aspects: Dict) -> List[Dict]:
        """Generate temporal inferences from entities and query context."""
        inferences = []
        
        # Infer temporal patterns
        if len(entities) >= 2:
            # Look for temporal sequences
            chronological = self._determine_chronological_order(entities)
            if len(chronological) >= 2:
                inferences.append({
                    "type": "temporal_sequence",
                    "sequence": [item["entity"] for item in chronological],
                    "confidence": 0.8
                })
        
        # Infer causal relationships
        if "temporal_relations" in temporal_aspects:
            for relation in temporal_aspects["temporal_relations"]:
                if relation["relation"] in ["precedes", "follows"]:
                    inferences.append({
                        "type": "potential_causality",
                        "source": relation["source"],
                        "target": relation["target"],
                        "temporal_relation": relation["relation"],
                        "confidence": 0.6
                    })
        
        return inferences
    
    def create_memory_timeline(self, entity_id: str) -> Dict:
        """Create a timeline of an entity's memory evolution."""
        try:
            # Get entity and its temporal history
            entity = self.neo4j_service.find_nodes("Entity", {"id": entity_id})
            if not entity:
                return {"error": "Entity not found"}
            
            # Get temporal relationships
            temporal_rels = self._get_entity_temporal_relationships(entity_id)
            
            # Build timeline
            timeline = self._build_entity_timeline(entity[0], temporal_rels)
            
            return {
                "entity_id": entity_id,
                "entity_name": entity[0].get("name"),
                "timeline": timeline,
                "created_at": timeline[0]["timestamp"] if timeline else None,
                "last_updated": timeline[-1]["timestamp"] if timeline else None
            }
            
        except Exception as e:
            logger.error(f"Failed to create memory timeline: {e}")
            return {"error": str(e)}
    
    def _get_entity_temporal_relationships(self, entity_id: str) -> List[Dict]:
        """Get temporal relationships for an entity."""
        try:
            temporal_rels = self.neo4j_service.get_temporal_relationships(entity_id)
            return temporal_rels
        except Exception as e:
            logger.error(f"Failed to get entity temporal relationships: {e}")
            return []
    
    def _build_entity_timeline(self, entity: Dict, temporal_rels: List[Dict]) -> List[Dict]:
        """Build timeline from entity and temporal relationships."""
        timeline = []
        
        # Add entity creation
        if "created_at" in entity:
            timeline.append({
                "timestamp": entity["created_at"],
                "event": "entity_created",
                "description": f"Entity '{entity['name']}' was created",
                "type": "creation"
            })
        
        # Add temporal relationship events
        for rel in temporal_rels:
            if "valid_from" in rel:
                timeline.append({
                    "timestamp": rel["valid_from"],
                    "event": "relationship_started",
                    "description": f"Relationship '{rel.get('temporal_type', 'unknown')}' started",
                    "type": "relationship",
                    "relationship_data": rel
                })
            
            if "valid_to" in rel:
                timeline.append({
                    "timestamp": rel["valid_to"],
                    "event": "relationship_ended",
                    "description": f"Relationship '{rel.get('temporal_type', 'unknown')}' ended",
                    "type": "relationship",
                    "relationship_data": rel
                })
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline
    
    def get_temporal_statistics(self) -> Dict:
        """Get statistics about temporal reasoning."""
        return {
            "supported_temporal_relations": list(self.temporal_edge_types.keys()),
            "time_periods": {k: str(v) for k, v in self.time_periods.items()},
            "temporal_reasoning_methods": [
                "temporal_logic",
                "chronological_ordering", 
                "time_constraint_application",
                "temporal_inference"
            ]
        }
