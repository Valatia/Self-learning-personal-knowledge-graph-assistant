"""
Advanced relation extraction agent for REXI.
"""

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np

# Try to import spaCy, but handle gracefully if it fails
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

from rexi.models.relationships import Relationship, RelationshipType
from rexi.models.entities import Entity, EntityType
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class RelationExtractor:
    """Advanced relation extraction using dependency parsing and ML models."""
    
    def __init__(self):
        """Initialize relation extractor."""
        self.nlp = None
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self._load_spacy_model()
        
        # Relation patterns based on dependency parsing
        self.dependency_patterns = {
            RelationshipType.ENABLES: [
                {"dep": "nsubj", "pos": "NOUN", "child_dep": "dobj"},
                {"dep": "agent", "pos": "VERB", "child_dep": "pobj"}
            ],
            RelationshipType.CAUSES: [
                {"dep": "nsubj", "pos": "NOUN", "child_dep": "advcl"},
                {"dep": "agent", "pos": "VERB", "child_dep": "nsubj"}
            ],
            RelationshipType.IMPROVES: [
                {"dep": "nsubj", "pos": "NOUN", "child_dep": "acomp"},
                {"dep": "dobj", "pos": "NOUN", "child_dep": "advmod"}
            ],
            RelationshipType.DEPENDS_ON: [
                {"dep": "nsubj", "pos": "NOUN", "child_dep": "prep"},
                {"dep": "pobj", "pos": "NOUN", "child_dep": "det"}
            ],
            RelationshipType.USED_IN: [
                {"dep": "nsubjpass", "pos": "NOUN", "child_dep": "agent"},
                {"dep": "agent", "pos": "NOUN", "child_dep": "pobj"}
            ],
            RelationshipType.PART_OF: [
                {"dep": "nsubj", "pos": "NOUN", "child_dep": "prep"},
                {"dep": "pobj", "pos": "NOUN", "child_dep": "det"}
            ]
        }
        
        # Lexical patterns for relation extraction
        self.lexical_patterns = {
            RelationshipType.ENABLES: [
                r"(\w+)\s+(enables|allows|facilitates|makes possible)\s+(\w+)",
                r"(\w+)\s+can\s+(\w+)",
                r"(\w+)\s+helps\s+(\w+)"
            ],
            RelationshipType.CAUSES: [
                r"(\w+)\s+(causes|leads to|results in|produces)\s+(\w+)",
                r"(\w+)\s+(because|due to|as a result of)\s+(\w+)",
                r"(\w+)\s+(trigger|triggered)\s+(\w+)"
            ],
            RelationshipType.IMPROVES: [
                r"(\w+)\s+(improves|enhances|boosts|optimizes)\s+(\w+)",
                r"(\w+)\s+(better|superior|enhanced)\s+(\w+)",
                r"(\w+)\s+(upgrade|upgraded)\s+(\w+)"
            ],
            RelationshipType.DEPENDS_ON: [
                r"(\w+)\s+(depends on|relies on|requires|needs)\s+(\w+)",
                r"(\w+)\s+(based on|built on)\s+(\w+)",
                r"(\w+)\s+(without|without)\s+(\w+)"
            ],
            RelationshipType.CONTRADICTS: [
                r"(\w+)\s+(contradicts|opposes|conflicts with)\s+(\w+)",
                r"(\w+)\s+(however|but|although)\s+(\w+)",
                r"(\w+)\s+(not|never)\s+(\w+)"
            ],
            RelationshipType.LEARNED_FROM: [
                r"(\w+)\s+(learned from|studied from|trained by)\s+(\w+)",
                r"(\w+)\s+(based on|inspired by)\s+(\w+)",
                r"(\w+)\s+(derived from)\s+(\w+)"
            ],
            RelationshipType.RELATED_TO: [
                r"(\w+)\s+(related to|associated with|connected to)\s+(\w+)",
                r"(\w+)\s+(similar to|like)\s+(\w+)",
                r"(\w+)\s+(and|or)\s+(\w+)"
            ]
        }
    
    def _load_spacy_model(self):
        """Load spaCy model with dependency parsing."""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy large model for relation extraction")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy medium model for relation extraction")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model for relation extraction")
                except OSError:
                    logger.error("Could not load any spaCy model")
                    self.nlp = None
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations from text using multiple methods."""
        if not self.nlp:
            logger.warning("spaCy model not loaded, using fallback extraction")
            return self._fallback_extraction(text, entities)
        
        try:
            doc = self.nlp(text)
            relations = []
            
            # Extract relations using dependency parsing
            dependency_relations = self._extract_dependency_relations(doc, entities)
            relations.extend(dependency_relations)
            
            # Extract relations using lexical patterns
            lexical_relations = self._extract_lexical_relations(text, entities)
            relations.extend(lexical_relations)
            
            # Extract relations using LLM for complex cases
            if self.llm_service.is_available():
                llm_relations = self._extract_llm_relations(text, entities)
                relations.extend(llm_relations)
            
            # Remove duplicates and score relations
            relations = self._deduplicate_relations(relations)
            relations = self._score_relations(relations, doc)
            
            # Sort by confidence
            relations.sort(key=lambda x: x["confidence"], reverse=True)
            
            return relations
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return self._fallback_extraction(text, entities)
    
    def _extract_dependency_relations(self, doc, entities: List[Dict]) -> List[Dict]:
        """Extract relations using dependency parsing."""
        relations = []
        
        # Create entity lookup by position
        entity_positions = {}
        for entity in entities:
            for i in range(entity["start"], entity["end"]):
                entity_positions[i] = entity
        
        # Analyze dependency patterns
        for token in doc:
            for relation_type, patterns in self.dependency_patterns.items():
                for pattern in patterns:
                    if self._matches_pattern(token, pattern):
                        # Find source and target entities
                        source_entity = self._find_entity_for_token(token, entity_positions)
                        target_entity = self._find_entity_for_token(token.head, entity_positions)
                        
                        if source_entity and target_entity and source_entity != target_entity:
                            relations.append({
                                "source": source_entity["text"],
                                "target": target_entity["text"],
                                "type": relation_type,
                                "confidence": 0.7,
                                "source_method": "dependency_parsing",
                                "evidence": token.text + " " + token.head.text,
                                "position": (token.idx, token.head.idx)
                            })
        
        return relations
    
    def _matches_pattern(self, token, pattern: Dict) -> bool:
        """Check if token matches dependency pattern."""
        for key, value in pattern.items():
            if key == "dep" and token.dep_ != value:
                return False
            elif key == "pos" and token.pos_ != value:
                return False
            elif key == "child_dep" and token.head.dep_ != value:
                return False
        return True
    
    def _find_entity_for_token(self, token, entity_positions: Dict) -> Optional[Dict]:
        """Find entity that contains this token."""
        return entity_positions.get(token.i)
    
    def _extract_lexical_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations using lexical patterns."""
        import re
        relations = []
        
        # Create entity lookup
        entity_names = {entity["text"].lower(): entity for entity in entities}
        
        for relation_type, patterns in self.lexical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_text = match.group(1).strip()
                    target_text = match.group(3).strip()
                    
                    # Find matching entities
                    source_entity = entity_names.get(source_text.lower())
                    target_entity = entity_names.get(target_text.lower())
                    
                    if source_entity and target_entity and source_entity != target_entity:
                        relations.append({
                            "source": source_entity["text"],
                            "target": target_entity["text"],
                            "type": relation_type,
                            "confidence": 0.6,
                            "source_method": "lexical_pattern",
                            "evidence": match.group(0),
                            "position": (match.start(), match.end())
                        })
        
        return relations
    
    def _extract_llm_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations using LLM for complex cases."""
        try:
            # Prepare entity list for LLM
            entity_list = "\n".join([f"- {e['text']} ({e['type'].value})" for e in entities])
            
            system_prompt = f"""
            You are an expert relation extractor. Extract relationships between the following entities:
            
            {entity_list}
            
            Possible relation types:
            - enables: one thing enables another
            - causes: one thing causes another
            - improves: one thing improves another
            - depends_on: one thing depends on another
            - contradicts: one thing contradicts another
            - learned_from: learning relationship
            - used_in: usage relationship
            - part_of: part-of relationship
            - related_to: general relatedness
            - precedes: temporal precedence
            - follows: temporal succession
            - inspired_by: inspiration relationship
            - supports: support relationship
            - applied_to: application relationship
            
            Return relations in JSON format with:
            - source: source entity name
            - target: target entity name
            - type: relation type
            - confidence: confidence score (0-1)
            - evidence: supporting text snippet
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text: {text}"}
            ]
            
            response = self.llm_service.chat_completion(
                messages,
                temperature=0.1,
                max_tokens=1500
            )
            
            # Parse LLM response
            import json
            try:
                llm_relations = json.loads(response)
                if isinstance(llm_relations, list):
                    # Add source method and validate entities
                    validated_relations = []
                    for rel in llm_relations:
                        if self._validate_llm_relation(rel, entities):
                            rel["source_method"] = "llm"
                            validated_relations.append(rel)
                    return validated_relations
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM relation extraction response")
            
            return []
            
        except Exception as e:
            logger.error(f"LLM relation extraction failed: {e}")
            return []
    
    def _validate_llm_relation(self, relation: Dict, entities: List[Dict]) -> bool:
        """Validate LLM-extracted relation against entity list."""
        entity_names = {e["text"] for e in entities}
        
        return (relation.get("source") in entity_names and 
                relation.get("target") in entity_names and
                relation.get("type") in [rt.value for rt in RelationshipType])
    
    def _fallback_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Fallback relation extraction using simple heuristics."""
        relations = []
        
        # Simple co-occurrence based relations
        entity_names = [e["text"] for e in entities]
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities appear close together
                distance = abs(entity1["start"] - entity2["start"])
                if distance < 100:  # Within 100 characters
                    relations.append({
                        "source": entity1["text"],
                        "target": entity2["text"],
                        "type": RelationshipType.RELATED_TO,
                        "confidence": 0.3,
                        "source_method": "co_occurrence",
                        "evidence": text[min(entity1["start"], entity2["start"]):max(entity1["end"], entity2["end"])],
                        "position": (entity1["start"], entity2["start"])
                    })
        
        return relations
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """Remove duplicate relations."""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # Create a key for deduplication (order-independent)
            key = tuple(sorted([relation["source"], relation["target"], relation["type"].value]))
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # Update existing relation with higher confidence
                for i, existing in enumerate(unique_relations):
                    existing_key = tuple(sorted([existing["source"], existing["target"], existing["type"].value]))
                    if existing_key == key:
                        unique_relations[i]["confidence"] = max(existing["confidence"], relation["confidence"])
                        break
        
        return unique_relations
    
    def _score_relations(self, relations: List[Dict], doc) -> List[Dict]:
        """Score relations based on various factors."""
        for relation in relations:
            base_score = relation["confidence"]
            
            # Adjust based on extraction method
            method_scores = {
                "dependency_parsing": 0.8,
                "llm": 0.9,
                "lexical_pattern": 0.6,
                "co_occurrence": 0.3
            }
            
            method_score = method_scores.get(relation["source_method"], 0.5)
            
            # Adjust based on relation type (some types more reliable)
            type_scores = {
                RelationshipType.ENABLES: 0.8,
                RelationshipType.CAUSES: 0.7,
                RelationshipType.IMPROVES: 0.7,
                RelationshipType.DEPENDS_ON: 0.8,
                RelationshipType.RELATED_TO: 0.5
            }
            
            type_score = type_scores.get(relation["type"], 0.6)
            
            # Adjust based on evidence quality
            evidence_score = self._calculate_evidence_score(relation.get("evidence", ""))
            
            # Calculate final confidence
            final_confidence = base_score * method_score * type_score * evidence_score
            relation["confidence"] = min(final_confidence, 1.0)
        
        return relations
    
    def _calculate_evidence_score(self, evidence: str) -> float:
        """Calculate evidence quality score."""
        if not evidence:
            return 0.5
        
        # Longer evidence is generally better
        length_score = min(len(evidence.split()) / 10.0, 1.0)
        
        # Check for strong relation indicators
        strong_indicators = ["enables", "causes", "improves", "depends on", "requires"]
        indicator_score = 0.7 if any(indicator in evidence.lower() for indicator in strong_indicators) else 0.5
        
        return (length_score + indicator_score) / 2
    
    def resolve_relations(self, relations: List[Dict], entities: List[Entity]) -> List[Relationship]:
        """Resolve extracted relations against entities."""
        resolved_relations = []
        
        # Create entity lookup
        entity_lookup = {entity.name: entity for entity in entities}
        
        for relation_data in relations:
            source_entity = entity_lookup.get(relation_data["source"])
            target_entity = entity_lookup.get(relation_data["target"])
            
            if source_entity and target_entity:
                relationship = Relationship(
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    type=relation_data["type"],
                    confidence=relation_data["confidence"],
                    properties={
                        "source_method": relation_data["source_method"],
                        "evidence": relation_data.get("evidence", ""),
                        "extraction_method": "advanced_relation_extraction"
                    }
                )
                resolved_relations.append(relationship)
        
        return resolved_relations
    
    def get_extraction_statistics(self) -> Dict:
        """Get statistics about relation extraction performance."""
        return {
            "spacy_model_loaded": self.nlp is not None,
            "dependency_patterns_count": len(self.dependency_patterns),
            "lexical_patterns_count": len(self.lexical_patterns),
            "supported_relation_types": list(RelationshipType),
            "extraction_methods": ["dependency_parsing", "lexical_patterns", "llm", "co_occurrence"]
        }
