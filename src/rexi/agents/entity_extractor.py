"""
Advanced entity extraction agent for REXI.
"""

import spacy
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np

from rexi.models.entities import Entity, EntityType
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Advanced entity extraction using spaCy and custom models."""
    
    def __init__(self):
        """Initialize entity extractor."""
        self.nlp = None
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self._load_spacy_model()
        
        # Entity type mappings
        self.spacy_to_rexi_mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.TOPIC,  # Geopolitical entities as topics
            "PRODUCT": EntityType.TOOL,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.PROJECT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.SKILL,
            "DATE": None,  # Handle separately
            "TIME": None,  # Handle separately
            "MONEY": None,  # Handle separately
            "QUANTITY": None,  # Handle separately
            "ORDINAL": None,  # Handle separately
            "CARDINAL": None   # Handle separately
        }
        
        # Custom patterns for specific entity types
        self.custom_patterns = {
            EntityType.SKILL: [
                [{"LOWER": {"REGEX": r"^(python|java|javascript|react|vue|angular|docker|kubernetes|aws|azure|gcp)$"}}],
                [{"LOWER": {"REGEX": r"^(machine learning|deep learning|neural networks|nlp|computer vision)$"}}],
                [{"LOWER": {"REGEX": r"^(data analysis|statistics|research|writing|communication)$"}}]
            ],
            EntityType.CONCEPT: [
                [{"LOWER": {"REGEX": r"^(ai|artificial intelligence|blockchain|cryptocurrency|web3|metaverse)$"}}],
                [{"LOWER": {"REGEX": r"^(algorithm|architecture|design pattern|methodology|framework)$"}}]
            ],
            EntityType.PROJECT: [
                [{"LOWER": {"REGEX": r"^(project|initiative|program|campaign|study|research)$"}}]
            ]
        }
    
    def _load_spacy_model(self):
        """Load spaCy model with custom components."""
        try:
            # Try to load the large model for better accuracy
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy large model")
        except OSError:
            try:
                # Fallback to medium model
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy medium model")
            except OSError:
                try:
                    # Fallback to small model
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model")
                except OSError:
                    logger.error("Could not load any spaCy model")
                    self.nlp = None
                    return
        
        # Add custom patterns to the pipeline
        if self.nlp:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            # Convert custom patterns to spaCy format
            spacy_patterns = []
            for entity_type, patterns in self.custom_patterns.items():
                for pattern in patterns:
                    spacy_patterns.append({"label": entity_type.value, "pattern": pattern})
            
            ruler.add_patterns(spacy_patterns)
    
    def extract_entities(self, text: str, min_confidence: float = 0.5) -> List[Dict]:
        """Extract entities from text using spaCy and custom models."""
        if not self.nlp:
            logger.warning("spaCy model not loaded, using fallback extraction")
            return self._fallback_extraction(text)
        
        try:
            doc = self.nlp(text)
            entities = []
            
            # Extract spaCy entities
            for ent in doc.ents:
                entity_type = self._map_spacy_to_rexi(ent.label_)
                if entity_type and self._is_valid_entity(ent.text, entity_type):
                    
                    # Calculate confidence based on context and model
                    confidence = self._calculate_entity_confidence(ent, doc)
                    
                    if confidence >= min_confidence:
                        entities.append({
                            "text": ent.text,
                            "type": entity_type,
                            "confidence": confidence,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "source": "spacy",
                            "context": self._get_entity_context(doc, ent)
                        })
            
            # Extract custom entities using patterns
            custom_entities = self._extract_custom_entities(doc, min_confidence)
            entities.extend(custom_entities)
            
            # Remove duplicates and merge overlapping entities
            entities = self._merge_overlapping_entities(entities)
            
            # Sort by confidence (descending)
            entities.sort(key=lambda x: x["confidence"], reverse=True)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._fallback_extraction(text)
    
    def _map_spacy_to_rexi(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to REXI entity types."""
        return self.spacy_to_rexi_mapping.get(spacy_label)
    
    def _is_valid_entity(self, text: str, entity_type: EntityType) -> bool:
        """Validate entity based on text and type."""
        text = text.strip()
        
        # Basic validation
        if len(text) < 2:
            return False
        
        # Skip common false positives
        if text.lower() in ["the", "and", "or", "but", "in", "on", "at", "to", "for"]:
            return False
        
        # Type-specific validation
        if entity_type == EntityType.PERSON:
            # Basic person name validation
            words = text.split()
            if len(words) < 2:
                return False
            # Check if words start with capital letters
            return all(word[0].isupper() if word else False for word in words)
        
        elif entity_type == EntityType.ORGANIZATION:
            # Organization names should be longer
            return len(text.split()) >= 2
        
        return True
    
    def _calculate_entity_confidence(self, entity, doc) -> float:
        """Calculate confidence score for an entity."""
        base_confidence = 0.7  # Base confidence for spaCy entities
        
        # Adjust based on entity length (longer entities often more specific)
        length_factor = min(len(entity.text.split()) / 3.0, 1.0)
        
        # Adjust based on context (entities in rich contexts more reliable)
        context_score = self._calculate_context_score(doc, entity)
        
        # Adjust based on entity type (some types more reliable)
        type_confidence = {
            EntityType.PERSON: 0.8,
            EntityType.ORGANIZATION: 0.75,
            EntityType.CONCEPT: 0.6,
            EntityType.SKILL: 0.65,
            EntityType.PROJECT: 0.7,
            EntityType.TOOL: 0.75
        }.get(entity._.get("rexi_type"), 0.7)
        
        confidence = base_confidence * length_factor * context_score * type_confidence
        return min(confidence, 1.0)
    
    def _calculate_context_score(self, doc, entity) -> float:
        """Calculate context quality score."""
        # Count surrounding tokens
        start = max(0, entity.start - 5)
        end = min(len(doc), entity.end + 5)
        context_tokens = doc[start:end]
        
        # More tokens = richer context
        context_length = len(context_tokens)
        if context_length < 3:
            return 0.5
        elif context_length < 7:
            return 0.7
        else:
            return 0.9
    
    def _get_entity_context(self, doc, entity, window: int = 50) -> str:
        """Get context around entity."""
        start = max(0, entity.start_char - window)
        end = min(len(doc.text), entity.end_char + window)
        return doc.text[start:end].strip()
    
    def _extract_custom_entities(self, doc, min_confidence: float) -> List[Dict]:
        """Extract entities using custom patterns."""
        entities = []
        
        # Extract using custom patterns
        for ent in doc.ents:
            if hasattr(ent, '_') and hasattr(ent._, 'rexi_type'):
                entity_type = ent._.rexi_type
                if entity_type in self.custom_patterns:
                    confidence = 0.8  # Higher confidence for pattern matches
                    
                    if confidence >= min_confidence:
                        entities.append({
                            "text": ent.text,
                            "type": entity_type,
                            "confidence": confidence,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "source": "custom_pattern",
                            "context": self._get_entity_context(doc, ent)
                        })
        
        return entities
    
    def _merge_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge overlapping entities, keeping highest confidence."""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x["start"])
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap
            if next_entity["start"] < current["end"]:
                # Overlap detected, keep the one with higher confidence
                if next_entity["confidence"] > current["confidence"]:
                    current = next_entity
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def _fallback_extraction(self, text: str) -> List[Dict]:
        """Fallback extraction using basic patterns."""
        entities = []
        
        # Basic pattern matching for common entity types
        import re
        
        # Skills (technical terms)
        skill_patterns = [
            r'\b(python|java|javascript|react|vue|angular|docker|kubernetes|aws|azure|gcp)\b',
            r'\b(machine learning|deep learning|neural networks|nlp|computer vision)\b',
            r'\b(data analysis|statistics|research|writing|communication)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "type": EntityType.SKILL,
                    "confidence": 0.6,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "fallback",
                    "context": text[max(0, match.start()-20):match.end()+20]
                })
        
        # Organizations (capitalized multi-word terms)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        matches = re.finditer(org_pattern, text)
        for match in matches:
            entities.append({
                "text": match.group(),
                "type": EntityType.ORGANIZATION,
                "confidence": 0.5,
                "start": match.start(),
                "end": match.end(),
                "source": "fallback",
                "context": text[max(0, match.start()-20):match.end()+20]
            })
        
        return entities
    
    def extract_entities_with_llm(self, text: str, context: str = "") -> List[Dict]:
        """Extract entities using LLM for complex cases."""
        if not self.llm_service.is_available():
            logger.warning("LLM service not available for entity extraction")
            return []
        
        try:
            # Create prompt for LLM
            system_prompt = """
            You are an expert entity extractor. Extract entities from the text and classify them into these types:
            - person: People or individuals
            - concept: Abstract concepts or ideas  
            - skill: Abilities or capabilities
            - topic: Subject areas or topics
            - project: Projects or initiatives
            - tool: Tools or software
            - organization: Companies or groups
            - event: Events or occurrences
            - idea: Ideas or proposals
            - task: Tasks or activities
            - goal: Goals or objectives
            
            Return entities in JSON format with:
            - text: entity text
            - type: entity type
            - confidence: confidence score (0-1)
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nText: {text}"}
            ]
            
            response = self.llm_service.chat_completion(
                messages, 
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse LLM response
            import json
            try:
                llm_entities = json.loads(response)
                if isinstance(llm_entities, list):
                    return llm_entities
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM entity extraction response")
            
            return []
            
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            return []
    
    def resolve_entities(self, entities: List[Dict], existing_entities: List[Entity]) -> List[Entity]:
        """Resolve extracted entities against existing knowledge graph."""
        resolved_entities = []
        
        for entity_data in entities:
            # Try to match with existing entities
            matched_entity = self._find_matching_entity(entity_data, existing_entities)
            
            if matched_entity:
                # Update existing entity
                matched_entity.confidence = max(matched_entity.confidence, entity_data["confidence"])
                resolved_entities.append(matched_entity)
            else:
                # Create new entity
                entity = Entity(
                    name=entity_data["text"],
                    type=entity_data["type"],
                    confidence=entity_data["confidence"],
                    properties={
                        "source": entity_data["source"],
                        "context": entity_data.get("context", ""),
                        "extraction_method": "advanced_ner"
                    }
                )
                
                # Generate embedding for the entity
                entity_text = f"{entity.name} {entity.properties.get('context', '')}"
                entity.embedding = self.embedding_service.encode_text(entity_text)
                
                resolved_entities.append(entity)
        
        return resolved_entities
    
    def _find_matching_entity(self, entity_data: Dict, existing_entities: List[Entity]) -> Optional[Entity]:
        """Find matching entity in existing knowledge graph."""
        entity_text = entity_data["text"].lower().strip()
        entity_type = entity_data["type"]
        
        for existing_entity in existing_entities:
            # Exact name match
            if existing_entity.name.lower().strip() == entity_text and existing_entity.type == entity_type:
                return existing_entity
            
            # Similarity match using embeddings
            if existing_entity.type == entity_type and existing_entity.embedding:
                # Generate embedding for new entity
                new_embedding = self.embedding_service.encode_text(entity_text)
                similarity = self.embedding_service.compute_similarity(
                    existing_entity.embedding, 
                    new_embedding
                )
                
                if similarity > 0.85:  # High similarity threshold
                    return existing_entity
        
        return None
    
    def get_extraction_statistics(self) -> Dict:
        """Get statistics about entity extraction performance."""
        return {
            "spacy_model_loaded": self.nlp is not None,
            "custom_patterns_count": len(self.custom_patterns),
            "supported_entity_types": list(EntityType),
            "extraction_methods": ["spacy", "custom_patterns", "llm", "fallback"]
        }
