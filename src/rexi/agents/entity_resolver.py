"""
Entity resolution agent for REXI - handles deduplication and merging.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher
import re

from rexi.models.entities import Entity, EntityType
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class EntityResolver:
    """Advanced entity resolution with multiple similarity metrics."""
    
    def __init__(self):
        """Initialize entity resolver."""
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
        # Similarity thresholds
        self.name_similarity_threshold = 0.85
        self.embedding_similarity_threshold = 0.90
        self.context_similarity_threshold = 0.80
        
        # Alias patterns
        self.alias_patterns = {
            # Common abbreviations
            r"^(AI|ML|DL|NLP|CV|API|UI|UX|SQL|NoSQL|HTTP|HTTPS|JSON|XML|HTML|CSS|JS|TS)$": "abbreviation",
            # Acronyms (all caps)
            r"^[A-Z]{2,}$": "acronym",
            # Person name variations
            r"^(Dr|Mr|Mrs|Ms|Prof)\.\s+([A-Z][a-z]+)": "person_title",
            # Organization suffixes
            r"(.+)(Inc|Corp|LLC|Ltd|Co|Company|Corporation)$": "organization_suffix",
        }
        
        # Common entity variations
        self.common_variations = {
            "machine learning": ["ML", "Machine Learning"],
            "artificial intelligence": ["AI", "Artificial Intelligence"],
            "python": ["Python", "PY"],
            "javascript": ["JavaScript", "JS", "js"],
            "react": ["React", "React.js", "ReactJS"],
            "docker": ["Docker", "docker"],
            "kubernetes": ["Kubernetes", "k8s", "K8s"],
            "amazon web services": ["AWS", "Amazon Web Services"],
            "microsoft azure": ["Azure", "Microsoft Azure"],
            "google cloud platform": ["GCP", "Google Cloud Platform"],
        }
    
    def resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """Resolve and merge duplicate entities."""
        if not entities:
            return []
        
        logger.info(f"Resolving {len(entities)} entities")
        
        # Group entities by type for more efficient resolution
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)
        
        resolved_entities = []
        
        for entity_type, type_entities in entities_by_type.items():
            # Resolve within each entity type
            resolved_type_entities = self._resolve_entities_by_type(type_entities, entity_type)
            resolved_entities.extend(resolved_type_entities)
        
        logger.info(f"Resolved to {len(resolved_entities)} unique entities")
        return resolved_entities
    
    def _resolve_entities_by_type(self, entities: List[Entity], entity_type: EntityType) -> List[Entity]:
        """Resolve entities within a specific type."""
        if len(entities) <= 1:
            return entities
        
        # Create similarity matrix
        similarity_matrix = self._compute_similarity_matrix(entities)
        
        # Find clusters of similar entities
        clusters = self._find_entity_clusters(similarity_matrix, entities)
        
        # Merge entities within each cluster
        merged_entities = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged_entities.append(entities[cluster[0]])
            else:
                merged_entity = self._merge_entity_cluster([entities[i] for i in cluster])
                merged_entities.append(merged_entity)
        
        return merged_entities
    
    def _compute_similarity_matrix(self, entities: List[Entity]) -> np.ndarray:
        """Compute similarity matrix between entities."""
        n = len(entities)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._compute_entity_similarity(entities[i], entities[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _compute_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Compute similarity between two entities using multiple metrics."""
        similarities = []
        
        # Name similarity
        name_sim = self._compute_name_similarity(entity1.name, entity2.name)
        similarities.append(("name", name_sim, 0.3))
        
        # Embedding similarity
        if entity1.embedding and entity2.embedding:
            embed_sim = self.embedding_service.compute_similarity(
                entity1.embedding, entity2.embedding
            )
            similarities.append(("embedding", embed_sim, 0.4))
        
        # Context similarity
        context_sim = self._compute_context_similarity(entity1, entity2)
        similarities.append(("context", context_sim, 0.2))
        
        # Type match (already ensured by grouping)
        similarities.append(("type", 1.0, 0.1))
        
        # Weighted combination
        total_weight = sum(weight for _, _, weight in similarities)
        weighted_similarity = sum(score * weight for _, score, weight in similarities) / total_weight
        
        return weighted_similarity
    
    def _compute_name_similarity(self, name1: str, name2: str) -> float:
        """Compute name similarity using multiple methods."""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        if name1 == name2:
            return 1.0
        
        # Exact match after normalization
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)
        
        if norm1 == norm2:
            return 0.95
        
        # Check common variations
        for key, variations in self.common_variations.items():
            if name1 in variations and name2 in variations:
                return 0.9
        
        # Sequence matcher similarity
        sequence_sim = SequenceMatcher(None, name1, name2).ratio()
        
        # Check for substring matches
        substring_sim = 0.0
        if name1 in name2 or name2 in name1:
            substring_sim = 0.8
        
        # Check for alias patterns
        alias_sim = self._check_alias_similarity(name1, name2)
        
        # Combine similarities
        return max(sequence_sim, substring_sim, alias_sim)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Handle common patterns
        name = re.sub(r'\b(inc|corp|llc|ltd|co|company|corporation)\b', '', name)
        name = re.sub(r'\b(dr|mr|mrs|ms|prof)\.\s*', '', name)
        
        return name.lower()
    
    def _check_alias_similarity(self, name1: str, name2: str) -> float:
        """Check if names match alias patterns."""
        for pattern, pattern_type in self.alias_patterns.items():
            match1 = re.match(pattern, name1, re.IGNORECASE)
            match2 = re.match(pattern, name2, re.IGNORECASE)
            
            if match1 and match2:
                if pattern_type == "abbreviation":
                    return 0.9
                elif pattern_type == "acronym":
                    return 0.85
                elif pattern_type == "person_title":
                    # Compare the actual names (without titles)
                    name1_core = match1.group(2) if len(match1.groups()) > 0 else match1.group(1)
                    name2_core = match2.group(2) if len(match2.groups()) > 0 else match2.group(1)
                    return SequenceMatcher(None, name1_core.lower(), name2_core.lower()).ratio()
                elif pattern_type == "organization_suffix":
                    # Compare the core names (without suffixes)
                    name1_core = match1.group(1)
                    name2_core = match2.group(1)
                    return SequenceMatcher(None, name1_core.lower(), name2_core.lower()).ratio()
        
        return 0.0
    
    def _compute_context_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Compute context similarity between entities."""
        context1 = entity1.properties.get("context", "")
        context2 = entity2.properties.get("context", "")
        
        if not context1 or not context2:
            return 0.5
        
        # Simple word overlap similarity
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _find_entity_clusters(self, similarity_matrix: np.ndarray, entities: List[Entity]) -> List[List[int]]:
        """Find clusters of similar entities using similarity matrix."""
        n = len(entities)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if not visited[i]:
                cluster = self._find_connected_component(i, similarity_matrix, visited)
                clusters.append(cluster)
        
        return clusters
    
    def _find_connected_component(self, start_idx: int, similarity_matrix: np.ndarray, visited: List[bool]) -> List[int]:
        """Find connected component using DFS."""
        stack = [start_idx]
        component = []
        
        while stack:
            idx = stack.pop()
            if not visited[idx]:
                visited[idx] = True
                component.append(idx)
                
                # Find similar entities
                for j, similarity in enumerate(similarity_matrix[idx]):
                    if j != idx and similarity >= self.embedding_similarity_threshold and not visited[j]:
                        stack.append(j)
        
        return component
    
    def _merge_entity_cluster(self, entities: List[Entity]) -> Entity:
        """Merge a cluster of similar entities."""
        if len(entities) == 1:
            return entities[0]
        
        # Choose the best entity as the base (highest confidence)
        base_entity = max(entities, key=lambda e: e.confidence)
        
        # Merge properties
        merged_properties = {}
        for entity in entities:
            merged_properties.update(entity.properties)
        
        # Merge source references
        all_source_refs = []
        for entity in entities:
            all_source_refs.extend(entity.source_references)
        merged_source_refs = list(set(all_source_refs))  # Remove duplicates
        
        # Merge embeddings (average)
        merged_embedding = None
        embeddings = [e.embedding for e in entities if e.embedding]
        if embeddings:
            merged_embedding = np.mean(embeddings, axis=0).tolist()
        
        # Create merged entity
        merged_entity = Entity(
            id=base_entity.id,  # Keep the base entity's ID
            name=self._choose_best_name(entities),
            type=base_entity.type,
            description=self._merge_descriptions(entities),
            confidence=max(e.confidence for e in entities),
            created_at=min(e.created_at for e in entities),
            updated_at=max(e.updated_at for e in entities),
            source_references=merged_source_refs,
            properties=merged_properties,
            privacy_level=min(e.privacy_level for e in entities),  # Most restrictive
            embedding=merged_embedding
        )
        
        # Add merge metadata
        merged_entity.properties["merged_from"] = [e.id for e in entities if e.id != base_entity.id]
        merged_entity.properties["merge_count"] = len(entities)
        
        return merged_entity
    
    def _choose_best_name(self, entities: List[Entity]) -> str:
        """Choose the best name from merged entities."""
        # Prefer the most common name
        name_counts = defaultdict(int)
        for entity in entities:
            name_counts[entity.name] += entity.confidence  # Weight by confidence
        
        best_name = max(name_counts.items(), key=lambda x: x[1])[0]
        return best_name
    
    def _merge_descriptions(self, entities: List[Entity]) -> str:
        """Merge descriptions from multiple entities."""
        descriptions = [e.description for e in entities if e.description]
        
        if not descriptions:
            return ""
        
        if len(descriptions) == 1:
            return descriptions[0]
        
        # Choose the longest description, or combine if similar length
        longest_desc = max(descriptions, key=len)
        
        # If other descriptions add significant information, combine them
        other_descs = [d for d in descriptions if d != longest_desc and len(d) > len(longest_desc) * 0.5]
        
        if other_descs:
            combined = longest_desc + " " + " ".join(other_descs)
            return combined[:500]  # Limit length
        
        return longest_desc
    
    def create_alias_mapping(self, entities: List[Entity]) -> Dict[str, str]:
        """Create mapping of entity aliases to canonical names."""
        alias_mapping = {}
        
        for entity in entities:
            canonical_name = entity.name
            
            # Add variations
            for key, variations in self.common_variations.items():
                if canonical_name.lower() == key.lower():
                    for variation in variations:
                        alias_mapping[variation] = canonical_name
                elif canonical_name in variations:
                    alias_mapping[canonical_name] = key
            
            # Add normalized version
            normalized = self._normalize_name(canonical_name)
            if normalized != canonical_name.lower():
                alias_mapping[normalized] = canonical_name
        
        return alias_mapping
    
    def resolve_with_llm(self, entities: List[Entity]) -> List[Entity]:
        """Use LLM for complex entity resolution cases."""
        if not self.llm_service.is_available():
            logger.warning("LLM service not available for entity resolution")
            return entities
        
        # Group potentially ambiguous entities
        ambiguous_groups = self._find_ambiguous_groups(entities)
        
        resolved_entities = list(entities)  # Start with all entities
        
        for group in ambiguous_groups:
            if len(group) > 1:
                # Use LLM to resolve ambiguity
                merged_entity = self._llm_resolve_group(group)
                if merged_entity:
                    # Remove original entities and add merged one
                    for entity in group:
                        if entity in resolved_entities:
                            resolved_entities.remove(entity)
                    resolved_entities.append(merged_entity)
        
        return resolved_entities
    
    def _find_ambiguous_groups(self, entities: List[Entity]) -> List[List[Entity]]:
        """Find groups of potentially ambiguous entities."""
        groups = []
        
        # Group by type and similar names
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)
        
        for entity_type, type_entities in entities_by_type.items():
            # Find entities with similar names but below automatic merge threshold
            for i, entity1 in enumerate(type_entities):
                similar_group = [entity1]
                
                for entity2 in type_entities[i+1:]:
                    similarity = self._compute_name_similarity(entity1.name, entity2.name)
                    if 0.6 < similarity < 0.85:  # Similar but not confidently the same
                        similar_group.append(entity2)
                
                if len(similar_group) > 1:
                    groups.append(similar_group)
        
        return groups
    
    def _llm_resolve_group(self, entities: List[Entity]) -> Optional[Entity]:
        """Use LLM to resolve a group of ambiguous entities."""
        try:
            # Prepare entity descriptions for LLM
            entity_descriptions = []
            for i, entity in enumerate(entities):
                desc = f"{i+1}. Name: {entity.name}\n"
                desc += f"   Type: {entity.type.value}\n"
                desc += f"   Description: {entity.description or 'N/A'}\n"
                desc += f"   Context: {entity.properties.get('context', 'N/A')}\n"
                entity_descriptions.append(desc)
            
            system_prompt = """
            You are an expert entity resolver. Determine if the following entities represent the same concept or different concepts.
            
            If they represent the same concept, provide a merged entity with the best name and description.
            If they represent different concepts, indicate that they should remain separate.
            
            Respond in JSON format:
            {
                "should_merge": true/false,
                "merged_entity": {
                    "name": "merged name",
                    "description": "merged description"
                } (only if should_merge is true)
            }
            """
            
            user_prompt = "Entities to resolve:\n\n" + "\n".join(entity_descriptions)
            
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
                
                if result.get("should_merge", False):
                    # Create merged entity
                    base_entity = max(entities, key=lambda e: e.confidence)
                    merged_data = result.get("merged_entity", {})
                    
                    merged_entity = Entity(
                        id=base_entity.id,
                        name=merged_data.get("name", base_entity.name),
                        type=base_entity.type,
                        description=merged_data.get("description", base_entity.description),
                        confidence=max(e.confidence for e in entities),
                        created_at=min(e.created_at for e in entities),
                        updated_at=max(e.updated_at for e in entities),
                        source_references=list(set(sum([e.source_references for e in entities], []))),
                        properties={**base_entity.properties, "llm_merged": True}
                    )
                    
                    return merged_entity
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM entity resolution response")
            
        except Exception as e:
            logger.error(f"LLM entity resolution failed: {e}")
        
        return None
    
    def get_resolution_statistics(self) -> Dict:
        """Get statistics about entity resolution."""
        return {
            "name_similarity_threshold": self.name_similarity_threshold,
            "embedding_similarity_threshold": self.embedding_similarity_threshold,
            "context_similarity_threshold": self.context_similarity_threshold,
            "alias_patterns_count": len(self.alias_patterns),
            "common_variations_count": len(self.common_variations),
            "resolution_methods": ["name_similarity", "embedding_similarity", "context_similarity", "llm_resolution"]
        }
