"""
Semantic role labeling agent for REXI - identifies semantic roles in text.
"""

import spacy
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np

from rexi.models.entities import Entity, EntityType
from rexi.models.relationships import Relationship, RelationshipType
from rexi.services.embedding_service import EmbeddingService
from rexi.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class SemanticRoleLabeler:
    """Advanced semantic role labeling using spaCy and custom patterns."""
    
    def __init__(self):
        """Initialize semantic role labeler."""
        self.nlp = None
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self._load_spacy_model()
        
        # Semantic role definitions
        self.semantic_roles = {
            "agent": "The entity that performs the action",
            "patient": "The entity that is affected by the action",
            "theme": "The entity that is moved or changed",
            "experiencer": "The entity that experiences a mental state",
            "instrument": "The tool or means used to perform the action",
            "location": "The place where the action occurs",
            "destination": "The place toward which something moves",
            "source": "The place from which something moves",
            "time": "When the action occurs",
            "manner": "How the action is performed",
            "purpose": "Why the action is performed",
            "cause": "What causes the action",
            "result": "What results from the action",
            "beneficiary": "The entity for whom the action is performed"
        }
        
        # Verb patterns for different action types
        self.verb_patterns = {
            "causative": [
                "cause", "make", "enable", "allow", "force", "compel",
                "induce", "provoke", "stimulate", "trigger", "generate"
            ],
            "transfer": [
                "give", "send", "transfer", "move", "bring", "take",
                "deliver", "transport", "carry", "ship", "provide"
            ],
            "communication": [
                "say", "tell", "explain", "describe", "announce", "report",
                "communicate", "inform", "mention", "state", "declare"
            ],
            "creation": [
                "create", "make", "build", "construct", "develop", "design",
                "produce", "generate", "form", "establish", "invent"
            ],
            "consumption": [
                "eat", "drink", "consume", "use", "utilize", "employ",
                "apply", "operate", "run", "execute", "perform"
            ],
            "perception": [
                "see", "hear", "feel", "smell", "taste", "perceive",
                "notice", "observe", "detect", "sense", "experience"
            ],
            "cognitive": [
                "think", "believe", "know", "understand", "remember", "learn",
                "realize", "recognize", "discover", "figure", "solve"
            ],
            "social": [
                "help", "support", "assist", "cooperate", "collaborate", "work",
                "partner", "team", "join", "participate", "contribute"
            ]
        }
        
        # Preposition patterns for semantic roles
        self.preposition_patterns = {
            "location": ["in", "at", "on", "inside", "outside", "within", "among"],
            "destination": ["to", "into", "onto", "toward", "towards"],
            "source": ["from", "out of", "away from", "off"],
            "instrument": ["with", "by", "using", "through", "via"],
            "time": ["at", "on", "in", "during", "while", "before", "after"],
            "purpose": ["for", "to", "in order to", "so as to"],
            "beneficiary": ["for", "to", "on behalf of"]
        }
    
    def _load_spacy_model(self):
        """Load spaCy model with dependency parsing."""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy large model for SRL")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy medium model for SRL")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model for SRL")
                except OSError:
                    logger.error("Could not load any spaCy model for SRL")
                    self.nlp = None
    
    def extract_semantic_roles(self, text: str, entities: List[Dict] = None) -> List[Dict]:
        """Extract semantic roles from text."""
        if not self.nlp:
            logger.warning("spaCy model not loaded, using fallback SRL")
            return self._fallback_srl(text, entities)
        
        try:
            doc = self.nlp(text)
            semantic_roles = []
            
            # Process each clause/sentence
            for sent in doc.sents:
                sent_roles = self._extract_sentence_roles(sent, entities)
                semantic_roles.extend(sent_roles)
            
            # Post-process and filter roles
            semantic_roles = self._filter_and_score_roles(semantic_roles)
            
            return semantic_roles
            
        except Exception as e:
            logger.error(f"Semantic role labeling failed: {e}")
            return self._fallback_srl(text, entities)
    
    def _extract_sentence_roles(self, sent, entities: List[Dict] = None) -> List[Dict]:
        """Extract semantic roles from a single sentence."""
        roles = []
        
        # Find the main verb (predicate)
        main_verb = self._find_main_verb(sent)
        if not main_verb:
            return roles
        
        # Determine action type
        action_type = self._classify_action_type(main_verb.lemma_)
        
        # Extract roles based on dependency parsing
        verb_roles = self._extract_dependency_roles(main_verb, sent, action_type)
        roles.extend(verb_roles)
        
        # Extract roles based on prepositional patterns
        prep_roles = self._extract_prepositional_roles(main_verb, sent, action_type)
        roles.extend(prep_roles)
        
        # Map entities to roles if provided
        if entities:
            roles = self._map_entities_to_roles(roles, entities)
        
        return roles
    
    def _find_main_verb(self, sent) -> Optional:
        """Find the main verb of the sentence."""
        # Look for root verb
        if sent.root.pos_ == "VERB":
            return sent.root
        
        # Look for auxiliary verbs
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "AUX":
                # Find the main verb connected to this auxiliary
                for child in token.children:
                    if child.pos_ == "VERB":
                        return child
        
        # Fallback: return first verb
        for token in sent:
            if token.pos_ == "VERB":
                return token
        
        return None
    
    def _classify_action_type(self, verb_lemma: str) -> str:
        """Classify the type of action based on verb."""
        for action_type, verbs in self.verb_patterns.items():
            if verb_lemma in verbs:
                return action_type
        
        return "general"
    
    def _extract_dependency_roles(self, verb, sent, action_type: str) -> List[Dict]:
        """Extract roles using dependency parsing."""
        roles = []
        
        # Agent (nsubj)
        for child in verb.children:
            if child.dep_ == "nsubj":
                roles.append({
                    "role": "agent",
                    "text": child.text,
                    "lemma": child.lemma_,
                    "pos": child.pos_,
                    "dependency": child.dep_,
                    "confidence": 0.8,
                    "source": "dependency_parsing",
                    "action_type": action_type
                })
        
        # Patient/Direct Object (dobj)
        for child in verb.children:
            if child.dep_ == "dobj":
                role = "patient" if action_type in ["causative", "transfer", "consumption"] else "theme"
                roles.append({
                    "role": role,
                    "text": child.text,
                    "lemma": child.lemma_,
                    "pos": child.pos_,
                    "dependency": child.dep_,
                    "confidence": 0.8,
                    "source": "dependency_parsing",
                    "action_type": action_type
                })
        
        # Indirect Object (iobj)
        for child in verb.children:
            if child.dep_ == "dative" or (child.dep_ == "prep" and child.lemma_ in ["to", "for"]):
                role = "beneficiary" if child.lemma_ == "for" else "recipient"
                roles.append({
                    "role": role,
                    "text": child.text,
                    "lemma": child.lemma_,
                    "pos": child.pos_,
                    "dependency": child.dep_,
                    "confidence": 0.7,
                    "source": "dependency_parsing",
                    "action_type": action_type
                })
        
        return roles
    
    def _extract_prepositional_roles(self, verb, sent, action_type: str) -> List[Dict]:
        """Extract roles using prepositional patterns."""
        roles = []
        
        for token in sent:
            if token.pos_ == "ADP" and token.dep_ == "prep":
                # Find the object of the preposition
                prep_obj = None
                for child in token.children:
                    if child.dep_ == "pobj":
                        prep_obj = child
                        break
                
                if prep_obj:
                    # Determine role based on preposition
                    role = self._map_preposition_to_role(token.lemma_)
                    
                    if role:
                        roles.append({
                            "role": role,
                            "text": prep_obj.text,
                            "lemma": prep_obj.lemma_,
                            "pos": prep_obj.pos_,
                            "dependency": token.dep_,
                            "preposition": token.lemma_,
                            "confidence": 0.6,
                            "source": "prepositional_pattern",
                            "action_type": action_type
                        })
        
        return roles
    
    def _map_preposition_to_role(self, prep: str) -> Optional[str]:
        """Map preposition to semantic role."""
        for role, prepositions in self.preposition_patterns.items():
            if prep in prepositions:
                return role
        return None
    
    def _map_entities_to_roles(self, roles: List[Dict], entities: List[Dict]) -> List[Dict]:
        """Map extracted entities to semantic roles."""
        entity_lookup = {entity["text"].lower(): entity for entity in entities}
        
        for role in roles:
            role_text_lower = role["text"].lower()
            
            # Direct entity match
            if role_text_lower in entity_lookup:
                role["entity"] = entity_lookup[role_text_lower]
                role["entity_match_type"] = "direct"
                continue
            
            # Partial entity match
            for entity_text, entity in entity_lookup.items():
                if role_text_lower in entity_text or entity_text in role_text_lower:
                    role["entity"] = entity
                    role["entity_match_type"] = "partial"
                    break
        
        return roles
    
    def _filter_and_score_roles(self, roles: List[Dict]) -> List[Dict]:
        """Filter and score semantic roles."""
        filtered_roles = []
        
        for role in roles:
            # Filter low-confidence roles
            if role["confidence"] < 0.5:
                continue
            
            # Adjust confidence based on role importance
            role_importance = {
                "agent": 1.0,
                "patient": 0.9,
                "theme": 0.8,
                "instrument": 0.7,
                "location": 0.6,
                "time": 0.5
            }
            
            importance = role_importance.get(role["role"], 0.5)
            role["confidence"] = min(role["confidence"] * importance, 1.0)
            
            # Add role description
            role["description"] = self.semantic_roles.get(role["role"], "Unknown role")
            
            filtered_roles.append(role)
        
        # Sort by confidence
        filtered_roles.sort(key=lambda x: x["confidence"], reverse=True)
        
        return filtered_roles
    
    def _fallback_srl(self, text: str, entities: List[Dict] = None) -> List[Dict]:
        """Fallback semantic role labeling using simple patterns."""
        import re
        
        roles = []
        
        # Simple verb-object patterns
        verb_object_patterns = [
            r"(\w+)\s+(\w+)",  # Simple verb-object
            r"(\w+)\s+(the|a|an)\s+(\w+)"  # Verb with article
        ]
        
        for pattern in verb_object_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    verb = groups[0]
                    obj = groups[-1]
                    
                    # Add agent role (subject would need more context)
                    roles.append({
                        "role": "agent",
                        "text": "unknown",
                        "confidence": 0.3,
                        "source": "fallback_pattern",
                        "action_type": "general"
                    })
                    
                    # Add patient/theme role
                    role = "patient" if verb in ["eat", "use", "consume"] else "theme"
                    roles.append({
                        "role": role,
                        "text": obj,
                        "confidence": 0.4,
                        "source": "fallback_pattern",
                        "action_type": "general"
                    })
        
        return roles
    
    def extract_srl_with_llm(self, text: str, entities: List[Dict] = None) -> List[Dict]:
        """Extract semantic roles using LLM for complex cases."""
        if not self.llm_service.is_available():
            logger.warning("LLM service not available for SRL")
            return []
        
        try:
            # Prepare entity list for LLM
            entity_list = ""
            if entities:
                entity_list = "\n".join([f"- {e['text']} ({e['type'].value})" for e in entities])
            
            system_prompt = f"""
            You are an expert semantic role labeler. Identify the semantic roles in the text and classify them.
            
            Semantic roles to identify:
            - agent: The entity that performs the action
            - patient: The entity that is affected by the action
            - theme: The entity that is moved or changed
            - experiencer: The entity that experiences a mental state
            - instrument: The tool or means used to perform the action
            - location: The place where the action occurs
            - destination: The place toward which something moves
            - source: The place from which something moves
            - time: When the action occurs
            - manner: How the action is performed
            - purpose: Why the action is performed
            - cause: What causes the action
            - result: What results from the action
            - beneficiary: The entity for whom the action is performed
            
            {f'Entities to consider:\n{entity_list}\n' if entity_list else ''}
            
            Return semantic roles in JSON format with:
            - role: semantic role type
            - text: text filling the role
            - confidence: confidence score (0-1)
            - action_type: type of action (causative, transfer, communication, etc.)
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text: {text}"}
            ]
            
            response = self.llm_service.chat_completion(
                messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse LLM response
            import json
            try:
                llm_roles = json.loads(response)
                if isinstance(llm_roles, list):
                    # Add source method and validate
                    validated_roles = []
                    for role in llm_roles:
                        if role.get("role") in self.semantic_roles:
                            role["source"] = "llm"
                            role["description"] = self.semantic_roles[role["role"]]
                            validated_roles.append(role)
                    return validated_roles
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM SRL response")
            
            return []
            
        except Exception as e:
            logger.error(f"LLM semantic role labeling failed: {e}")
            return []
    
    def convert_roles_to_relationships(self, roles: List[Dict], entities: List[Entity]) -> List[Relationship]:
        """Convert semantic roles to relationships between entities."""
        relationships = []
        
        # Create entity lookup
        entity_lookup = {entity.name.lower(): entity for entity in entities}
        
        # Group roles by action/predicate
        role_groups = defaultdict(list)
        for role in roles:
            # Group by action type or create a generic grouping
            group_key = role.get("action_type", "general")
            role_groups[group_key].append(role)
        
        for action_type, action_roles in role_groups.items():
            # Find agent and patient/theme
            agent_roles = [r for r in action_roles if r["role"] == "agent"]
            patient_roles = [r for r in action_roles if r["role"] in ["patient", "theme"]]
            
            # Create relationships
            for agent_role in agent_roles:
                agent_entity = entity_lookup.get(agent_role["text"].lower())
                
                if not agent_entity and "entity" in agent_role:
                    agent_entity = entity_lookup.get(agent_role["entity"]["text"].lower())
                
                if agent_entity:
                    for patient_role in patient_roles:
                        patient_entity = entity_lookup.get(patient_role["text"].lower())
                        
                        if not patient_entity and "entity" in patient_role:
                            patient_entity = entity_lookup.get(patient_role["entity"]["text"].lower())
                        
                        if patient_entity and agent_entity.id != patient_entity.id:
                            # Determine relationship type based on action and roles
                            rel_type = self._map_action_to_relationship(action_type, agent_role["role"], patient_role["role"])
                            
                            relationship = Relationship(
                                source_entity_id=agent_entity.id,
                                target_entity_id=patient_entity.id,
                                type=rel_type,
                                confidence=min(agent_role["confidence"], patient_role["confidence"]),
                                properties={
                                    "semantic_roles": [agent_role["role"], patient_role["role"]],
                                    "action_type": action_type,
                                    "extraction_method": "semantic_role_labeling",
                                    "evidence": f"{agent_role['text']} {action_type} {patient_role['text']}"
                                }
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _map_action_to_relationship(self, action_type: str, source_role: str, target_role: str) -> RelationshipType:
        """Map action and roles to relationship type."""
        mapping = {
            ("causative", "agent", "patient"): RelationshipType.CAUSES,
            ("causative", "agent", "theme"): RelationshipType.ENABLES,
            ("transfer", "agent", "theme"): RelationshipType.USED_IN,
            ("communication", "agent", "theme"): RelationshipType.RELATED_TO,
            ("creation", "agent", "theme"): RelationshipType.PART_OF,
            ("consumption", "agent", "patient"): RelationshipType.DEPENDS_ON,
            ("social", "agent", "beneficiary"): RelationshipType.SUPPORTS,
        }
        
        return mapping.get((action_type, source_role, target_role), RelationshipType.RELATED_TO)
    
    def get_srl_statistics(self) -> Dict:
        """Get statistics about semantic role labeling."""
        return {
            "spacy_model_loaded": self.nlp is not None,
            "supported_roles": list(self.semantic_roles.keys()),
            "action_types": list(self.verb_patterns.keys()),
            "preposition_patterns": {k: len(v) for k, v in self.preposition_patterns.items()},
            "extraction_methods": ["dependency_parsing", "prepositional_patterns", "llm", "fallback"]
        }
