"""
AI agents for REXI knowledge graph processing.
"""

# Temporarily disable entity extractor due to spaCy compatibility issues
# from .entity_extractor import EntityExtractor
# from .relation_extractor import RelationExtractor
from .entity_resolver import EntityResolver
from .memory_evolution import MemoryEvolutionEngine
from .temporal_reasoning import TemporalReasoningEngine
from .self_learning import SelfLearningEngine

__all__ = [
    # "EntityExtractor",
    # "RelationExtractor", 
    "EntityResolver",
    "MemoryEvolutionEngine",
    "TemporalReasoningEngine",
    "SelfLearningEngine"
]
