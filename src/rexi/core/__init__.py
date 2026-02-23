"""
Core modules for REXI knowledge graph system.
"""

from .knowledge_graph import KnowledgeGraph
from .ingestion import IngestionEngine
from .reasoning import ReasoningEngine
from .memory_evolution import MemoryEvolutionCore

__all__ = [
    "KnowledgeGraph",
    "IngestionEngine", 
    "ReasoningEngine",
    "MemoryEvolutionCore"
]
