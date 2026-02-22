"""
Core modules for REXI knowledge graph system.
"""

from rexi.core.knowledge_graph import KnowledgeGraph
from rexi.core.ingestion import IngestionEngine
from rexi.core.reasoning import ReasoningEngine
from rexi.core.memory import MemoryEvolutionEngine

__all__ = [
    "KnowledgeGraph",
    "IngestionEngine",
    "ReasoningEngine", 
    "MemoryEvolutionEngine",
]
