"""
REXI - Self-Learning Personal Knowledge Graph Assistant

A sophisticated AI system that transforms personal data into structured knowledge
with reasoning capabilities.
"""

__version__ = "0.1.0"
__author__ = "REXI Team"
__email__ = "team@rexi.ai"

from rexi.core.knowledge_graph import KnowledgeGraph
from rexi.core.ingestion import IngestionEngine
from rexi.core.reasoning import ReasoningEngine

__all__ = [
    "KnowledgeGraph",
    "IngestionEngine", 
    "ReasoningEngine",
]
