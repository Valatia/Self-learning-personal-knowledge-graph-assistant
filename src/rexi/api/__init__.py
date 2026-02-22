"""
API endpoints for REXI system.
"""

from rexi.api.main import app
from rexi.api.routers import documents, entities, relationships, reasoning

__all__ = ["app", "documents", "entities", "relationships", "reasoning"]
