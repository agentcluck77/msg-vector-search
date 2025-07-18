"""
Core Package
Provides the main functionality for the MCP server
"""

from .database import SeaTalkDatabase, MessageProcessor
from .embeddings import EmbeddingProcessor
from .search import SemanticSearchEngine

__all__ = [
    'SeaTalkDatabase',
    'MessageProcessor',
    'EmbeddingProcessor',
    'SemanticSearchEngine'
] 