"""
Vritti - Episodic Memory System for AI Agents.

Stop AI agents from repeating the same mistakes.

Vritti captures failures, analyzes patterns, and prevents AI coding assistants 
from making the same errors twice. Built on KyroDB for vector storage and 
multi-modal embeddings.

Key Features:
    - Episodic memory for failure episodes
    - Multi-perspective LLM analysis
    - Semantic search and retrieval
    - Skills promotion from successful fixes
    - Multi-tenant customer isolation

Example:
    >>> from src import get_settings
    >>> settings = get_settings()
    >>> print(settings.kyrodb.text_address)
"""

from src.config import get_settings

__all__ = ["get_settings"]
