"""
Adapters for RAGFlow.

This package contains adapters that implement the core interfaces
for various external libraries and services.
"""

from ragflow.adapters.chunking_strategies import RecursiveCharacterTextSplitterAdapter
from ragflow.adapters.embedding_models import SentenceTransformersAdapter
from ragflow.adapters.llms import GeminiAdapter
from ragflow.adapters.retrieval_strategies import SimpleSimilarityRetriever
from ragflow.adapters.vector_stores import ChromaDBAdapter

__all__ = [
    "ChromaDBAdapter",
    "SentenceTransformersAdapter",
    "GeminiAdapter",
    "RecursiveCharacterTextSplitterAdapter",
    "SimpleSimilarityRetriever",
]

"""Adapter implementations for various RAGFlow components."""
