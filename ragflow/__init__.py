"""RAGFlow: A high-level framework for Retrieval Augmented Generation.

RAGFlow streamlines the development of RAG applications by providing
intuitive interfaces and pre-configured pipelines built on top of LangChain.
"""

import sys

# Re-export core interfaces for easier imports
from ragflow.core.interfaces import (
    ChunkingStrategyInterface,
    EmbeddingModelInterface,
    LLMInterface,
    RetrievalStrategyInterface,
    VectorStoreInterface,
)

# Re-export default pipeline
from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

__version__ = "0.1.0"

__import__("pysqlite3")

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

__all__ = [
    "ChunkingStrategyInterface",
    "DefaultRAGPipeline",
    "EmbeddingModelInterface",
    "LLMInterface",
    "RetrievalStrategyInterface",
    "VectorStoreInterface",
]
