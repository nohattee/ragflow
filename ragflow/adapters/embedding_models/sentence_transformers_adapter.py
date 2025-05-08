"""Sentence Transformers adapter for generating embeddings.

This module provides an adapter for using Sentence Transformers models
to generate embeddings for documents and queries.
"""

from typing import List

from ragflow.core.interfaces import EmbeddingModelInterface


class SentenceTransformersAdapter(EmbeddingModelInterface):
    """Adapter for Sentence Transformers embedding models.

    This adapter wraps SentenceTransformers models to implement the
    EmbeddingModelInterface, providing consistent embedding generation
    for documents and queries.

    Attributes:
        model_name: Name of the SentenceTransformers model to use.
        model: The loaded SentenceTransformers model instance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the SentenceTransformers adapter.

        Args:
            model_name: Name of the SentenceTransformers model to use.
                Defaults to "all-MiniLM-L6-v2".
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformers model.

        Raises:
            ImportError: If the sentence_transformers package is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError(
                "The sentence_transformers package is required to use this adapter. "
                "Please install it with `pip install sentence-transformers`."
            )

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
            documents: List of text strings to generate embeddings for.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty.")

        # Encode the documents and convert to a list of lists
        embeddings = self.model.encode(documents)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a query string.

        Args:
            query: The query text to embed.

        Returns:
            The embedding vector for the query.

        Raises:
            ValueError: If query is empty.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        # Encode the query and convert to a list
        embedding = self.model.encode(query)
        return embedding.tolist()
