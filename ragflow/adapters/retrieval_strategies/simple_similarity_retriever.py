"""Simple similarity-based retrieval strategy adapter.

This module provides an adapter implementing the RetrievalStrategyInterface
for simple similarity-based retrieval.
"""

from typing import Any, Dict, List

from ragflow.core.interfaces import RetrievalStrategyInterface, VectorStoreInterface


class SimpleSimilarityRetriever(RetrievalStrategyInterface):
    """Simple similarity-based retrieval strategy.

    Adapter for a simple similarity-based retrieval strategy that implements
    the RetrievalStrategyInterface.

    This adapter uses a vector store to retrieve documents based on their
    similarity to the query.
    """

    def __init__(self, vector_store: VectorStoreInterface, k: int = 4):
        """Initialize the SimpleSimilarityRetriever.

        Args:
            vector_store: The vector store to retrieve documents from.
            k: Number of documents to retrieve (default: 4).
        """
        self.vector_store = vector_store
        self.k = k

    def retrieve(
        self, query: str, vector_store: VectorStoreInterface, k: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the vector store.

        Args:
            query: The query to find relevant documents for.
            vector_store: The vector store to retrieve documents from.
                If provided, this overrides the vector_store set in the constructor.
            k: Number of documents to retrieve.
                If provided, this overrides the k value set in the constructor.

        Returns:
            List of retrieved documents.
        """
        # Use the provided vector_store if available, otherwise use the one from the constructor
        target_store = vector_store or self.vector_store

        # Use the provided k if available, otherwise use the one from the constructor
        target_k = k if k is not None else self.k

        # Perform similarity search
        return target_store.similarity_search(query, k=target_k)

    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: Query string

        Returns:
            List of relevant documents
        """
        return self.retrieve(query, None)

    def get_relevant_documents_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query with relevance scores.

        Args:
            query: Query string

        Returns:
            List of tuples containing (document, score)
        """
        # For the default adapter, we'll use a simple similarity search
        # and then assign a placeholder score of 1.0 to each result
        # This is a simplification, as we don't have direct access to scores
        # in the basic VectorStoreInterface

        documents = self.retrieve(query, None)

        # Create a list of tuples with documents and placeholder scores
        # In a more advanced implementation, these would be actual relevance scores
        # For now, we apply a descending score that starts at 1.0 and decreases
        # This assumes the vector store returns results in order of relevance
        results = []
        for i, doc in enumerate(documents):
            score = 1.0 - (i * (0.9 / max(len(documents), 1)))
            results.append({**doc, "score": score})

        return results
