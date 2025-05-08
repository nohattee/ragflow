"""Unit tests for the SentenceTransformersAdapter."""

import numpy as np
import pytest
from ragflow.adapters.embedding_models.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)


@pytest.mark.integration
class TestSentenceTransformersAdapter:
    """Integration tests for SentenceTransformersAdapter."""

    def test_initialization(self):
        """Test initialization of the adapter."""
        # Initialize with a real model (small model for quick tests)
        adapter = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")
        assert adapter.model_name == "all-MiniLM-L6-v2"
        assert adapter.model is not None

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        adapter = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")
        texts = ["This is a test document.", "This is another test document."]
        embeddings = adapter.embed_documents(texts)

        # Check that we got the right number of embeddings
        assert len(embeddings) == 2

        # Check that each embedding is a vector of the expected dimension
        assert len(embeddings[0]) > 0
        assert len(embeddings[1]) > 0

        # Check that embeddings are different (meaning they captured semantic differences)
        assert not np.array_equal(embeddings[0], embeddings[1])

    def test_embed_query(self):
        """Test embedding a single query."""
        adapter = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")
        query = "What is the meaning of life?"
        embedding = adapter.embed_query(query)

        # Check that the embedding is a vector of the expected dimension
        assert len(embedding) > 0

        # Verify that embedding a single string produces the same result as embedding it in a list
        embedding_as_list = adapter.embed_documents([query])[0]
        assert np.array_equal(embedding, embedding_as_list)

    def test_empty_inputs(self):
        """Test that empty inputs raise appropriate errors."""
        adapter = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")

        # Empty document list should raise ValueError
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            adapter.embed_documents([])

        # Empty query string should raise ValueError
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            adapter.embed_query("")

        # Whitespace-only query should raise ValueError
        with pytest.raises(ValueError, match="Query string cannot be empty"):
            adapter.embed_query("   ")
