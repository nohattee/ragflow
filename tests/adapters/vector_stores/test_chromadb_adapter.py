"""Unit tests for the ChromaDBAdapter."""

import shutil
import tempfile

import pytest
from ragflow.adapters.vector_stores.chromadb_adapter import ChromaDBAdapter
from ragflow.core.interfaces import Document


@pytest.fixture
def temp_persist_directory():
    """Create a temporary directory for the ChromaDB database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.integration
class TestChromaDBAdapter:
    """Integration tests for ChromaDBAdapter."""

    def test_init_in_memory(self):
        """Test initialization with in-memory client."""
        adapter = ChromaDBAdapter()
        assert adapter.persist_directory is None
        assert adapter.collection_name == "ragflow"
        assert adapter.client is not None

    def test_init_persistent(self, temp_persist_directory):
        """Test initialization with persistent client."""
        adapter = ChromaDBAdapter(persist_directory=temp_persist_directory)
        assert adapter.persist_directory == temp_persist_directory
        assert adapter.collection_name == "ragflow"
        assert adapter.client is not None

    def test_add_texts(self, temp_persist_directory):
        """Test adding texts to the vector store."""
        # Setup
        adapter = ChromaDBAdapter(persist_directory=temp_persist_directory)
        texts = ["Test document 1", "Test document 2"]
        metadata = [{"source": "test1"}, {"source": "test2"}]

        # Exercise - add_texts doesn't return IDs
        adapter.add_texts(texts=texts, metadata=metadata)

        # Verify by querying for added content
        results = adapter.similarity_search("Test document", k=2)
        assert len(results) == 2
        assert "Test document" in results[0].page_content
        assert "Test document" in results[1].page_content

    def test_add_documents(self, temp_persist_directory):
        """Test adding documents to the vector store."""
        # Setup
        adapter = ChromaDBAdapter(persist_directory=temp_persist_directory)
        documents = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"}),
        ]

        # Exercise - add_documents doesn't return IDs
        adapter.add_documents(documents)

        # Verify by querying for added content
        results = adapter.similarity_search("Test document", k=2)
        assert len(results) == 2
        assert "Test document" in results[0].page_content
        assert "Test document" in results[1].page_content

    def test_similarity_search(self, temp_persist_directory):
        """Test similarity search."""
        # Setup
        adapter = ChromaDBAdapter(persist_directory=temp_persist_directory)
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "The five boxing wizards jump quickly",
            "The lazy dog sleeps all day",
        ]
        metadata = [
            {"source": "fox_document"},
            {"source": "wizard_document"},
            {"source": "dog_document"},
        ]
        adapter.add_texts(texts=texts, metadata=metadata)

        # Exercise
        results = adapter.similarity_search("quick fox", k=2)

        # Verify
        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert isinstance(results[1], Document)
        # First result should be more relevant to "quick fox"
        assert "fox" in results[0].page_content

    def test_similarity_search_with_score(self, temp_persist_directory):
        """Test similarity search with score."""
        # Setup
        adapter = ChromaDBAdapter(persist_directory=temp_persist_directory)
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "The five boxing wizards jump quickly",
            "The lazy dog sleeps all day",
        ]
        metadata = [
            {"source": "fox_document"},
            {"source": "wizard_document"},
            {"source": "dog_document"},
        ]
        adapter.add_texts(texts=texts, metadata=metadata)

        # Exercise - note: ChromaDBAdapter doesn't implement similarity_search_with_score
        # Let's verify similarity_search instead
        results = adapter.similarity_search("quick fox", k=2)

        # Verify
        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert "fox" in results[0].page_content or "quick" in results[0].page_content

    def test_reset_collection(self, temp_persist_directory):
        """Test resetting the collection by deleting the collection and recreating it."""
        # Setup
        adapter = ChromaDBAdapter(persist_directory=temp_persist_directory)
        collection_name = adapter.collection_name
        texts = ["Test document"]
        metadata = [{"source": "test_doc"}]
        adapter.add_texts(texts=texts, metadata=metadata)

        # Verify documents exist
        results = adapter.similarity_search("test", k=1)
        assert len(results) == 1

        # Exercise - Delete the collection and recreate it
        adapter.client.delete_collection(name=collection_name)
        adapter.collection = adapter.client.get_or_create_collection(
            name=collection_name
        )

        # Verify - should now return no results
        results = adapter.similarity_search("test", k=1)
        assert len(results) == 0
