"""Integration tests for the DefaultRAGPipeline."""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from ragflow.core.errors import APIKeyError, VectorStoreError
from ragflow.core.interfaces import Document
from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline


@pytest.fixture
def gemini_api_key():
    """Get the Gemini API key from environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def temp_persist_directory():
    """Create a temporary directory for the ChromaDB database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="RAGFlow is a Python library for building RAG applications.",
            metadata={"source": "documentation", "page": 1},
        ),
        Document(
            page_content="Retrieval Augmented Generation combines search with generative AI.",
            metadata={"source": "article", "page": 5},
        ),
        Document(
            page_content="Large language models can be enhanced using external knowledge retrieval.",
            metadata={"source": "research", "page": 12},
        ),
    ]


@pytest.mark.integration
class TestDefaultRAGPipeline:
    """Integration tests for the DefaultRAGPipeline class."""

    def test_init_with_api_key(self, gemini_api_key):
        """Test initialization with API key."""
        # Setup & Exercise
        pipeline = DefaultRAGPipeline(api_key=gemini_api_key)

        # Verify the pipeline was created correctly
        # Assuming the API key is not stored in config for security reasons
        assert pipeline.llm is not None
        assert pipeline.chunking_strategy is not None
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None

        # Verify the configuration was stored correctly
        assert (
            pipeline.config["model_name"] == "gemini-2.5-flash-preview-04-17"
        )  # Default
        assert pipeline.config["temperature"] == 0.7  # Default
        assert pipeline.config["chunk_size"] == 1000  # Default
        assert pipeline.config["chunk_overlap"] == 200  # Default

    def test_init_with_custom_config(self, gemini_api_key, temp_persist_directory):
        """Test initialization with custom configuration."""
        # Setup
        custom_config = {
            "model_name": "gemini-2.5-flash-preview-04-17",
            "temperature": 0.5,
            "chunk_size": 500,
            "chunk_overlap": 100,
            "persist_directory": temp_persist_directory,
        }

        # Exercise
        pipeline = DefaultRAGPipeline(api_key=gemini_api_key, **custom_config)

        # Verify configuration
        assert pipeline.config["model_name"] == "gemini-2.5-flash-preview-04-17"
        assert pipeline.config["temperature"] == 0.5
        assert pipeline.config["chunk_size"] == 500
        assert pipeline.config["chunk_overlap"] == 100
        assert pipeline.config["persist_directory"] == temp_persist_directory

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        # Exercise/Verify
        with pytest.raises(APIKeyError):
            DefaultRAGPipeline(api_key=None)

    def clean_db(self, persist_directory):
        """Clean up test database directory."""
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

    def test_add_texts(self, gemini_api_key, temp_persist_directory):
        """Test adding texts to the pipeline."""
        # Setup
        pipeline = DefaultRAGPipeline(
            api_key=gemini_api_key,
            persist_directory=temp_persist_directory,
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50,
        )

        # Sample texts
        texts = [
            "RAGFlow is a Python library for building RAG applications.",
            "Retrieval Augmented Generation combines search with generative AI.",
        ]

        # Exercise
        pipeline.add_texts(texts, metadata=[{"source": "test1"}, {"source": "test2"}])

        # Verify - Check if vector store has content
        collection_size = len(pipeline.vector_store.collection.get()["ids"])
        assert collection_size > 0, (
            "Vector store should contain documents after adding texts"
        )

    def test_add_documents(
        self, gemini_api_key, temp_persist_directory, sample_documents
    ):
        """Test adding documents to the pipeline."""
        # Setup
        pipeline = DefaultRAGPipeline(
            api_key=gemini_api_key,
            persist_directory=temp_persist_directory,
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50,
        )

        # Exercise
        pipeline.add_documents(sample_documents)

        # Verify - Check if vector store has content
        collection_size = len(pipeline.vector_store.collection.get()["ids"])
        assert collection_size > 0, (
            "Vector store should contain documents after adding documents"
        )

        # Verify metadata is preserved (at least in some form)
        collection_data = pipeline.vector_store.collection.get()
        assert "metadatas" in collection_data
        assert len(collection_data["metadatas"]) > 0
        for metadata in collection_data["metadatas"]:
            assert metadata is not None, "Metadata should be preserved"

    def test_query(self, gemini_api_key, temp_persist_directory, sample_documents):
        """Test querying the pipeline."""
        # Setup - Create pipeline and add documents
        pipeline = DefaultRAGPipeline(
            api_key=gemini_api_key,
            persist_directory=temp_persist_directory,
            chunk_size=500,
            chunk_overlap=50,
            temperature=0.1,  # Lower temperature for more deterministic results
        )

        # Add sample documents
        pipeline.add_documents(sample_documents)

        # Exercise - Query the pipeline
        result = pipeline.query("What is RAGFlow?")

        # Verify - Check that we got a non-empty result that makes sense
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # The response should mention RAG or Python since that's in our sample data
        assert any(
            term in result.lower()
            for term in ["rag", "python", "retrieval", "generation"]
        )

    def test_from_existing_db(
        self, gemini_api_key, temp_persist_directory, sample_documents
    ):
        """Test creating pipeline from existing DB."""
        # Setup - Create an initial pipeline and add data
        initial_pipeline = DefaultRAGPipeline(
            api_key=gemini_api_key,
            persist_directory=temp_persist_directory,
            chunk_size=500,
            chunk_overlap=50,
        )

        # Add sample documents to create the vector store
        initial_pipeline.add_documents(sample_documents)

        # Get initial collection stats to compare later
        initial_collection = initial_pipeline.vector_store.collection.get()
        initial_count = len(initial_collection["ids"])

        # Exercise - Create a new pipeline from the existing DB
        new_pipeline = DefaultRAGPipeline.from_existing_db(
            persist_directory=temp_persist_directory, api_key=gemini_api_key
        )

        # Verify - New pipeline should have the same documents
        new_collection = new_pipeline.vector_store.collection.get()
        new_count = len(new_collection["ids"])

        assert new_count == initial_count, (
            "New pipeline should have same number of documents as initial pipeline"
        )
        assert new_pipeline.config["persist_directory"] == temp_persist_directory

    def test_from_existing_db_nonexistent(self, gemini_api_key):
        """Test error when DB directory doesn't exist."""
        # Setup
        with patch("os.path.exists", return_value=False), pytest.raises(
            VectorStoreError
        ):
            # Exercise/Verify
            DefaultRAGPipeline.from_existing_db(
                persist_directory="/nonexistent/path", api_key=gemini_api_key
            )

    def test_get_config(self, gemini_api_key):
        """Test getting the pipeline configuration."""
        # Setup
        pipeline = DefaultRAGPipeline(
            api_key=gemini_api_key, chunk_size=500, retrieval_k=5
        )

        # Exercise
        config = pipeline.get_config()

        # Verify
        assert config["chunk_size"] == 500
        assert config["retrieval_k"] == 5
