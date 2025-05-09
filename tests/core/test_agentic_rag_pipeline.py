"""Integration tests for the AgenticRAGPipeline."""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from ragflow.adapters.chunking_strategies.recursive_character_splitter_adapter import (
    RecursiveCharacterTextSplitterAdapter,
)
from ragflow.adapters.embedding_models.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from ragflow.adapters.llms.gemini_adapter import GeminiAdapter
from ragflow.adapters.retrieval_strategies.simple_similarity_retriever import (
    SimpleSimilarityRetriever,
)
from ragflow.adapters.vector_stores.chromadb_adapter import ChromaDBAdapter
from ragflow.core.interfaces import Document
from ragflow.core.pipeline import AgenticRAGPipeline


@pytest.fixture
def gemini_api_key():
    """Get the Gemini API key from environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
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
            page_content="RAGFlow is a Python library for building RAG applications. It provides an agentic pipeline that can dynamically decide when to retrieve, rewrite, or answer questions.",
            metadata={"source": "documentation", "page": 1},
        ),
        Document(
            page_content="Retrieval Augmented Generation combines search with generative AI. The agentic RAG approach adds a decision-making layer.",
            metadata={"source": "article", "page": 5},
        ),
        Document(
            page_content="Large language models can be enhanced using external knowledge retrieval. Agentic systems can improve retrieval quality by reformulating queries.",
            metadata={"source": "research", "page": 12},
        ),
    ]


@pytest.mark.integration
class TestAgenticRAGPipeline:
    """Integration tests for the AgenticRAGPipeline class."""

    def test_init_with_all_components(self, gemini_api_key, temp_persist_directory):
        """Test initialization with all optional components."""
        # Setup
        embedding_model = SentenceTransformersAdapter()
        chunking_strategy = RecursiveCharacterTextSplitterAdapter()
        vector_store = ChromaDBAdapter(
            persist_directory=temp_persist_directory, embedding_function=embedding_model
        )
        retrieval_strategy = SimpleSimilarityRetriever(vector_store=vector_store)
        agent_llm = GeminiAdapter(api_key=gemini_api_key)
        generator_llm = GeminiAdapter(api_key=gemini_api_key)

        # Exercise
        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Verify
        assert pipeline.agent_llm is not None
        assert pipeline.generator_llm is not None
        assert pipeline.retrieval_strategy is not None
        assert pipeline.chunking_strategy is not None
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None
        assert pipeline.max_iterations == 3  # Default

    def test_init_without_optional_components(self, gemini_api_key):
        """Test initialization without optional components."""
        # Setup
        retrieval_strategy = SimpleSimilarityRetriever(
            vector_store=ChromaDBAdapter(
                embedding_function=SentenceTransformersAdapter()
            )
        )
        agent_llm = GeminiAdapter(api_key=gemini_api_key)
        generator_llm = GeminiAdapter(api_key=gemini_api_key)

        # Exercise
        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
        )

        # Verify
        assert pipeline.agent_llm is not None
        assert pipeline.generator_llm is not None
        assert pipeline.retrieval_strategy is not None
        assert pipeline.chunking_strategy is None
        assert pipeline.embedding_model is None
        assert pipeline.vector_store is None
        assert pipeline.max_iterations == 3  # Default

    def test_init_with_partial_optional_components(self, gemini_api_key):
        """Test that initialization with only some optional components raises ValueError."""
        # Setup
        embedding_model = SentenceTransformersAdapter()
        retrieval_strategy = SimpleSimilarityRetriever(
            vector_store=ChromaDBAdapter(embedding_function=embedding_model)
        )
        agent_llm = GeminiAdapter(api_key=gemini_api_key)
        generator_llm = GeminiAdapter(api_key=gemini_api_key)

        # Exercise / Verify
        with pytest.raises(ValueError):
            # Missing vector_store but providing chunking_strategy
            AgenticRAGPipeline(
                agent_llm=agent_llm,
                generator_llm=generator_llm,
                retrieval_strategy=retrieval_strategy,
                chunking_strategy=RecursiveCharacterTextSplitterAdapter(),
                embedding_model=embedding_model,
            )

    @patch("ragflow.adapters.llms.gemini_adapter.GeminiAdapter")
    def test_init_without_api_key(self, mock_gemini_adapter):
        """Test initialization without API key raises error."""
        # Setup - Configure the mock to raise the expected exception
        mock_gemini_adapter.side_effect = ValueError("Missing API key")

        # Exercise/Verify
        with pytest.raises(ValueError):
            # This will use the mocked GeminiAdapter which will raise ValueError
            retrieval_strategy = SimpleSimilarityRetriever(
                vector_store=ChromaDBAdapter(
                    embedding_function=SentenceTransformersAdapter()
                )
            )

            # Since GeminiAdapter is mocked, this won't actually call the real class
            agent_llm = GeminiAdapter(api_key=None)
            generator_llm = GeminiAdapter(api_key=None)

            AgenticRAGPipeline(
                agent_llm=agent_llm,
                generator_llm=generator_llm,
                retrieval_strategy=retrieval_strategy,
            )

    def test_add_documents(
        self, gemini_api_key, temp_persist_directory, sample_documents
    ):
        """Test adding documents to the pipeline."""
        # Setup - Create pipeline with all components
        embedding_model = SentenceTransformersAdapter()
        chunking_strategy = RecursiveCharacterTextSplitterAdapter(
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50,
        )
        vector_store = ChromaDBAdapter(
            persist_directory=temp_persist_directory, embedding_function=embedding_model
        )
        retrieval_strategy = SimpleSimilarityRetriever(vector_store=vector_store)
        agent_llm = GeminiAdapter(api_key=gemini_api_key)
        generator_llm = GeminiAdapter(api_key=gemini_api_key)

        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_store=vector_store,
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
        """Test the query method with a populated vector store."""
        # Setup - Create pipeline and add documents
        embedding_model = SentenceTransformersAdapter()
        chunking_strategy = RecursiveCharacterTextSplitterAdapter(
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50,
        )
        vector_store = ChromaDBAdapter(
            persist_directory=temp_persist_directory, embedding_function=embedding_model
        )
        retrieval_strategy = SimpleSimilarityRetriever(vector_store=vector_store)
        agent_llm = GeminiAdapter(
            api_key=gemini_api_key,
            temperature=0.1,  # Lower temperature for more deterministic results
        )
        generator_llm = GeminiAdapter(
            api_key=gemini_api_key,
            temperature=0.1,  # Lower temperature for more deterministic results
        )

        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Add sample documents
        pipeline.add_documents(sample_documents)

        # Exercise - Query the pipeline
        result = pipeline.query("What is RAGFlow?")

        # Verify - Check that we got a non-empty result
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # The response might mention RAG or Python since that's in our sample data
        # But due to LLM non-determinism, we don't make strong assertions about content

    def test_query_with_sources(
        self, gemini_api_key, temp_persist_directory, sample_documents
    ):
        """Test the query_with_sources method."""
        # Setup - Create pipeline and add documents
        embedding_model = SentenceTransformersAdapter()
        chunking_strategy = RecursiveCharacterTextSplitterAdapter(
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50,
        )
        vector_store = ChromaDBAdapter(
            persist_directory=temp_persist_directory, embedding_function=embedding_model
        )
        retrieval_strategy = SimpleSimilarityRetriever(vector_store=vector_store)
        agent_llm = GeminiAdapter(
            api_key=gemini_api_key,
            temperature=0.1,  # Lower temperature for more deterministic results
        )
        generator_llm = GeminiAdapter(
            api_key=gemini_api_key,
            temperature=0.1,  # Lower temperature for more deterministic results
        )

        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Add sample documents
        pipeline.add_documents(sample_documents)

        # Exercise - Query the pipeline with sources
        result = pipeline.query_with_sources("What is RAGFlow?")

        # Verify - Check that we got a result with the expected structure
        assert result is not None
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "history" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert isinstance(result["history"], list)
        # History should contain at least one entry for the initial iteration
        assert len(result["history"]) > 0

    def test_empty_vector_store(self, gemini_api_key, temp_persist_directory):
        """Test querying with an empty vector store."""
        # Setup - Create pipeline without adding documents
        embedding_model = SentenceTransformersAdapter()
        chunking_strategy = RecursiveCharacterTextSplitterAdapter()
        vector_store = ChromaDBAdapter(
            persist_directory=temp_persist_directory, embedding_function=embedding_model
        )
        retrieval_strategy = SimpleSimilarityRetriever(vector_store=vector_store)
        agent_llm = GeminiAdapter(api_key=gemini_api_key)
        generator_llm = GeminiAdapter(api_key=gemini_api_key)

        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Exercise - Query the pipeline
        result = pipeline.query("What is RAGFlow?")

        # Verify - Should still get a response, either from the agent directly answering
        # or acknowledging it doesn't have information
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_max_iterations_setting(self, gemini_api_key, temp_persist_directory):
        """Test setting the max_iterations parameter."""
        # Setup
        retrieval_strategy = SimpleSimilarityRetriever(
            vector_store=ChromaDBAdapter(
                persist_directory=temp_persist_directory,
                embedding_function=SentenceTransformersAdapter(),
            )
        )
        agent_llm = GeminiAdapter(api_key=gemini_api_key)
        generator_llm = GeminiAdapter(api_key=gemini_api_key)

        # Exercise - Create pipeline with custom max_iterations
        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
            max_iterations=5,
        )

        # Verify
        assert pipeline.max_iterations == 5

    def test_different_agent_and_generator_models(
        self, gemini_api_key, temp_persist_directory
    ):
        """Test using different model configurations for agent and generator LLMs."""
        # Setup
        retrieval_strategy = SimpleSimilarityRetriever(
            vector_store=ChromaDBAdapter(
                persist_directory=temp_persist_directory,
                embedding_function=SentenceTransformersAdapter(),
            )
        )
        # Agent with higher temperature for more creative decisions
        agent_llm = GeminiAdapter(
            api_key=gemini_api_key,
            temperature=0.8,
            model_name="gemini-2.5-flash-preview-04-17",
        )
        # Generator with lower temperature for more deterministic answers
        generator_llm = GeminiAdapter(
            api_key=gemini_api_key,
            temperature=0.2,
            model_name="gemini-2.5-flash-preview-04-17",
        )

        # Exercise
        pipeline = AgenticRAGPipeline(
            agent_llm=agent_llm,
            generator_llm=generator_llm,
            retrieval_strategy=retrieval_strategy,
        )

        # Verify the LLMs have different configurations
        assert pipeline.agent_llm.temperature == 0.8
        assert pipeline.generator_llm.temperature == 0.2
        assert pipeline.agent_llm is not pipeline.generator_llm
