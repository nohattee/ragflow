"""
Default RAG Pipeline implementation for RAGFlow.

This module provides a pre-configured RAGPipeline implementation that uses the default
component stack specified in the PRD:

- Vector Store: ChromaDBAdapter (implementing VectorStoreInterface)
- Embedding Model: SentenceTransformersAdapter (implementing EmbeddingModelInterface)
- LLM: GeminiAdapter (implementing LLMInterface)
- Chunking Strategy: RecursiveCharacterTextSplitterAdapter (implementing ChunkingStrategyInterface)
- Retrieval Strategy: SimilarityRetrievalStrategy (implementing RetrievalStrategyInterface)

This ready-to-use pipeline enables developers to quickly get started with RAG
without having to configure each component individually.
"""

import os
from typing import Any, Dict, List, Optional

from ..adapters.chunking_strategies.recursive_character_splitter_adapter import (
    RecursiveCharacterTextSplitterAdapter,
)
from ..adapters.embedding_models.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from ..adapters.llms.gemini_adapter import GeminiAdapter
from ..adapters.retrieval_strategies.simple_similarity_retriever import (
    SimpleSimilarityRetriever,
)
from ..adapters.vector_stores.chromadb_adapter import ChromaDBAdapter
from ..core.errors import APIKeyError, ConfigurationError, VectorStoreError
from ..core.pipeline import RAGPipeline


class DefaultRAGPipeline(RAGPipeline):
    """
    A pre-configured RAG pipeline that uses the default component stack.

    This class provides a convenient way to create a fully functional RAG pipeline
    with sensible defaults, requiring minimal configuration from users. It
    abstracts away the complexity of setting up individual components.

    The DefaultRAGPipeline uses the following components:
    - ChromaDBAdapter for vector storage
    - SentenceTransformersAdapter for generating embeddings
    - GeminiAdapter for the language model
    - RecursiveCharacterTextSplitterAdapter for chunking documents
    - SimpleSimilarityRetriever for retrieving relevant documents

    You can configure these components through the constructor parameters.

    Examples:
        Basic usage with environment variable for API key:
        ```python
        # Set GEMINI_API_KEY environment variable first
        pipeline = DefaultRAGPipeline()
        pipeline.add_texts(["Document 1", "Document 2"])
        answer = pipeline.query("What is the content of the documents?")
        ```

        Providing API key directly:
        ```python
        pipeline = DefaultRAGPipeline(api_key="your-gemini-api-key")
        ```

        Customizing component parameters:
        ```python
        pipeline = DefaultRAGPipeline(
            chunk_size=500, chunk_overlap=50, retrieval_k=5, temperature=0.5
        )
        ```
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        api_key: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retrieval_k: int = 4,
        collection_name: str = "ragflow",
        separators: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        top_k: int = 40,
        model_name: str = "gemini-2.5-flash-preview-04-17",
        **kwargs,
    ):
        r"""
        Initialize the DefaultRAGPipeline with configurable parameters.

        Args:
            persist_directory: Directory where ChromaDB will store its data
            api_key: Gemini API key (if not provided, will look for GEMINI_API_KEY env var)
            embedding_model_name: Name of the SentenceTransformers model to use
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            retrieval_k: Number of documents to retrieve for each query
            collection_name: Name of the ChromaDB collection to use
            separators: List of separators to use for text splitting (default: ["\n\n", "\n", " ", ""])
            temperature: Controls randomness in LLM generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens for LLM to generate
            top_p: Nucleus sampling parameter for LLM (0.0 to 1.0)
            top_k: Top-k sampling parameter for LLM
            model_name: Name of the Gemini model to use
            **kwargs: Additional arguments to pass to the underlying components

        Raises:
            APIKeyError: If no Gemini API key is provided or found in environment
            ConfigurationError: If any parameter has an invalid value
            VectorStoreError: If there's an issue initializing the vector store
        """
        # Use API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key is None:
                raise APIKeyError("Gemini", "GEMINI_API_KEY")

        # Validate key parameters
        if chunk_size <= 0:
            raise ConfigurationError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigurationError("chunk_overlap must be between 0 and chunk_size")
        if retrieval_k <= 0:
            raise ConfigurationError("retrieval_k must be greater than 0")
        if temperature < 0.0 or temperature > 1.0:
            raise ConfigurationError("temperature must be between 0.0 and 1.0")
        if top_p <= 0.0 or top_p > 1.0:
            raise ConfigurationError("top_p must be between 0.0 and 1.0")
        if top_k <= 0:
            raise ConfigurationError("top_k must be greater than 0")

        # Get additional component-specific parameters from kwargs
        chunker_kwargs = kwargs.get("chunker_kwargs", {})
        embedder_kwargs = kwargs.get("embedder_kwargs", {})
        vector_store_kwargs = kwargs.get("vector_store_kwargs", {})
        retriever_kwargs = kwargs.get("retriever_kwargs", {})
        llm_kwargs = kwargs.get("llm_kwargs", {})

        # Initialize components
        try:
            chunking_strategy = RecursiveCharacterTextSplitterAdapter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                **chunker_kwargs,
            )

            embedding_model = SentenceTransformersAdapter(
                model_name=embedding_model_name, **embedder_kwargs
            )

            vector_store = ChromaDBAdapter(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                **vector_store_kwargs,
            )

            retrieval_strategy = SimpleSimilarityRetriever(
                vector_store=vector_store, k=retrieval_k, **retriever_kwargs
            )

            llm = GeminiAdapter(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                **llm_kwargs,
            )
        except Exception as e:
            # Convert any component initialization error to a more specific RAGFlow error
            if "chroma" in str(e).lower() or "persist" in str(e).lower():
                raise VectorStoreError(f"Error initializing ChromaDB: {str(e)}")
            elif "model_name" in str(e).lower() or "sentence" in str(e).lower():
                raise ConfigurationError(f"Error with embedding model: {str(e)}")
            elif "api_key" in str(e).lower() or "gemini" in str(e).lower():
                raise APIKeyError("Gemini", "GEMINI_API_KEY")
            else:
                raise ConfigurationError(f"Error initializing components: {str(e)}")

        # Initialize the base RAGPipeline with our components
        super().__init__(
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            vector_store=vector_store,
            retrieval_strategy=retrieval_strategy,
            llm=llm,
        )

        # Store the configuration for reference
        self.config = {
            "persist_directory": persist_directory,
            "embedding_model_name": embedding_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_k": retrieval_k,
            "collection_name": collection_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "model_name": model_name,
        }

    @classmethod
    def from_existing_db(
        cls,
        persist_directory: str,
        api_key: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "ragflow",
        **kwargs,
    ) -> "DefaultRAGPipeline":
        """
        Create a DefaultRAGPipeline from an existing ChromaDB database.

        This factory method makes it easy to resume using a previously
        created pipeline with an existing vector database.

        Args:
            persist_directory: Path to the existing ChromaDB directory
            api_key: Gemini API key (if not provided, will look for GEMINI_API_KEY env var)
            embedding_model_name: Name of the SentenceTransformers model to use
            collection_name: Name of the ChromaDB collection to use
            **kwargs: Additional arguments to pass to the pipeline

        Returns:
            A configured DefaultRAGPipeline instance

        Raises:
            VectorStoreError: If the ChromaDB directory doesn't exist or is invalid
            APIKeyError: If no Gemini API key is provided or found in environment
        """
        # Check if the directory exists
        if not os.path.exists(persist_directory):
            raise VectorStoreError(
                f"Vector store directory does not exist: {persist_directory}"
            )

        return cls(
            persist_directory=persist_directory,
            api_key=api_key,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the pipeline.

        Returns:
            Dictionary containing the configuration parameters
        """
        return self.config.copy()
