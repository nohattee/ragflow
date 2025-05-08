"""
Customization Examples for RAGFlow

This module demonstrates the various ways to customize RAGFlow:
1. Configuring parameters of the DefaultRAGPipeline
2. Using component-specific kwargs for deeper customization
3. Building custom pipelines from individual adapters
4. Creating configuration profiles for different use cases
"""

import os
from typing import Optional

from dotenv import load_dotenv
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
from ragflow.core.pipeline import RAGPipeline

# Import RAGFlow components and pipeline
from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

# Load environment variables from .env file (if present)
load_dotenv()

# Sample documents for testing
SAMPLE_DOCS = [
    Document(
        page_content="RAGFlow is a high-level framework for building Retrieval Augmented Generation applications in Python.",
        metadata={"source": "documentation", "section": "introduction"},
    ),
    Document(
        page_content="Retrieval Augmented Generation (RAG) is a technique that enhances large language models by retrieving external knowledge.",
        metadata={"source": "documentation", "section": "concepts"},
    ),
    Document(
        page_content="Vector databases store embeddings that represent the semantic meaning of text, enabling similarity search.",
        metadata={"source": "documentation", "section": "components"},
    ),
]


def example_1_basic_customization():
    """
    Example 1: Basic customization of DefaultRAGPipeline parameters.

    This example shows how to customize the DefaultRAGPipeline by passing
    different parameter values during initialization.
    """
    print("\n=== Example 1: Basic DefaultRAGPipeline Customization ===")

    # Get API key from environment or use your own
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        return

    # Create a pipeline with custom configuration parameters
    pipeline = DefaultRAGPipeline(
        # Vector store configuration
        persist_directory="./custom_db",
        collection_name="custom_collection",
        # Embedding model configuration
        embedding_model_name="all-MiniLM-L6-v2",
        # Chunking configuration
        chunk_size=500,  # Smaller chunks than default
        chunk_overlap=100,  # Custom overlap
        # Retrieval configuration
        retrieval_k=5,  # Retrieve more documents than default
        # LLM configuration
        api_key=api_key,
        model_name="gemini-pro",
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=200,
        top_p=0.92,
    )

    # Add documents and query
    pipeline.add_documents(SAMPLE_DOCS)

    answer = pipeline.query("What is RAGFlow?")
    print("Query: What is RAGFlow?")
    print(f"Answer: {answer}")

    # Print the configuration
    print("\nPipeline Configuration:")
    config = pipeline.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")


def example_2_component_specific_kwargs():
    """
    Example 2: Using component-specific kwargs for deeper customization.

    This example demonstrates how to use the component-specific kwargs
    feature of DefaultRAGPipeline for more granular control.
    """
    print("\n=== Example 2: Component-Specific kwargs Customization ===")

    # Get API key from environment or use your own
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        return

    # Create a pipeline with component-specific kwargs
    pipeline = DefaultRAGPipeline(
        api_key=api_key,
        persist_directory="./advanced_db",
        # Specify additional kwargs for each component type
        chunker_kwargs={"separators": ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]},
        embedder_kwargs={
            # SentenceTransformers-specific parameters
            "device": "cpu"  # Force CPU usage for embeddings
        },
        vector_store_kwargs={
            # ChromaDB-specific parameters
            "anonymized_telemetry": False
        },
        retriever_kwargs={
            # Additional retriever configuration
            "search_type": "similarity"  # Example parameter
        },
        llm_kwargs={
            # Gemini-specific parameters
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                }
            ]
        },
    )

    # Add documents and query
    pipeline.add_documents(SAMPLE_DOCS)

    answer = pipeline.query("How does RAG enhance language models?")
    print("Query: How does RAG enhance language models?")
    print(f"Answer: {answer}")


def example_3_custom_pipeline_with_individual_adapters():
    """
    Example 3: Building a custom pipeline with individual adapters.

    This example shows how to instantiate and configure individual
    adapters and then combine them into a custom RAGPipeline.
    """
    print("\n=== Example 3: Custom Pipeline with Individual Adapters ===")

    # Get API key from environment or use your own
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        return

    # 1. Create a custom chunking strategy
    chunker = RecursiveCharacterTextSplitterAdapter(
        chunk_size=250,  # Very small chunks
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
    )

    # 2. Create a custom embedding model
    embedder = SentenceTransformersAdapter(
        # Different model than the default
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 3. Create a custom vector store
    vector_store = ChromaDBAdapter(
        collection_name="custom_pipeline_example",
        persist_directory="./custom_pipeline_db",
        # Pass the embedder directly to the vector store
        embedding_function=embedder,
    )

    # 4. Create a custom retrieval strategy
    retriever = SimpleSimilarityRetriever(
        vector_store=vector_store,
        k=3,  # Retrieve only 3 documents
    )

    # 5. Create a custom LLM
    llm = GeminiAdapter(
        api_key=api_key,
        model_name="gemini-pro",
        temperature=0.1,  # Very low temperature for factual responses
        max_tokens=100,  # Short answers
        top_p=0.9,
        top_k=20,
    )

    # 6. Assemble the custom pipeline
    custom_pipeline = RAGPipeline(
        chunking_strategy=chunker,
        embedding_model=embedder,
        vector_store=vector_store,
        retrieval_strategy=retriever,
        llm=llm,
    )

    # Add documents and query
    custom_pipeline.add_documents(SAMPLE_DOCS)

    answer = custom_pipeline.query("What are vector databases used for?")
    print("Query: What are vector databases used for?")
    print(f"Answer: {answer}")

    # Demonstrate that we can also directly use the components
    print("\nDirect Component Usage Examples:")

    # Use the chunker directly
    chunks = chunker.split_text(
        "This is a test document. It contains multiple sentences. We want to see how it gets split."
    )
    print(f"  Direct chunking result: {len(chunks)} chunks")

    # Use the embedder directly
    embedding = embedder.embed_query("What is RAG?")
    print(f"  Direct embedding result: vector of size {len(embedding)}")

    # Use the LLM directly
    direct_response = llm.generate("Briefly explain what vector embeddings are.")
    print(f"  Direct LLM result: {direct_response[:50]}...")


def example_4_configuration_profiles():
    """
    Example 4: Creating and using configuration profiles.

    This example demonstrates how to create reusable configuration
    profiles for different use cases.
    """
    print("\n=== Example 4: Configuration Profiles ===")

    # Get API key from environment or use your own
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        return

    # Define configuration profiles for different use cases
    config_profiles = {
        "low_resource": {
            # Configuration optimized for low resource usage
            "persist_directory": "./low_resource_db",
            "collection_name": "low_resource",
            "embedding_model_name": "all-MiniLM-L6-v2",  # Small, fast model
            "chunk_size": 1500,  # Larger chunks, fewer embeddings
            "chunk_overlap": 100,
            "retrieval_k": 2,  # Retrieve fewer documents
            "temperature": 0.7,
            "max_tokens": 100,  # Shorter responses
            "model_name": "gemini-pro",
        },
        "high_accuracy": {
            # Configuration optimized for accuracy
            "persist_directory": "./high_accuracy_db",
            "collection_name": "high_accuracy",
            "embedding_model_name": "all-mpnet-base-v2",  # More accurate model
            "chunk_size": 500,  # Smaller chunks for more precise retrieval
            "chunk_overlap": 150,  # Higher overlap
            "retrieval_k": 8,  # Retrieve more context
            "temperature": 0.2,  # Lower temperature for more deterministic responses
            "max_tokens": 300,  # Longer responses
            "model_name": "gemini-pro",
        },
    }

    # Function to create a pipeline from a profile
    def create_pipeline_from_profile(profile_name: str) -> Optional[DefaultRAGPipeline]:
        if profile_name not in config_profiles:
            print(f"Profile '{profile_name}' not found")
            return None

        config = config_profiles[profile_name].copy()
        config["api_key"] = api_key

        return DefaultRAGPipeline(**config)

    # Create pipelines using the profiles
    low_resource_pipeline = create_pipeline_from_profile("low_resource")
    high_accuracy_pipeline = create_pipeline_from_profile("high_accuracy")

    if low_resource_pipeline and high_accuracy_pipeline:
        # Add the same documents to both pipelines
        low_resource_pipeline.add_documents(SAMPLE_DOCS)
        high_accuracy_pipeline.add_documents(SAMPLE_DOCS)

        # Compare the results
        query = "What is the relationship between RAG and vector databases?"

        low_resource_answer = low_resource_pipeline.query(query)
        high_accuracy_answer = high_accuracy_pipeline.query(query)

        print(f"Query: {query}")
        print(f"\nLow Resource Profile Answer:\n{low_resource_answer}")
        print(f"\nHigh Accuracy Profile Answer:\n{high_accuracy_answer}")

        # Print configurations for comparison
        print("\nLow Resource Configuration:")
        low_config = low_resource_pipeline.get_config()
        for key, value in low_config.items():
            print(f"  {key}: {value}")

        print("\nHigh Accuracy Configuration:")
        high_config = high_accuracy_pipeline.get_config()
        for key, value in high_config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_customization()
    example_2_component_specific_kwargs()
    example_3_custom_pipeline_with_individual_adapters()
    example_4_configuration_profiles()
