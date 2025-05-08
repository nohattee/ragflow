"""
Configuration Examples for RAGFlow

This module provides examples of how to configure RAGFlow components and pipelines
for various use cases.
"""

import os

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
from ragflow.core.pipeline import RAGPipeline

# Import RAGFlow components and pipeline
from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

# Load environment variables from .env file (if present)
load_dotenv()


def basic_pipeline_example():
    """
    Example of setting up a basic pipeline with minimal configuration.
    """
    # Ensure API key is set in environment or provide directly
    if "GEMINI_API_KEY" not in os.environ:
        print(
            "Please set the GEMINI_API_KEY environment variable or provide it directly"
        )
        return

    # Create a basic pipeline with defaults
    pipeline = DefaultRAGPipeline()

    # Add some documents
    docs = [
        "RAGFlow is a framework for building RAG applications",
        "Retrieval Augmented Generation combines search with LLMs",
        "Vector databases are used to store embeddings for semantic search",
    ]

    pipeline.add_texts(docs)

    # Query the pipeline
    answer = pipeline.query("What is RAGFlow used for?")
    print(answer)


def custom_pipeline_example():
    """
    Example of setting up a pipeline with custom configuration.
    """
    # Ensure API key is set in environment or provide directly
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Please set the GEMINI_API_KEY environment variable or provide it directly"
        )
        return

    # Create a pipeline with custom configuration
    pipeline = DefaultRAGPipeline(
        persist_directory="./custom_db",
        api_key=api_key,
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=3,
    )

    # Add some documents
    docs = [
        "RAGFlow is a framework for building RAG applications",
        "Retrieval Augmented Generation combines search with LLMs",
        "Vector databases are used to store embeddings for semantic search",
    ]

    pipeline.add_texts(docs)

    # Query the pipeline
    answer = pipeline.query("What is RAGFlow used for?")
    print(answer)


def advanced_pipeline_example():
    """
    Example of setting up an advanced pipeline with manually configured components.
    """
    # Ensure API key is set in environment or provide directly
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Please set the GEMINI_API_KEY environment variable or provide it directly"
        )
        return

    # Configure each component manually
    chunker = RecursiveCharacterTextSplitterAdapter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
    )

    embedder = SentenceTransformersAdapter(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_store = ChromaDBAdapter(
        collection_name="advanced_example", persist_directory="./advanced_db"
    )

    retriever = SimpleSimilarityRetriever(vector_store=vector_store, k=5)

    llm = GeminiAdapter(
        api_key=api_key,
        model_name="gemini-pro",
        temperature=0.3,
        max_tokens=150,
        top_p=0.92,
        top_k=30,
    )

    # Create a pipeline with the custom components
    pipeline = RAGPipeline(
        chunking_strategy=chunker,
        embedding_model=embedder,
        vector_store=vector_store,
        retrieval_strategy=retriever,
        llm=llm,
    )

    # Add some documents
    docs = [
        "RAGFlow is a framework for building RAG applications",
        "Retrieval Augmented Generation combines search with LLMs",
        "Vector databases are used to store embeddings for semantic search",
    ]

    pipeline.add_texts(docs)

    # Query the pipeline
    answer = pipeline.query("What is RAGFlow used for?")
    print(answer)


def using_existing_database():
    """
    Example of setting up a pipeline using an existing database.
    """
    # Ensure API key is set in environment or provide directly
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Please set the GEMINI_API_KEY environment variable or provide it directly"
        )
        return

    # Use an existing ChromaDB database
    pipeline = DefaultRAGPipeline.from_existing_db(
        persist_directory="./existing_db",
        api_key=api_key,
        embedding_model_name="all-MiniLM-L6-v2",
    )

    # Query the pipeline (using the existing database)
    answer = pipeline.query("What is RAGFlow used for?")
    print(answer)


if __name__ == "__main__":
    print("Basic pipeline example:")
    basic_pipeline_example()

    print("\nCustom pipeline example:")
    custom_pipeline_example()

    print("\nAdvanced pipeline example:")
    advanced_pipeline_example()

    # Uncomment if you have an existing database
    # print("\nUsing existing database example:")
    # using_existing_database()
