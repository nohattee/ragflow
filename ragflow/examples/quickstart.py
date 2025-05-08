"""
RAGFlow Quickstart Example

This example demonstrates how to get started with RAGFlow in just a few lines of code,
showcasing the library's focus on developer experience and ease of use.
"""

import os

from dotenv import load_dotenv
from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline
from ragflow.utils.helpers import format_sources

# Load environment variables from .env file if present
load_dotenv()


def quickstart():
    """Simple quickstart example for RAGFlow."""
    # Step 1: Create a default RAG pipeline
    # This uses ChromaDB, SentenceTransformers, and Gemini with sensible defaults
    pipeline = DefaultRAGPipeline(
        # API key can be provided directly or set as GEMINI_API_KEY environment variable
        api_key=os.getenv("GEMINI_API_KEY"),
        # Customize any parameters as needed
        temperature=0.5,
    )

    # Step 2: Add some documents (simple string examples)
    pipeline.add_texts(
        [
            "RAGFlow is a Python library designed to streamline the development of "
            "Retrieval Augmented Generation (RAG) applications. It provides a high-level, "
            "flexible, and extensible framework.",
            "Retrieval Augmented Generation (RAG) is a technique that combines retrieval "
            "of relevant documents with text generation by large language models. This "
            "helps ground LLM outputs in factual information.",
            "RAGFlow offers sensible defaults and pre-configured pipelines to get developers "
            "started quickly, while also allowing for customization and extension as needs evolve.",
        ]
    )

    # Step 3: Query the pipeline
    question = "What is RAGFlow and how does it relate to RAG?"
    answer = pipeline.query(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Step 4: Advanced querying with sources
    print("\n--- With Sources ---")
    result = pipeline.query_with_sources("What are the main benefits of using RAGFlow?")

    print(f"Answer: {result['answer']}")
    print("\nSources:")
    print(format_sources(result, include_content=True))

    # Step 5: Using helper functions to load documents from files
    # Uncomment this section if you have text files to load
    """
    try:
        documents = load_text_files("./sample_data", recursive=True)
        pipeline.add_documents(documents)
        print(f"\nAdded {len(documents)} documents from files")
    except Exception as e:
        print(f"Error loading documents: {e}")
    """

    print("\nQuickstart complete! 🚀")


if __name__ == "__main__":
    quickstart()
