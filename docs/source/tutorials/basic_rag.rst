=================
Basic RAG Tutorial
=================

This tutorial will guide you through building a simple Retrieval Augmented Generation (RAG) application using RAGFlow. By the end, you'll understand how to set up a RAG pipeline, add documents, and query the system for answers.

What is RAG?
-----------

Retrieval Augmented Generation (RAG) is an approach that combines information retrieval with text generation to create more accurate, factual, and contextually relevant responses. The basic RAG process works as follows:

1. **Document Ingestion**: Process and store documents in a vector database
2. **Query Processing**: Convert a user query into a vector representation
3. **Retrieval**: Find the most relevant documents that match the query
4. **Generation**: Send the query and retrieved documents to an LLM to generate a response

RAG has several advantages over using an LLM alone:

- **Up-to-date information**: RAG can incorporate recent documents that weren't in the LLM's training data
- **Factual grounding**: Responses are based on specific documents, reducing hallucinations
- **Source attribution**: The system can cite the sources used to generate responses
- **Domain specialization**: RAG can be used to create domain-specific assistants without fine-tuning an LLM

Setting Up Your Environment
--------------------------

First, let's set up our environment with the necessary dependencies:

.. code-block:: bash

    pip install ragflow

    # If you're using Gemini API (default in this tutorial)
    export GEMINI_API_KEY=your-api-key-here

    # Or for environment variable persistence
    echo "GEMINI_API_KEY=your-api-key-here" > .env

Creating a Simple RAG Pipeline
-----------------------------

RAGFlow makes it easy to get started with just a few lines of code:

.. code-block:: python

    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

    # Create a pipeline using the default components
    pipeline = DefaultRAGPipeline()

    # You can also specify parameters to customize the components
    # pipeline = DefaultRAGPipeline(
    #     chunk_size=500,         # Document chunk size
    #     chunk_overlap=50,       # Overlap between chunks
    #     temperature=0.3,        # LLM temperature
    #     retrieval_k=4           # Number of documents to retrieve
    # )

The DefaultRAGPipeline uses these components with sensible defaults:

- **ChromaDB**: Vector database for storing and retrieving documents
- **SentenceTransformers**: Embedding model for converting text to vector representations
- **RecursiveCharacterTextSplitter**: Chunking strategy for breaking documents into manageable pieces
- **SimpleSimilarityRetriever**: Retrieval strategy based on vector similarity
- **GeminiAdapter**: LLM interface for generating responses (requires API key)

Adding Documents
---------------

Next, let's add some documents to our RAG pipeline:

.. code-block:: python

    # Add simple text strings
    pipeline.add_texts([
        "RAGFlow is a Python library for building RAG applications. "
        "It simplifies the process of creating robust retrieval systems.",

        "Chunking is the process of breaking documents into smaller pieces "
        "that can be processed independently. This improves retrieval "
        "granularity and context relevance.",

        "Vector embedding models convert text into numerical vectors that "
        "capture semantic meaning, enabling similarity search operations."
    ])

    # Add texts with metadata
    pipeline.add_texts(
        texts=[
            "Retrieval strategies determine how documents are fetched based on a query.",
            "LLMs generate coherent text based on prompts and context."
        ],
        metadata=[
            {"source": "concepts.md", "section": "retrieval"},
            {"source": "concepts.md", "section": "generation"}
        ]
    )

For real-world applications, you'll likely want to load documents from files. RAGFlow provides utilities for this:

.. code-block:: python

    from ragflow.utils.document_loaders import load_text_files

    # Load all .txt files from a directory
    documents = load_text_files("./data", recursive=True)
    pipeline.add_documents(documents)

Querying the Pipeline
-------------------

Now that we have documents in our pipeline, we can ask questions:

.. code-block:: python

    # Ask a simple question
    question = "What is chunking and why is it important?"
    answer = pipeline.query(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

For transparency, you can also retrieve the source documents used to generate the answer:

.. code-block:: python

    from ragflow.utils.helpers import format_sources

    # Get answer with sources
    result = pipeline.query_with_sources(
        "How do vector embeddings enable document retrieval?"
    )

    print(f"Answer: {result['answer']}")
    print("\nSources:")
    print(format_sources(result, include_content=True))

Complete Example
--------------

Let's put everything together in a complete example:

.. code-block:: python

    """
    Basic RAG application using RAGFlow
    """
    import os
    from dotenv import load_dotenv

    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline
    from ragflow.utils.helpers import format_sources

    # Load environment variables from .env file if present
    load_dotenv()

    # Create a pipeline
    pipeline = DefaultRAGPipeline(
        # API key can be provided directly or set as an environment variable
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5,
        chunk_size=1000,
        chunk_overlap=100,
        retrieval_k=3
    )

    # Add documents
    pipeline.add_texts([
        "RAGFlow is a Python library designed to streamline the development of "
        "Retrieval Augmented Generation (RAG) applications. It provides a high-level, "
        "flexible, and extensible framework.",

        "Retrieval Augmented Generation (RAG) is a technique that combines retrieval "
        "of relevant documents with text generation by large language models. This "
        "helps ground LLM outputs in factual information.",

        "RAGFlow offers sensible defaults and pre-configured pipelines to get developers "
        "started quickly, while also allowing for customization and extension as needs evolve.",

        "The chunking process in RAG systems divides documents into smaller, manageable pieces. "
        "This is important because: 1) It allows for more granular retrieval, 2) It helps match "
        "specific parts of documents to queries, and 3) It enables working within LLM context limits."
    ])

    # Simple query
    question = "What is RAGFlow and how does it relate to RAG?"
    answer = pipeline.query(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Query with sources
    print("\n--- With Sources ---")
    result = pipeline.query_with_sources("Why is chunking important in RAG systems?")

    print(f"Answer: {result['answer']}")
    print("\nSources:")
    print(format_sources(result, include_content=True))

Understanding What's Happening
-----------------------------

Let's explore each step of the process:

1. **Document Processing**:
   - When you call `pipeline.add_texts()`, the documents are split into chunks
   - Each chunk is converted into a vector embedding
   - The embeddings and document content are stored in ChromaDB

2. **Query Processing**:
   - When you call `pipeline.query()`, your question is converted to an embedding
   - The retrieval strategy finds documents with embeddings similar to your query
   - The relevant documents and your question are sent to the LLM
   - The LLM generates an answer based on the question and retrieved context

Best Practices
-------------

To build effective RAG applications with RAGFlow, consider these best practices:

**Document Processing**:
- Choose an appropriate chunk size for your content (typically 500-1500 characters)
- Include enough chunk overlap to preserve context across boundaries
- Store meaningful metadata with your documents for attribution

**Retrieval Quality**:
- Experiment with different embedding models for your specific domain
- Adjust the number of retrieved documents (`retrieval_k`) based on your needs
- Consider using hybrid retrieval for complex queries

**Response Generation**:
- Tune the LLM temperature based on the task (lower for factual queries, higher for creative responses)
- Provide clear, specific questions to get better answers
- Consider post-processing responses to ensure consistent formatting

Next Steps
---------

Now that you've built a basic RAG application, you can:

- Explore customizing individual components (see :doc:`../user_guide/customization`)
- Learn how to handle different document types (see :doc:`document_loading`)
- Develop a custom adapter for a specific use case (see :doc:`../advanced/custom_adapters`)
- Optimize RAG performance (see :doc:`../advanced/performance_tuning`)

For more detailed information on RAGFlow's components, check the :doc:`../api/core` documentation.
