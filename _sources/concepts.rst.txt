========
Concepts
========

This page explains the key concepts in Retrieval Augmented Generation (RAG) and how RAGFlow implements them.

What is RAG?
-----------

Retrieval Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge. In a standard RAG system:

1. A user submits a query
2. The system retrieves relevant documents or chunks from a knowledge base
3. The retrieved information is combined with the query and sent to an LLM
4. The LLM generates a response based on both the query and the retrieved context

RAG helps solve several limitations of LLMs:

* **Knowledge cutoff**: LLMs are trained on data up to a certain date and don't have access to newer information
* **Hallucinations**: LLMs can generate plausible but incorrect information
* **Domain-specific knowledge**: LLMs may have limited knowledge in specialized domains
* **Source verification**: RAG allows responses to cite sources for factual claims

Core Components
-------------

A RAG system comprises several key components, each with a specific function in the pipeline:

Document Processing
~~~~~~~~~~~~~~~~~

Before information can be retrieved, it must be processed:

* **Document Loading**: Converting various file formats (PDF, DOCX, HTML, etc.) into a common text format
* **Chunking**: Splitting long documents into smaller, manageable chunks that are suitable for embedding and retrieval
* **Metadata**: Adding additional information to chunks to help with filtering and attribution

Embedding and Storage
~~~~~~~~~~~~~~~~~~~

To enable efficient retrieval, documents are converted to vector representations and stored:

* **Embedding Models**: Neural networks that convert text into numerical vectors that capture semantic meaning
* **Vector Stores**: Specialized databases that store vector embeddings and allow for similarity search
* **Indexing**: Techniques to organize vectors for efficient retrieval

Retrieval
~~~~~~~~

When a query is received, relevant information is retrieved:

* **Query Embedding**: Converting the user's query into a vector representation
* **Similarity Search**: Finding document chunks whose embeddings are most similar to the query embedding
* **Reranking**: Further refining retrieved results to improve relevance
* **Filtering**: Limiting results based on metadata or other criteria

Generation
~~~~~~~~~

Finally, the retrieved information is used to generate a response:

* **Context Integration**: Combining retrieved documents with the user's query
* **Prompt Engineering**: Formatting the context and query for optimal LLM performance
* **LLM Generation**: Using an LLM to create a coherent, accurate response based on the provided information
* **Response Processing**: Formatting the response, extracting citations, etc.

RAGFlow Architecture
------------------

RAGFlow implements these core RAG components through a modular, interface-driven architecture:

Core Interfaces
~~~~~~~~~~~~~

RAGFlow defines five key interfaces that correspond to the main components of a RAG system:

* **ChunkingStrategyInterface**: Defines how documents are split into chunks
* **EmbeddingModelInterface**: Defines how text is converted to vector embeddings
* **VectorStoreInterface**: Defines how embeddings are stored and retrieved
* **RetrievalStrategyInterface**: Defines how relevant documents are retrieved for a query
* **LLMInterface**: Defines how responses are generated based on retrieved context

Default Adapters
~~~~~~~~~~~~~~

RAGFlow provides default implementations for each interface:

* **RecursiveCharacterTextSplitterAdapter**: Splits text based on character separators
* **SentenceTransformersAdapter**: Uses the Sentence Transformers library for embeddings
* **ChromaDBAdapter**: Uses ChromaDB for vector storage
* **SimpleSimilarityRetriever**: Retrieves documents based on embedding similarity
* **GeminiAdapter**: Uses Google's Gemini model for text generation

Pipeline
~~~~~~~

These components are orchestrated by the RAGFlow pipeline:

* **RAGPipeline**: The base pipeline class that defines the workflow between components
* **DefaultRAGPipeline**: A pre-configured pipeline using the default adapters

Extension Points
~~~~~~~~~~~~~

RAGFlow is designed to be extensible:

* Custom adapters can be created by implementing any of the core interfaces
* The default pipeline can be customized through configuration
* Individual components can be used independently or combined in custom ways

Understanding these concepts will help you effectively use RAGFlow and customize it to your specific needs. For practical examples, see the :doc:`quick_start` guide and the various :doc:`tutorials/basic_rag`.
