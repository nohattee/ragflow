"""
Core interfaces for RAGFlow.

This module defines the fundamental interfaces that form the building blocks
of the RAG (Retrieval Augmented Generation) pipeline. Each interface represents
a component with a specific role in the pipeline:

- ChunkingStrategyInterface: Splitting documents into smaller chunks
- EmbeddingModelInterface: Converting text into vector embeddings
- VectorStoreInterface: Storing and retrieving vector embeddings
- RetrievalStrategyInterface: Retrieving relevant documents based on a query
- LLMInterface: Generating text based on context and prompts

These interfaces promote modularity, testability, and extensibility by
defining standard contracts that concrete implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Document:
    """
    A document with text content and optional metadata.

    This class represents the core unit of text data in RAGFlow, containing
    both the actual content (text) and associated metadata. The metadata can
    hold information such as source, author, creation date, or any other
    relevant attributes.

    Documents are used throughout the RAG pipeline and are especially important
    during retrieval, where metadata can help filter results or provide
    attribution information.

    Attributes:
        page_content (str): The text content of the document.
        metadata (Dict[str, Any]): Optional metadata associated with the document.
            Common keys include 'source', 'author', 'created_at', etc.

    Examples:
    --------
        Creating a simple document:

        .. code-block:: python

            doc = Document(page_content="This is a sample document.")

        Creating a document with metadata:

        .. code-block:: python

            doc = Document(
                page_content="This document contains important information.",
                metadata={
                    "source": "research_paper.pdf",
                    "author": "Smith, J.",
                    "year": 2024,
                    "page": 42,
                },
            )
    """

    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Document.

        Args:
            page_content: The text content of the document. This is the actual
                textual information that will be processed and retrieved.
            metadata: Optional metadata associated with the document. This can
                include information like source, author, creation date, page number,
                or any other relevant attributes. Defaults to an empty dictionary.
        """
        self.page_content = page_content
        self.metadata = metadata or {}


class ChunkingStrategyInterface(ABC):
    """
    Interface for splitting documents into smaller chunks.

    A chunking strategy is responsible for dividing documents into smaller,
    more manageable pieces that can be processed independently. This is
    particularly important for:
    1. Large documents that might exceed context limits of LLMs
    2. Improving retrieval granularity by breaking content into focused segments
    3. Optimizing embedding and storage efficiency

    Different chunking strategies can be implemented based on the specific needs:
    - Splitting by character/token count
    - Splitting by semantic units (paragraphs, sections)
    - Splitting with overlaps to maintain context across chunks
    - Using language-aware chunking for better semantics

    Implementations of this interface should ensure that:
    - The original document metadata is preserved or appropriately modified in each chunk
    - The text is split in a way that maintains semantic coherence when possible
    - The chunking parameters (size, overlap) can be configured

    Examples:
    --------
        Using a chunking strategy:

        .. code-block:: python

            chunker = RecursiveCharacterTextSplitterAdapter(
                chunk_size=1000, chunk_overlap=200
            )
            documents = [Document(page_content="A very long document...")]
            chunks = chunker.split_documents(documents)
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        This method takes a list of Document objects, splits their content
        into smaller chunks, and returns a new list of Document objects
        representing those chunks. The metadata from the original documents
        should be preserved or appropriately modified in each chunk.

        Args:
            documents: List of Document objects to split. Each document's
                page_content will be split into chunks.

        Returns:
            List of Document objects representing the chunks. This will typically
            contain more documents than the input list, as each input document
            may produce multiple chunks.

        Examples:
        --------
            .. code-block:: python

                chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=1000)
                long_docs = [
                    Document(
                        page_content="Long document 1...",
                        metadata={"source": "doc1.txt"},
                    ),
                    Document(
                        page_content="Long document 2...",
                        metadata={"source": "doc2.txt"},
                    ),
                ]
                chunks = chunker.split_documents(long_docs)
                # chunks will contain multiple Document objects, each with a portion of the
                # original text and the preserved or modified metadata
        """
        pass

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split a text string into smaller chunks.

        This method takes a single text string and splits it into a list of
        smaller text chunks according to the strategy's algorithm. This is
        a lower-level method that doesn't deal with Document objects and
        their metadata.

        Args:
            text: A text string to split into chunks.

        Returns:
            List of text chunks. The original text is divided into multiple
            smaller strings according to the chunking strategy.

        Examples:
        --------
            .. code-block:: python

                chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=1000)
                long_text = "This is a very long document that needs to be split..."
                chunks = chunker.split_text(long_text)
                # chunks will be a list of strings, each up to 1000 characters
        """
        pass


class EmbeddingModelInterface(ABC):
    """
    Interface for embedding models that convert text into vector representations.

    Embedding models transform text into numerical vector representations that
    capture semantic meaning, enabling similarity searches and other vector
    operations. These embeddings are at the core of RAG systems, as they allow
    for efficient retrieval of relevant content based on semantic similarity.

    Different embedding models have different characteristics:
    - Varying vector dimensions (e.g., 384, 768, 1536)
    - Different training data and optimization targets
    - Trade-offs between speed, accuracy, and resource requirements
    - Support for different languages and domains

    Implementations of this interface should focus on:
    - Providing a consistent way to generate embeddings across different models
    - Handling batching for efficiency when embedding multiple documents
    - Proper normalization and preprocessing of input text
    - Appropriate error handling for model-specific issues

    Examples:
    --------
        Using an embedding model:

        .. code-block:: python

            embedder = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")

            # Embed a single query
            query_embedding = embedder.embed_query("What is RAGFlow?")

            # Embed multiple documents
            doc_embeddings = embedder.embed_documents(
                [
                    "RAGFlow is a framework for building RAG applications.",
                    "It uses various components like embedding models and vector stores.",
                ]
            )
    """

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a query string.

        This method converts a single query string into a vector embedding
        that represents its semantic meaning. The embedding can then be used
        for similarity search against document embeddings.

        Args:
            query: The query text to embed. This can be a question, phrase,
                or any text that needs to be converted to a vector representation.

        Returns:
            List[float]: The vector embedding of the query.

        Examples:
        --------
            .. code-block:: python

                embedder = SentenceTransformersAdapter()
                query_embedding = embedder.embed_query("How does RAG work?")
                # query_embedding will be a list of floats (e.g., [0.1, 0.2, ...])
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        This method converts multiple document strings into vector embeddings
        that represent their semantic meanings. These embeddings can be stored
        in a vector database for later retrieval.

        Implementations should handle batching efficiently to optimize performance
        when embedding large numbers of documents.

        Args:
            documents: List of document texts to embed. Each string in this list
                will be converted to a separate embedding vector.

        Returns:
            List[List[float]]: A list of embeddings, where each inner list is a
            vector embedding for a document. The order of embeddings corresponds
            to the order of the input documents.

        Examples:
        --------
            .. code-block:: python

                embedder = SentenceTransformersAdapter()
                texts_to_embed = [
                    "RAGFlow provides flexible interfaces.",
                    "Embeddings are crucial for semantic search.",
                ]
                document_embeddings = embedder.embed_documents(texts_to_embed)
                # document_embeddings will be a list of lists of floats
        """
        pass


class VectorStoreInterface(ABC):
    """
    Interface for vector stores that store and retrieve embeddings.

    Vector stores are specialized databases that efficiently store and
    search vector embeddings, enabling similarity search operations. They
    are a critical component in RAG systems for retrieving documents that
    are semantically similar to a query.

    Key capabilities of vector stores include:
    - Efficient storage of high-dimensional vector embeddings
    - Fast approximate nearest neighbor (ANN) search algorithms
    - Ability to store metadata alongside vectors for filtering
    - Support for various distance metrics (cosine, Euclidean, dot product)
    - Persistence and scaling of vector collections

    Common vector store implementations include:
    - ChromaDB, Pinecone, Weaviate, Qdrant, Milvus, Faiss
    - PostgreSQL with pgvector extension
    - Redis with RediSearch/RedisVL
    - In-memory implementations for testing

    Implementations of this interface should ensure:
    - Consistent handling of documents and their embeddings
    - Proper error handling for database operations
    - Efficient vector search functionality
    - Appropriate metadata storage and filtering
    - Efficiently update or delete existing embeddings

    Examples:
    --------
        Using a vector store:

        .. code-block:: python

            embedder = SentenceTransformersAdapter()
            vector_store = ChromaDBAdapter(embedding_function=embedder)

            # Add documents
            vector_store.add_texts(
                ["Document 1 about apples", "Document 2 about oranges"],
                metadata=[{"source": "doc1"}, {"source": "doc2"}],
            )

            # Search for similar documents
            results = vector_store.similarity_search("Tell me about fruits", k=1)
            print(results)
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        This method stores document content and metadata in the vector store.
        If embeddings are provided, they are used directly; otherwise, the
        vector store will generate embeddings using its embedding function.

        Args:
            documents: List of document dictionaries to store. Each dictionary
                should contain at least a 'page_content' key with the document text
                and a 'metadata' key with a dictionary of associated metadata.
            embeddings: Optional pre-computed embeddings for the documents. If provided,
                these should be a list of embedding vectors (list of floats) with the
                same length as the documents list. If not provided, the vector store
                will use its embedding function to generate embeddings.

        Returns:
            None

        Examples:
        --------
            .. code-block:: python

                embedder = SentenceTransformersAdapter()
                vector_store = ChromaDBAdapter(embedding_function=embedder)
                docs_to_add = [
                    {
                        "page_content": "RAGFlow is a framework for RAG applications.",
                        "metadata": {"source": "readme.md", "section": "intro"},
                    },
                    {
                        "page_content": "Vector stores enable efficient similarity search.",
                        "metadata": {"source": "guide.md", "section": "concepts"},
                    },
                ]

                # Add documents without providing embeddings (will be generated)
                vector_store.add_documents(docs_to_add)

                # Or with pre-computed embeddings
                doc_texts = [doc["page_content"] for doc in docs_to_add]
                embeddings = embedder.embed_documents(doc_texts)
                vector_store.add_documents(docs_to_add, embeddings=embeddings)
        """
        pass

    @abstractmethod
    def add_texts(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add text strings with optional metadata to the vector store.

        This is a convenience method that takes raw text strings and optional
        metadata, converts them to the document format, and stores them in
        the vector store.

        Args:
            texts: List of text strings to add. Each string will be stored
                as the content of a document.
            metadata: Optional list of metadata dictionaries, one for each text.
                If provided, this list should have the same length as the texts list.
                If not provided, empty metadata will be used for all texts.

        Returns:
            None

        Examples:
        --------
            .. code-block:: python

                embedder = SentenceTransformersAdapter()
                vector_store = ChromaDBAdapter(embedding_function=embedder)
                texts = ["First document text", "Second document text"]
                metadata = [{"source": "doc1"}, {"source": "doc2"}]
                vector_store.add_texts(texts, metadata=metadata)
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents similar to the query.

        This method converts the query string to an embedding and then
        finds the most similar documents in the vector store based on
        a distance metric (typically cosine similarity).

        Args:
            query: The query string to find similar documents for. This
                will be converted to an embedding using the vector store's
                embedding function.
            k: Number of similar documents to return. Defaults to 5.

        Returns:
            A list of Document objects, ranked by similarity.

        Examples:
        --------
            .. code-block:: python

                embedder = SentenceTransformersAdapter()
                vector_store = ChromaDBAdapter(embedding_function=embedder)
                query_embedding = embedder.embed_query("Search query")
                # Assume documents have been added
                results = vector_store.similarity_search_by_vector(query_embedding, k=2)
                for doc in results:
                    print(doc.page_content)
        """
        pass

    @abstractmethod
    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Document]:
        """
        Retrieve documents similar to the provided embedding vector.

        This method allows searching with a pre-computed embedding vector
        instead of a text query. This is useful when you've already generated
        an embedding using a specific model or when you want to ensure exact
        control over the embedding used for search.

        Args:
            embedding: The query vector embedding (list of floats).
            k: The number of similar documents to retrieve. Defaults to 4.

        Returns:
            A list of Document objects, ranked by similarity.

        Examples:
        --------
            .. code-block:: python

                embedder = SentenceTransformersAdapter()
                vector_store = ChromaDBAdapter(embedding_function=embedder)
                query_embedding = embedder.embed_query("Search query")
                # Assume documents have been added
                results = vector_store.similarity_search_by_vector(query_embedding, k=2)
                for doc in results:
                    print(doc.page_content)
        """
        pass


class RetrievalStrategyInterface(ABC):
    """
    Interface for retrieving relevant documents based on a query.

    A retrieval strategy is responsible for selecting the most relevant documents
    from a corpus based on a user query. This is a critical component in RAG systems
    as it determines what context information will be provided to the LLM.

    Different retrieval strategies can be implemented based on specific needs:
    - Simple similarity-based retrieval using vector embeddings
    - Hybrid retrieval combining vector similarity with keyword search
    - Re-ranking strategies that apply additional filtering or scoring
    - Multi-step retrieval that iteratively refines results

    Implementations of this interface should focus on:
    - Integrating with a vector store or other search mechanism
    - Applying appropriate filtering or re-ranking logic if needed
    - Handling different query types and complexities
    - Efficiently querying the underlying vector store or search index

    Examples:
    --------
        Using a retrieval strategy:

        .. code-block:: python

            embedder = SentenceTransformersAdapter()
            vector_store = ChromaDBAdapter(embedding_function=embedder)
            # Add documents to vector_store first...
            retriever = SimpleSimilarityRetriever(vector_store=vector_store, k=5)
            relevant_docs = retriever.get_relevant_documents(
                "What are the key features of RAGFlow?"
            )
    """

    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents that are relevant to the given query.

        This method searches through the available documents and returns
        those that are most relevant to answering the provided query.
        The exact matching algorithm depends on the implementation.

        Args:
            query: The search query to find relevant documents for.
                This is typically a question or information need expressed
                as a natural language string.

        Returns:
            A list of Document objects considered relevant to the query.

        Examples:
        --------
            .. code-block:: python

                retriever = SimpleSimilarityRetriever(vector_store=my_vector_store, k=3)
                documents = retriever.get_relevant_documents(
                    "How to build a RAG pipeline?"
                )
                # documents will be a list of up to 3 Document objects
        """
        pass

    @abstractmethod
    def get_relevant_documents_with_scores(
        self, query: str
    ) -> List[tuple[Document, float]]:
        """
        Retrieve relevant documents along with their relevance scores.

        Similar to get_relevant_documents(), but also returns a relevance
        score for each document. This allows consumers to make decisions
        based on the confidence of the retrieval system.

        Args:
            query: The search query to find relevant documents for.
                This is typically a question or information need expressed
                as a natural language string.

        Returns:
            A list of tuples, each containing a Document object and its
            corresponding relevance score (float) representing its relevance.

        Examples:
        --------
            .. code-block:: python

                retriever = SimpleSimilarityRetriever(vector_store=my_vector_store)
                results = retriever.get_relevant_documents_with_scores("Query text")
                for doc, score in results:
                    print(f"Score: {score}, Content: {doc.page_content}")
        """
        pass


class LLMInterface(ABC):
    """
    Interface for language models that generate text based on prompts and context.

    A language model is responsible for the "generation" part of Retrieval Augmented
    Generation. It takes a user query along with relevant context documents and
    generates a coherent, informative response.

    This interface abstracts away the details of different LLM providers and models,
    allowing RAGFlow to work with various backends:
    - OpenAI models (GPT-3.5, GPT-4, etc.)
    - Google models (Gemini, PaLM, etc.)
    - Anthropic models (Claude)
    - Open-source models (Llama, Mistral, etc.)
    - Local models running on-premises

    Implementations of this interface should focus on:
    - Managing communication with the LLM API or local model
    - Crafting appropriate prompts that incorporate the retrieved context
    - Handling API-specific parameters like temperature, max tokens, etc.
    - Proper error handling for API quotas, rate limits, etc.
    - Handling of rate limits, API errors, and retries if applicable

    Examples:
    --------
        Using an LLM interface:

        .. code-block:: python

            llm = GeminiAdapter(api_key="YOUR_API_KEY")
            answer = llm.generate("What is the capital of France?")
            print(answer)  # Expected: Paris

        Generating with context:

        .. code-block:: python

            llm = GeminiAdapter(api_key="your-api-key")
            vector_store = ChromaDBAdapter(...)
            retriever = SimpleSimilarityRetriever(vector_store=vector_store)

            # Get relevant documents for a query
            query = "What is the impact of chunking on RAG performance?"
            relevant_docs = retriever.get_relevant_documents(query)

            # Generate answer using context
            answer = llm.generate_with_context(query, relevant_docs)
            print(answer)
    """

    @abstractmethod
    def generate(
        self, prompt: str, context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate text based on a prompt.

        This basic generation method takes a prompt and optional context
        dictionary and returns generated text from the language model.

        Args:
            prompt: The prompt string to generate text from. This can be a
                question, instruction, or any text that the LLM should
                continue or respond to.
            context: Optional additional context or parameters to pass to the
                language model. This can include configuration options or
                additional information that influences generation.

        Returns:
            A string containing the generated text.

        Examples:
        --------
            .. code-block:: python

                llm = GeminiAdapter(api_key="your-api-key", temperature=0.7)

                # Simple prompt without context
                response = llm.generate("Write a haiku about programming.")

                # With additional context/parameters
                response = llm.generate(
                    prompt="What is machine learning?",
                    context={"max_tokens": 100, "style": "concise"},
                )
        """
        pass

    @abstractmethod
    def generate_with_context(self, query: str, context: List[Document]) -> str:
        """
        Generate a response to a query using retrieved context documents.

        This is the core RAG method that combines the user query with
        relevant retrieved documents to generate an informed response.
        The implementation should properly format the prompt to include
        both the query and the context in a way that the LLM can use.

        Args:
            query: The user's query or question.
            context: A list of Document objects providing relevant context for the query.
                These are typically retrieved from a vector store.

        Returns:
            A string containing the generated answer.

        Examples:
        --------
            .. code-block:: python

                llm = GeminiAdapter(api_key="YOUR_API_KEY")
                retrieved_documents = [
                    Document(page_content="The sky is blue during the day."),
                    Document(page_content="The sun is a star."),
                ]
                answer = llm.generate_with_context(
                    "Why is the sky blue?", context=retrieved_documents
                )
                # Answer will be based on the provided documents and the LLM's knowledge
        """
        pass
