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
        Creating a simple document:
        ```python
        doc = Document(page_content="This is a sample document.")
        ```

        Creating a document with metadata:
        ```python
        doc = Document(
            page_content="This document contains important information.",
            metadata={
                "source": "research_paper.pdf",
                "author": "Smith, J.",
                "year": 2024,
                "page": 42,
            },
        )
        ```
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
        Using a chunking strategy:
        ```python
        chunker = RecursiveCharacterTextSplitterAdapter(
            chunk_size=1000, chunk_overlap=200
        )
        documents = [Document(page_content="A very long document...")]
        chunks = chunker.split_documents(documents)
        ```
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
            ```python
            chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=1000)
            long_docs = [
                Document(
                    page_content="Long document 1...", metadata={"source": "doc1.txt"}
                ),
                Document(
                    page_content="Long document 2...", metadata={"source": "doc2.txt"}
                ),
            ]
            chunks = chunker.split_documents(long_docs)
            # chunks will contain multiple Document objects, each with a portion of the
            # original text and the preserved or modified metadata
            ```
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
            ```python
            chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=1000)
            long_text = "This is a very long document that needs to be split..."
            chunks = chunker.split_text(long_text)
            # chunks will be a list of strings, each up to 1000 characters
            ```
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
        Using an embedding model:
        ```python
        embedder = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")

        # Embed a single query
        query_embedding = embedder.embed_query("What is RAGFlow?")

        # Embed multiple documents
        doc_embeddings = embedder.embed_documents(
            [
                "RAGFlow is a framework for building RAG applications.",
                "It provides interfaces for embedding, storage, and retrieval.",
            ]
        )
        ```
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
            A list of floats representing the embedding vector. The dimension
            of this vector depends on the specific embedding model implementation.

        Examples:
            ```python
            embedder = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")
            query = "How does RAG improve search quality?"
            query_embedding = embedder.embed_query(query)

            # The embedding is a vector of floating point numbers
            print(f"Embedding dimension: {len(query_embedding)}")
            print(f"First few values: {query_embedding[:5]}")
            ```
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
            List of embedding vectors, one for each input document. Each vector
            is a list of floats with the same dimension as those produced by
            embed_query().

        Examples:
            ```python
            embedder = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")

            documents = [
                "RAGFlow provides a simple interface for RAG applications.",
                "Vector databases store embeddings for efficient retrieval.",
                "Language models generate text based on retrieved context.",
            ]

            embeddings = embedder.embed_documents(documents)

            print(f"Generated {len(embeddings)} embeddings")
            print(f"Each embedding has {len(embeddings[0])} dimensions")
            ```
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

    Examples:
        Using a vector store:
        ```python
        # Create an embedding model first
        embedder = SentenceTransformersAdapter()

        # Create and use a vector store
        vector_store = ChromaDBAdapter(
            collection_name="my_documents",
            persist_directory="./data/chroma_db",
            embedding_function=embedder,
        )

        # Add documents
        vector_store.add_texts(
            texts=["Document 1 content", "Document 2 content"],
            metadata=[{"source": "file1.txt"}, {"source": "file2.txt"}],
        )

        # Search
        results = vector_store.similarity_search("search query", k=2)
        ```
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
            ```python
            # Create a vector store
            vector_store = ChromaDBAdapter(
                collection_name="my_documents", embedding_function=embedder
            )

            # Prepare documents
            documents = [
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
            vector_store.add_documents(documents)

            # Or with pre-computed embeddings
            doc_texts = [doc["page_content"] for doc in documents]
            embeddings = embedder.embed_documents(doc_texts)
            vector_store.add_documents(documents, embeddings=embeddings)
            ```
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
            ```python
            vector_store = ChromaDBAdapter(
                collection_name="my_documents", embedding_function=embedder
            )

            # Add texts with metadata
            vector_store.add_texts(
                texts=[
                    "RAGFlow provides a simple interface for RAG applications.",
                    "Chunking strategies split documents into manageable pieces.",
                ],
                metadata=[
                    {"source": "readme.md", "section": "overview"},
                    {"source": "concepts.md", "section": "chunking"},
                ],
            )

            # Add texts without metadata
            vector_store.add_texts(
                ["More document content here", "And another example"]
            )
            ```
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
            List of documents most similar to the query, ordered by similarity
            (most similar first). Each document is a dictionary containing at
            least 'page_content' (the document text) and 'metadata' (a dictionary
            of associated metadata).

        Examples:
            ```python
            vector_store = ChromaDBAdapter(
                collection_name="my_documents", embedding_function=embedder
            )

            # After adding documents to the vector store
            results = vector_store.similarity_search(
                query="How does RAG improve search quality?",
                k=3,  # Return the top 3 most similar documents
            )

            # Process the results
            for doc in results:
                print("Content:", doc["page_content"])
                print("Source:", doc["metadata"].get("source", "Unknown"))
                print("---")
            ```
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
            embedding: The embedding vector to search with. This should be a list
                of floating-point numbers with the same dimension as the document
                embeddings stored in the vector store.
            k: Number of similar documents to return. Defaults to 4.

        Returns:
            List of Document objects most similar to the provided embedding,
            ordered by similarity (most similar first).

        Examples:
            ```python
            # Create embedding model and vector store
            embedder = SentenceTransformersAdapter()
            vector_store = ChromaDBAdapter(
                collection_name="my_documents", embedding_function=embedder
            )

            # First, create a query embedding
            query = "What is the role of embeddings in RAG?"
            query_embedding = embedder.embed_query(query)

            # Then search with that embedding
            results = vector_store.similarity_search_by_vector(
                embedding=query_embedding, k=5
            )

            # Process the results
            for doc in results:
                print(doc.page_content)
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print("---")
            ```
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
    - Selecting high-quality, relevant documents for the query
    - Providing appropriate relevance scoring when possible
    - Configurable parameters to control retrieval behavior (e.g., number of results)

    Examples:
        Using a retrieval strategy:
        ```python
        # Create a vector store first
        vector_store = ChromaDBAdapter(...)

        # Create and use a retrieval strategy
        retriever = SimpleSimilarityRetriever(vector_store=vector_store, k=5)
        relevant_docs = retriever.get_relevant_documents("What is RAG?")
        ```
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
            A list of Document objects that are most relevant to the query.
            The documents are typically ordered by relevance (most relevant first).

        Examples:
            ```python
            retriever = SimpleSimilarityRetriever(vector_store=vector_store)
            query = "What are the benefits of RAG systems?"
            relevant_docs = retriever.get_relevant_documents(query)

            # Access the content of the first relevant document
            if relevant_docs:
                print(relevant_docs[0].page_content)
                print(relevant_docs[0].metadata)  # Source information, etc.
            ```
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
            corresponding relevance score (typically a float between 0 and 1,
            where higher values indicate greater relevance).

        Examples:
            ```python
            retriever = SimpleSimilarityRetriever(vector_store=vector_store)
            query = "How does chunking affect RAG performance?"
            results = retriever.get_relevant_documents_with_scores(query)

            # Process results with score filtering
            for doc, score in results:
                if score > 0.8:  # Only use high-confidence matches
                    print(f"Relevance: {score:.2f}")
                    print(doc.page_content)
            ```
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

    Examples:
        Using an LLM adapter:
        ```python
        llm = GeminiAdapter(api_key="your-api-key")
        response = llm.generate("What is the capital of France?")
        print(response)  # "The capital of France is Paris."
        ```
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
            A string containing the generated text response from the LLM.

        Examples:
            ```python
            llm = GeminiAdapter(api_key="your-api-key", temperature=0.7)

            # Simple prompt without context
            response = llm.generate("Write a haiku about programming.")

            # With additional context/parameters
            response = llm.generate(
                prompt="What is machine learning?",
                context={"max_tokens": 100, "style": "concise"},
            )
            ```
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
            query: The user query or question to answer.
            context: A list of Document objects providing relevant context
                for answering the query. These documents typically come
                from a retrieval system.

        Returns:
            A string containing the generated response that answers the
            query based on the provided context.

        Examples:
            ```python
            # Set up components
            llm = GeminiAdapter(api_key="your-api-key")
            vector_store = ChromaDBAdapter(...)
            retriever = SimpleSimilarityRetriever(vector_store=vector_store)

            # Get relevant documents for a query
            query = "What is the impact of chunking on RAG performance?"
            relevant_docs = retriever.get_relevant_documents(query)

            # Generate answer using context
            answer = llm.generate_with_context(query, relevant_docs)
            print(answer)
            ```
        """
        pass
