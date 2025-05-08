"""
Core pipeline implementation for RAGFlow.

This module provides the base RAGPipeline class that orchestrates the
Retrieval Augmented Generation (RAG) workflow by composing components
that implement the core interfaces.

The RAGPipeline coordinates document loading, chunking, embedding, storage,
retrieval, and generation to enable powerful question-answering capabilities
over custom document collections.
"""

from typing import Any, Dict, List, Optional

from .interfaces import (
    ChunkingStrategyInterface,
    Document,
    EmbeddingModelInterface,
    LLMInterface,
    RetrievalStrategyInterface,
    VectorStoreInterface,
)


class RAGPipeline:
    """
    Base RAG pipeline that orchestrates the full RAG workflow.

    The RAGPipeline coordinates all components in the RAG process:
    1. Splitting documents into chunks using a chunking strategy
    2. Embedding those chunks using an embedding model
    3. Storing embeddings in a vector store
    4. Retrieving relevant context based on queries
    5. Generating answers using an LLM based on retrieved context

    This design allows each component to be easily replaced, enabling
    customization and extension of the RAG pipeline.

    Examples:
        Creating a custom RAG pipeline:
        ```python
        from ragflow.adapters.chunking_strategies import (
            RecursiveCharacterTextSplitterAdapter,
        )
        from ragflow.adapters.embedding_models import SentenceTransformersAdapter
        from ragflow.adapters.vector_stores import ChromaDBAdapter
        from ragflow.adapters.retrieval_strategies import SimpleSimilarityRetriever
        from ragflow.adapters.llms import GeminiAdapter
        from ragflow.core.pipeline import RAGPipeline

        # Create the component instances
        chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=1000)
        embedder = SentenceTransformersAdapter()
        vector_store = ChromaDBAdapter(embedding_function=embedder)
        retriever = SimpleSimilarityRetriever(vector_store=vector_store)
        llm = GeminiAdapter(api_key="your-api-key")

        # Create the pipeline
        pipeline = RAGPipeline(
            chunking_strategy=chunker,
            embedding_model=embedder,
            vector_store=vector_store,
            retrieval_strategy=retriever,
            llm=llm,
        )

        # Use the pipeline
        pipeline.add_documents([Document(page_content="Example document")])
        answer = pipeline.query("What is in the document?")
        ```
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategyInterface,
        embedding_model: EmbeddingModelInterface,
        vector_store: VectorStoreInterface,
        retrieval_strategy: RetrievalStrategyInterface,
        llm: LLMInterface,
    ):
        """
        Initialize the RAG pipeline with its component parts.

        This constructor assembles all the required components into a cohesive
        pipeline that can process documents and answer queries. Each component
        must implement its respective interface from the core.interfaces module.

        Args:
            chunking_strategy: Strategy for splitting documents into chunks.
                Must implement the ChunkingStrategyInterface.
            embedding_model: Model for generating vector embeddings.
                Must implement the EmbeddingModelInterface.
            vector_store: Store for saving and retrieving embeddings.
                Must implement the VectorStoreInterface.
            retrieval_strategy: Strategy for retrieving relevant documents.
                Must implement the RetrievalStrategyInterface.
            llm: Language model for generating answers.
                Must implement the LLMInterface.

        Examples:
            ```python
            # Create component instances
            chunker = RecursiveCharacterTextSplitterAdapter()
            embedder = SentenceTransformersAdapter()
            vector_store = ChromaDBAdapter(embedding_function=embedder)
            retriever = SimpleSimilarityRetriever(vector_store=vector_store)
            llm = GeminiAdapter(api_key="your-api-key")

            # Initialize the pipeline
            pipeline = RAGPipeline(
                chunking_strategy=chunker,
                embedding_model=embedder,
                vector_store=vector_store,
                retrieval_strategy=retriever,
                llm=llm,
            )
            ```
        """
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy
        self.llm = llm

    def add_documents(self, documents: List[Document]) -> None:
        """
        Process and add documents to the pipeline.

        This method handles the complete document ingestion process:
        1. Splits the documents into smaller chunks using the configured chunking strategy
        2. Generates embeddings for these chunks using the embedding model
        3. Adds the chunks and their embeddings to the vector store

        This prepares the documents for later retrieval when answering queries.

        Args:
            documents: List of Document objects to add to the pipeline.
                Each Document should have page_content (text) and optional
                metadata.

        Returns:
            None

        Examples:
            ```python
            # Create a pipeline
            pipeline = RAGPipeline(...)

            # Create Document objects
            documents = [
                Document(
                    page_content="RAGFlow is a framework for building RAG applications.",
                    metadata={"source": "introduction.txt"},
                ),
                Document(
                    page_content="Vector stores enable efficient similarity search.",
                    metadata={"source": "concepts.txt"},
                ),
            ]

            # Add documents to the pipeline
            pipeline.add_documents(documents)
            ```
        """
        # Split documents into chunks
        chunked_documents = self.chunking_strategy.split_documents(documents)

        # Add chunked documents to the vector store
        self.vector_store.add_documents(chunked_documents)

    def add_texts(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Process and add text strings to the pipeline.

        This is a convenience method that:
        1. Converts texts to Document objects (with optional metadata)
        2. Splits them into smaller chunks
        3. Adds those chunks to the vector store

        It's a simpler alternative to creating Document objects manually
        when you have raw text strings.

        Args:
            texts: List of text strings to add to the pipeline.
            metadata: Optional list of metadata dictionaries, one for each text.
                If provided, must have the same length as texts. If not provided,
                empty metadata will be used for all texts.

        Returns:
            None

        Examples:
            ```python
            # Create a pipeline
            pipeline = RAGPipeline(...)

            # Add texts with metadata
            pipeline.add_texts(
                texts=[
                    "RAGFlow is a framework for building RAG applications.",
                    "Chunking strategies split documents into manageable pieces.",
                ],
                metadata=[
                    {"source": "introduction.txt", "page": 1},
                    {"source": "concepts.txt", "page": 5},
                ],
            )

            # Add texts without metadata
            pipeline.add_texts(["More text content here", "And another example"])
            ```
        """
        # Convert texts to Documents
        documents = []
        for i, text in enumerate(texts):
            m = metadata[i] if metadata and i < len(metadata) else {}
            documents.append(Document(page_content=text, metadata=m))

        # Add documents to the pipeline
        self.add_documents(documents)

    def query(self, question: str) -> str:
        """
        Process a query through the RAG pipeline to generate an answer.

        This method implements the core RAG workflow:
        1. Retrieves relevant documents using the retrieval strategy
        2. Passes the question and retrieved documents to the LLM
        3. Returns the generated answer

        The quality of the answer depends on:
        - The relevance of the retrieved documents
        - The capabilities of the LLM
        - The pre-processing of documents (chunking)

        Args:
            question: The query/question to answer. This can be a natural
                language question, a statement, or any text that requires
                a response based on the stored documents.

        Returns:
            A string containing the generated answer based on retrieved context.

        Examples:
            ```python
            # Create and initialize a pipeline
            pipeline = RAGPipeline(...)

            # Add some documents
            pipeline.add_texts(
                [
                    "RAGFlow is a Python framework for building RAG applications.",
                    "It provides interfaces for document processing and retrieval.",
                ]
            )

            # Ask a question
            answer = pipeline.query("What is RAGFlow?")
            print(answer)
            # Output might be: "RAGFlow is a Python framework designed for
            # building Retrieval Augmented Generation (RAG) applications.
            # It provides interfaces for document processing and retrieval."
            ```
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieval_strategy.get_relevant_documents(question)

        # Generate answer using LLM and retrieved context
        answer = self.llm.generate_with_context(question, relevant_docs)

        return answer

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Process a query and return both the answer and source documents.

        Similar to query(), but also returns the source documents used
        to generate the answer, enabling citation and verification. This
        is useful for:
        - Providing attribution for information
        - Allowing users to verify the sources
        - Debugging the RAG pipeline's retrieval performance

        Args:
            question: The query/question to answer. This can be a natural
                language question, a statement, or any text that requires
                a response based on the stored documents.

        Returns:
            A dictionary containing:
            - 'answer': The generated response as a string
            - 'sources': List of Document objects used to generate the answer

        Examples:
            ```python
            # Create and initialize a pipeline
            pipeline = RAGPipeline(...)

            # Add some documents with source information
            pipeline.add_texts(
                texts=[
                    "RAGFlow was created in 2024.",
                    "RAGFlow supports multiple vector databases.",
                ],
                metadata=[
                    {"source": "history.txt", "page": 1},
                    {"source": "features.txt", "page": 3},
                ],
            )

            # Ask a question with sources
            result = pipeline.query_with_sources("When was RAGFlow created?")

            # Print the answer
            print(result["answer"])  # "RAGFlow was created in 2024."

            # Print the sources
            for doc in result["sources"]:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content}")
            ```
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieval_strategy.get_relevant_documents(question)

        # Generate answer using LLM and retrieved context
        answer = self.llm.generate_with_context(question, relevant_docs)

        # Return both the answer and the source documents
        return {"answer": answer, "sources": relevant_docs}
