"""
Custom Adapters Example for RAGFlow

This module demonstrates how to create and use custom adapters:
1. HuggingFaceEmbeddingAdapter - A custom adapter for Hugging Face embeddings
2. PineconeVectorStoreAdapter - A custom adapter for Pinecone vector store
3. CustomRAGPipeline that uses these custom adapters
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from ragflow.adapters.chunking_strategies.recursive_character_splitter_adapter import (
    RecursiveCharacterTextSplitterAdapter,
)
from ragflow.adapters.llms.gemini_adapter import GeminiAdapter

# Import RAGFlow interfaces and components
from ragflow.core.interfaces import (
    Document,
    EmbeddingModelInterface,
    VectorStoreInterface,
)
from ragflow.core.pipeline import RAGPipeline

# Load environment variables
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


class HuggingFaceEmbeddingAdapter(EmbeddingModelInterface):
    """
    Custom adapter for Hugging Face text embeddings that implements EmbeddingModelInterface.

    This adapter uses Hugging Face's transformers library to generate embeddings
    from text using any compatible Hugging Face model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the Hugging Face embedding adapter.

        Args:
            model_name: Name of the Hugging Face model to use
            max_length: Maximum sequence length
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional kwargs to pass to the tokenizer and model
        """
        # These imports are placed inside the init to make them optional
        # and avoid import errors if transformers is not installed
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The HuggingFaceEmbeddingAdapter requires the transformers and torch packages. "
                "Please install them with `pip install transformers torch`."
            )

        self.model_name = model_name
        self.max_length = max_length
        self.device = device

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(device)

        # Store needed imports for later use
        self.torch = torch

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text input.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate the embedding
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            # Use the mean of the last hidden state as the embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        # Convert to list of floats and return
        return embedding.cpu().tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a query string.

        Args:
            query: The query text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        return self._get_embedding(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document texts to embed

        Returns:
            List of embedding vectors, one for each input document
        """
        return [self._get_embedding(doc) for doc in documents]


class PineconeVectorStoreAdapter(VectorStoreInterface):
    """
    Custom adapter for Pinecone vector database that implements VectorStoreInterface.

    This adapter uses Pinecone's Python client to store document embeddings
    and perform similarity searches.
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        namespace: str = "",
        embedding_function: Optional[EmbeddingModelInterface] = None,
        **kwargs,
    ):
        """
        Initialize the Pinecone vector store adapter.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            namespace: Optional namespace within the index
            embedding_function: Function to generate embeddings
            **kwargs: Additional kwargs to pass to Pinecone
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "The PineconeVectorStoreAdapter requires the pinecone-client package. "
                "Please install it with `pip install pinecone-client`."
            )

        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_function = embedding_function

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Get or create the index
        if index_name not in pinecone.list_indexes():
            raise ValueError(
                f"Index '{index_name}' does not exist. Please create it first."
            )

        self.index = pinecone.Index(index_name)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        self.add_texts(texts, metadata)

    def add_texts(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add text strings with optional metadata to the vector store.

        Args:
            texts: List of text strings to add
            metadata: Optional list of metadata dicts, one for each text
        """
        if not texts:
            return

        # Ensure we have an embedding function
        if self.embedding_function is None:
            raise ValueError("No embedding function provided")

        # Generate embeddings
        embeddings = self.embedding_function.embed_documents(texts)

        # Prepare vectors for Pinecone
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadata[i] if metadata else {}
            # Add the text to the metadata
            metadata["text"] = text
            vectors.append(
                {
                    "id": f"doc_{i}_{hash(text) % 10000}",
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        # Upsert vectors to Pinecone
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search using a query string.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of documents most similar to the query
        """
        # Ensure we have an embedding function
        if self.embedding_function is None:
            raise ValueError("No embedding function provided")

        # Generate embedding for the query
        query_embedding = self.embedding_function.embed_query(query)

        # Perform the similarity search
        return self.similarity_search_by_vector(query_embedding, k)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Document]:
        """
        Perform a similarity search using a vector embedding.

        Args:
            embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of documents most similar to the query embedding
        """
        # Query Pinecone
        results = self.index.query(
            vector=embedding, namespace=self.namespace, top_k=k, include_metadata=True
        )

        # Convert results to Document objects
        documents = []
        for match in results.matches:
            metadata = dict(match.metadata)
            text = metadata.pop("text", "")
            documents.append(Document(page_content=text, metadata=metadata))

        return documents


def custom_adapters_example():
    """
    Example of creating and using custom adapters with RAGFlow.

    This example builds a custom RAG pipeline using our HuggingFaceEmbeddingAdapter
    and a simulated PineconeVectorStoreAdapter.
    """
    print("\n=== Custom Adapters Example ===")

    # Get API keys from environment
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        return

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
    pinecone_index = os.environ.get("PINECONE_INDEX", "ragflow-demo")

    # Check if we can run the Pinecone example
    has_pinecone = all([pinecone_api_key, pinecone_environment, pinecone_index])

    try:
        # 1. Create our custom Hugging Face embedding adapter
        custom_embedder = HuggingFaceEmbeddingAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=512,
            device="cpu",
        )
        print("✓ Initialized HuggingFaceEmbeddingAdapter")

        # 2. Option 1: Create a RAGPipeline with the custom embedding adapter
        #    but standard vector store
        from ragflow.adapters.retrieval_strategies.simple_similarity_retriever import (
            SimpleSimilarityRetriever,
        )
        from ragflow.adapters.vector_stores.chromadb_adapter import ChromaDBAdapter

        vector_store = ChromaDBAdapter(
            collection_name="custom_adapter_example",
            persist_directory="./custom_adapter_db",
            embedding_function=custom_embedder,
        )

        retriever = SimpleSimilarityRetriever(vector_store=vector_store, k=2)

        chunker = RecursiveCharacterTextSplitterAdapter(
            chunk_size=500, chunk_overlap=100
        )

        llm = GeminiAdapter(api_key=gemini_api_key, temperature=0.2)

        # Create the pipeline with our custom embedding adapter
        custom_pipeline = RAGPipeline(
            chunking_strategy=chunker,
            embedding_model=custom_embedder,
            vector_store=vector_store,
            retrieval_strategy=retriever,
            llm=llm,
        )

        # Add documents and query
        custom_pipeline.add_documents(SAMPLE_DOCS)

        print("\nTesting pipeline with custom HuggingFaceEmbeddingAdapter:")
        answer = custom_pipeline.query("What is RAGFlow?")
        print("Query: What is RAGFlow?")
        print(f"Answer: {answer}")

        # 3. Option 2: Only run this if Pinecone credentials are available
        if has_pinecone:
            print("\nTesting with PineconeVectorStoreAdapter:")

            try:
                # Create the Pinecone adapter
                pinecone_store = PineconeVectorStoreAdapter(
                    api_key=pinecone_api_key,
                    environment=pinecone_environment,
                    index_name=pinecone_index,
                    namespace="custom_adapter_example",
                    embedding_function=custom_embedder,
                )

                # Create a retriever with the Pinecone store
                pinecone_retriever = SimpleSimilarityRetriever(
                    vector_store=pinecone_store, k=2
                )

                # Create a pipeline with both custom adapters
                pinecone_pipeline = RAGPipeline(
                    chunking_strategy=chunker,
                    embedding_model=custom_embedder,
                    vector_store=pinecone_store,
                    retrieval_strategy=pinecone_retriever,
                    llm=llm,
                )

                # Add documents and query
                pinecone_pipeline.add_documents(SAMPLE_DOCS)

                answer = pinecone_pipeline.query("What are vector databases used for?")
                print("Query: What are vector databases used for?")
                print(f"Answer: {answer}")

            except Exception as e:
                print(f"Error using Pinecone adapter: {str(e)}")
                print("This part of the example requires a valid Pinecone index.")
        else:
            print(
                "\nSkipping PineconeVectorStoreAdapter example - no credentials provided."
            )
            print(
                "To run this example, set PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX env vars."
            )

    except ImportError as e:
        print(f"Import error: {str(e)}")
        print(
            "This example requires additional packages. Please install them as described in the error message."
        )


if __name__ == "__main__":
    custom_adapters_example()
