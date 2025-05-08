# RAGFlow Custom Adapters Guide

This guide explains how to extend RAGFlow by implementing custom adapters for various components. The interface-driven architecture of RAGFlow makes it straightforward to add support for new vector stores, embedding models, LLMs, chunking strategies, or retrieval approaches.

## Understanding RAGFlow's Interface Architecture

RAGFlow uses an interface-driven design pattern where each component type is defined by an abstract interface:

- **`ChunkingStrategyInterface`**: For splitting documents into smaller chunks
- **`EmbeddingModelInterface`**: For generating vector embeddings from text
- **`VectorStoreInterface`**: For storing and retrieving vector embeddings
- **`RetrievalStrategyInterface`**: For retrieving relevant documents from a vector store
- **`LLMInterface`**: For generating text using large language models

To add support for a new tool or service (like a new embedding model or vector database), you create an *adapter* that implements the appropriate interface. This adapter translates between RAGFlow's standardized interface methods and the specific API of the tool you're integrating.

## Steps for Creating a Custom Adapter

### 1. Choose the Interface to Implement

First, determine which interface your adapter should implement:

```python
from ragflow.core.interfaces import (
    ChunkingStrategyInterface,
    EmbeddingModelInterface,
    VectorStoreInterface,
    RetrievalStrategyInterface,
    LLMInterface
)
```

### 2. Create a New Adapter Class

Create a new Python file in the appropriate subdirectory of `ragflow/adapters/`. Your class should inherit from the relevant interface.

### 3. Implement Required Methods

Implement all abstract methods defined in the interface. Refer to the interface docstrings for method signatures and expected behavior.

### 4. Add Proper Documentation

Add docstrings to your adapter class and methods, following the same style as the existing adapters.

## Example: Custom Embedding Model Adapter

Here's an example of creating a custom adapter for a hypothetical embedding model:

```python
"""
Custom embedding model adapter for EmbeddingModelInterface.

This module provides an implementation of EmbeddingModelInterface using
a custom embedding model to generate embeddings.
"""

from typing import List
import numpy as np
# Import your embedding model library
import custom_embeddings_library

from ragflow.core.interfaces import EmbeddingModelInterface

class CustomEmbeddingModelAdapter(EmbeddingModelInterface):
    """
    Adapter for a custom embedding model that implements the EmbeddingModelInterface.

    This adapter uses [describe your embedding model/library] to generate embeddings
    for queries and documents.
    """

    def __init__(self, model_name: str = "default-model", **kwargs):
        """
        Initialize the custom embedding model adapter.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional arguments to pass to the model
        """
        self.model_name = model_name
        self.model = custom_embeddings_library.load_model(model_name, **kwargs)

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a query string.

        Args:
            query: The query text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        # Implement the embedding logic specific to your model
        embedding = self.model.encode(query)
        return embedding.tolist()  # Ensure the return value is a List[float]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document texts to embed

        Returns:
            List of embedding vectors, one for each input document
        """
        # Implement batch embedding logic
        embeddings = self.model.encode_batch(documents)
        return [embedding.tolist() for embedding in embeddings]
```

## Example: Custom Vector Store Adapter

Here's an example of a custom vector store adapter:

```python
"""
Custom vector store adapter for VectorStoreInterface.

This module provides an implementation of VectorStoreInterface using
a custom vector database to store and retrieve embeddings.
"""

from typing import List, Dict, Any, Optional
# Import your vector database library
import custom_vector_db

from ragflow.core.interfaces import VectorStoreInterface, Document, EmbeddingModelInterface

class CustomVectorStoreAdapter(VectorStoreInterface):
    """
    Adapter for a custom vector database that implements the VectorStoreInterface.

    This adapter uses [describe your vector database] to store document embeddings
    and perform similarity searches.
    """

    def __init__(
        self,
        collection_name: str = "ragflow",
        embedding_function: Optional[EmbeddingModelInterface] = None,
        **kwargs
    ):
        """
        Initialize the custom vector store adapter.

        Args:
            collection_name: Name of the collection to use
            embedding_function: Function to generate embeddings (if None, must be provided in add_texts)
            **kwargs: Additional arguments to pass to the vector database
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Initialize your vector database client
        self.client = custom_vector_db.Client(**kwargs)
        self.collection = self.client.get_or_create_collection(collection_name)

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
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
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

        # Add to your vector database
        # (Adjust this to match your database's API)
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadata[i] if metadata else {}
            self.collection.add(
                id=f"doc_{i}",
                vector=embedding,
                metadatas={"text": text, **metadata}
            )

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
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
        self,
        embedding: List[float],
        k: int = 4
    ) -> List[Document]:
        """
        Perform a similarity search using a vector embedding.

        Args:
            embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of documents most similar to the query embedding
        """
        # Use your vector database's search method
        # (Adjust this to match your database's API)
        results = self.collection.search(
            vector=embedding,
            limit=k
        )

        # Convert results to Document objects
        documents = []
        for result in results:
            metadata = dict(result.metadata)
            text = metadata.pop("text", "")
            documents.append(Document(page_content=text, metadata=metadata))

        return documents
```

## Best Practices for Custom Adapters

1. **Follow Interface Contracts**: Ensure your adapter fully implements all methods defined in the interface with the correct signatures.

2. **Error Handling**: Implement proper error handling and provide helpful error messages when things go wrong.

3. **Configuration Options**: Make your adapter configurable through constructor parameters.

4. **Documentation**: Add comprehensive docstrings to your adapter class and methods.

5. **Type Hints**: Use proper type hints to improve code readability and enable static type checking.

6. **Testing**: Write unit tests for your adapter to ensure it works correctly and follows the interface contract.

## Testing Your Custom Adapter

Test your adapter both in isolation and as part of a RAGPipeline:

```python
# Test in isolation
adapter = YourCustomAdapter()
result = adapter.some_method(input_data)

# Test as part of a pipeline
from ragflow.core.pipeline import RAGPipeline

pipeline = RAGPipeline(
    # ... other components ...
    your_component_type=YourCustomAdapter()
)

# Try the pipeline with your adapter
pipeline.add_documents(documents)
response = pipeline.query("Test query")
```

## Contributing Adapters to RAGFlow

If you've created a useful adapter that others might benefit from, consider contributing it back to the RAGFlow project. Follow these steps:

1. Ensure your code follows the project's coding style and has proper documentation.
2. Add appropriate unit tests for your adapter.
3. Create a pull request with your new adapter.

For more details on contributing, see our [Contribution Guidelines](contributing.md).

## Additional Resources

- [RAGFlow Core Interfaces](../core/interfaces.py) - The source code for the interfaces you'll be implementing
- [Existing Adapter Implementations](../adapters/) - Examples of how other adapters are implemented
- [DefaultRAGPipeline](../pipelines/default_rag_pipeline.py) - See how adapters are used in the pipeline
