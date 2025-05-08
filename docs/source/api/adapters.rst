=============
Adapters
=============

This page documents the adapter implementations for each component interface in RAGFlow. Each adapter implements one of the core interfaces defined in the :doc:`../api/core` module.

Adapters are the concrete implementations that connect RAGFlow to specific technologies and services. They allow the framework to be extended with new components without changing the core pipeline logic.

Chunking Strategies
-----------------

Chunking strategy adapters implement the ``ChunkingStrategyInterface`` and are responsible for splitting documents into smaller, manageable pieces.

.. automodule:: ragflow.adapters.chunking_strategies.recursive_character_splitter_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Embedding Models
--------------

Embedding model adapters implement the ``EmbeddingModelInterface`` and convert text into vector representations that capture semantic meaning.

.. automodule:: ragflow.adapters.embedding_models.sentence_transformers_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Vector Stores
-----------

Vector store adapters implement the ``VectorStoreInterface`` and provide storage and retrieval of vector embeddings.

.. automodule:: ragflow.adapters.vector_stores.chromadb_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Retrieval Strategies
------------------

Retrieval strategy adapters implement the ``RetrievalStrategyInterface`` and are responsible for finding the most relevant documents for a query.

.. automodule:: ragflow.adapters.retrieval_strategies.simple_similarity_retriever
   :members:
   :undoc-members:
   :show-inheritance:

LLMs
----

LLM adapters implement the ``LLMInterface`` and generate responses based on queries and retrieved context.

.. automodule:: ragflow.adapters.llms.gemini_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Creating Custom Adapters
---------------------

For detailed instructions on how to create your own adapters to extend RAGFlow's capabilities, see the :doc:`../advanced/custom_adapters` guide.

Example:

.. code-block:: python

   from ragflow.core.interfaces import EmbeddingModelInterface

   class MyCustomEmbeddingModel(EmbeddingModelInterface):
       """
       A custom implementation of the EmbeddingModelInterface.
       """

       def __init__(self, model_name: str = "my-custom-model"):
           self.model_name = model_name
           # Initialize your custom embedding model

       def embed_query(self, query: str) -> list[float]:
           # Implement query embedding logic
           pass

       def embed_documents(self, documents: list[str]) -> list[list[float]]:
           # Implement document embedding logic
           pass
