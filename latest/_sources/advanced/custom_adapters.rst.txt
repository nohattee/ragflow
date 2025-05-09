=====================
Creating Custom Adapters
=====================

This guide explains how to extend RAGFlow by implementing custom adapters for various components. The interface-driven architecture of RAGFlow makes it straightforward to add support for new vector stores, embedding models, LLMs, chunking strategies, or retrieval approaches.

Understanding RAGFlow's Interface Architecture
--------------------------------------------

RAGFlow uses an interface-driven design pattern where each component type is defined by an abstract interface:

- **ChunkingStrategyInterface**: For splitting documents into smaller chunks
- **EmbeddingModelInterface**: For generating vector embeddings from text
- **VectorStoreInterface**: For storing and retrieving vector embeddings
- **RetrievalStrategyInterface**: For retrieving relevant documents from a vector store
- **LLMInterface**: For generating text using large language models

To add support for a new tool or service (like a new embedding model or vector database), you create an *adapter* that implements the appropriate interface. This adapter translates between RAGFlow's standardized interface methods and the specific API of the tool you're integrating.

Steps for Creating a Custom Adapter
---------------------------------

1. Choose the Interface to Implement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, determine which interface your adapter should implement:

.. code-block:: python

   from ragflow.core.interfaces import (
       ChunkingStrategyInterface,
       EmbeddingModelInterface,
       VectorStoreInterface,
       RetrievalStrategyInterface,
       LLMInterface
   )

2. Create a New Adapter Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new Python file in the appropriate subdirectory under ``ragflow/adapters/``. Your class should inherit from the relevant interface.

3. Implement Required Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement all abstract methods defined in the interface. Refer to the interface docstrings for method signatures and expected behavior.

4. Add Proper Documentation
~~~~~~~~~~~~~~~~~~~~~~~~

Add docstrings to your adapter class and methods, following the same style as the existing adapters.

Example: Custom Embedding Model Adapter
-------------------------------------

Here's an example of creating a custom adapter for a hypothetical embedding model:

.. code-block:: python

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

Example: Custom Vector Store Adapter
----------------------------------

Here's an example of a custom vector store adapter:

.. code-block:: python

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
                   embedding=embedding,
                   text=text,
                   metadatas=metadata
               )

       def similarity_search(
           self,
           query: str,
           k: int = 4,
           **kwargs
       ) -> List[Document]:
           """
           Perform a similarity search for a query string.

           Args:
               query: Query string
               k: Number of results to return
               **kwargs: Additional search parameters

           Returns:
               List of Documents most similar to the query
           """
           # Ensure we have an embedding function
           if self.embedding_function is None:
               raise ValueError("No embedding function provided")

           # Generate embedding for the query
           query_embedding = self.embedding_function.embed_query(query)

           # Search the vector database
           # (Adjust this to match your database's API)
           results = self.collection.search(
               embedding=query_embedding,
               limit=k,
               **kwargs
           )

           # Convert results to Documents
           documents = []
           for result in results:
               doc = Document(
                   page_content=result["text"],
                   metadata=result["metadata"]
               )
               documents.append(doc)

           return documents

Example: Custom Document Loader Adapter
-------------------------------------

Document loaders are utilities that help load content from various sources into the RAG pipeline. While they don't directly implement one of the core interfaces, they follow a similar pattern:

.. code-block:: python

   """
   Custom document loader for specialized file formats.

   This module provides a document loader for handling custom file formats
   not supported by the default loaders.
   """

   from typing import List, Optional
   import specialized_parser  # Hypothetical parser for your file format

   from ragflow.core.interfaces import Document

   def load_specialized_files(
       directory: str,
       recursive: bool = True,
       glob_pattern: str = "*.special",
       encoding: str = "utf-8"
   ) -> List[Document]:
       """
       Load documents from specialized file format.

       Args:
           directory: Path to the directory containing the files
           recursive: Whether to search subdirectories
           glob_pattern: Pattern to match files
           encoding: Text encoding to use

       Returns:
           List of Document objects
       """
       import os
       import glob

       # Find all matching files
       if recursive:
           pattern = os.path.join(directory, "**", glob_pattern)
           files = glob.glob(pattern, recursive=True)
       else:
           pattern = os.path.join(directory, glob_pattern)
           files = glob.glob(pattern)

       documents = []

       for file_path in files:
           try:
               # Use your specialized parser
               parsed_content = specialized_parser.parse_file(file_path, encoding=encoding)

               # Extract content and metadata
               text_content = parsed_content.get_text()
               metadata = {
                   "source": file_path,
                   "filename": os.path.basename(file_path),
                   "filetype": "specialized",
                   "author": parsed_content.get_author(),
                   "created_date": parsed_content.get_created_date()
               }

               # Create document
               doc = Document(page_content=text_content, metadata=metadata)
               documents.append(doc)

           except Exception as e:
               print(f"Error processing {file_path}: {e}")

       return documents

Best Practices for Adapter Development
------------------------------------

When creating custom adapters, follow these best practices to ensure they work well within the RAGFlow ecosystem:

1. **Thorough Docstrings**

   Write comprehensive docstrings that explain:
   - The purpose of the adapter
   - Required dependencies
   - All initialization parameters
   - Method parameters and return values
   - Usage examples

2. **Error Handling**

   Implement robust error handling:
   - Gracefully handle failures from the underlying library
   - Provide clear error messages that help users diagnose issues
   - Validate parameters early to prevent obscure errors later

   .. code-block:: python

      def embed_query(self, query: str) -> List[float]:
          """Generate embedding for query."""
          if not query or not isinstance(query, str):
              raise ValueError("Query must be a non-empty string")

          try:
              embedding = self.model.encode(query)
              return embedding.tolist()
          except Exception as e:
              raise RuntimeError(f"Error generating embedding: {e}") from e

3. **Type Annotations**

   Use proper type annotations for all parameters and return values. This improves IDE support and helps catch errors early.

4. **Testing Strategy**

   Create thorough tests for your adapter:

   .. code-block:: python

      import unittest
      from ragflow.core.interfaces import Document
      from ragflow.adapters.your_module import YourAdapter

      class TestYourAdapter(unittest.TestCase):
          """Tests for YourAdapter."""

          def setUp(self):
              """Set up test fixtures."""
              self.adapter = YourAdapter()

          def test_interface_compliance(self):
              """Test that all interface methods are properly implemented."""
              # Test method implementations

          def test_basic_functionality(self):
              """Test basic adapter functionality."""
              # Test core functionality

          def test_error_handling(self):
              """Test adapter error handling."""
              # Test error scenarios

5. **Configuration Options**

   Provide sensible defaults but allow for configuration:
   - Make key parameters customizable
   - Document the impact of different configurations
   - When possible, use environment variables for sensitive information

6. **Performance Considerations**

   Optimize for performance where appropriate:
   - Implement batching for operations that support it
   - Consider caching results when appropriate
   - Document any performance characteristics users should be aware of

7. **Dependency Management**

   Clearly document and manage dependencies:
   - List all required packages in your documentation
   - Specify version requirements if needed
   - Consider making dependencies optional with helpful error messages

Example: Complete Custom LLM Adapter
----------------------------------

Here's a more complete example of a custom LLM adapter with best practices implemented:

.. code-block:: python

   """
   Adapter for the MyCustomLLM API.

   This module provides an implementation of LLMInterface for the
   MyCustomLLM API, enabling its use in RAGFlow pipelines.

   Requirements:
       - mycustomllm-python package (>= 1.2.0)
       - API key set as MYCUSTOMLLM_API_KEY environment variable
   """

   import os
   import logging
   from typing import List, Dict, Any, Optional, Union
   from functools import lru_cache

   # Import the library (with error handling)
   try:
       import mycustomllm
   except ImportError:
       raise ImportError(
           "The mycustomllm-python package is required. "
           "Install it with: pip install mycustomllm-python>=1.2.0"
       )

   from ragflow.core.interfaces import LLMInterface, Document

   logger = logging.getLogger(__name__)

   class MyCustomLLMAdapter(LLMInterface):
       """
       Adapter for MyCustomLLM that implements the LLMInterface.

       This adapter allows using MyCustomLLM as the language model in a RAG pipeline.

       Example:
           ```python
           from ragflow.adapters.llms.mycustomllm_adapter import MyCustomLLMAdapter

           llm = MyCustomLLMAdapter(
               api_key="your-api-key",
               model="mycustomllm-large",
               temperature=0.7
           )

           response = llm.generate("What is RAG?")
           print(response)
           ```
       """

       DEFAULT_MODEL = "mycustomllm-base"
       DEFAULT_PROMPT_TEMPLATE = """
       Answer the following question based on the provided context.

       Context:
       {context}

       Question: {question}

       Answer:
       """

       def __init__(
           self,
           api_key: Optional[str] = None,
           model: str = DEFAULT_MODEL,
           temperature: float = 0.5,
           max_tokens: int = 500,
           prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
           **kwargs
       ):
           """
           Initialize the MyCustomLLM adapter.

           Args:
               api_key: API key for MyCustomLLM. If None, will try to use
                   the MYCUSTOMLLM_API_KEY environment variable.
               model: Model name to use. Defaults to "mycustomllm-base".
               temperature: Temperature parameter controlling randomness.
                   Range from 0.0 to 1.0, where 0 is deterministic.
               max_tokens: Maximum number of tokens to generate.
               prompt_template: Template string for formatting context and questions.
                   Use {context} and {question} placeholders.
               **kwargs: Additional parameters to pass to the model.

           Raises:
               ValueError: If no API key is provided and none is found in
                   the environment variables.
           """
           # Get API key from arguments or environment
           self.api_key = api_key or os.environ.get("MYCUSTOMLLM_API_KEY")
           if not self.api_key:
               raise ValueError(
                   "No API key provided. Either pass an api_key parameter or "
                   "set the MYCUSTOMLLM_API_KEY environment variable."
               )

           # Validate and store parameters
           if temperature < 0.0 or temperature > 1.0:
               raise ValueError("Temperature must be between 0.0 and 1.0")

           self.model = model
           self.temperature = temperature
           self.max_tokens = max_tokens
           self.prompt_template = prompt_template
           self.additional_params = kwargs

           # Initialize client
           self.client = mycustomllm.Client(api_key=self.api_key)

           logger.info(f"Initialized MyCustomLLMAdapter with model: {model}")

       @lru_cache(maxsize=32)
       def _get_model(self):
           """
           Get and cache the model.

           Returns:
               The loaded model instance
           """
           return self.client.get_model(self.model)

       def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
           """
           Generate text based on a prompt.

           Args:
               prompt: The prompt string to generate text from
               context: Optional additional context or parameters

           Returns:
               A string containing the generated text response

           Raises:
               RuntimeError: If an error occurs during generation
           """
           try:
               # Get cached model
               model = self._get_model()

               # Prepare parameters
               params = {
                   "temperature": self.temperature,
                   "max_tokens": self.max_tokens,
                   **self.additional_params
               }

               # Update with any context parameters
               if context:
                   params.update(context)

               # Generate response
               response = model.generate(prompt, **params)

               return response.text
           except Exception as e:
               logger.error(f"Error generating text: {e}")
               raise RuntimeError(f"Error generating text: {e}") from e

       def generate_with_context(self, query: str, context: List[Document]) -> str:
           """
           Generate a response to a query using retrieved context documents.

           Args:
               query: The user query or question to answer
               context: A list of Document objects providing relevant context

           Returns:
               A string containing the generated response
           """
           # Format context documents into a string
           context_text = "\n\n".join(
               f"Document {i+1}:\n{doc.page_content}"
               for i, doc in enumerate(context)
           )

           # Format the prompt using the template
           prompt = self.prompt_template.format(
               context=context_text,
               question=query
           )

           # Generate response
           return self.generate(prompt)

Testing Custom Adapters
---------------------

Create a test file for your adapter in the appropriate test directory. Here's an example:

.. code-block:: python

   """
   Tests for the MyCustomLLMAdapter.
   """

   import unittest
   from unittest.mock import patch, MagicMock
   import os

   from ragflow.core.interfaces import Document
   from ragflow.adapters.llms.mycustomllm_adapter import MyCustomLLMAdapter

   class TestMyCustomLLMAdapter(unittest.TestCase):
       """Test suite for MyCustomLLMAdapter."""

       def setUp(self):
           """Set up test fixtures."""
           # Use a test API key for tests
           self.api_key = "test-api-key"
           # Create a patcher for the mycustomllm library
           self.client_patcher = patch("mycustomllm.Client")
           self.mock_client = self.client_patcher.start()

           # Set up the mock client
           self.mock_model = MagicMock()
           self.mock_model.generate.return_value = MagicMock(
               text="This is a test response."
           )
           self.mock_client.return_value.get_model.return_value = self.mock_model

           # Create adapter with test configuration
           self.adapter = MyCustomLLMAdapter(
               api_key=self.api_key,
               model="test-model",
               temperature=0.5
           )

       def tearDown(self):
           """Tear down test fixtures."""
           self.client_patcher.stop()

       def test_init_with_api_key_param(self):
           """Test initialization with API key parameter."""
           adapter = MyCustomLLMAdapter(api_key=self.api_key)
           self.assertEqual(adapter.api_key, self.api_key)
           self.mock_client.assert_called_with(api_key=self.api_key)

       def test_init_with_environment_variable(self):
           """Test initialization with environment variable."""
           with patch.dict(os.environ, {"MYCUSTOMLLM_API_KEY": "env-api-key"}):
               adapter = MyCustomLLMAdapter()
               self.assertEqual(adapter.api_key, "env-api-key")

       def test_generate(self):
           """Test generate method."""
           response = self.adapter.generate("Test prompt")
           self.assertEqual(response, "This is a test response.")

           # Verify the model was called with correct parameters
           self.mock_model.generate.assert_called_with(
               "Test prompt",
               temperature=0.5,
               max_tokens=500
           )

       def test_generate_with_context(self):
           """Test generate_with_context method."""
           # Create test documents
           documents = [
               Document(page_content="Document content 1"),
               Document(page_content="Document content 2")
           ]

           # Call method
           response = self.adapter.generate_with_context(
               "Test question",
               documents
           )

           # Verify result
           self.assertEqual(response, "This is a test response.")

           # Verify the expected prompt was generated
           expected_context = (
               "Document 1:\nDocument content 1\n\n"
               "Document 2:\nDocument content 2"
           )
           self.mock_model.generate.assert_called_once()
           call_args = self.mock_model.generate.call_args[0][0]
           self.assertIn(expected_context, call_args)
           self.assertIn("Test question", call_args)

       def test_error_handling(self):
           """Test error handling."""
           # Make the mock model raise an exception
           self.mock_model.generate.side_effect = Exception("API error")

           # Verify exception is properly propagated
           with self.assertRaises(RuntimeError):
               self.adapter.generate("Test prompt")

Contributing Your Adapter to RAGFlow
----------------------------------

If you've developed a useful adapter that others might benefit from, consider contributing it to the RAGFlow project. Here's how:

1. Ensure your code follows the project coding standards
2. Add thorough documentation and tests
3. Fork the RAGFlow repository and create a new branch for your adapter
4. Create a pull request with a clear description of your adapter
5. Respond to any feedback during the code review process

Before contributing, please read the full :doc:`../development/contributing` guide for more details.
