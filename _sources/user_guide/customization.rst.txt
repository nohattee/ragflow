====================
Customizing RAGFlow
====================

RAGFlow is designed to be highly customizable to meet diverse requirements. This guide explains the different approaches to customization, from simple parameter adjustments to building entirely custom pipelines.

Customizing the Default Pipeline
-------------------------------

The ``DefaultRAGPipeline`` class is designed to be easily customizable through its constructor parameters. You can adjust settings for all components without having to instantiate them individually.

.. code-block:: python

   from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

   pipeline = DefaultRAGPipeline(
       # Vector store configuration
       persist_directory="./my_db",
       collection_name="my_collection",

       # Embedding model configuration
       embedding_model_name="all-MiniLM-L6-v2",

       # Chunking configuration
       chunk_size=500,
       chunk_overlap=100,

       # Retrieval configuration
       retrieval_k=5,

       # LLM configuration
       api_key="your-api-key",
       model_name="gemini-pro",
       temperature=0.3,
       max_tokens=200
   )

Component-Specific Customization
------------------------------

For more advanced customization, the ``DefaultRAGPipeline`` accepts component-specific keyword arguments that are passed directly to the underlying adapters:

.. code-block:: python

   pipeline = DefaultRAGPipeline(
       api_key="your-api-key",

       # Pass custom parameters to specific components
       chunker_kwargs={
           "separators": ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
       },

       embedder_kwargs={
           "device": "cpu"  # SentenceTransformers-specific parameter
       },

       vector_store_kwargs={
           "anonymized_telemetry": False  # ChromaDB-specific parameter
       },

       retriever_kwargs={
           "search_type": "similarity"  # Retriever-specific parameter
       },

       llm_kwargs={
           "safety_settings": [
               {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
           ]  # Gemini-specific parameter
       }
   )

Building Custom Pipelines
-----------------------

For maximum flexibility, you can instantiate each adapter separately and combine them into a custom pipeline:

.. code-block:: python

   from ragflow.core.pipeline import RAGPipeline
   from ragflow.adapters.chunking_strategies.recursive_character_splitter_adapter import RecursiveCharacterTextSplitterAdapter
   from ragflow.adapters.embedding_models.sentence_transformers_adapter import SentenceTransformersAdapter
   from ragflow.adapters.vector_stores.chromadb_adapter import ChromaDBAdapter
   from ragflow.adapters.retrieval_strategies.simple_similarity_retriever import SimpleSimilarityRetriever
   from ragflow.adapters.llms.gemini_adapter import GeminiAdapter

   # Create custom components
   chunker = RecursiveCharacterTextSplitterAdapter(
       chunk_size=250,
       chunk_overlap=50,
       separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
   )

   embedder = SentenceTransformersAdapter(
       model_name="paraphrase-multilingual-MiniLM-L12-v2"
   )

   vector_store = ChromaDBAdapter(
       collection_name="custom_collection",
       persist_directory="./custom_db",
       embedding_function=embedder
   )

   retriever = SimpleSimilarityRetriever(
       vector_store=vector_store,
       k=3
   )

   llm = GeminiAdapter(
       api_key="your-api-key",
       model_name="gemini-pro",
       temperature=0.1,
       max_tokens=100
   )

   # Assemble the custom pipeline
   custom_pipeline = RAGPipeline(
       chunking_strategy=chunker,
       embedding_model=embedder,
       vector_store=vector_store,
       retrieval_strategy=retriever,
       llm=llm
   )

Using Independent Components
--------------------------

Each adapter can also be used independently outside of a pipeline:

.. code-block:: python

   # Use the chunker directly
   chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=500, chunk_overlap=100)
   chunks = chunker.split_text("This is a long document that needs to be split into chunks.")

   # Use the embedding model directly
   embedder = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")
   embedding = embedder.embed_query("What is RAG?")

   # Use the LLM directly
   llm = GeminiAdapter(api_key="your-api-key")
   response = llm.generate("Explain what vector embeddings are.")

Configuration Profiles
--------------------

For reusable configurations, you can create configuration profiles:

.. code-block:: python

   config_profiles = {
       "low_resource": {
           "persist_directory": "./low_resource_db",
           "embedding_model_name": "all-MiniLM-L6-v2",  # Small, fast model
           "chunk_size": 1500,  # Larger chunks, fewer embeddings
           "retrieval_k": 2,  # Retrieve fewer documents
           "temperature": 0.7,
           "max_tokens": 100,  # Shorter responses
       },
       "high_accuracy": {
           "persist_directory": "./high_accuracy_db",
           "embedding_model_name": "all-mpnet-base-v2",  # More accurate model
           "chunk_size": 500,  # Smaller chunks for precise retrieval
           "retrieval_k": 8,  # Retrieve more context
           "temperature": 0.2,  # Lower temperature for factual responses
           "max_tokens": 300,  # Longer responses
       }
   }

   # Create a pipeline from a profile
   def create_pipeline_from_profile(profile_name, api_key):
       if profile_name not in config_profiles:
           raise ValueError(f"Profile '{profile_name}' not found")

       config = config_profiles[profile_name].copy()
       config["api_key"] = api_key

       return DefaultRAGPipeline(**config)

   # Usage
   pipeline = create_pipeline_from_profile("high_accuracy", "your-api-key")

Extending with Custom Adapters
----------------------------

For advanced use cases, you can create your own adapters by implementing the core interfaces of RAGFlow. This allows you to integrate new vector stores, embedding models, LLMs, chunking strategies, or retrieval approaches not included in the default implementation.

See the :doc:`../advanced/custom_adapters` guide for detailed instructions and examples for implementing your own adapters.

Examples
-------

For complete working examples of these customization approaches, see:

* ``ragflow/examples/customization_examples.py``
* ``ragflow/examples/configuration_examples.py``
* ``ragflow/examples/custom_adapters_example.py``
