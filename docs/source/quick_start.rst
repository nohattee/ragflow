==========
Quick Start
==========

This guide will help you get started with RAGFlow in just a few minutes. We'll cover the basic steps to create a simple Retrieval Augmented Generation (RAG) pipeline and use it to answer questions.

Prerequisites
------------

Before starting, make sure you have:

1. Installed RAGFlow (see :doc:`installation`)
2. An API key for an LLM (default is Gemini)

Creating Your First RAG Pipeline
-------------------------------

Let's start by creating a simple RAG pipeline using RAGFlow's default components:

.. code-block:: python

   from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

   # Create a default RAG pipeline
   # This uses ChromaDB, SentenceTransformers, and Gemini with sensible defaults
   pipeline = DefaultRAGPipeline(
       # API key can be provided directly or set as GEMINI_API_KEY environment variable
       api_key="your-api-key-here",
       # Optional: customize any parameters as needed
       temperature=0.5
   )

Adding Documents
--------------

Next, let's add some documents to the pipeline:

.. code-block:: python

   # Add documents as simple strings
   pipeline.add_texts([
       "RAGFlow is a Python library designed to streamline the development of "
       "Retrieval Augmented Generation (RAG) applications. It provides a high-level, "
       "flexible, and extensible framework.",

       "Retrieval Augmented Generation (RAG) is a technique that combines retrieval "
       "of relevant documents with text generation by large language models. This "
       "helps ground LLM outputs in factual information.",

       "RAGFlow offers sensible defaults and pre-configured pipelines to get developers "
       "started quickly, while also allowing for customization and extension as needs evolve."
   ])

Querying the Pipeline
-------------------

Now you can ask questions about the documents:

.. code-block:: python

   # Ask a simple question
   question = "What is RAGFlow and how does it relate to RAG?"
   answer = pipeline.query(question)

   print(f"Question: {question}")
   print(f"Answer: {answer}")

Getting Answers with Sources
--------------------------

RAGFlow can also provide the sources used to generate an answer:

.. code-block:: python

   from ragflow.utils.helpers import format_sources

   # Ask a question and get sources
   result = pipeline.query_with_sources(
       "What are the main benefits of using RAGFlow?"
   )

   print(f"Answer: {result['answer']}")
   print("\nSources:")
   print(format_sources(result, include_content=True))

Loading Documents from Files
--------------------------

RAGFlow includes helper functions to load documents from files:

.. code-block:: python

   from ragflow.utils.helpers import load_text_files

   # Load documents from files
   documents = load_text_files("./sample_data", recursive=True)
   pipeline.add_documents(documents)
   print(f"Added {len(documents)} documents from files")

Complete Example
--------------

Here's a complete example that you can run:

.. code-block:: python

   import os
   from dotenv import load_dotenv

   from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline
   from ragflow.utils.helpers import format_sources

   # Load environment variables from .env file if present
   load_dotenv()

   def quickstart():
       """Simple quickstart example for RAGFlow."""

       # Step 1: Create a default RAG pipeline
       pipeline = DefaultRAGPipeline(
           api_key=os.getenv("GEMINI_API_KEY"),
           temperature=0.5
       )

       # Step 2: Add some documents
       pipeline.add_texts([
           "RAGFlow is a Python library designed to streamline the development of "
           "Retrieval Augmented Generation (RAG) applications. It provides a high-level, "
           "flexible, and extensible framework.",

           "Retrieval Augmented Generation (RAG) is a technique that combines retrieval "
           "of relevant documents with text generation by large language models. This "
           "helps ground LLM outputs in factual information.",

           "RAGFlow offers sensible defaults and pre-configured pipelines to get developers "
           "started quickly, while also allowing for customization and extension as needs evolve."
       ])

       # Step 3: Query the pipeline
       question = "What is RAGFlow and how does it relate to RAG?"
       answer = pipeline.query(question)

       print(f"Question: {question}")
       print(f"Answer: {answer}")

       # Step 4: Advanced querying with sources
       print("\n--- With Sources ---")
       result = pipeline.query_with_sources(
           "What are the main benefits of using RAGFlow?"
       )

       print(f"Answer: {result['answer']}")
       print("\nSources:")
       print(format_sources(result, include_content=True))

       print("\nQuickstart complete! 🚀")


   if __name__ == "__main__":
       quickstart()

Next Steps
---------

Now that you've created your first RAG pipeline with RAGFlow, you can:

* Learn more about :doc:`concepts` in RAG and RAGFlow
* Explore :doc:`user_guide/customization` to adapt the pipeline to your needs
* Check out the :doc:`tutorials/basic_rag` for more detailed examples
* See the :doc:`api/pipelines` reference for more detailed information on the DefaultRAGPipeline
