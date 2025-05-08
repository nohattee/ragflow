Welcome to RAGFlow Documentation
===============================

RAGFlow is a high-level, flexible, and extensible framework built on top of the powerful Langchain library, designed to streamline the development of Retrieval Augmented Generation (RAG) applications.

Features
--------

* **Simplified Setup**: Enable developers to get a RAG system up and running quickly with minimal code.
* **Opinionated Best Practices**: Embed sensible defaults and a well-structured pipeline reflecting common best practices in RAG development.
* **Flexible Architecture**: Allow for customization of individual components or the entire pipeline to suit specific needs.
* **Clean Interface Design**: Use a modular, interface-driven architecture that promotes extensibility and testability.

Getting Started
--------------

To get started with RAGFlow, check out the :doc:`installation` guide and the :doc:`quick_start` tutorial.

.. code-block:: python

   from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

   # Create a default RAG pipeline with minimal configuration
   pipeline = DefaultRAGPipeline(
       api_key="your-api-key",
   )

   # Add some documents
   pipeline.add_texts([
       "RAGFlow is a Python library for Retrieval Augmented Generation.",
       "It provides a high-level, flexible, and extensible framework.",
   ])

   # Query the pipeline
   answer = pipeline.query("What is RAGFlow?")
   print(answer)

Table of Contents
----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quick_start
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/default_pipeline
   user_guide/customization
   user_guide/component_configuration

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_rag
   tutorials/document_loading
   tutorials/custom_pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/pipelines
   api/adapters
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   advanced/custom_adapters
   advanced/extending_ragflow
   advanced/performance_tuning

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/architecture
   development/roadmap

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
