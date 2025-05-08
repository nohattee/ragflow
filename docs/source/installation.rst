============
Installation
============

RAGFlow can be installed using pip, the Python package manager. The library requires Python 3.8 or later.

Basic Installation
-----------------

To install the base RAGFlow package:

.. code-block:: bash

   pip install ragflow

This will install RAGFlow with its core dependencies, which include:

* langchain>=0.0.267
* sentence-transformers>=2.2.2

Installation with Optional Dependencies
--------------------------------------

RAGFlow supports several optional dependencies for different use cases:

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

For development, including linting and code formatting tools:

.. code-block:: bash

   pip install ragflow[dev]

Testing Dependencies
~~~~~~~~~~~~~~~~~~~

For running the test suite:

.. code-block:: bash

   pip install ragflow[test]

Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

For building the documentation:

.. code-block:: bash

   pip install ragflow[docs]

Installing from Source
---------------------

To install RAGFlow from source:

.. code-block:: bash

   git clone https://github.com/ragflow/ragflow.git
   cd ragflow
   pip install -e .

For development installation:

.. code-block:: bash

   pip install -e ".[dev,test,docs]"

Dependencies
-----------

Core Dependencies
~~~~~~~~~~~~~~~~

* Python >= 3.8
* langchain >= 0.0.267
* sentence-transformers >= 2.2.2

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~

Depending on your use case, you might need to install additional dependencies:

* For using ChromaDB:

  .. code-block:: bash

     pip install chromadb

* For using Gemini models:

  .. code-block:: bash

     pip install google-generativeai

* For other vector stores or LLMs, install the appropriate packages as needed.

Verify Installation
------------------

You can verify your installation by importing RAGFlow in Python:

.. code-block:: python

   import ragflow
   print(ragflow.__version__)

Next Steps
----------

After installation, check out the :doc:`quick_start` guide to get started with RAGFlow.
