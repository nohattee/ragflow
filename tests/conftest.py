"""Configuration file for pytest.

This file defines shared fixtures and configurations that can be used
across all test modules in the project.
"""

# Import the SQLite patch early to handle ChromaDB's SQLite requirements
try:
    import os
    import sys

    import pysqlite3

    # Save the original sqlite3 module
    sys.modules["sqlite3_original"] = sys.modules.get("sqlite3")

    # Replace sqlite3 with pysqlite3
    sys.modules["sqlite3"] = pysqlite3

    # Set environment variable to avoid SQLite version checks in ChromaDB
    os.environ["CHROMA_DANGEROUS_DISABLE_SQLITE_VERSION_CHECK"] = "1"

    print(f"Patched sqlite3 with pysqlite3 {pysqlite3.sqlite_version}")
except ImportError:
    print("Warning: Could not import db_patch. SQLite version may cause issues.")

from unittest.mock import Mock

import pytest
from ragflow.core.interfaces import Document


@pytest.fixture
def sample_documents():
    """Create a list of sample documents for testing."""
    return [
        Document(page_content="Document 1 content", metadata={"source": "doc1.txt"}),
        Document(page_content="Document 2 content", metadata={"source": "doc2.txt"}),
        Document(page_content="Document 3 content", metadata={"source": "doc3.txt"}),
    ]


@pytest.fixture
def sample_texts():
    """Create a list of sample text strings for testing."""
    return ["Document 1 content", "Document 2 content", "Document 3 content"]


@pytest.fixture
def sample_metadata():
    """Create a list of sample metadata dictionaries for testing."""
    return [{"source": "doc1.txt"}, {"source": "doc2.txt"}, {"source": "doc3.txt"}]


@pytest.fixture
def sample_embeddings():
    """Create a list of sample embeddings for testing."""
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model that implements EmbeddingModelInterface."""
    mock = Mock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    mock.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    return mock
