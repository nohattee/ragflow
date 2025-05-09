"""
Utility functions for RAGFlow.

This module provides helper functions for common operations when working with RAGFlow,
designed to reduce boilerplate code and improve developer experience.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.errors import ConfigurationError, DocumentProcessingError
from ..core.interfaces import Document
from ..pipelines.default_rag_pipeline import DefaultRAGPipeline


def load_text_files(
    directory_path: str,
    file_pattern: str = "*.txt",
    encoding: str = "utf-8",
    recursive: bool = False,
    metadata_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> List[Document]:
    """
    Load text files from a directory into Document objects.

    Args:
        directory_path: Path to the directory containing text files
        file_pattern: Glob pattern to match files (default: "*.txt")
        encoding: File encoding to use (default: "utf-8")
        recursive: Whether to search subdirectories recursively (default: False)
        metadata_fn: Optional function to generate metadata from a file path
            Function signature should be: (filepath: str) -> Dict[str, Any]

    Returns:
        List of Document objects, one for each file

    Raises:
        DocumentProcessingError: If there's an issue reading or processing files

    Examples:
        Basic usage with default settings:
        ```python
        docs = load_text_files("./data")
        pipeline.add_documents(docs)
        ```

        With custom metadata function:
        ```python
        def create_metadata(filepath):
            filename = os.path.basename(filepath)
            return {"filename": filename, "date_added": "2025-05-07"}


        docs = load_text_files("./data", metadata_fn=create_metadata)
        ```
    """
    try:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            raise DocumentProcessingError(f"Directory does not exist: {directory_path}")

        # Default metadata function uses the filename as a title
        if metadata_fn is None:

            def metadata_fn(filepath):
                return {"source": os.path.basename(filepath)}

        documents = []
        glob_pattern = f"**/{file_pattern}" if recursive else file_pattern

        for file_path in path.glob(glob_pattern):
            if file_path.is_file():
                try:
                    with open(file_path, encoding=encoding) as file:
                        content = file.read()
                    metadata = metadata_fn(str(file_path))
                    documents.append(Document(page_content=content, metadata=metadata))
                except Exception as e:
                    raise DocumentProcessingError(
                        f"Error reading file {file_path}: {str(e)}"
                    )

        if not documents:
            raise DocumentProcessingError(
                f"No files matching pattern '{file_pattern}' found in {directory_path}"
            )

        return documents
    except Exception as e:
        if isinstance(e, DocumentProcessingError):
            raise
        raise DocumentProcessingError(f"Error loading text files: {str(e)}")


def create_pipeline_from_env(
    collection_name: str = "ragflow",
    persist_directory: str = "./",
    env_var: str = "GEMINI_API_KEY",
    **kwargs,
) -> DefaultRAGPipeline:
    """
    Create a DefaultRAGPipeline using an API key from an environment variable.

    This is a convenience function for the common case of creating a pipeline
    with an API key stored in an environment variable.

    Args:
        collection_name: Name of the ChromaDB collection to use
        persist_directory: Directory where ChromaDB will store its data
        env_var: Name of the environment variable containing the API key
        **kwargs: Additional parameters to pass to DefaultRAGPipeline constructor

    Returns:
        A configured DefaultRAGPipeline instance

    Raises:
        ConfigurationError: If the environment variable is not set

    Examples:
        ```python
        # Assuming GEMINI_API_KEY is set in the environment
        pipeline = create_pipeline_from_env()
        pipeline.add_texts(["Document 1", "Document 2"])
        answer = pipeline.query("What is in the documents?")
        ```
    """
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ConfigurationError(
            f"Environment variable {env_var} is not set. "
            f"Set the variable or provide api_key directly to DefaultRAGPipeline."
        )

    return DefaultRAGPipeline(
        api_key=api_key,
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs,
    )


def format_sources(
    result: Dict[str, Any], include_content: bool = False, max_chars: int = 100
) -> str:
    r"""
    Format the sources from a query_with_sources result into a readable string.

    Args:
        result: The result dictionary from query_with_sources()
        include_content: Whether to include the content of each source (default: False)
        max_chars: Maximum characters to show for each source content (default: 100)

    Returns:
        A formatted string with source information

    Examples:
        ```python
        result = pipeline.query_with_sources("What is RAGFlow?")
        print(result["answer"])
        print("\nSources:")
        print(format_sources(result))
        ```
    """
    if "sources" not in result or not result["sources"]:
        return "No sources available"

    sources = result["sources"]
    formatted = []

    for i, doc in enumerate(sources, 1):
        source_info = f"Source {i}:"

        # Add metadata if available
        if doc.metadata:
            meta_items = []
            for key, value in doc.metadata.items():
                meta_items.append(f"{key}: {value}")
            source_info += " " + ", ".join(meta_items)

        formatted.append(source_info)

        # Add content if requested
        if include_content:
            content = doc.page_content
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            formatted.append(f"  Content: {content}")

    return "\n".join(formatted)
