"""Recursive character text splitter adapter for chunking documents.

This module provides an adapter that implements the ChunkingStrategyInterface
using Langchain's RecursiveCharacterTextSplitter.
"""

from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from ragflow.core.interfaces import ChunkingStrategyInterface, Document


class RecursiveCharacterTextSplitterAdapter(ChunkingStrategyInterface):
    """Adapter for RecursiveCharacterTextSplitter.

    This adapter uses Langchain's RecursiveCharacterTextSplitter to implement
    the ChunkingStrategyInterface.

    This adapter uses RecursiveCharacterTextSplitter to split documents into
    smaller chunks based on specified chunk size and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        r"""Initialize the RecursiveCharacterTextSplitter adapter.

        Args:
            chunk_size: The target size of each text chunk (in characters)
            chunk_overlap: The number of characters of overlap between chunks
            separators: List of separators to use for splitting, in order of priority.
                       If None, defaults to ["\n\n", "\n", " ", ""]
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self._text_splitter = self._create_splitter()

    def _create_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create a RecursiveCharacterTextSplitter instance.

        Returns:
            Configured RecursiveCharacterTextSplitter.
        """
        kwargs = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.separators is not None:
            kwargs["separators"] = self.separators

        return RecursiveCharacterTextSplitter(**kwargs)

    def split_text(self, text: str) -> List[str]:
        """Split a text string into smaller chunks.

        Args:
            text: The text to split into chunks.

        Returns:
            List of text chunks.
        """
        return self._text_splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into smaller chunks.

        Args:
            documents: The documents to split into chunks.

        Returns:
            List of document chunks.
        """
        # Convert RAGFlow Document objects to Langchain-compatible format
        langchain_docs = []
        for doc in documents:
            langchain_doc = LangchainDocument(
                page_content=doc.page_content, metadata=doc.metadata
            )
            langchain_docs.append(langchain_doc)

        # Split the documents using Langchain's splitter
        split_docs = self._text_splitter.split_documents(langchain_docs)

        # Convert back to RAGFlow Document objects
        result_docs = []
        for doc in split_docs:
            result_doc = Document(page_content=doc.page_content, metadata=doc.metadata)
            result_docs.append(result_doc)

        return result_docs
