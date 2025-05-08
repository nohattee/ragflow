"""
ChromaDB adapter for VectorStoreInterface.

This module provides an implementation of VectorStoreInterface using ChromaDB
as the underlying vector store.
"""

from typing import Any, Dict, List, Optional

import chromadb

from ragflow.core.interfaces import Document, VectorStoreInterface


class ChromaDBAdapter(VectorStoreInterface):
    """
    Adapter for ChromaDB that implements the VectorStoreInterface.

    This adapter uses ChromaDB to store and query vector embeddings. It supports
    both in-memory and persistent storage options.
    """

    def __init__(
        self,
        collection_name: str = "ragflow",
        persist_directory: Optional[str] = None,
        embedding_function=None,
    ):
        """
        Initialize the ChromaDB adapter.

        Args:
            collection_name: Name of the ChromaDB collection to use
            persist_directory: Directory to persist the ChromaDB data. If None,
                              uses in-memory storage
            embedding_function: Function used to embed documents. If None,
                               uses the embedding function from the collection
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Create or get collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
        """
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        ids = [str(i) for i in range(len(documents))]

        # If documents already have id in metadata, use it
        for i, m in enumerate(metadata):
            if "id" in m:
                ids[i] = str(m["id"])

        self.collection.add(documents=texts, metadatas=metadata, ids=ids)

    def add_texts(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add text strings with optional metadata to the vector store.

        Args:
            texts: List of text strings to add
            metadata: Optional list of metadata dicts, one for each text
        """
        if metadata is None:
            metadata = [{} for _ in texts]

        ids = [str(i) for i in range(len(texts))]

        # If metadata contain id, use it
        for i, m in enumerate(metadata):
            if "id" in m:
                ids[i] = str(m["id"])

        self.collection.add(documents=texts, metadatas=metadata, ids=ids)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search using a query string.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of documents most similar to the query
        """
        results = self.collection.query(
            query_texts=[query], n_results=k, include=["documents", "metadatas"]
        )

        documents = []
        for i in range(len(results["documents"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            documents.append(doc)

        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Document]:
        """
        Perform a similarity search using a vector embedding.

        Args:
            embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of documents most similar to the query embedding
        """
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas"],
        )

        documents = []
        for i in range(len(results["documents"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            documents.append(doc)

        return documents
