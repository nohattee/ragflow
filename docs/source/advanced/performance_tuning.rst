===================
Performance Tuning
===================

This guide provides detailed strategies and techniques for optimizing the performance of RAG systems built with RAGFlow. Performance in RAG systems can be measured in terms of:

1. **Retrieval Quality**: How well the system finds relevant documents
2. **Generation Quality**: How accurately and coherently the system answers questions
3. **Latency**: How quickly the system responds to queries
4. **Throughput**: How many queries the system can handle simultaneously
5. **Resource Usage**: How efficiently the system uses memory, CPU, and storage

We'll explore ways to optimize each component of your RAG pipeline to achieve the best balance of these performance aspects for your specific use case.

Optimizing Document Chunking
---------------------------

Chunking directly impacts both retrieval quality and resource efficiency. The right chunking strategy can make or break your RAG system's performance.

Chunk Size Selection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ragflow.adapters.chunking_strategies import RecursiveCharacterTextSplitterAdapter
    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

    # Smaller chunks for more granular retrieval
    small_chunk_pipeline = DefaultRAGPipeline(
        chunk_size=300,
        chunk_overlap=30,
        # Other parameters...
    )

    # Medium chunks for balance (often the sweet spot)
    medium_chunk_pipeline = DefaultRAGPipeline(
        chunk_size=800,
        chunk_overlap=150,
        # Other parameters...
    )

    # Larger chunks for more context per chunk
    large_chunk_pipeline = DefaultRAGPipeline(
        chunk_size=1500,
        chunk_overlap=250,
        # Other parameters...
    )

**Best Practices for Chunk Size:**

- **Start with 500-1000 characters** as a baseline chunk size
- For **short, factual content** (e.g., product descriptions), use **smaller chunks** (200-500 chars)
- For **narrative or contextual content** (e.g., articles, documentation), use **larger chunks** (800-1500 chars)
- Always include **chunk overlap** (typically 10-20% of chunk size) to maintain context across chunk boundaries

Custom Chunking Strategies
~~~~~~~~~~~~~~~~~~~~~~~~

For more specialized needs, implement a custom chunking strategy:

.. code-block:: python

    from ragflow.core.interfaces import ChunkingStrategyInterface, Document
    from typing import List
    import re

    class SectionBasedChunker(ChunkingStrategyInterface):
        """Split documents based on section headers (e.g., Markdown headers)."""

        def __init__(self, min_size: int = 200, default_separator: str = "\n\n"):
            self.min_size = min_size
            self.default_separator = default_separator
            self.section_pattern = re.compile(r'^#{1,6}\s+(.+?)$', re.MULTILINE)

        def split_text(self, text: str) -> List[str]:
            # Find all section headers
            sections = []
            last_pos = 0

            for match in self.section_pattern.finditer(text):
                if match.start() - last_pos > self.min_size:
                    # Add text since last section header as a chunk
                    sections.append(text[last_pos:match.start()].strip())
                last_pos = match.start()

            # Add final section
            if len(text) - last_pos > self.min_size:
                sections.append(text[last_pos:].strip())

            return sections if sections else [text]

        def split_documents(self, documents: List[Document]) -> List[Document]:
            chunked_docs = []

            for doc in documents:
                chunks = self.split_text(doc.page_content)

                for i, chunk in enumerate(chunks):
                    if not chunk:  # Skip empty chunks
                        continue

                    # Copy metadata and add chunk info
                    metadata = doc.metadata.copy()
                    metadata["chunk"] = i

                    chunked_docs.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))

            return chunked_docs

Use this custom chunker in your pipeline:

.. code-block:: python

    from ragflow.core.pipeline import RAGPipeline

    # Create other components
    # ...

    # Use custom chunker
    section_chunker = SectionBasedChunker(min_size=300)

    pipeline = RAGPipeline(
        chunking_strategy=section_chunker,
        embedding_model=embedder,
        vector_store=vector_store,
        retrieval_strategy=retriever,
        llm=llm
    )

Optimizing Embedding Models
-------------------------

The embedding model affects retrieval quality, latency, and resource usage.

Model Selection Trade-offs
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ragflow.adapters.embedding_models import SentenceTransformersAdapter

    # Lightweight, fast model (384 dimensions) - good for quick testing or small datasets
    fast_embedder = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")

    # Balanced model (768 dimensions) - good balance of quality and performance
    balanced_embedder = SentenceTransformersAdapter(model_name="all-mpnet-base-v2")

    # High accuracy model (768+ dimensions) - best quality but slower and more resource-intensive
    accurate_embedder = SentenceTransformersAdapter(model_name="text-embedding-ada-002")  # OpenAI

**Benchmark different models** on your specific data and queries:

.. code-block:: python

    import time
    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

    def benchmark_embedding_model(model_name, test_queries):
        # Initialize pipeline with specific embedding model
        pipeline = DefaultRAGPipeline(
            embedding_model_name=model_name,
            # Other parameters...
        )

        # Add same test documents to each pipeline
        pipeline.add_texts(test_documents)

        # Measure query time
        start_time = time.time()
        results = [pipeline.query(q) for q in test_queries]
        end_time = time.time()

        return {
            "model": model_name,
            "avg_query_time": (end_time - start_time) / len(test_queries),
            "results": results
        }

    # Test different models
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2"]
    test_queries = ["What is RAG?", "How do embeddings work?", "Explain chunking strategies"]

    benchmarks = [benchmark_embedding_model(model, test_queries) for model in models]

    # Compare results
    for b in benchmarks:
        print(f"Model: {b['model']}, Avg Query Time: {b['avg_query_time']:.4f}s")

Batching for Performance
~~~~~~~~~~~~~~~~~~~~~

When embedding large document sets, use batching:

.. code-block:: python

    from tqdm import tqdm

    def add_documents_in_batches(pipeline, documents, batch_size=32):
        """Add documents to pipeline in batches for better performance."""
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(documents), batch_size), total=total_batches):
            batch = documents[i:i + batch_size]
            pipeline.add_documents(batch)
            # Optional: add a small delay to avoid overwhelming the system
            # time.sleep(0.1)

    # Load a large document set
    all_documents = load_text_files("./data")

    # Add in batches
    add_documents_in_batches(pipeline, all_documents, batch_size=64)

Optimizing Vector Stores
----------------------

The vector store impacts retrieval quality, speed, and scalability.

Choosing the Right Vector Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RAGFlow supports different vector stores through adapters. Each has different performance characteristics:

.. code-block:: python

    from ragflow.adapters.vector_stores import ChromaDBAdapter, FaissAdapter
    from ragflow.adapters.embedding_models import SentenceTransformersAdapter

    # Create embedding model
    embedder = SentenceTransformersAdapter()

    # ChromaDB - good for persistence and metadata filtering
    chroma_store = ChromaDBAdapter(
        collection_name="my_documents",
        persist_directory="./",
        embedding_function=embedder
    )

    # FAISS - excellent for fast similarity search on larger datasets
    faiss_store = FaissAdapter(
        embedding_function=embedder,
        index_type="Flat"  # Can be "Flat" (exact) or "IVF" (approximate) or "HNSW" (hierarchical)
    )

**Vector Store Selection Guidelines:**

- For **small to medium datasets** (< 100k chunks), **ChromaDB** is a good default choice
- For **large datasets** (100k+ chunks), consider **FAISS** with approximate nearest neighbor algorithms
- For **heavy filtering on metadata**, prefer **ChromaDB** or **Weaviate**
- For **production deployments**, consider managed options like **Pinecone** or **Qdrant**

Index Parameters and ANN Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large-scale vector stores, tune ANN (Approximate Nearest Neighbor) parameters:

.. code-block:: python

    # FAISS with IVF (Inverted File) for better scaling
    # The nlist parameter controls the number of centroids (partitions)
    # Higher values = more partitions = faster search but potentially lower accuracy
    faiss_ivf_store = FaissAdapter(
        embedding_function=embedder,
        index_type="IVF",
        nlist=100,  # Number of partitions (rule of thumb: sqrt(n) where n is total vectors)
        nprobe=10   # Number of partitions to search (higher = more accurate but slower)
    )

    # FAISS with HNSW (Hierarchical Navigable Small World)
    # Good for high-dimensional vectors
    faiss_hnsw_store = FaissAdapter(
        embedding_function=embedder,
        index_type="HNSW",
        M=16,       # Number of connections per layer (higher = more accurate but more memory)
        ef_construction=200  # Search depth during construction (higher = more accurate but slower build)
    )

Optimizing Retrieval Strategies
----------------------------

The retrieval strategy affects both quality and speed of retrievals.

Tuning Retrieval Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ragflow.adapters.retrieval_strategies import SimpleSimilarityRetriever
    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

    # Retrieve more documents for complex queries
    pipeline_more_docs = DefaultRAGPipeline(
        retrieval_k=8,  # Retrieve 8 documents per query
        # Other parameters...
    )

    # Use a custom retriever with more control
    custom_retriever = SimpleSimilarityRetriever(
        vector_store=vector_store,
        k=5,
        score_threshold=0.7  # Only return documents with similarity above threshold
    )

Implementing Advanced Retrieval Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For better retrieval quality, implement a hybrid retrieval strategy:

.. code-block:: python

    from ragflow.core.interfaces import RetrievalStrategyInterface, Document
    from typing import List, Tuple
    import re

    class HybridRetriever(RetrievalStrategyInterface):
        """Combines semantic search with keyword matching for better retrieval."""

        def __init__(self, vector_store, k=4, keyword_boost=0.2):
            self.vector_store = vector_store
            self.k = k
            self.keyword_boost = keyword_boost

        def _extract_keywords(self, query):
            """Extract important keywords from query."""
            # Simple implementation - could be improved with NLP techniques
            stop_words = {"the", "a", "an", "in", "on", "at", "is", "are", "and", "or", "to", "of"}
            words = re.findall(r'\b\w+\b', query.lower())
            return {w for w in words if w not in stop_words and len(w) > 3}

        def _keyword_score(self, doc, keywords):
            """Calculate keyword match score for a document."""
            content = doc.page_content.lower()
            matches = sum(1 for kw in keywords if kw in content)
            return matches / max(1, len(keywords))

        def get_relevant_documents(self, query: str) -> List[Document]:
            # Get vector similarity results
            vector_results = self.vector_store.similarity_search(query, k=self.k*2)

            # Extract keywords from query
            keywords = self._extract_keywords(query)

            # Score documents by combining vector similarity with keyword matching
            scored_docs = []
            for i, doc in enumerate(vector_results):
                # Vector score (approximated by position, best=1.0, worst=0.0)
                vector_score = 1.0 - (i / len(vector_results))

                # Keyword score
                kw_score = self._keyword_score(doc, keywords)

                # Combined score
                combined_score = vector_score + (self.keyword_boost * kw_score)
                scored_docs.append((doc, combined_score))

            # Sort by combined score and take top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:self.k]]

        def get_relevant_documents_with_scores(self, query: str) -> List[Tuple[Document, float]]:
            # Similar to above but return the scores as well
            vector_results = self.vector_store.similarity_search(query, k=self.k*2)
            keywords = self._extract_keywords(query)

            scored_docs = []
            for i, doc in enumerate(vector_results):
                vector_score = 1.0 - (i / len(vector_results))
                kw_score = self._keyword_score(doc, keywords)
                combined_score = vector_score + (self.keyword_boost * kw_score)
                scored_docs.append((doc, combined_score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:self.k]

Optimizing LLM Generation
-----------------------

Tune LLM parameters to balance quality, latency, and cost.

Temperature and Sampling Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

    # More deterministic/factual responses
    factual_pipeline = DefaultRAGPipeline(
        temperature=0.1,  # Low temperature for more deterministic outputs
        top_p=0.9,        # Nucleus sampling parameter
        top_k=40,         # Limit to top 40 tokens when sampling
        # Other parameters...
    )

    # More creative/varied responses
    creative_pipeline = DefaultRAGPipeline(
        temperature=0.8,  # Higher temperature for more variety
        top_p=0.95,
        top_k=50,
        # Other parameters...
    )

Maximum Token Management
~~~~~~~~~~~~~~~~~~~~~

Control token usage for better performance and cost management:

.. code-block:: python

    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

    # Limit output length for cost/performance optimization
    pipeline = DefaultRAGPipeline(
        max_tokens=150,  # Maximum tokens to generate in response
        # Other parameters...
    )

**Tips for Token Usage:**

- For **summaries and short answers**, set max_tokens between 100-250
- For **detailed explanations**, use 300-800 max_tokens
- Consider the **pricing model** of your LLM provider when setting limits
- For systems with many queries, lower max_tokens to **reduce costs and latency**

Prompt Engineering
~~~~~~~~~~~~~~~

Customize the prompt template for better results:

.. code-block:: python

    from ragflow.adapters.llms import GeminiAdapter

    # Custom prompt template
    template = """
    Answer the following question based only on the provided context.
    If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

    Context:
    {context}

    Question: {question}

    Answer:
    """

    # Create LLM with custom prompt template
    llm = GeminiAdapter(
        api_key="your-api-key",
        prompt_template=template,
        temperature=0.3
    )

System-Level Optimizations
------------------------

Optimize the overall RAG system for production use cases.

Caching Strategies
~~~~~~~~~~~~~~~

Implement response caching to improve performance for repeated queries:

.. code-block:: python

    import hashlib
    import json
    from functools import lru_cache

    class CachedRAGPipeline:
        """A wrapper around RAGPipeline that adds response caching."""

        def __init__(self, pipeline, cache_size=100):
            self.pipeline = pipeline
            self.query_cache = {}
            self._cached_query = lru_cache(maxsize=cache_size)(self._query_impl)

        def _hash_query(self, query):
            """Create a hash for the query string."""
            return hashlib.md5(query.encode()).hexdigest()

        def _query_impl(self, query_hash):
            """Implementation of query that will be cached."""
            return self.query_cache[query_hash]

        def query(self, question):
            query_hash = self._hash_query(question)

            # Check if in cache
            if query_hash in self.query_cache:
                return self._cached_query(query_hash)

            # Not in cache, perform query
            result = self.pipeline.query(question)

            # Store in cache
            self.query_cache[query_hash] = result

            return result

    # Use the cached pipeline
    cached_pipeline = CachedRAGPipeline(pipeline)

    # This will be slow (first time)
    response1 = cached_pipeline.query("What is RAG?")

    # This will be fast (cached)
    response2 = cached_pipeline.query("What is RAG?")

Asynchronous Processing
~~~~~~~~~~~~~~~~~~

For high-throughput systems, implement asynchronous processing:

.. code-block:: python

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    class AsyncRAGPipeline:
        """A wrapper for asynchronous processing of RAG queries."""

        def __init__(self, pipeline, max_workers=4):
            self.pipeline = pipeline
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

        async def query_async(self, question):
            """Async version of query method."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.pipeline.query,
                question
            )

        async def process_batch(self, questions):
            """Process multiple questions concurrently."""
            tasks = [self.query_async(q) for q in questions]
            return await asyncio.gather(*tasks)

    # Usage:
    async def main():
        # Create and setup pipeline
        pipeline = DefaultRAGPipeline()
        pipeline.add_texts(documents)

        # Create async wrapper
        async_pipeline = AsyncRAGPipeline(pipeline)

        # Process multiple queries concurrently
        questions = [
            "What is RAG?",
            "How do embeddings work?",
            "Explain vector databases"
        ]

        results = await async_pipeline.process_batch(questions)

        for q, r in zip(questions, results):
            print(f"Q: {q}")
            print(f"A: {r}")
            print()

    # Run the async function
    asyncio.run(main())

Monitoring and Observability
~~~~~~~~~~~~~~~~~~~~~~~~~

Implement monitoring for your RAG system:

.. code-block:: python

    import time
    import logging

    class MonitoredRAGPipeline:
        """RAG pipeline with performance monitoring."""

        def __init__(self, pipeline):
            self.pipeline = pipeline
            self.logger = logging.getLogger("rag_monitor")

            # Configure logging
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            # Performance metrics
            self.query_times = []
            self.retrieval_counts = []

        def add_documents(self, documents):
            start = time.time()
            self.pipeline.add_documents(documents)
            elapsed = time.time() - start

            self.logger.info(f"Added {len(documents)} documents in {elapsed:.2f} seconds")

        def query(self, question):
            # Time the query
            start_time = time.time()

            # Get both answer and sources
            result = self.pipeline.query_with_sources(question)

            # Calculate elapsed time
            elapsed = time.time() - start_time
            self.query_times.append(elapsed)

            # Log retrieval stats
            num_sources = len(result["sources"])
            self.retrieval_counts.append(num_sources)

            self.logger.info(
                f"Query processed in {elapsed:.2f}s, retrieved {num_sources} documents, "
                f"avg query time: {sum(self.query_times)/len(self.query_times):.2f}s"
            )

            return result["answer"]

        def get_performance_stats(self):
            """Get performance statistics."""
            if not self.query_times:
                return {"error": "No queries processed yet"}

            return {
                "avg_query_time": sum(self.query_times) / len(self.query_times),
                "max_query_time": max(self.query_times),
                "min_query_time": min(self.query_times),
                "total_queries": len(self.query_times),
                "avg_docs_retrieved": sum(self.retrieval_counts) / len(self.retrieval_counts)
            }

    # Use the monitored pipeline
    monitored_pipeline = MonitoredRAGPipeline(pipeline)
    monitored_pipeline.add_documents(documents)

    answer = monitored_pipeline.query("What is RAG?")

    # Get performance stats
    stats = monitored_pipeline.get_performance_stats()
    print(json.dumps(stats, indent=2))

Comprehensive RAG System Performance Checklist
-------------------------------------------

Use this checklist when optimizing your RAG system:

1. **Embedding Model Selection**
   - [ ] Benchmark different embedding models on your dataset
   - [ ] Balance dimension size vs. quality vs. speed
   - [ ] Consider domain-specific models for specialized content

2. **Chunking Optimization**
   - [ ] Test different chunk sizes (300, 800, 1500 chars)
   - [ ] Adjust chunk overlap (10-20% of chunk size)
   - [ ] Consider custom chunking for structured documents

3. **Vector Store Tuning**
   - [ ] Select appropriate vector store for dataset size
   - [ ] Configure ANN parameters for large datasets
   - [ ] Implement efficient metadata filtering

4. **Retrieval Improvements**
   - [ ] Tune k value (number of retrieved documents)
   - [ ] Consider hybrid retrieval approaches
   - [ ] Implement re-ranking if needed

5. **LLM Optimization**
   - [ ] Adjust temperature for factual vs. creative responses
   - [ ] Set appropriate max_tokens limit
   - [ ] Optimize prompt templates
   - [ ] Consider model quantization for on-premise LLMs

6. **System-Level Optimizations**
   - [ ] Implement response caching
   - [ ] Use batching for document processing
   - [ ] Enable asynchronous query handling for high throughput
   - [ ] Add monitoring and logging

7. **Evaluation**
   - [ ] Measure retrieval precision and recall
   - [ ] Evaluate answer correctness and relevance
   - [ ] Monitor latency and resource usage
   - [ ] Track costs for API-based components

By following these guidelines and tuning techniques, you can significantly improve the performance, quality, and efficiency of your RAGFlow-based applications.
