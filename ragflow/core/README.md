# RAGFlow Core Architecture

## Overview

The core of RAGFlow is built around a clean, modular architecture based on interfaces. This design enables flexibility, extensibility, and testability while providing a simple, intuitive API for developers.

## Key Components

### Interfaces

RAGFlow's core interfaces define the contracts that concrete implementations must follow:

1. **ChunkingStrategyInterface**: Responsible for splitting documents into smaller chunks for more effective retrieval and processing.
   - Methods: `split_documents()`, `split_text()`

2. **EmbeddingModelInterface**: Converts text into vector embeddings that capture semantic meaning.
   - Methods: `embed_query()`, `embed_documents()`

3. **VectorStoreInterface**: Stores and retrieves vector embeddings, supporting similarity search operations.
   - Methods: `add_documents()`, `add_texts()`, `similarity_search()`, `similarity_search_by_vector()`

4. **RetrievalStrategyInterface**: Retrieves relevant documents based on a query.
   - Methods: `get_relevant_documents()`, `get_relevant_documents_with_scores()`

5. **LLMInterface**: Generates text responses based on prompts and context.
   - Methods: `generate()`, `generate_with_context()`

### Document Class

The `Document` class represents the fundamental unit of text in RAGFlow, containing:
- `page_content`: The actual text content
- `metadata`: Associated metadata such as source, timestamps, etc.

### RAGPipeline

The `RAGPipeline` class orchestrates the entire RAG workflow by composing the above interfaces:

1. **Initialization**: The pipeline is initialized with concrete implementations of each interface.
2. **Document Processing**:
   - Documents are split into chunks using the `chunking_strategy`
   - Chunks are embedded and stored in the `vector_store`
3. **Query Processing**:
   - Query is passed to the `retrieval_strategy` to get relevant documents
   - Retrieved documents and the original query are sent to the `llm` for answer generation

## Flow Diagram

```
┌─────────────┐          ┌─────────────────┐          ┌──────────────────┐
│  Documents  │──────────►  ChunkingStrategy  │──────────►  VectorStore  │
└─────────────┘          └─────────────────┘          └──────────────────┘
                                                              │
                                                              │
┌─────────────┐          ┌─────────────────┐          ┌────────────────────┐
│   Answer    │◄─────────┤       LLM       │◄─────────┤  RetrievalStrategy │
└─────────────┘          └─────────────────┘          └────────────────────┘
                                  ▲                           ▲
                                  │                           │
                              ┌─────────────┐                 │
                              │    Query    │─────────────────┘
                              └─────────────┘
```

## Benefits of This Architecture

1. **Modularity**: Each component can be developed, tested, and replaced independently.
2. **Extensibility**: New implementations of any interface can be added without changing the pipeline.
3. **Testability**: Mock implementations can be used for testing each component in isolation.
4. **Clarity**: The clear separation of concerns makes the system easier to understand and maintain.
5. **Configurability**: Different configurations can be created by swapping implementations.

## Usage Example

```python
# Initialize components
chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=1000, chunk_overlap=200)
embeddings = SentenceTransformersAdapter(model_name="all-MiniLM-L6-v2")
vector_store = ChromaDBAdapter(persist_directory="./data/chroma_db")
retriever = SimilarityRetrievalStrategy(vector_store=vector_store, k=4)
llm = GeminiAdapter(api_key=os.environ["GEMINI_API_KEY"])

# Create the pipeline
pipeline = RAGPipeline(
    chunking_strategy=chunker,
    embedding_model=embeddings,
    vector_store=vector_store,
    retrieval_strategy=retriever,
    llm=llm
)

# Add documents
documents = [Document(page_content="...", metadata={"source": "example.txt"})]
pipeline.add_documents(documents)

# Query the pipeline
answer = pipeline.query("What is RAGFlow?")
print(answer)

# Get answer with sources
result = pipeline.query_with_sources("What is RAGFlow?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```
