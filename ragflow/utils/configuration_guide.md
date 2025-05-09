# RAGFlow Configuration Guide

This guide explains how to configure RAGFlow components and pipelines to meet your specific requirements.

## Configuration Philosophy

RAGFlow uses Python class initializers as the primary configuration mechanism. This provides several benefits:
- **Type Safety**: Parameters are validated at instantiation time
- **IDE Support**: Auto-completion and parameter hints in your IDE
- **Explicit Configuration**: Configuration is clearly visible in your code
- **Flexibility**: Easy to override defaults at any level

## Configuring Individual Components

### Vector Store: ChromaDBAdapter

```python
from ragflow.adapters.vector_stores.chromadb_adapter import ChromaDBAdapter

# With default settings (in-memory storage)
vector_store = ChromaDBAdapter()

# With persistent storage
vector_store = ChromaDBAdapter(
    collection_name="my_docs",  # Default: "ragflow"
    persist_directory="./my_chroma_db"  # Default: None (in-memory)
)

# With custom embedding function
vector_store = ChromaDBAdapter(
    collection_name="my_docs",
    persist_directory="./my_chroma_db",
    embedding_function=my_embedding_function  # Optional: custom embedding function
)
```

### Embedding Model: SentenceTransformersAdapter

```python
from ragflow.adapters.embedding_models.sentence_transformers_adapter import SentenceTransformersAdapter

# With default model
embedding_model = SentenceTransformersAdapter()  # Uses "all-MiniLM-L6-v2"

# With custom model
embedding_model = SentenceTransformersAdapter(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # Different model
)
```

### LLM: GeminiAdapter

```python
from ragflow.adapters.llms.gemini_adapter import GeminiAdapter

# Minimal configuration (API key required)
llm = GeminiAdapter(api_key="YOUR_GEMINI_API_KEY")

# Full configuration
llm = GeminiAdapter(
    api_key="YOUR_GEMINI_API_KEY",
    model_name="gemini-pro",  # Default: "gemini-pro"
    temperature=0.5,  # Default: 0.7
    max_tokens=300,   # Default: None (model decides)
    top_p=0.9,        # Default: 0.95
    top_k=20          # Default: 40
)

# Using environment variable for API key
import os
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
llm = GeminiAdapter(api_key=os.environ.get("GEMINI_API_KEY"))
```

### Chunking Strategy: RecursiveCharacterTextSplitterAdapter

```python
from ragflow.adapters.chunking_strategies.recursive_character_splitter_adapter import RecursiveCharacterTextSplitterAdapter

# With default settings
chunker = RecursiveCharacterTextSplitterAdapter()  # chunk_size=1000, chunk_overlap=200

# With custom settings
chunker = RecursiveCharacterTextSplitterAdapter(
    chunk_size=500,   # Default: 1000
    chunk_overlap=50, # Default: 200
    separators=["\n\n", "\n", ".", " ", ""]  # Default: ["\n\n", "\n", " ", ""]
)
```

### Retrieval Strategy: SimpleSimilarityRetriever

```python
from ragflow.adapters.retrieval_strategies.simple_similarity_retriever import SimpleSimilarityRetriever

# Minimal configuration (vector store required)
retriever = SimpleSimilarityRetriever(vector_store=my_vector_store)

# With custom k value
retriever = SimpleSimilarityRetriever(
    vector_store=my_vector_store,
    k=10  # Default: 4
)
```

## Configuring the DefaultRAGPipeline

The `DefaultRAGPipeline` allows you to configure all components through a single interface:

```python
from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

# Minimal configuration
pipeline = DefaultRAGPipeline(api_key="YOUR_GEMINI_API_KEY")

# Standard configuration
pipeline = DefaultRAGPipeline(
    persist_directory="./",  # Default: "./"
    api_key="YOUR_GEMINI_API_KEY",       # Required (or via GEMINI_API_KEY env var)
    embedding_model_name="all-MiniLM-L6-v2",  # Default: "all-MiniLM-L6-v2"
    chunk_size=800,                      # Default: 1000
    chunk_overlap=100,                   # Default: 200
    retrieval_k=5                        # Default: 4
)

# Using environment variable for API key
import os
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
pipeline = DefaultRAGPipeline()  # Will use GEMINI_API_KEY from environment
```

### From Existing Database

If you have a pre-populated ChromaDB database, you can initialize the pipeline using it:

```python
pipeline = DefaultRAGPipeline.from_existing_db(
    persist_directory="./my_existing_db",
    api_key="YOUR_GEMINI_API_KEY"
)
```

## Handling Sensitive Data

For sensitive data like API keys, we recommend these approaches (in order of preference):

1. **Environment Variables**: Store API keys as environment variables
   ```python
   import os
   os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
   pipeline = DefaultRAGPipeline()  # Uses GEMINI_API_KEY from env
   ```

2. **Dotenv Files**: Use `.env` files with the python-dotenv package
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Loads variables from .env file
   pipeline = DefaultRAGPipeline()  # Uses GEMINI_API_KEY from .env
   ```

3. **Secret Management Services**: For production, use a service like AWS Secrets Manager
   ```python
   import boto3
   client = boto3.client('secretsmanager')
   response = client.get_secret_value(SecretId='GeminiApiKey')
   api_key = response['SecretString']
   pipeline = DefaultRAGPipeline(api_key=api_key)
   ```

Never hardcode API keys in your source code, especially if it's version controlled.

## Advanced Configuration

For advanced scenarios where you need full control, you can:

1. Create each component with custom configuration
2. Manually initialize the RAGPipeline with these components

```python
from ragflow.core.pipeline import RAGPipeline
from ragflow.adapters.chunking_strategies.recursive_character_splitter_adapter import RecursiveCharacterTextSplitterAdapter
from ragflow.adapters.embedding_models.sentence_transformers_adapter import SentenceTransformersAdapter
from ragflow.adapters.vector_stores.chromadb_adapter import ChromaDBAdapter
from ragflow.adapters.retrieval_strategies.simple_similarity_retriever import SimpleSimilarityRetriever
from ragflow.adapters.llms.gemini_adapter import GeminiAdapter

# Configure each component
chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=500, chunk_overlap=50)
embedder = SentenceTransformersAdapter(model_name="paraphrase-multilingual-MiniLM-L12-v2")
vector_store = ChromaDBAdapter(collection_name="custom_collection", persist_directory="./custom_db")
retriever = SimpleSimilarityRetriever(vector_store=vector_store, k=8)
llm = GeminiAdapter(api_key=os.environ.get("GEMINI_API_KEY"), temperature=0.3)

# Create pipeline with custom components
pipeline = RAGPipeline(
    chunking_strategy=chunker,
    embedding_model=embedder,
    vector_store=vector_store,
    retrieval_strategy=retriever,
    llm=llm
)
```

This approach gives you maximum flexibility while still leveraging the RAGFlow architecture.
