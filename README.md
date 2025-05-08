# RAGFlow

A high-level framework for Retrieval Augmented Generation (RAG) applications, built on top of Langchain.

## Features

- Simple, intuitive API for building RAG pipelines
- Modular design with clean interfaces
- Pre-configured components with sensible defaults
- Extensible architecture for custom components
- Comprehensive documentation and examples

## Installation

```bash
pip install ragflow
```

## Quick Start

```python
from ragflow import DefaultRAGPipeline

# Create a pipeline with default settings
pipeline = DefaultRAGPipeline.from_defaults(
    persist_directory="./data/chroma_db",
    api_key="your-gemini-api-key"  # or set GEMINI_API_KEY env var
)

# Add documents
pipeline.add_documents([
    {"text": "RAGFlow is a framework for building RAG applications."},
    {"text": "Retrieval Augmented Generation enhances LLMs with external knowledge."},
])

# Query the pipeline
answer = pipeline.query("What is RAGFlow?")
print(answer)
```

## Documentation

Comprehensive documentation is available at [docs.ragflow.ai](https://docs.ragflow.ai) including:

- API Reference
- Tutorials
- User Guides
- Advanced Customization
- Development Guidelines

### Building Documentation Locally

To build the documentation locally:

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Generate HTML documentation
cd docs
sphinx-build -b html source build/html

# View documentation
# Open build/html/index.html in your browser
```

## Development

### Code Quality Standards

This project uses a comprehensive set of tools to maintain high code quality:

1. **Ruff**: For linting and formatting Python code
2. **Pre-commit hooks**: To automate code quality checks
3. **Bandit**: For security vulnerability scanning

### Setup Development Environment

1. Clone the repository
2. Install development dependencies:

```bash
pip install -e ".[dev,test]"
pre-commit install
```

### Code Style Guide

We follow these principles:

- PEP 8 for Python code style
- Google style for docstrings
- Maximum line length of 88 characters
- Type hints for all function signatures
- Clear, descriptive variable names
- Comprehensive docstrings for all public APIs

### Running Tests

```bash
pytest
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. The hooks check for:

- Code formatting (Ruff)
- Import ordering (Ruff)
- Linting errors (Ruff)
- Security issues (Bandit)
- Missing docstrings (Ruff)
- File formatting issues (trailing whitespace, etc.)

You can manually run the hooks with:

```bash
pre-commit run --all-files
```

## Project Structure

```
ragflow/
├── adapters/            # Component implementations
│   ├── chunking_strategies/
│   ├── embedding_models/
│   ├── llms/
│   ├── retrieval_strategies/
│   └── vector_stores/
├── core/                # Core interfaces and abstractions
├── pipelines/           # Pre-configured pipelines
└── utils/               # Helper utilities
```

## License

MIT
