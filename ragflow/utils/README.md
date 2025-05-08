# RAGFlow Utilities

This directory contains various utilities and guides for the RAGFlow library.

## Configuration Guide

The `configuration_guide.md` document provides a comprehensive overview of RAGFlow's configuration mechanisms, including:

- Configuring individual components (adapters)
- Using the DefaultRAGPipeline's configuration options
- Handling sensitive data (API keys)
- Setting up advanced, custom configurations

RAGFlow's configuration philosophy focuses on:

1. **Simplicity**: Simple configuration through constructor parameters
2. **Flexibility**: Ability to customize every aspect of the pipeline
3. **Security**: Multiple options for safe handling of API keys
4. **Modularity**: Component-specific configuration that doesn't affect other parts

## Examples

For practical examples of configuration, see the `examples/configuration_examples.py` module, which demonstrates:

- Basic configuration (using defaults)
- Custom configuration of DefaultRAGPipeline
- Advanced configuration with custom components
- Using an existing database

## Configuration Best Practices

1. **API Keys**: Never hardcode API keys in source code. Use environment variables or dotenv files.
2. **Persistent Storage**: Use named collections and specify a persistent directory for ChromaDB to enable data reuse.
3. **Parameters**: Tune chunking parameters (`chunk_size` and `chunk_overlap`) based on your document characteristics.
4. **Component-Specific Kwargs**: Use the component-specific kwargs (`chunker_kwargs`, `embedder_kwargs`, etc.) for deeper customization.
