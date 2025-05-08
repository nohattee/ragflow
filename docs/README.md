# RAGFlow Documentation

This directory contains the documentation for RAGFlow, built using Sphinx.

## Structure

- `source/`: Contains the source files for the documentation
- `build/`: Will contain the generated documentation after building

## Building the Documentation

To build the documentation, follow these steps:

1. Install the documentation dependencies:
   ```bash
   pip install ragflow[docs]
   ```

2. Build the HTML documentation:
   ```bash
   cd docs
   sphinx-build -b html source build/html
   ```

3. View the documentation by opening `build/html/index.html` in a web browser.

## Documentation Sections

The documentation is organized into the following sections:

- **Getting Started**: Installation instructions, quick start guide, and core concepts
- **User Guide**: Detailed information on using RAGFlow components and customizing pipelines
- **Tutorials**: Step-by-step guides for common use cases
- **API Reference**: Auto-generated documentation for RAGFlow's classes and methods
- **Advanced**: Advanced topics like creating custom adapters and extending RAGFlow
- **Development**: Information for contributors and developers

## Contributing to the Documentation

If you're contributing to the documentation, please:

1. Follow the reStructuredText syntax for .rst files
2. Test your changes by building the documentation locally
3. Keep code examples up-to-date with the current API
4. Run spell check on your contributions
5. Ensure all code examples are runnable and produce the expected output

## Hosting

The documentation is designed to be hosted on Read the Docs, which automatically builds and hosts the documentation from the repository.
