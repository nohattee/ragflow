# Code Quality Guide

This document outlines the code quality practices and standards for the RAGFlow project.

## Overview

Maintaining high code quality is essential for:

- Ensuring maintainability for future development
- Making onboarding easier for new contributors
- Reducing bugs and technical debt
- Providing a consistent experience for library users

## Tools

RAGFlow uses the following tools to maintain code quality:

### Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter and formatter, written in Rust. We use Ruff for:

- Code formatting (replacing Black)
- Import sorting (replacing isort)
- Linting (replacing Flake8 and many plugins)
- Static analysis

Ruff configuration is in `pyproject.toml` under the `[tool.ruff]` section.

### Pre-commit

[Pre-commit](https://pre-commit.com/) runs checks automatically before each commit, ensuring code quality standards are maintained. We use pre-commit to run:

- Ruff (formatting and linting)
- Various file checks (trailing whitespace, file endings)
- Security scanning with Bandit

Pre-commit configuration is in `.pre-commit-config.yaml`.

### Bandit

[Bandit](https://github.com/PyCQA/bandit) is a security-focused linter that identifies common security issues in Python code. Configuration is in `pyproject.toml` under the `[tool.bandit]` section.

## Standards

### Python Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Maximum line length: 88 characters
- Use meaningful variable and function names
- Maintain consistent indentation (4 spaces)
- Group imports in this order: standard library, third-party, local
- Use double quotes for strings by default

### Documentation

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Document all public API functions, classes, and methods
- Include type hints in function signatures
- Document parameters, return values, and raised exceptions
- Provide examples for complex functions or classes

Example docstring:
```python
def embed_query(self, query: str) -> List[float]:
    """Generate an embedding for a query string.

    Args:
        query: The query text to embed.

    Returns:
        The embedding vector for the query.

    Raises:
        ValueError: If query is empty.
    """
```

### Interface Design

- Follow the interface-driven design pattern
- Keep interfaces minimal but complete
- Document the expected behavior in interface docstrings
- Ensure all implementations fully satisfy their interfaces

### Error Handling

- Use custom exception types for different error categories
- Provide informative error messages
- Handle expected exceptions gracefully
- Document all exceptions that might be raised

## Development Workflow

1. Install development dependencies:
   ```bash
   pip install -e ".[dev,test]"
   pre-commit install
   ```

2. Write code following the standards

3. Run pre-commit before committing:
   ```bash
   pre-commit run --all-files
   ```

4. Run tests to ensure functionality:
   ```bash
   pytest
   ```

5. Commit only after all checks pass

## Common Issues and Solutions

### Line Too Long (E501)

Solutions:
- Break long strings over multiple lines
- Use parentheses for line continuation
- Assign intermediate results to variables

### Missing Docstrings (D10*)

Solutions:
- Always document classes and public methods
- Follow Google style format
- Include all required sections

### Security Issues (S*)

Solutions:
- Review Bandit warnings carefully
- Understand the security implications
- Implement proper mitigations

## Continuous Integration

Our CI pipeline enforces these standards by:
- Running pre-commit checks on all PRs
- Running the test suite
- Checking documentation build
- Verifying package installation
