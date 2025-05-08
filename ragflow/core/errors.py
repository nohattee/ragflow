"""
Error classes for RAGFlow.

This module defines a hierarchy of exception classes used throughout the RAGFlow library,
providing clear and informative error messages for common failure scenarios.
"""


class RAGFlowError(Exception):
    """Base exception class for all RAGFlow-related errors."""

    pass


class ConfigurationError(RAGFlowError):
    """Raised when there's an issue with component or pipeline configuration."""

    pass


class VectorStoreError(RAGFlowError):
    """Raised when operations with the vector store fail."""

    pass


class EmbeddingError(RAGFlowError):
    """Raised when embedding generation fails."""

    pass


class LLMError(RAGFlowError):
    """Raised when LLM interaction fails."""

    pass


class ChunkingError(RAGFlowError):
    """Raised when document chunking fails."""

    pass


class RetrievalError(RAGFlowError):
    """Raised when document retrieval fails."""

    pass


class MissingDependencyError(RAGFlowError):
    """Raised when a required dependency is missing."""

    def __init__(self, dependency, installation_guide=None):
        """
        Initialize a MissingDependencyError.

        Args:
            dependency: The name of the missing dependency
            installation_guide: Optional installation instructions
        """
        message = f"Missing required dependency: {dependency}"
        if installation_guide:
            message += f"\n\nInstallation guide:\n{installation_guide}"
        super().__init__(message)


class APIKeyError(ConfigurationError):
    """Raised when an API key is missing or invalid."""

    def __init__(self, api_name, env_var=None):
        """
        Initialize an APIKeyError.

        Args:
            api_name: The name of the API
            env_var: The environment variable that should contain the API key
        """
        message = f"Missing or invalid API key for {api_name}"
        if env_var:
            message += (
                f". Set the {env_var} environment variable or provide the key directly."
            )
        super().__init__(message)


class InvalidArgumentError(ConfigurationError):
    """Raised when an invalid argument is provided to a function or method."""

    pass


class DocumentProcessingError(RAGFlowError):
    """Raised when processing documents fails."""

    pass
