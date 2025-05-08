# RAGFlow API Design Principles

This document outlines the design principles and best practices for the RAGFlow library's API.
These guidelines should be followed to ensure a consistent and developer-friendly experience.

## Core API Design Principles

1. **Simplicity First**
   - The most common operations should be achievable with minimal code
   - Default behaviors should be sensible and work "out of the box"
   - Advanced customization should be available but not required

2. **Explicit over Implicit**
   - Parameter names should be clear and self-documenting
   - Avoid hidden behaviors or "magic"
   - Error messages should explain what went wrong and how to fix it

3. **Progressive Complexity**
   - Start simple (DefaultRAGPipeline) for common use cases
   - Allow deeper customization (component configuration) for intermediate users
   - Enable full flexibility (custom adapters, pipeline construction) for advanced users

4. **Consistent Interface**
   - Methods with similar purposes should have similar signatures
   - Use consistent parameter naming across components
   - Follow Python naming conventions (snake_case for methods, PascalCase for classes)

## Error Handling Guidelines

1. **Use the RAGFlow exception hierarchy**
   - All exceptions should inherit from `RAGFlowError`
   - Use specific subclasses for different error categories
   - Avoid raising built-in exceptions directly

2. **Error messages should be:**
   - Clear: Describe what went wrong
   - Actionable: Suggest how to fix the issue
   - Context-aware: Include relevant parameter values or component names

3. **Validation**
   - Validate inputs early, before performing operations
   - Check for invalid arguments, file existence, and API credentials

## Documentation Best Practices

1. **All public APIs must have docstrings that include:**
   - Clear description of functionality
   - All parameters with types and descriptions
   - Return values with types and descriptions
   - Exceptions that may be raised
   - Usage examples

2. **Examples should demonstrate:**
   - Basic usage with defaults
   - Common customization scenarios
   - Handling errors appropriately

3. **Internal implementation details should be marked private with underscore prefix**
   - `_private_method()` or `_private_attribute`

## Backwards Compatibility

1. **Adding Features**
   - New parameters should have sensible defaults
   - Use keyword arguments for all optional parameters
   - When adding parameters to existing methods, place them after current parameters

2. **Deprecation Process**
   - Mark deprecated features with warnings
   - Keep deprecated features working for at least one major version cycle
   - Document alternatives in the deprecation message

## Minimizing Boilerplate

1. **Helper Functions**
   - Provide utilities for common operations in the `utils` package
   - Focus on reducing repetitive code patterns

2. **Sensible Defaults**
   - DefaultRAGPipeline should require minimal configuration
   - Component-specific parameters should have reasonable default values

3. **Factory Methods**
   - Use class methods like `from_existing_db()` for common initialization patterns
   - Consider adding more factory methods for other common scenarios

## Testing Recommendations

1. **Test Error Cases**
   - Verify that invalid inputs result in appropriate exceptions
   - Check that error messages are helpful

2. **Test with Real-world Examples**
   - Create tests using realistic document formats and queries
   - Verify that the API handles edge cases gracefully

3. **Test Configuration Variations**
   - Verify that different configuration options work as expected
   - Test combinations of parameters to ensure they interact correctly

---

By following these principles, we'll create a consistent and intuitive API that
makes it easy for developers to build powerful RAG applications with minimal friction.
