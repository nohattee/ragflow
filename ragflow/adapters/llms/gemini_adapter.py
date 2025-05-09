"""
Gemini adapter for LLMInterface.

This module provides an implementation of LLMInterface using Google's Gemini
API to generate text.
"""

from typing import List, Optional

from google import genai
from google.genai import types

from ragflow.core.interfaces import Document, LLMInterface


class GeminiAdapter(LLMInterface):
    """
    Adapter for Google's Gemini API that implements the LLMInterface.

    This adapter uses the Gemini API to generate text responses based on
    prompts and context.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        top_k: int = 40,
    ):
        """
        Initialize the Gemini adapter.

        Args:
            api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use (default: "gemini-2.0-flash")
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate (optional)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k

        # Configure the Gemini API Client
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """
        Generate text based on a prompt string.

        Args:
            prompt: Input prompt string

        Returns:
            Generated text
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
            ),
        )
        return response.text

    def generate_with_context(self, query: str, context: List[Document]) -> str:
        """
        Generate text based on a query and context documents.

        Args:
            query: User query
            context: List of context documents

        Returns:
            Generated text that answers the query using the provided context
        """
        # Format the context into a string
        context_str = "\n\n".join([doc.page_content for doc in context])

        # Create a prompt that includes both the context and the query
        prompt = f"""
        Context:
        {context_str}

        Question:
        {query}

        Based only on the context provided above, answer the question.
        If the context doesn't contain the answer, say "I don't have enough information to answer this question."
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
            ),
        )
        return response.text
