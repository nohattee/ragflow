"""Unit tests for the AgenticRAGPipeline."""

from unittest.mock import Mock, patch

from ragflow.core.interfaces import Document
from ragflow.core.pipeline import AgenticRAGPipeline


class TestAgenticRAGPipelineUnit:
    """Unit tests for the AgenticRAGPipeline class using mocks."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock dependencies
        self.mock_agent_llm = Mock()
        self.mock_generator_llm = Mock()
        self.mock_retrieval_strategy = Mock()
        self.mock_chunking_strategy = Mock()
        self.mock_embedding_model = Mock()
        self.mock_vector_store = Mock()

        # Set up the pipeline with mocks
        self.pipeline = AgenticRAGPipeline(
            agent_llm=self.mock_agent_llm,
            generator_llm=self.mock_generator_llm,
            retrieval_strategy=self.mock_retrieval_strategy,
            chunking_strategy=self.mock_chunking_strategy,
            embedding_model=self.mock_embedding_model,
            vector_store=self.mock_vector_store,
            max_iterations=3,
        )

    def test_decide_retrieval_parse_valid_json(self):
        """Test that _decide_retrieval correctly parses valid JSON from the agent LLM."""
        # Setup
        self.mock_agent_llm.generate.return_value = (
            '{"action": "retrieve", "query": "test query"}'
        )

        # Exercise
        result = self.pipeline._decide_retrieval("What is X?", ["Previous attempt"])

        # Verify
        self.mock_agent_llm.generate.assert_called_once()
        assert result == {"action": "retrieve", "query": "test query"}

    def test_decide_retrieval_parse_invalid_json(self):
        """Test that _decide_retrieval handles invalid JSON with a fallback."""
        # Setup
        self.mock_agent_llm.generate.return_value = "This is not JSON"

        # Exercise
        result = self.pipeline._decide_retrieval("What is X?", [])

        # Verify
        self.mock_agent_llm.generate.assert_called_once()
        assert result == {"action": "answer", "content": "This is not JSON"}

    def test_decide_retrieval_missing_action(self):
        """Test that _decide_retrieval handles JSON without 'action' field with a fallback."""
        # Setup
        self.mock_agent_llm.generate.return_value = '{"content": "Some content"}'

        # Exercise
        result = self.pipeline._decide_retrieval("What is X?", [])

        # Verify
        self.mock_agent_llm.generate.assert_called_once()
        assert result == {"action": "answer", "content": '{"content": "Some content"}'}

    def test_check_relevance_true(self):
        """Test that _check_relevance returns True when agent indicates relevance."""
        # Setup
        documents = [Document(page_content="test content", metadata={})]
        self.mock_agent_llm.generate.return_value = "Yes"

        # Exercise
        result = self.pipeline._check_relevance("Question?", documents)

        # Verify
        self.mock_agent_llm.generate.assert_called_once()
        assert result is True

    def test_check_relevance_false(self):
        """Test that _check_relevance returns False when agent indicates irrelevance."""
        # Setup
        documents = [Document(page_content="irrelevant content", metadata={})]
        self.mock_agent_llm.generate.return_value = "No"

        # Exercise
        result = self.pipeline._check_relevance("Question?", documents)

        # Verify
        self.mock_agent_llm.generate.assert_called_once()
        assert result is False

    def test_check_relevance_empty_docs(self):
        """Test that _check_relevance returns False when documents list is empty."""
        # Setup
        documents = []

        # Exercise
        result = self.pipeline._check_relevance("Question?", documents)

        # Verify
        # Agent LLM should not be called at all
        self.mock_agent_llm.generate.assert_not_called()
        assert result is False

    def test_rewrite_query(self):
        """Test that _rewrite_query returns the agent's response."""
        # Setup
        self.mock_agent_llm.generate.return_value = "Rewritten question"

        # Exercise
        result = self.pipeline._rewrite_query("Original question", "Feedback")

        # Verify
        self.mock_agent_llm.generate.assert_called_once()
        assert result == "Rewritten question"

    def test_generate_final_answer(self):
        """Test that _generate_final_answer calls the generator LLM with the right parameters."""
        # Setup
        documents = [Document(page_content="relevant content", metadata={})]
        self.mock_generator_llm.generate_with_context.return_value = "Final answer"

        # Exercise
        result = self.pipeline._generate_final_answer("Question?", documents)

        # Verify
        self.mock_generator_llm.generate_with_context.assert_called_once_with(
            "Question?", documents
        )
        assert result == "Final answer"

    def test_agentic_loop_successful_retrieval(self):
        """Test the agentic loop with a successful retrieval first time."""
        # Setup
        question = "What is X?"
        documents = [Document(page_content="X is Y", metadata={})]

        # Mock agent LLM to choose retrieve
        self.mock_agent_llm.generate.side_effect = [
            '{"action": "retrieve", "query": "X details"}',  # First decision
            "Yes",  # Relevance check
        ]

        # Mock retrieval strategy to return documents
        self.mock_retrieval_strategy.get_relevant_documents.return_value = documents

        # Mock generator LLM to return an answer
        self.mock_generator_llm.generate_with_context.return_value = (
            "X is Y according to sources"
        )

        # Exercise
        result = self.pipeline._agentic_loop(question)

        # Verify
        assert result["answer"] == "X is Y according to sources"
        assert result["sources"] == documents
        assert (
            len(result["history"]) == 3
        )  # Initial iteration + retrieve action + found relevant

    def test_agentic_loop_irrelevant_retrieval_then_rewrite(self):
        """Test the agentic loop with irrelevant retrieval, followed by rewrite and success."""
        # Setup
        question = "What is X?"
        documents_irrelevant = [Document(page_content="Irrelevant", metadata={})]
        documents_relevant = [Document(page_content="X is Y", metadata={})]

        # Mock agent LLM with multiple side effects
        self.mock_agent_llm.generate.side_effect = [
            '{"action": "retrieve", "query": "X details"}',  # First decision
            "No",  # First relevance check (irrelevant)
            "What is X in context?",  # Rewrite query
            '{"action": "retrieve", "query": "X in context"}',  # Second decision
            "Yes",  # Second relevance check (relevant)
        ]

        # Mock retrieval strategy to return different documents on each call
        self.mock_retrieval_strategy.get_relevant_documents.side_effect = [
            documents_irrelevant,
            documents_relevant,
        ]

        # Mock generator LLM to return an answer
        self.mock_generator_llm.generate_with_context.return_value = "X is Y in context"

        # Exercise
        result = self.pipeline._agentic_loop(question)

        # Verify
        assert result["answer"] == "X is Y in context"
        assert result["sources"] == documents_relevant
        assert len(result["history"]) > 3  # Should have multiple steps in history

    def test_agentic_loop_agent_answers_directly(self):
        """Test the agentic loop when agent decides to answer directly."""
        # Setup
        question = "What is X?"

        # Mock agent LLM to choose answer
        self.mock_agent_llm.generate.return_value = (
            '{"action": "answer", "content": "X is a variable"}'
        )

        # Exercise
        result = self.pipeline._agentic_loop(question)

        # Verify
        assert result["answer"] == "X is a variable"
        assert result["sources"] == []  # No sources when answering directly
        assert len(result["history"]) == 2  # Initial iteration + answer action

    def test_agentic_loop_agent_decides_to_end(self):
        """Test the agentic loop when agent decides to end."""
        # Setup
        question = "What is X?"

        # Mock agent LLM to choose end
        self.mock_agent_llm.generate.return_value = (
            '{"action": "end", "content": "Cannot determine what X is"}'
        )

        # Exercise
        result = self.pipeline._agentic_loop(question)

        # Verify
        assert result["answer"] == "Cannot determine what X is"
        assert result["sources"] == []  # No sources when ending
        assert len(result["history"]) == 2  # Initial iteration + end action

    def test_agentic_loop_bad_rewrite(self):
        """Test the agentic loop when rewrite fails (returns same query)."""
        # Setup
        question = "What is X?"
        documents_irrelevant = [Document(page_content="Irrelevant", metadata={})]

        # Mock agent LLM with multiple side effects
        self.mock_agent_llm.generate.side_effect = [
            '{"action": "retrieve", "query": "X details"}',  # First decision
            "No",  # Relevance check (irrelevant)
            "What is X?",  # Rewrite returns same query (bad rewrite)
        ]

        # Mock retrieval strategy
        self.mock_retrieval_strategy.get_relevant_documents.return_value = (
            documents_irrelevant
        )

        # Exercise
        result = self.pipeline._agentic_loop(question)

        # Verify
        assert "could not find relevant information" in result["answer"].lower()
        assert result["sources"] == []
        assert len(result["history"]) > 2

    def test_agentic_loop_max_iterations_reached(self):
        """Test the agentic loop when max iterations is reached without resolution."""
        # Setup
        question = "What is X?"
        [Document(page_content="Irrelevant", metadata={})]

        # Configure the pipeline with a low max_iterations
        self.pipeline.max_iterations = 2

        # Mock agent LLM to keep choosing rewrite
        self.mock_agent_llm.generate.side_effect = [
            '{"action": "rewrite", "new_query": "What is X ver 2?"}',  # First iteration
            '{"action": "rewrite", "new_query": "What is X ver 3?"}',  # Second iteration
            "No",  # Final relevance check (shouldn't be called due to max_iterations)
        ]

        # Exercise
        result = self.pipeline._agentic_loop(question)

        # Verify
        assert "tried multiple steps" in result["answer"].lower()
        assert len(result["history"]) > 2
        assert "Max iterations reached." in result["history"]

    def test_agentic_loop_unknown_action(self):
        """Test the agentic loop when agent returns an unknown action."""
        # Setup
        question = "What is X?"

        # Mock agent LLM to return an unknown action
        self.mock_agent_llm.generate.return_value = (
            '{"action": "unknown_action", "data": "some data"}'
        )

        # Exercise
        result = self.pipeline._agentic_loop(question)

        # Verify
        assert "unexpected situation" in result["answer"].lower()
        assert "Unknown agent action" in result["history"][-1]

    def test_query_calls_agentic_loop(self):
        """Test that query() calls _agentic_loop and returns answer."""
        # Setup
        with patch.object(self.pipeline, "_agentic_loop") as mock_agentic_loop:
            mock_agentic_loop.return_value = {
                "answer": "Test answer",
                "sources": [],
                "history": [],
            }

            # Exercise
            result = self.pipeline.query("Test question")

            # Verify
            mock_agentic_loop.assert_called_once_with("Test question")
            assert result == "Test answer"

    def test_query_with_sources_calls_agentic_loop(self):
        """Test that query_with_sources() calls _agentic_loop and returns full result."""
        # Setup
        expected_result = {"answer": "Test answer", "sources": [], "history": []}

        with patch.object(self.pipeline, "_agentic_loop") as mock_agentic_loop:
            mock_agentic_loop.return_value = expected_result

            # Exercise
            result = self.pipeline.query_with_sources("Test question")

            # Verify
            mock_agentic_loop.assert_called_once_with("Test question")
            assert result == expected_result
