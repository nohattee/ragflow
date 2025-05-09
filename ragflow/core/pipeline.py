"""
Core pipeline implementation for RAGFlow.

This module provides the base RAGPipeline class that orchestrates the
Retrieval Augmented Generation (RAG) workflow by composing components
that implement the core interfaces.

The RAGPipeline coordinates document loading, chunking, embedding, storage,
retrieval, and generation to enable powerful question-answering capabilities
over custom document collections.
"""

from typing import Any, Dict, List, Optional

from .interfaces import (
    ChunkingStrategyInterface,
    Document,
    EmbeddingModelInterface,
    LLMInterface,
    RetrievalStrategyInterface,
    VectorStoreInterface,
)


class RAGPipeline:
    """
    Base RAG pipeline that orchestrates the full RAG workflow.

    The RAGPipeline coordinates all components in the RAG process:
    1. Splitting documents into chunks using a chunking strategy
    2. Embedding those chunks using an embedding model
    3. Storing embeddings in a vector store
    4. Retrieving relevant context based on queries
    5. Generating answers using an LLM based on retrieved context

    This design allows each component to be easily replaced, enabling
    customization and extension of the RAG pipeline.

    Examples:
    --------
        Creating a custom RAG pipeline:

        .. code-block:: python

            from ragflow.adapters.chunking_strategies import (
                RecursiveCharacterTextSplitterAdapter,
            )
            from ragflow.adapters.embedding_models import SentenceTransformersAdapter
            from ragflow.adapters.vector_stores import ChromaDBAdapter
            from ragflow.adapters.retrieval_strategies import SimpleSimilarityRetriever
            from ragflow.adapters.llms import GeminiAdapter
            from ragflow.core.pipeline import RAGPipeline

            # Create the component instances
            chunker = RecursiveCharacterTextSplitterAdapter(chunk_size=1000)
            embedder = SentenceTransformersAdapter()
            vector_store = ChromaDBAdapter(embedding_function=embedder)
            retriever = SimpleSimilarityRetriever(vector_store=vector_store)
            llm = GeminiAdapter(api_key="your-api-key")

            # Create the pipeline
            pipeline = RAGPipeline(
                chunking_strategy=chunker,
                embedding_model=embedder,
                vector_store=vector_store,
                retrieval_strategy=retriever,
                llm=llm,
            )

            # Use the pipeline
            pipeline.add_documents([Document(page_content="Example document")])
            answer = pipeline.query("What is in the document?")
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategyInterface,
        embedding_model: EmbeddingModelInterface,
        vector_store: VectorStoreInterface,
        retrieval_strategy: RetrievalStrategyInterface,
        llm: LLMInterface,
    ):
        """
        Initialize the RAG pipeline with its component parts.

        This constructor assembles all the required components into a cohesive
        pipeline that can process documents and answer queries. Each component
        must implement its respective interface from the core.interfaces module.

        Args:
            chunking_strategy: Strategy for splitting documents into chunks.
                Must implement the ChunkingStrategyInterface.
            embedding_model: Model for generating vector embeddings.
                Must implement the EmbeddingModelInterface.
            vector_store: Store for saving and retrieving embeddings.
                Must implement the VectorStoreInterface.
            retrieval_strategy: Strategy for retrieving relevant documents.
                Must implement the RetrievalStrategyInterface.
            llm: Language model for generating answers.
                Must implement the LLMInterface.

        Examples:
        --------
            .. code-block:: python

                # Create component instances
                chunker = RecursiveCharacterTextSplitterAdapter()
                embedder = SentenceTransformersAdapter()
                vector_store = ChromaDBAdapter(embedding_function=embedder)
                retriever = SimpleSimilarityRetriever(vector_store=vector_store)
                llm = GeminiAdapter(api_key="your-api-key")

                # Initialize the pipeline
                pipeline = RAGPipeline(
                    chunking_strategy=chunker,
                    embedding_model=embedder,
                    vector_store=vector_store,
                    retrieval_strategy=retriever,
                    llm=llm,
                )
        """
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.retrieval_strategy = retrieval_strategy
        self.llm = llm

    def add_documents(self, documents: List[Document]) -> None:
        """
        Process and add documents to the pipeline.

        This method handles the complete document ingestion process:
        1. Splits the documents into smaller chunks using the configured chunking strategy
        2. Generates embeddings for these chunks using the embedding model
        3. Adds the chunks and their embeddings to the vector store

        This prepares the documents for later retrieval when answering queries.

        Args:
            documents: List of Document objects to add to the pipeline.
                Each Document should have page_content (text) and optional
                metadata.

        Returns:
            None

        Examples:
        --------
            .. code-block:: python

                # Create a pipeline
                pipeline = RAGPipeline(...)

                # Create Document objects
                documents = [
                    Document(
                        page_content="RAGFlow is a framework for building RAG applications.",
                        metadata={"source": "introduction.txt"},
                    ),
                    Document(
                        page_content="Vector stores enable efficient similarity search.",
                        metadata={"source": "concepts.txt"},
                    ),
                ]

                # Add documents to the pipeline
                pipeline.add_documents(documents)
        """
        # Split documents into chunks
        chunked_documents = self.chunking_strategy.split_documents(documents)

        # Add chunked documents to the vector store
        self.vector_store.add_documents(chunked_documents)

    def add_texts(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Process and add text strings to the pipeline.

        This is a convenience method that:
        1. Converts texts to Document objects (with optional metadata)
        2. Splits them into smaller chunks
        3. Adds those chunks to the vector store

        It's a simpler alternative to creating Document objects manually
        when you have raw text strings.

        Args:
            texts: List of text strings to add to the pipeline.
            metadata: Optional list of metadata dictionaries, one for each text.
                If provided, must have the same length as texts. If not provided,
                empty metadata will be used for all texts.

        Returns:
            None

        Examples:
        --------
            .. code-block:: python

                # Create a pipeline
                pipeline = RAGPipeline(...)

                # Add texts
                pipeline.add_texts(
                    ["Some text here.", "Another piece of text."],
                    metadata=[{"source": "text_source_1"}, {"source": "text_source_2"}],
                )
        """
        # Convert texts to Documents
        documents = []
        for i, text in enumerate(texts):
            m = metadata[i] if metadata and i < len(metadata) else {}
            documents.append(Document(page_content=text, metadata=m))

        # Add documents to the pipeline
        self.add_documents(documents)

    def query(self, question: str) -> str:
        """
        Process a query through the RAG pipeline to generate an answer.

        This method implements the core RAG workflow:
        1. Retrieves relevant documents using the retrieval strategy
        2. Passes the question and retrieved documents to the LLM
        3. Returns the generated answer

        The quality of the answer depends on:
        - The relevance of the retrieved documents
        - The capabilities of the LLM
        - The pre-processing of documents (chunking)

        Args:
            question: The query/question to answer. This can be a natural
                language question, a statement, or any text that requires
                a response based on the stored documents.

        Returns:
            str: The generated answer to the question.

        Examples:
        --------
            .. code-block:: python

                # Create and populate a pipeline
                pipeline = RAGPipeline(...)
                pipeline.add_texts(["Paris is the capital of France."])

                # Query the pipeline
                answer = pipeline.query("What is the capital of France?")
                print(answer)
                # Output might be: "Paris is the capital of France."
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieval_strategy.get_relevant_documents(question)

        # Generate answer using LLM and retrieved context
        answer = self.llm.generate_with_context(question, relevant_docs)

        return answer

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Process a query and return both the answer and source documents.

        Similar to query(), but also returns the source documents used
        to generate the answer, enabling citation and verification. This
        is useful for:
        - Providing attribution for information
        - Allowing users to verify the sources
        - Debugging the RAG pipeline's retrieval performance

        Args:
            question: The query/question to answer. This can be a natural
                language question, a statement, or any text that requires
                a response based on the stored documents.

        Returns:
            A dictionary containing:
            - 'answer': The generated response as a string
            - 'sources': List of Document objects used to generate the answer

        Examples:
        --------
            .. code-block:: python

                # Create and populate a pipeline
                pipeline = RAGPipeline(...)
                pipeline.add_texts(["The Eiffel Tower is in Paris."])

                # Query the pipeline and get sources
                result = pipeline.query_with_sources("Where is the Eiffel Tower?")

                # Print the answer
                print(result["answer"])  # "The Eiffel Tower is in Paris."

                # Print the sources
                for source_doc in result["sources"]:
                    print(source_doc.page_content, source_doc.metadata)
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieval_strategy.get_relevant_documents(question)

        # Generate answer using LLM and retrieved context
        answer = self.llm.generate_with_context(question, relevant_docs)

        # Return both the answer and the source documents
        return {"answer": answer, "sources": relevant_docs}


class AgenticRAGPipeline(RAGPipeline):
    """
    An agentic RAG pipeline that uses an LLM-based agent.

    An agentic RAG pipeline that uses an LLM-based agent to decide when to retrieve
    documents, rewrite queries, and generate answers.

    This pipeline follows a more dynamic flow based on the agent's decisions.
    """

    def __init__(
        self,
        agent_llm: LLMInterface,
        generator_llm: LLMInterface,
        retrieval_strategy: RetrievalStrategyInterface,
        chunking_strategy: Optional[ChunkingStrategyInterface] = None,
        embedding_model: Optional[EmbeddingModelInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
        max_iterations: int = 3,
    ):
        """
        Initialize the AgenticRAGPipeline with the given components.

        Args:
            agent_llm: The LLM used for decision-making and query generation.
            generator_llm: The LLM used for final answer generation.
            retrieval_strategy: The strategy for retrieving relevant documents.
            chunking_strategy: The strategy for chunking documents.
            embedding_model: The embedding model for generating document embeddings.
            vector_store: The vector store for storing document embeddings.
            max_iterations: The maximum number of iterations for the agentic loop.
        """
        if chunking_strategy and embedding_model and vector_store:
            super().__init__(
                chunking_strategy=chunking_strategy,
                embedding_model=embedding_model,
                vector_store=vector_store,
                retrieval_strategy=retrieval_strategy,
                llm=generator_llm,
            )
        elif not (
            chunking_strategy is None
            and embedding_model is None
            and vector_store is None
        ):
            raise ValueError(
                "Either all of chunking_strategy, embedding_model, and vector_store must be provided, or none."
            )
        else:
            self.chunking_strategy = chunking_strategy
            self.embedding_model = embedding_model
            self.vector_store = vector_store
            self.retrieval_strategy = retrieval_strategy
            self.llm = generator_llm  # Base class uses self.llm for its query methods

        self.agent_llm = agent_llm
        self.generator_llm = (
            generator_llm  # Explicit reference for clarity in agentic methods
        )
        self.max_iterations = max_iterations

    def _decide_retrieval(self, question: str, history: List[str]) -> Dict[str, Any]:
        prompt = f"""You are an intelligent assistant. Given the question: "{question}"
History of previous attempts:
{history_str if (history_str := "\\n".join(history)) else "No previous attempts."}

What is the best next step?
Your available actions are:
1. **retrieve**: If you need more information to answer the question. Provide the search query.
2. **rewrite**: If the current question is not good for retrieval and needs reformulation. Provide the new query.
3. **answer**: If you have enough information or believe you can answer directly. Provide the answer.
4. **end**: If you've tried and cannot answer the question.

Respond with a JSON object with "action" and relevant keys (e.g., "query" for retrieve, "new_query" for rewrite, "content" for answer/end).
Example for retrieve: {{"action": "retrieve", "query": "details about X"}}
Example for answer: {{"action": "answer", "content": "The answer is Y."}}
"""
        response_text = self.agent_llm.generate(prompt)
        try:
            import json

            decision = json.loads(response_text)
            if "action" not in decision:
                return {"action": "answer", "content": response_text}  # Fallback
            return decision
        except json.JSONDecodeError:
            print(f"Warning: Agent LLM did not return valid JSON: {response_text}")
            return {"action": "answer", "content": response_text}  # Fallback

    def _check_relevance(self, question: str, documents: List[Document]) -> bool:
        if not documents:
            return False
        context_str = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""Given the question: "{question}"
And the following retrieved context:
---
{context_str[:2000]}
---
Is the retrieved context relevant and sufficient to answer the question?
Answer with only "Yes" or "No".
"""
        relevance_response = self.agent_llm.generate(prompt).strip().lower()
        return "yes" in relevance_response

    def _rewrite_query(self, original_question: str, feedback: str) -> str:
        prompt = f"""The previous attempt to answer the question "{original_question}" was not successful.
Feedback: {feedback}
Please rewrite the question to be more effective for information retrieval or to clarify the user's intent.
Return only the rewritten question.
Rewritten Question:"""
        rewritten_query = self.agent_llm.generate(prompt).strip()
        return rewritten_query

    def _generate_final_answer(self, question: str, documents: List[Document]) -> str:
        return self.generator_llm.generate_with_context(question, documents)

    def _agentic_loop(self, question: str) -> Dict[str, Any]:
        """
        Core agentic loop to process a question.

        Returns a dictionary containing "answer", "sources", and "history".
        """
        current_query = question
        history: List[str] = []
        final_docs_for_answer: List[Document] = []
        answer = "Could not determine an answer through the agentic process."  # Default answer

        for iteration in range(self.max_iterations):
            history.append(
                f"Iteration {iteration + 1}, Current query: '{current_query}'"
            )
            decision = self._decide_retrieval(current_query, history)
            action = decision.get("action")

            if action == "retrieve":
                search_query = decision.get("query", current_query)
                history.append(
                    f"Action: Retrieve documents with query: '{search_query}'"
                )
                retrieved_docs = self.retrieval_strategy.get_relevant_documents(
                    search_query
                )
                if self._check_relevance(current_query, retrieved_docs):
                    history.append(
                        "Action: Documents found relevant. Generating answer."
                    )
                    final_docs_for_answer = retrieved_docs
                    answer = self._generate_final_answer(
                        current_query, final_docs_for_answer
                    )
                    return {
                        "answer": answer,
                        "sources": final_docs_for_answer,
                        "history": history,
                    }
                else:
                    history.append("Action: Documents found not relevant.")
                    # Provide feedback for rewrite based on irrelevance
                    feedback = "Retrieved documents were not relevant to the query."
                    # Allow agent to decide next step, which could be rewrite
                    # If agent decides to rewrite, it will happen in next iteration based on history.
                    # Or, force a rewrite here:
                    current_query = self._rewrite_query(
                        question, feedback
                    )  # Use original question for context
                    if (
                        not current_query or current_query == question
                    ):  # Avoid infinite loop on bad rewrite
                        history.append(
                            "Rewrite did not change query or failed. Ending with no relevant docs."
                        )
                        answer = "I could not find relevant information after attempting retrieval and rewrite."
                        return {"answer": answer, "sources": [], "history": history}
                    final_docs_for_answer = []  # Reset docs as query changed
                    continue  # Continue to next iteration with rewritten query

            elif action == "rewrite":
                new_query = decision.get("new_query")
                if not new_query or new_query == current_query:
                    history.append(
                        f"Action: Rewrite failed or produced same query ('{new_query}')."
                    )
                    # Attempt to answer with any last known good documents or end
                    if final_docs_for_answer and self._check_relevance(
                        question, final_docs_for_answer
                    ):
                        answer = self._generate_final_answer(
                            question, final_docs_for_answer
                        )
                    else:
                        answer = "I tried to rewrite the question but could not improve it to find an answer."
                    return {
                        "answer": answer,
                        "sources": final_docs_for_answer,
                        "history": history,
                    }

                history.append(
                    f"Action: Rewrite query from '{current_query}' to '{new_query}'"
                )
                current_query = new_query
                final_docs_for_answer = []  # Reset retrieved docs for new query
                continue

            elif action == "answer":
                answer = decision.get("content", "I'm not sure how to respond to that.")
                history.append(f"Action: Answer directly. Content: '{answer[:50]}...'")
                return {
                    "answer": answer,
                    "sources": final_docs_for_answer,
                    "history": history,
                }  # Sources might be from a previous step

            elif action == "end":
                answer = decision.get(
                    "content",
                    "I am unable to answer your question with the available information.",
                )
                history.append(f"Action: End. Content: '{answer}'")
                return {"answer": answer, "sources": [], "history": history}

            else:
                answer = "I encountered an unexpected situation and cannot proceed."
                history.append(f"Action: Unknown agent action: {action}. Ending.")
                return {"answer": answer, "sources": [], "history": history}

        history.append("Max iterations reached.")
        # Fallback if max_iterations reached without a definitive answer
        if final_docs_for_answer and self._check_relevance(
            question, final_docs_for_answer
        ):  # Use original question for relevance check
            answer = self._generate_final_answer(
                question, final_docs_for_answer
            )  # Use original question for generation
        else:
            answer = "I've tried multiple steps but could not find a satisfactory answer to your question."
        return {"answer": answer, "sources": final_docs_for_answer, "history": history}

    def query(self, question: str) -> str:  # type: ignore
        """
        Processes a query using the agentic loop.

        Args:
            question: The user's question.

        Returns:
            The generated answer.
        """
        result = self._agentic_loop(question)
        return result["answer"]

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """Processes a query using the agentic loop and returns answer with sources and history."""
        return self._agentic_loop(question)
