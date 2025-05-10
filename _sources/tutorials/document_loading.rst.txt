======================
Document Loading Tutorial
======================

This tutorial explains how to load various document types into RAGFlow for processing and retrieval. Properly loading and preprocessing documents is a critical first step in building an effective RAG system.

Supported Document Types
----------------------

RAGFlow provides utilities to load multiple document formats:

- Plain text files (`.txt`)
- PDF documents (`.pdf`)
- Markdown files (`.md`)
- HTML/web pages
- CSV and spreadsheet data
- JSON and structured data

Basic Document Loading
--------------------

Let's start with the simplest case - loading plain text files:

.. code-block:: python

    from ragflow.utils.document_loaders import load_text_files
    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline

    # Initialize your pipeline
    pipeline = DefaultRAGPipeline()

    # Load all text files from a directory
    documents = load_text_files(
        directory="./data/text",
        recursive=True,  # Include subdirectories
        glob_pattern="*.txt"  # Only .txt files
    )

    # Add the documents to the pipeline
    pipeline.add_documents(documents)

    print(f"Loaded {len(documents)} documents")

    # Preview the first document
    if documents:
        print(f"First document content: {documents[0].page_content[:100]}...")
        print(f"First document metadata: {documents[0].metadata}")

The `load_text_files` function returns a list of `Document` objects with the document content in the `page_content` field and metadata in the `metadata` dictionary. The metadata includes the file path, filename, and other file-specific information.

Loading PDF Documents
-------------------

Processing PDF documents requires the `pdfplumber` or `pypdf` library:

.. code-block:: bash

    pip install ragflow[pdf]  # Installs the PDF dependencies

Then you can load PDF documents:

.. code-block:: python

    from ragflow.utils.document_loaders import load_pdf_files

    # Load PDF documents
    pdf_documents = load_pdf_files(
        directory="./data/pdfs",
        recursive=True,
        include_page_numbers=True  # Adds page numbers to metadata
    )

    pipeline.add_documents(pdf_documents)

Loading Web Pages
---------------

For loading web pages, you'll need the `requests` and `beautifulsoup4` libraries:

.. code-block:: bash

    pip install ragflow[web]  # Installs web loading dependencies

Then you can load web content:

.. code-block:: python

    from ragflow.utils.document_loaders import load_web_pages

    # Define URLs to load
    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://example.com/another-page"
    ]

    # Load web pages
    web_documents = load_web_pages(urls=urls)

    pipeline.add_documents(web_documents)

    # Or load a single page:
    from ragflow.utils.document_loaders import load_web_page

    single_page_docs = load_web_page(url="https://example.com/important-data")
    pipeline.add_documents(single_page_docs)

Loading Markdown Files
--------------------

Markdown files can be loaded similarly to text files:

.. code-block:: python

    from ragflow.utils.document_loaders import load_markdown_files

    markdown_documents = load_markdown_files(
        directory="./docs/markdown",
        recursive=True,
        extract_metadata=True  # Extract YAML frontmatter as metadata
    )

    pipeline.add_documents(markdown_documents)

Working with Structured Data
--------------------------

For CSV, JSON, or other structured data:

.. code-block:: python

    from ragflow.utils.document_loaders import load_csv_file, load_json_file

    # Load CSV file
    csv_documents = load_csv_file(
        file_path="./data/products.csv",
        column_to_text="description",  # Column containing main text
        metadata_columns=["product_id", "category", "price"]  # Add these as metadata
    )

    # Load JSON file
    json_documents = load_json_file(
        file_path="./data/articles.json",
        jq_pattern=".items[]",  # JQ pattern to extract items (optional)
        content_key="text",  # Key containing text content
        metadata_keys=["author", "date", "tags"]  # Keys to include as metadata
    )

    # Add all documents
    pipeline.add_documents(csv_documents)
    pipeline.add_documents(json_documents)

Handling Document Metadata
------------------------

Metadata is critical for document retrieval and attribution. Here's how to work with it:

.. code-block:: python

    from ragflow.core.interfaces import Document

    # Create a document with custom metadata
    custom_doc = Document(
        page_content="This is a document with custom metadata.",
        metadata={
            "source": "manual input",
            "author": "Jane Smith",
            "date": "2025-03-15",
            "category": "example",
            "importance": "high"
        }
    )

    # Add a list of custom documents
    custom_documents = [
        Document(page_content="First document", metadata={"source": "doc1.txt", "category": "A"}),
        Document(page_content="Second document", metadata={"source": "doc2.txt", "category": "B"})
    ]

    pipeline.add_documents(custom_documents)

Combining Multiple Document Sources
---------------------------------

In real applications, you often need to combine documents from multiple sources:

.. code-block:: python

    # Load documents from various sources
    text_docs = load_text_files("./data/text")
    pdf_docs = load_pdf_files("./data/pdfs")
    web_docs = load_web_pages(["https://example.com/page1", "https://example.com/page2"])

    # Combine all documents
    all_documents = text_docs + pdf_docs + web_docs

    # Add specific metadata to each source
    for doc in text_docs:
        doc.metadata["source_type"] = "text_file"

    for doc in pdf_docs:
        doc.metadata["source_type"] = "pdf"

    for doc in web_docs:
        doc.metadata["source_type"] = "web"

    # Add all documents to the pipeline
    pipeline.add_documents(all_documents)

Preprocessing Documents
---------------------

Before adding documents to the pipeline, you might want to preprocess them:

.. code-block:: python

    def preprocess_document(doc):
        """Apply custom preprocessing to a document."""
        # Convert content to lowercase
        content = doc.page_content.lower()

        # Remove specific patterns or clean text
        content = content.replace("unwanted text", "")

        # Return a new document with processed content
        return Document(
            page_content=content,
            metadata=doc.metadata
        )

    # Load documents
    documents = load_text_files("./data/text")

    # Preprocess each document
    processed_documents = [preprocess_document(doc) for doc in documents]

    # Add processed documents to pipeline
    pipeline.add_documents(processed_documents)

Complete Example
--------------

Here's a complete example that loads documents from multiple sources:

.. code-block:: python

    import os
    from dotenv import load_dotenv

    from ragflow.pipelines.default_rag_pipeline import DefaultRAGPipeline
    from ragflow.utils.document_loaders import (
        load_text_files,
        load_pdf_files,
        load_markdown_files,
        load_web_pages
    )

    # Load environment variables
    load_dotenv()

    # Initialize the pipeline
    pipeline = DefaultRAGPipeline(
        api_key=os.getenv("GEMINI_API_KEY"),
        chunk_size=1000,
        chunk_overlap=200
    )

    # Load documents from different sources
    text_docs = load_text_files("./data/text", recursive=True)
    pdf_docs = load_pdf_files("./data/pdfs", recursive=True)
    md_docs = load_markdown_files("./data/markdown", recursive=True)
    web_docs = load_web_pages([
        "https://example.com/page1",
        "https://example.com/page2"
    ])

    # Add source information to metadata
    for docs, source_type in [
        (text_docs, "text"),
        (pdf_docs, "pdf"),
        (md_docs, "markdown"),
        (web_docs, "web")
    ]:
        for doc in docs:
            doc.metadata["source_type"] = source_type

    # Combine all documents
    all_documents = text_docs + pdf_docs + md_docs + web_docs
    print(f"Loaded {len(all_documents)} total documents")

    # Add documents to the pipeline
    pipeline.add_documents(all_documents)

    # Test with a query
    question = "What information do we have about retrieval augmented generation?"
    answer = pipeline.query(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

Best Practices for Document Loading
---------------------------------

1. **Preserve Source Information:**
   - Always include source information in metadata (filename, URL, page number)
   - This enables proper attribution and verification of information

2. **Chunk Appropriately:**
   - PDF and web documents often need different chunking strategies than plain text
   - Consider the document structure when choosing chunk size and overlap

3. **Handle Encoding Issues:**
   - Be prepared to handle text encoding issues, especially with PDFs and web content
   - Use try/except blocks to handle problematic files

4. **Filter Irrelevant Content:**
   - For web pages, consider removing navigation elements, ads, and other non-content
   - For PDFs, you might need to filter out headers, footers, and page numbers

5. **Batch Processing:**
   - For large document collections, process files in batches
   - This prevents memory issues and allows for progress tracking

Troubleshooting
-------------

**PDF Loading Issues:**
- Ensure PDF files are not password-protected
- Some PDFs may be scanned images requiring OCR (not supported by default)
- Try both `pypdf` and `pdfplumber` backends to see which works better

**Web Loading Issues:**
- Some websites block scraping attempts
- Consider adding delays between requests with `load_web_pages(urls, delay_seconds=2)`
- Use a proper user-agent string with `load_web_pages(urls, user_agent="...")`

**Character Encoding:**
- Specify encoding when loading text files: `load_text_files(directory, encoding="utf-8")`
- For non-standard encodings, preprocess files before loading

Next Steps
---------

Now that you know how to load various document types into RAGFlow, you can:

- Explore customizing the chunking strategy for different document types
- Learn how to filter and process documents based on metadata
- Set up a document refresh strategy for regularly updated content

Check the :doc:`../advanced/custom_adapters` guide to learn how to create custom document loaders for specialized formats.
