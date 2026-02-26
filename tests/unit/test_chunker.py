
from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import DocumentChunker


def test_supported_extensions():
    """Test that DocumentChunker exposes supported extensions."""
    chunker = DocumentChunker()
    assert chunker.supported_extensions == [".md", ".txt"]


def test_chunk_ids_are_unique_across_different_paths():
    chunker = DocumentChunker(max_chars=100)
    content = "Sample content that is long enough to be a valid chunk."

    # Same filename, different directories
    doc1 = Document(filepath="docs/README.md", content=content, content_hash="hash1")
    doc2 = Document(filepath="src/README.md", content=content, content_hash="hash2")

    chunks1 = list(chunker.process(doc1))
    chunks2 = list(chunker.process(doc2))

    assert chunks1[0].id == "docs/README.md_chunk_0"
    assert chunks2[0].id == "src/README.md_chunk_0"
    assert chunks1[0].id != chunks2[0].id


def test_markdown_chunking_with_code_fences():
    chunker = DocumentChunker(max_chars=200)
    content = """# Title
This is some introductory text that should be grouped together if small enough.

```python
def foo():
    # This code block should be atomic
    return 42
```

Final paragraph.
"""
    doc = Document(filepath="test.md", content=content, content_hash="hash1")
    chunks = list(chunker.process(doc))

    assert len(chunks) == 3
    assert chunks[0].id == "test.md_chunk_0"
    assert "# Title" in chunks[0].text
    assert "introductory text" in chunks[0].text
    assert chunks[1].id == "test.md_chunk_1"
    assert "```python" in chunks[1].text
    assert "return 42" in chunks[1].text
    assert chunks[2].id == "test.md_chunk_2"
    assert "Final paragraph" in chunks[2].text


def test_markdown_chunking_large_prose_splitting():
    # Small max_chars to force splitting of prose
    chunker = DocumentChunker(max_chars=20)
    content = """This is a very long paragraph that definitely exceeds twenty characters.

And another one here."""

    doc = Document(filepath="test.md", content=content, content_hash="hash2")
    chunks = list(chunker.process(doc))

    # Since each paragraph is > 20, they should be separate chunks
    assert len(chunks) >= 2
    assert "very long paragraph" in chunks[0].text
    assert "another one" in chunks[1].text


def test_text_file_fallback():
    chunker = DocumentChunker(max_chars=500)
    # A .txt file uses double-newline splitting
    content = (
        "Paragraph 1 is here.\n\nParagraph 2 is here, and it is also quite short.\n\nParagraph 3."
    )

    doc = Document(filepath="data.txt", content=content, content_hash="hash3")
    chunks = list(chunker.process(doc))

    # Fits in one chunk
    assert len(chunks) == 1
    assert "Paragraph 1" in chunks[0].text
    assert "Paragraph 3" in chunks[0].text


def test_text_file_splitting():
    chunker = DocumentChunker(max_chars=30)
    content = "Paragraph 1 is quite long.\n\nParagraph 2 is also long."

    doc = Document(filepath="data.txt", content=content, content_hash="hash4")
    chunks = list(chunker.process(doc))

    assert len(chunks) == 2
    assert "Paragraph 1" in chunks[0].text
    assert "Paragraph 2" in chunks[1].text


def test_empty_document_yields_no_chunks():
    """Empty document should yield no chunks."""
    chunker = DocumentChunker(max_chars=100)
    doc = Document(filepath="empty.md", content="", content_hash="hash1")
    chunks = list(chunker.process(doc))
    assert len(chunks) == 0


def test_whitespace_only_document_yields_no_chunks():
    """Document with only whitespace should yield no chunks."""
    chunker = DocumentChunker(max_chars=100)
    doc = Document(filepath="whitespace.md", content="   \n\n  \t  \n", content_hash="hash2")
    chunks = list(chunker.process(doc))
    assert len(chunks) == 0


def test_content_below_threshold_filtered():
    """Content below 5-char threshold is filtered out from final chunks."""
    chunker = DocumentChunker(max_chars=100)
    # Mix of short and long content - short gets combined with long
    content = "Hi\n\nHello world\n\nX"

    doc = Document(filepath="short.md", content=content, content_hash="hash3")
    chunks = list(chunker.process(doc))

    # All paragraphs are combined into one chunk (under max_chars=100)
    # Since combined text is > 5 chars, it's kept
    assert len(chunks) == 1
    assert "Hello world" in chunks[0].text


def test_all_content_below_threshold_yields_no_chunks():
    """When all content is below 5 chars, no chunks are yielded."""
    chunker = DocumentChunker(max_chars=100)
    content = "Hi\n\nBye"  # Each paragraph < 5 chars, combined = "Hi\n\nBye" (7 chars)
    # Actually combined is 7 chars so it passes... let me try smaller
    content = "X\n\nY"  # Combined = "X\n\nY" (4 chars) - should be filtered

    doc = Document(filepath="tiny.md", content=content, content_hash="hash3")
    chunks = list(chunker.process(doc))

    assert len(chunks) == 0


def test_pure_code_fence_markdown():
    """Markdown with only code fence, no prose."""
    chunker = DocumentChunker(max_chars=200)
    content = """```python
def hello():
    return "world"
```"""

    doc = Document(filepath="code_only.md", content=content, content_hash="hash4")
    chunks = list(chunker.process(doc))

    assert len(chunks) == 1
    assert "```python" in chunks[0].text
    assert "def hello():" in chunks[0].text
    assert chunks[0].id == "code_only.md_chunk_0"


def test_txt_single_paragraph():
    """.txt file with single paragraph."""
    chunker = DocumentChunker(max_chars=500)
    content = "This is a single paragraph in a text file."

    doc = Document(filepath="single.txt", content=content, content_hash="hash5")
    chunks = list(chunker.process(doc))

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single paragraph in a text file."
    assert chunks[0].id == "single.txt_chunk_0"
    assert chunks[0].source == "single.txt"
