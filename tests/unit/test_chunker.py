from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import DocumentChunker


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
