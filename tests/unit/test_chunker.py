from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import DocumentChunker


def _chunks(content, **kw):
    # length_fn defaults to len (chars) -> deterministic, model-free.
    ch = DocumentChunker(target_tokens=120, max_tokens=240, min_tokens=8, **kw)
    return list(ch.process(Document(filepath="t.md", content=content, content_hash="h")))


# ---- KEPT UNCHANGED --------------------------------------------------------


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


# ---- ADAPTED (empty/whitespace/below-threshold) ----------------------------
# Intent preserved: empty/whitespace .md → no chunks; <5-char filter still
# tested on the .txt path where it remains in effect.


def test_empty_document_yields_no_chunks():
    """Empty .md document should yield no chunks."""
    chunker = DocumentChunker(max_chars=100)
    doc = Document(filepath="empty.md", content="", content_hash="hash1")
    chunks = list(chunker.process(doc))
    assert len(chunks) == 0


def test_whitespace_only_document_yields_no_chunks():
    """Whitespace-only .md document should yield no chunks."""
    chunker = DocumentChunker(max_chars=100)
    doc = Document(filepath="whitespace.md", content="   \n\n  \t  \n", content_hash="hash2")
    chunks = list(chunker.process(doc))
    assert len(chunks) == 0


def test_content_below_threshold_filtered_txt():
    """Content below 5-char threshold is filtered out from final .txt chunks.

    Adapted from test_content_below_threshold_filtered: the <5-char gate lives
    in _chunk_text so the intent is best exercised on the .txt path.
    """
    chunker = DocumentChunker(max_chars=100)
    # Three paragraphs under max_chars=100 → packed into one chunk (8 chars+),
    # which passes the >=5-char filter.
    content = "Hi\n\nHello world\n\nX"
    doc = Document(filepath="short.txt", content=content, content_hash="hash3")
    chunks = list(chunker.process(doc))
    assert len(chunks) == 1
    assert "Hello world" in chunks[0].text


def test_all_content_below_threshold_yields_no_chunks_txt():
    """When all combined content is <5 chars on the .txt path, no chunks are yielded.

    Adapted from test_all_content_below_threshold_yields_no_chunks to target
    the .txt path where the >=5-char filter is enforced.
    """
    chunker = DocumentChunker(max_chars=100)
    content = "X\n\nY"  # Combined = "X\n\nY" (4 chars) → filtered by >=5 rule
    doc = Document(filepath="tiny.txt", content=content, content_hash="hash3")
    chunks = list(chunker.process(doc))
    assert len(chunks) == 0


# ---- REPLACED (assert old greedy markdown behavior) -----------------------
# New tests assert heading-aware, token-sized, metadata-carrying behavior.


def test_heading_is_prepended_not_emitted_alone():
    content = "# Top\n\n## Setup\n\nInstall the package and run it.\n"
    chunks = _chunks(content)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text.startswith("Top > Setup")
    assert "Install the package" in c.text
    assert c.parent_scope == "Top > Setup"
    assert c.node_type == "section"
    # invariant: final text (heading path included) never exceeds max_tokens
    assert len(c.text) <= 240


def test_bare_heading_with_no_body_produces_no_chunk():
    content = "# Parent\n\n## Child\n\n### Grandchild\n\nReal content here.\n"
    chunks = _chunks(content)
    assert len(chunks) == 1
    assert chunks[0].parent_scope == "Parent > Child > Grandchild"


def test_code_fence_under_budget_stays_atomic():
    content = "## Code\n\n```python\ndef f():\n    return 1\n```\n"
    chunks = _chunks(content)
    assert len(chunks) == 1
    assert "```python" in chunks[0].text
    assert chunks[0].node_type == "code"


def test_metadata_line_range_is_one_based_inclusive():
    # lines: 0="## A", 1="", 2="body...". The body paragraph's markdown-it map
    # is [2, 3] (0-based, end-exclusive) -> 1-based inclusive "3-3".
    content = "## A\n\nbody text that is long enough.\n"
    c = _chunks(content)[0]
    assert c.line_range == "3-3"
    assert c.node_type == "section"


def test_pure_code_fence_markdown():
    """Markdown with only code fence, no prose."""
    chunks = _chunks("```python\ndef hello():\n    return \"world\"\n```")
    assert len(chunks) == 1
    assert "```python" in chunks[0].text
    assert "def hello():" in chunks[0].text
    assert chunks[0].id == "t.md_chunk_0"
    assert chunks[0].node_type == "code"
