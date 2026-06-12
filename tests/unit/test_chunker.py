from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import DocumentChunker
from dbs_vector.infrastructure.chunking.filters import FilterRegistry


def _chunks(content, **kw):
    # length_fn defaults to len (chars) -> deterministic, model-free.
    kw.setdefault("target_tokens", 120)
    kw.setdefault("max_tokens", 240)
    kw.setdefault("min_tokens", 8)
    ch = DocumentChunker(**kw)
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
    chunks = _chunks('```python\ndef hello():\n    return "world"\n```')
    assert len(chunks) == 1
    assert "```python" in chunks[0].text
    assert "def hello():" in chunks[0].text
    assert chunks[0].id == "t.md_chunk_0"
    assert chunks[0].node_type == "code"


def test_sibling_sections_produce_separate_chunks():
    content = "## A\n\nFirst section body goes here.\n\n## B\n\nSecond section body goes here.\n"
    chunks = _chunks(content)
    scopes = {c.parent_scope for c in chunks}
    assert scopes == {"A", "B"}
    assert len(chunks) == 2


def test_tiny_trailing_block_merges_into_previous():
    # target_tokens=40 forces the long paragraph and the tiny 'Hi.' into
    # separate packed units; 'Hi.' (< min_tokens=8) then folds back into the
    # previous unit -> a single chunk that still contains 'Hi.'.
    content = "## S\n\nThis is a sufficiently long first paragraph here.\n\nHi.\n"
    chunks = _chunks(content, target_tokens=40)
    assert len(chunks) == 1
    assert "Hi." in chunks[0].text


# ---- oversized splitting + filter behavior ---------------------------------


def test_oversized_code_fence_splits_by_lines_never_truncates():
    code = "\n".join(f"line_{i} = {i}" for i in range(400))  # ~ large
    content = f"## Big\n\n```python\n{code}\n```\n"
    chunks = _chunks(content, target_tokens=120, max_tokens=240)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 240  # final text incl. heading path + fences fits max
        assert "```python" in c.text
    # reassembled code retains all lines
    joined = "\n".join(c.text for c in chunks)
    assert "line_0 = 0" in joined and "line_399 = 399" in joined


def test_oversized_list_splits_at_item_boundaries():
    items = "\n".join(f"- item number {i} with some words" for i in range(80))
    content = f"## L\n\n{items}\n"
    chunks = _chunks(content, target_tokens=100, max_tokens=200)
    assert len(chunks) > 1
    # no chunk starts mid-item (every body line under heading begins with '- ')
    for c in chunks:
        body = c.text.split("\n\n", 1)[-1]
        first = body.splitlines()[0]
        assert first.startswith("- ")


def test_oversized_table_repeats_header_each_part():
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(60))
    content = f"## T\n\n| a | b |\n|---|---|\n{rows}\n"
    chunks = _chunks(content, target_tokens=100, max_tokens=200)
    assert len(chunks) > 1
    for c in chunks:
        assert "| a | b |" in c.text  # header repeated


def test_single_huge_line_is_char_windowed():
    blob = "x" * 5000  # one line, no internal boundary
    content = f"## Blob\n\n```text\n{blob}\n```\n"
    chunks = _chunks(content, target_tokens=200, max_tokens=400)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 400  # char-window guarantees the invariant


def test_excalidraw_file_yields_no_chunks():
    ch = DocumentChunker(filters=FilterRegistry.resolve(["excalidraw"]))
    out = list(
        ch.process(Document(filepath="d.excalidraw.md", content="# x\n\nbody\n", content_hash="h"))
    )
    assert out == []


def test_compressed_json_block_is_dropped():
    content = "## Drawing\n\nIntro line that is long enough to keep.\n\n```compressed-json\nN4KAblob\n```\n"
    ch = DocumentChunker(
        target_tokens=120,
        max_tokens=240,
        filters=FilterRegistry.resolve(["compressed_json"]),
    )
    chunks = list(ch.process(Document(filepath="t.md", content=content, content_hash="h")))
    assert all("compressed-json" not in c.text and "N4KAblob" not in c.text for c in chunks)
    assert any("Intro line" in c.text for c in chunks)


def test_pack_atoms_measured_chars_grow_linearly_not_quadratically():
    """Running-sum estimate ⇒ total characters passed to length_fn are O(total
    input chars), not O(atoms × chunk_size).

    NOTE: counting CALLS does not discriminate — the old code already made
    only ~2 calls per atom. The quadratic quantity is the SIZE of the strings
    re-measured (the candidate grows toward `target` before each flush), so we
    sum chars.
    """
    measured = {"chars": 0}

    def counting_len(s: str) -> int:
        measured["chars"] += len(s)
        return len(s)

    chunker = DocumentChunker(
        target_tokens=50, max_tokens=100, min_tokens=1, length_fn=counting_len
    )
    atoms = [f"word{i}" for i in range(200)]  # ~1,300 chars of input total
    total_input = sum(len(a) for a in atoms)
    chunker._pack_atoms(atoms, " ", target=50, max_=100)

    # Old code re-measures the growing candidate every step ⇒ ~7,000+ chars.
    # Running-sum: each atom once + the joiner once ⇒ ≈ total_input.
    assert measured["chars"] <= 2 * total_input


def test_running_estimate_nets_out_special_token_overhead():
    """With a tokenizer that adds 2 special tokens to EVERY measurement (like
    the production count_tokens with add_special_tokens=True), the running sum
    must not pay those specials once per atom — the joined text pays them
    exactly once. Uncorrected summing packs far below target."""

    def tok(s: str) -> int:
        return (len(s.split()) if s else 0) + 2  # content words + BOS/EOS

    chunker = DocumentChunker(target_tokens=20, max_tokens=40, min_tokens=1, length_fn=tok)
    atoms = ["w w w"] * 12  # 3 content tokens each; k joined atoms measure 3k+2
    out = chunker._pack_atoms(atoms, "\n\n", target=20, max_=40)

    # Corrected estimate: first atom 5, each addition +3 → 6 atoms reach
    # exactly 20 ≤ target → 2 groups. Uncorrected: first 5, each addition
    # +7 (joiner 2 + atom 5) → flushes at 3 atoms → 4 groups.
    assert len(out) == 2, f"expected 2 groups of 6 atoms, got {len(out)}: {out}"
    assert all(tok(g) <= 20 for g in out)  # the estimate never under-counted


def test_tiny_merge_uses_net_estimate_not_inflated_sum():
    """A chunk genuinely below min_tokens must still be folded into its
    neighbour when the estimate carries per-atom special-token inflation.

    Uncorrected summing sees the trailing 3-word fragment as 13 "tokens"
    (specials counted once per paragraph + per-join cost) >= min_tokens=10
    and emits it standalone; its true size is 5."""

    def tok(s: str) -> int:
        return (len(s.split()) if s else 0) + 2  # content words + BOS/EOS

    # Empty prefix measures tok("")=2, so eff_target = 16-2 = 14. Lead: 12
    # words -> tok 14 == eff_target (cannot absorb more). Then three 1-word
    # paragraphs that pack into one trailing fragment: uncorrected est
    # 3 +(2+3)+(2+3) = 13 >= min_tokens -> escapes the merge; net est is 5.
    lead = " ".join(f"word{i}" for i in range(12))
    chunker = DocumentChunker(target_tokens=16, max_tokens=60, min_tokens=10, length_fn=tok)
    doc = Document(
        filepath="t.md",
        content=f"{lead}\n\nx\n\ny\n\nz\n",
        content_hash="h",
    )
    chunks = list(chunker.process(doc))

    # The fragment "x\n\ny\n\nz" (true size 5 < min_tokens) must fold into
    # the previous chunk, not be emitted as a standalone sub-minimum chunk.
    assert len(chunks) == 1, f"expected tiny trailing chunk merged, got: {[c.text for c in chunks]}"
