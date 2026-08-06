from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import (
    DocumentChunker,
    _escape_collisions,
    _PackedFragment,
    _PackedSection,
    _PackedUnit,
    _render_boundary,
)
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


def test_oversized_fenced_code_containing_a_literal_fence_stays_balanced():
    inner = "\n".join(["```", "nested opener", "```"] + ["line " + "x" * 40 for _ in range(80)])
    chunks = _chunks(f"~~~py\n{inner}\n~~~\n", target_tokens=120, max_tokens=240)
    assert len(chunks) > 1
    for c in chunks:
        body = c.text.split("\n\n", 1)[-1] if c.parent_scope else c.text
        lines = body.split("\n")
        assert lines[0].startswith("(code, part "), "part marker must survive"
        opener = lines[1]
        delim = opener[: len(opener) - len(opener.lstrip("`"))]
        assert len(delim) >= 4, "opener must outrun the literal ``` in the body"
        assert lines[-1] == delim, "closing delimiter must equal the opener"
        assert len(c.text) <= 240, "the hard size invariant survives the wider fence"


def test_backtick_in_info_string_forces_a_tilde_fence():
    inner = "\n".join("line " + "x" * 40 for _ in range(80))
    chunks = _chunks("~~~py`variant\n" + inner + "\n~~~\n", target_tokens=120, max_tokens=240)
    for c in chunks:
        body = c.text.split("\n\n", 1)[-1] if c.parent_scope else c.text
        opener = [ln for ln in body.split("\n") if ln.strip()][1]
        assert opener.startswith("~~~"), "a backtick in the info string rules out a backtick fence"


def _assert_balanced_parts(chunks, max_tokens, min_parts):
    markers = [ln for c in chunks for ln in c.text.split("\n") if ln.startswith("(code, part ")]
    assert len(markers) >= min_parts, "fixture must actually reach a three-digit part count"
    for c in chunks:
        body = c.text.split("\n\n", 1)[-1] if c.parent_scope else c.text
        lines = [ln for ln in body.split("\n") if ln.strip()]
        opener = lines[1]
        delim = opener[: len(opener) - len(opener.lstrip("`"))]
        assert delim and lines[-1] == delim, "fence must stay balanced at 3-digit part counts"
        assert len(c.text) <= max_tokens


def test_hundred_plus_part_code_block_keeps_balanced_fences_at_tight_budget():
    """`(code, part 99/99)` under-reserves once the part count reaches three
    digits. With target == max there is no slack to absorb the wider marker,
    the rendered part exceeds max_tokens, and _compose falls back to
    _char_window — which slices mid-fence and destroys the balance guarantee
    the other tests assert.

    Line width is load-bearing. The OLD reservation is exactly 29 characters
    — `(code, part 99/99)` (18) + `\n```py\n` (7) + `\n``` ` (4) — leaving
    `bm = 31`. Lines of exactly 31 characters fill that budget precisely, so
    the real 3-digit marker (`(code, part 400/400)`, 20 chars vs the 18
    reserved) pushes each rendered part to 62 against a 60 cap and exposes the
    mid-fence fallback. Wider lines are char-windowed into UNDERFILLED pieces
    whose slack absorbs the extra digit, and the test passes against the broken
    splitter — verified: 35-character lines give 800 balanced parts, max length
    58/60, green before the fix.
    """
    inner = "\n".join("x" * 31 for _ in range(400))
    chunks = _chunks(f"```py\n{inner}\n```\n", target_tokens=60, max_tokens=60)
    _assert_balanced_parts(chunks, 60, 100)


def test_single_very_long_line_also_reaches_a_three_digit_part_count():
    """Falsifies the atom-count bound directly: ONE atom, hundreds of parts.

    `_pack_atoms` char-windows any atom longer than `max_` (document.py:346-350),
    so `len(inner) == 1` here while the block splits into ~400+ parts. A digit
    reservation derived from atom count would under-reserve badly. Minified
    JSON and base64 blobs are the real-world shape.

    The budget must stay FEASIBLE: at max_tokens=10 the reserved marker alone
    is 22 characters, `bm` collapses to 1, and every rendered part overflows
    into _char_window no matter how correct the implementation is — the test
    would be permanently red. 10k characters at max_tokens=60 leaves ~23
    characters of body per part, which still yields well over 100 parts.
    """
    inner = "x" * 10_000
    chunks = _chunks(f"```json\n{inner}\n```\n", target_tokens=60, max_tokens=60)
    _assert_balanced_parts(chunks, 60, 100)


def test_front_matter_title_reaches_the_breadcrumb():
    src = "---\ntitle: Ops Guide\n---\n\n# Backups\n\nRun the backup nightly.\n"
    chunks = _chunks(src)
    assert chunks[0].parent_scope == "Ops Guide > Backups"
    assert chunks[0].text.startswith("Ops Guide > Backups\n\n")


def test_packing_restructure_is_behaviour_neutral():
    """Golden output for a document exercising headings, prose, code and tables.

    Task 4 is a pure refactor; this pins the exact output so the restructure
    cannot silently move a boundary.
    """
    src = (
        "# Guide\n\nIntro prose that is reasonably long so it packs.\n\n"
        "## Setup\n\nStep one text.\n\n```sh\nmake install\n```\n\n"
        "## Table\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    )
    chunks = _chunks(src)
    assert [(c.node_type, c.parent_scope) for c in chunks] == [
        ("section", "Guide"),
        ("section", "Guide > Setup"),
        ("table", "Guide > Table"),
    ]
    assert chunks[0].text == "Guide\n\nIntro prose that is reasonably long so it packs."


def _table(rows: int) -> str:
    head = "| Name | Scope | Val |\n|---|---|---|\n"
    return head + "".join(f"| var_{i} | Global | {i} |\n" for i in range(rows))


def test_table_inside_admonition_repeats_the_header_on_every_chunk():
    """The correctness invariant from issue #7.

    An LLM handed headerless data rows infers column meanings from the values
    and states them confidently. Every part must carry the header.
    """
    indented = "".join("    " + ln + "\n" for ln in _table(60).strip().split("\n"))
    src = '# Reference\n\n!!! note "Full variable table"\n\n' + indented
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    assert len(chunks) > 1, "table must actually split for this test to mean anything"
    for c in chunks:
        assert "| Name | Scope | Val |" in c.text


def test_admonition_body_reaches_the_table_node_type():
    indented = "".join("    " + ln + "\n" for ln in _table(3).strip().split("\n"))
    src = '!!! note "Small"\n\n' + indented
    chunks = _chunks(src)
    assert chunks[0].node_type == "table"


def test_admonition_frame_reaches_the_breadcrumb():
    src = '# Guide\n\n!!! warning "Data loss risk"\n\n    Be careful here.\n'
    chunks = _chunks(src)
    assert chunks[0].parent_scope == "Guide > warning: Data loss risk"


def test_tab_indented_admonition_body_is_parsed_not_treated_as_code():
    """Genuinely red: today a tab-indented body is an indented code block, so
    the table never reaches _split_table. A single-line assertion on leading
    whitespace would pass today because `.strip()` already removes it — this
    asserts the STRUCTURE instead."""
    tabbed = "".join("\t" + ln + "\n" for ln in _table(3).strip().split("\n"))
    chunks = _chunks('!!! note "Tabbed"\n\n' + tabbed)
    assert chunks[0].node_type == "table"
    assert chunks[0].parent_scope == "note: Tabbed"


def test_frame_type_label_is_casefolded_but_title_casing_is_kept():
    src = '!!! Warning "Data Loss Risk"\n\n    Body.\n'
    chunks = _chunks(src)
    assert chunks[0].parent_scope == "warning: Data Loss Risk"


def test_two_adjacent_admonitions_do_not_pack_together():
    src = '!!! note "One"\n\n    First.\n\n!!! note "Two"\n\n    Second.\n'
    chunks = _chunks(src)
    assert len(chunks) == 2
    assert "Second." not in chunks[0].text


def test_prose_around_an_admonition_is_not_packed_under_its_frame():
    src = 'Before text.\n\n!!! note "Mid"\n\n    Inside.\n\nAfter text.\n'
    chunks = _chunks(src)
    framed = [c for c in chunks if c.parent_scope and "note" in c.parent_scope]
    assert len(framed) == 1
    assert "Before text." not in framed[0].text
    assert "After text." not in framed[0].text


def test_empty_admonition_disappears_entirely():
    src = '# H\n\n!!! note "Nothing here"\n\nAfter.\n'
    chunks = _chunks(src)
    # Assert presence FIRST: `all(...)` over an empty list is vacuously true,
    # so without this the test would pass if the chunker emitted nothing at all.
    assert [c.text for c in chunks if "After." in c.text], "the real content must survive"
    assert all("!!!" not in c.text for c in chunks)
    assert all(c.parent_scope == "H" for c in chunks)


def test_heading_inside_a_container_does_not_change_following_breadcrumbs():
    src = '# Guide\n\n!!! note "N"\n\n    ## Not A Real Heading\n\n    Inside.\n\nAfter text.\n'
    chunks = _chunks(src)
    after = [c for c in chunks if "After text." in c.text]
    assert after and after[0].parent_scope == "Guide"


def test_nested_container_wrapper_is_unprefixed_by_its_ancestor():
    src = '!!! note "N"\n\n    > quoted inside admonition\n'
    chunks = _chunks(src)
    assert "> quoted inside admonition" in chunks[0].text
    assert "    > quoted" not in chunks[0].text


def test_small_blockquote_keeps_its_markers_and_packs_with_prose():
    src = "# H\n\nBefore.\n\n> A short quote.\n\nAfter.\n"
    chunks = _chunks(src)
    assert len(chunks) == 1
    assert "> A short quote." in chunks[0].text
    assert chunks[0].parent_scope == "H"


def test_small_alert_blockquote_enters_scope_and_does_not_absorb_prose():
    src = "# H\n\nBefore.\n\n> [!WARNING]\n> Careful.\n\nAfter.\n"
    chunks = _chunks(src)
    framed = [c for c in chunks if c.parent_scope and "warning" in c.parent_scope]
    assert len(framed) == 1
    assert "Before." not in framed[0].text and "After." not in framed[0].text


def test_alert_type_is_casefolded():
    src = "> [!Warning]\n> Careful.\n"
    chunks = _chunks(src)
    assert chunks[0].parent_scope == "warning"


def test_oversized_blockquote_splits_on_block_boundaries_not_mid_fence():
    body = "\n".join(f"> line {i}" for i in range(200))
    src = f"> ```sql\n> SELECT 1;\n> ```\n>\n{body}\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    fenced = [c for c in chunks if "```" in c.text]
    for c in fenced:
        assert c.text.count("```") % 2 == 0, "a fence was split across chunks"


def test_expanded_ordinary_blockquote_gets_the_quote_frame():
    src = "> " + ("word " * 400) + "\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    assert all(c.parent_scope == "quote" for c in chunks)


def test_small_nested_quote_inside_an_oversized_quote_keeps_one_marker_level():
    """Selection is recursive: the outer quote is oversized so it expands, but
    the small inner quote still fits and stays atomic — so exactly one marker
    level was removed from it. Two oversized levels would expand twice and
    leave no markers, which is why this fixture makes only the outer one big."""
    src = "> " + ("word " * 400) + "\n>\n> > a short nested quote\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    joined = "\n".join(c.text for c in chunks)
    assert "> a short nested quote" in joined
    assert ">> a short nested quote" not in joined


def test_two_oversized_quote_levels_expand_fully():
    inner = "> > " + ("word " * 400) + "\n"
    chunks = _chunks(inner, target_tokens=120, max_tokens=240)
    assert all(not c.text.lstrip().startswith(">") for c in chunks)


def test_lazy_continuation_line_survives_unprefixing():
    src = "> first line\nlazy second line\n" + "> " + ("word " * 400) + "\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    joined = " ".join(c.text for c in chunks)
    assert "lazy second line" in joined


# ---- Re-homed from Task 5 (require blockquote registration + _select_form,
# both of which land in this task) -------------------------------------------


def test_empty_admonition_inside_an_expanded_blockquote_also_disappears():
    """The path _flatten_always_expand cannot reach: the admonition is only
    uncovered when _select_form expands the enclosing quote."""
    src = '> !!! note "Nothing here"\n>\n> ' + ("word " * 400) + "\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    assert chunks
    assert all("!!!" not in c.text for c in chunks)


def test_oversized_blockquote_inside_an_admonition_loses_both_prefixes():
    """Mixed syntaxes: the four-space admonition indent must be removed BEFORE
    the blockquote matcher runs, or `^ {0,3}>` never matches and the `>`
    markers survive into chunk text."""
    body = "\n".join("    > word " + "x" * 20 for _ in range(80))
    src = '!!! note "N"\n\n' + body + "\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    assert chunks
    for c in chunks:
        # Inspect BODY lines only. The breadcrumb prefix is legitimately
        # "note: N > quote" — its " > " separator would fail a naive
        # `">" not in c.text` check on the whole chunk.
        body_lines = (
            c.text.split("\n\n", 1)[1].split("\n") if c.parent_scope else c.text.split("\n")
        )
        assert all(not ln.lstrip().startswith(">") for ln in body_lines)
        assert c.parent_scope and c.parent_scope.startswith("note: N")


def test_admonition_inside_an_expanded_blockquote():
    inner = "\n".join("> " + "y" * 60 for _ in range(60))
    src = '> !!! warning "W"\n>\n>     inside the admonition\n>\n' + inner + "\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    framed = [c for c in chunks if c.parent_scope and "warning: W" in c.parent_scope]
    assert framed, "the nested admonition must be reachable once the quote expands"
    assert "inside the admonition" in "\n".join(c.text for c in framed)


def test_nested_alert_blockquote_is_detected_at_its_own_level():
    inner = "\n".join("> > word " + "z" * 20 for _ in range(80))
    src = "> " + ("pad " * 400) + "\n>\n> > [!WARNING]\n" + inner + "\n"
    chunks = _chunks(src, target_tokens=120, max_tokens=240)
    assert any(c.parent_scope and "warning" in c.parent_scope for c in chunks)


def test_filters_apply_to_children_of_an_expanded_container():
    from dbs_vector.infrastructure.chunking.filters import FilterRegistry

    # CompressedJsonFilter keys on the `compressed-json` info string
    # (filters.py:26) — a plain ```json fence is NOT dropped, and using one
    # here would fail the test even with the filter correctly placed.
    #
    # STRENGTHENED (moved from Task 5): the original assertion checked for
    # the absence of the full ~1400-char `big` blob, which char-windowing
    # (max_tokens=240) would strip out of every chunk regardless of whether
    # the filter fired at all — a false-positive risk. `marker` is a small,
    # distinctive, repeated fragment that WOULD survive char-windowing intact
    # if the filter fails to reach the expanded child, making this a real
    # detector: absent if the filter worked, present in some chunk if not.
    marker = "UNIQUE_MARKER_7f3a"
    big = '{"a":' + ",".join(f"{marker}{i}" for i in range(400)) + "}"
    src = "> ```compressed-json\n> " + big + "\n> ```\n> \n> " + ("word " * 400) + "\n"
    chunks = _chunks(
        src,
        target_tokens=120,
        max_tokens=240,
        filters=FilterRegistry.resolve(["compressed_json"]),
    )
    assert all(marker not in c.text for c in chunks)


def _unit(text, *, frames=(), verbatim=False):
    return _PackedUnit(
        fragments=(_PackedFragment(text=text, node_type="section", verbatim=verbatim),),
        node_type="section",
        start=0,
        end=1,
        scope=(),
        frames=frames,
        eff_target=100,
        eff_max=200,
        est=len(text),
    )


def test_render_boundary_emits_atx_parent_and_frames():
    section = _PackedSection(
        document_title=None,
        ancestors=((1, "Guide"), (2, "X")),
        heading=(3, "A", 4),
        units=(_unit("body"),),
    )
    out = _render_boundary(section, _unit("body", frames=("warning: Data loss risk",)))
    assert out == (
        "### A\n"
        '(dbs-vector context: parent="Guide > X")\n'
        '(dbs-vector context: frame="warning: Data loss risk")'
    )


def test_render_boundary_omits_empty_parent_and_frames():
    section = _PackedSection(None, (), (2, "B", 9), (_unit("body"),))
    assert _render_boundary(section, _unit("body")) == "## B"


def test_render_boundary_for_preamble_has_no_atx_line():
    section = _PackedSection("Ops Guide", (), None, (_unit("body"),))
    assert _render_boundary(section, _unit("body")) == '(dbs-vector context: parent="Ops Guide")'


def test_json_escaping_of_quotes_and_parens_in_values():
    section = _PackedSection(None, (), (2, 'He said ")x"', 1), (_unit("b"),))
    out = _render_boundary(section, _unit("b", frames=('note: a"b)c',)))
    assert '(dbs-vector context: frame="note: a\\"b)c")' in out


def test_authored_lookalike_line_gains_one_backslash():
    unit = _unit('(dbs-vector context: parent="fake")\nreal text')
    assert _escape_collisions(unit).text.startswith("\\(dbs-vector context:")


def test_existing_backslash_run_is_extended_not_replaced():
    unit = _unit("\\(dbs-vector context: x)")
    assert _escape_collisions(unit).text.startswith("\\\\(dbs-vector context:")


def test_verbatim_code_fragments_are_never_escaped():
    unit = _PackedUnit(
        fragments=(
            _PackedFragment('(dbs-vector context: parent="x")', "code", verbatim=True),
            _PackedFragment('(dbs-vector context: parent="y")', "section", verbatim=False),
        ),
        node_type="section",
        start=0,
        end=1,
        scope=(),
        frames=(),
        eff_target=100,
        eff_max=200,
        est=0,
    )
    out = _escape_collisions(unit)
    assert out.fragments[0].text == '(dbs-vector context: parent="x")'
    assert out.fragments[1].text.startswith("\\(dbs-vector context:")
