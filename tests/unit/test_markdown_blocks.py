from dbs_vector.infrastructure.chunking.markdown_blocks import MarkdownBlockParser


def test_parser_returns_heading_and_paragraph_blocks():
    blocks = MarkdownBlockParser().parse("# Title\n\nSome prose here.\n")
    assert [(b.node_type, b.text) for b in blocks] == [
        ("heading", "Title"),
        ("section", "Some prose here."),
    ]


def test_parser_keeps_fence_atomic_with_info():
    blocks = MarkdownBlockParser().parse("```sql\nSELECT 1;\n```\n")
    assert len(blocks) == 1
    assert blocks[0].node_type == "code"
    assert blocks[0].info == "sql"


def test_thematic_break_is_dropped():
    blocks = MarkdownBlockParser().parse("First part.\n\n---\n\nSecond part.\n")
    assert all("---" not in b.text for b in blocks)
    assert [b.text for b in blocks] == ["First part.", "Second part."]


def test_indented_code_becomes_a_fenced_code_block():
    """Asserts the exact stored body, not just node_type. An under-budget
    indented block never reaches _split_code, so if parse() does not normalize
    it here it stays mangled — first line dedented by .strip(), the rest not."""
    blocks = MarkdownBlockParser().parse("# T\n\n    indented one\n    indented two\n")
    code = [b for b in blocks if b.node_type == "code"]
    assert len(code) == 1
    assert code[0].info == ""
    assert code[0].text == "```\nindented one\nindented two\n```"


def test_indented_code_containing_a_literal_fence_uses_a_longer_delimiter():
    """A three-backtick wrapper would be closed early by the body's own ```
    line, spilling the remainder into prose."""
    # NOTE: parse() still returns a plain list here. `_ParsedDocument.blocks`
    # arrives in Task 3, which updates every Task 1-2 test in one pass.
    src = "# T\n\n    before\n    ```\n    inside\n    ```\n    after\n"
    code = [b for b in MarkdownBlockParser().parse(src) if b.node_type == "code"]
    assert len(code) == 1
    assert code[0].text.startswith("````")
    assert code[0].text.endswith("````")
    assert "\n```\ninside\n```\n" in code[0].text


def test_style_and_script_and_comment_html_blocks_are_dropped():
    src = (
        "<style>\nbody { color: red; }\n</style>\n\n"
        "<script>\nvar x = 1;\n</script>\n\n"
        "<!-- a comment -->\n\n"
        "Real prose.\n"
    )
    blocks = MarkdownBlockParser().parse(src)
    assert [b.text for b in blocks] == ["Real prose."]


def test_html_table_is_kept():
    """GUARD test, not a red test — this already passes and must keep passing.
    It pins the narrowness of _is_noise_html so a later widening of the regex
    cannot start eating real content."""
    src = "<table>\n<tr><td>keep me</td></tr>\n</table>\n"
    blocks = MarkdownBlockParser().parse(src)
    assert len(blocks) == 1
    assert "keep me" in blocks[0].text
