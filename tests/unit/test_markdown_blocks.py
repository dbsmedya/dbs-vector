import pytest

from dbs_vector.infrastructure.chunking.markdown_blocks import MarkdownBlockParser, choose_fence


def test_parser_returns_heading_and_paragraph_blocks():
    blocks = MarkdownBlockParser().parse("# Title\n\nSome prose here.\n").blocks
    assert [(b.node_type, b.text) for b in blocks] == [
        ("heading", "Title"),
        ("section", "Some prose here."),
    ]


def test_parser_keeps_fence_atomic_with_info():
    blocks = MarkdownBlockParser().parse("```sql\nSELECT 1;\n```\n").blocks
    assert len(blocks) == 1
    assert blocks[0].node_type == "code"
    assert blocks[0].info == "sql"


def test_thematic_break_is_dropped():
    blocks = MarkdownBlockParser().parse("First part.\n\n---\n\nSecond part.\n").blocks
    assert all("---" not in b.text for b in blocks)
    assert [b.text for b in blocks] == ["First part.", "Second part."]


def test_indented_code_becomes_a_fenced_code_block():
    """Asserts the exact stored body, not just node_type. An under-budget
    indented block never reaches _split_code, so if parse() does not normalize
    it here it stays mangled — first line dedented by .strip(), the rest not."""
    blocks = MarkdownBlockParser().parse("# T\n\n    indented one\n    indented two\n").blocks
    code = [b for b in blocks if b.node_type == "code"]
    assert len(code) == 1
    assert code[0].info == ""
    assert code[0].text == "```\nindented one\nindented two\n```"


def test_indented_code_containing_a_literal_fence_uses_a_safe_delimiter():
    """A backtick wrapper would be closed early by the body's own ``` line;
    the shorter tilde alternative stays balanced without wasting budget."""
    src = "# T\n\n    before\n    ```\n    inside\n    ```\n    after\n"
    code = [b for b in MarkdownBlockParser().parse(src).blocks if b.node_type == "code"]
    assert len(code) == 1
    assert code[0].text.startswith("~~~")
    assert code[0].text.endswith("~~~")
    assert "\n```\ninside\n```\n" in code[0].text


def test_choose_fence_prefers_the_shortest_safe_delimiter():
    assert choose_fence("`" * 100) == "~~~"
    assert choose_fence("~" * 100) == "```"


def test_style_and_script_and_comment_html_blocks_are_dropped():
    src = (
        "<style>\nbody { color: red; }\n</style>\n\n"
        "<script>\nvar x = 1;\n</script>\n\n"
        "<!-- a comment -->\n\n"
        "Real prose.\n"
    )
    blocks = MarkdownBlockParser().parse(src).blocks
    assert [b.text for b in blocks] == ["Real prose."]


def test_html_table_is_kept():
    """GUARD test, not a red test — this already passes and must keep passing.
    It pins the narrowness of _is_noise_html so a later widening of the regex
    cannot start eating real content."""
    src = "<table>\n<tr><td>keep me</td></tr>\n</table>\n"
    blocks = MarkdownBlockParser().parse(src).blocks
    assert len(blocks) == 1
    assert "keep me" in blocks[0].text


def test_front_matter_is_dropped_and_title_lifted():
    doc = MarkdownBlockParser().parse("---\ntitle: My Doc\nauthor: Z\n---\n\n# Heading\n\nBody.\n")
    assert doc.title == "My Doc"
    assert all("---" not in b.text for b in doc.blocks)
    assert all("author" not in b.text for b in doc.blocks)


def test_title_equal_to_first_h1_is_not_lifted():
    doc = MarkdownBlockParser().parse("---\ntitle: Same Name\n---\n\n# same name\n\nBody.\n")
    assert doc.title is None


@pytest.mark.parametrize(
    "front",
    [
        "---\ntitle: [a, b]\n---\n",  # non-scalar title
        "---\n- just\n- a list\n---\n",  # non-mapping document
        "---\ntitle: 'unterminated\n---\n",  # malformed YAML
        "---\nauthor: Z\n---\n",  # no title key
    ],
)
def test_front_matter_without_usable_title_yields_none_and_never_raises(front):
    doc = MarkdownBlockParser().parse(front + "\n# H\n\nBody.\n")
    assert doc.title is None
