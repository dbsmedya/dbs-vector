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
