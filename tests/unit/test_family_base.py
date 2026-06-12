from dbs_vector.mcp.families.base import render_with_budget


def test_render_under_budget_joins_all_blocks():
    header = "HEADER\n"
    blocks = ["block-a\n", "block-b\n"]
    out = render_with_budget(header, blocks, budget_bytes=10_000)
    assert out == "\n".join([header, "block-a\n", "block-b\n"])
    assert "elided" not in out


def test_render_over_budget_emits_elision_footer_and_stays_within_budget():
    header = "HEADER\n"
    big = "x" * 600_000 + "\n"
    blocks = [big, big, big]  # ~1.8 MB of bodies, budget 1 MB
    out = render_with_budget(header, blocks, budget_bytes=1_000_000)
    assert len(out.encode("utf-8")) <= 1_000_000
    assert "results elided due to MCP response size cap" in out


def test_render_footer_counts_omitted_relative_to_total():
    header = "H\n"
    big = "y" * 900_000 + "\n"
    blocks = [big, big, big]  # only the first fits under 1 MB
    out = render_with_budget(header, blocks, budget_bytes=1_000_000)
    # first block emitted, remaining 2 elided, total 3
    assert "2 of 3 results elided due to MCP response size cap" in out


def test_render_first_block_oversized_emits_only_header_and_footer():
    """When even blocks[0] overflows, the elision path fires at idx=0: header is
    kept (never popped), the footer reports all blocks omitted, output ≤ budget."""
    header = "H\n"
    big = "z" * 2_000_000 + "\n"  # single block far exceeds budget
    blocks = [big, big]
    out = render_with_budget(header, blocks, budget_bytes=1_000_000)
    assert len(out.encode("utf-8")) <= 1_000_000
    assert out.startswith(header)
    assert "block" not in out  # no body emitted
    assert "2 of 2 results elided due to MCP response size cap" in out


def test_render_footer_dropped_when_even_header_plus_footer_overflows():
    """If the footer cannot fit alongside the header alone, no footer is emitted
    and output is just the header (the header is never popped)."""
    header = "h" * 50 + "\n"
    blocks = ["x" * 100 + "\n"]
    # Budget smaller than header itself ⇒ even header+footer overflows.
    out = render_with_budget(header, blocks, budget_bytes=10)
    assert out == header
    assert "elided" not in out
