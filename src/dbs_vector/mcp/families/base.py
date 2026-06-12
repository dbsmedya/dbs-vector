"""SearchFamily Protocol: contract that each search family implements."""

from typing import Any, Protocol

from dbs_vector.services.search import SearchService


def _byte_len(value: str) -> int:
    return len(value.encode("utf-8"))


def render_with_budget(header: str, blocks: list[str], budget_bytes: int) -> str:
    """Join `header` + `blocks` under a UTF-8 byte budget, appending an
    elision footer when trailing blocks must be dropped to fit.

    Each block is appended whole; if the next block would push the joined
    payload over `budget_bytes`, emission stops and a footer reporting how
    many of the total blocks were elided is appended (popping already-added
    blocks if even the footer would overflow). Output is byte-identical to the
    prior inline SqlFamily implementation, so both families — and the future
    ImageFamily — can share one transport-ceiling guard.
    """
    output = [header]
    total = len(blocks)

    def _append_elision_footer(omitted: int) -> None:
        footer = f"[{omitted} of {total} results elided due to MCP response size cap]"
        while len(output) > 1 and _byte_len("\n".join([*output, footer])) > budget_bytes:
            output.pop()
            omitted += 1
            footer = f"[{omitted} of {total} results elided due to MCP response size cap]"
        if _byte_len("\n".join([*output, footer])) <= budget_bytes:
            output.append(footer)

    for idx, block in enumerate(blocks):
        # Re-joins the accumulated output each step (O(n·total_chars)); benign
        # because the byte budget caps total_chars at ~budget_bytes (~1 MB), so
        # n is bounded by budget_bytes / min_block_bytes.
        candidate = "\n".join([*output, block])
        if _byte_len(candidate) > budget_bytes:
            _append_elision_footer(total - idx)
            break
        output.append(block)
    return "\n".join(output)


class SearchFamily(Protocol):
    """Self-contained MCP-layer plugin for a class of search engines.

    Each family owns:
      - A search dispatcher (translate kwargs → service call → list of results).
      - A result formatter (translate results → human-readable string).
      - A handler factory (build a per-engine async function with a concrete
        signature that FastMCP will introspect for its tool schema).

    The handler signature returned by make_handler() IS the family's public
    argument schema. There is no separate args_model — duplication risks
    drift, and FastMCP's introspection works on the handler directly.
    """

    name: str  # e.g., "document", "sql"; must match a key in FamilyKeyRegistry

    def run_search(
        self,
        service: SearchService,
        query: str,
        limit: int,
        source_filter: str | None,
        **family_kwargs: Any,
    ) -> list[Any]:
        """Run the search and return the raw result list."""
        ...

    def format_results(self, results: list[Any], query: str, total_matching: int = 0) -> str:
        """Render results for an MCP tool's stdout.

        `total_matching` is the count of rows surviving non-semantic
        prefilters, regardless of whether they ranked above the similarity or
        FTS threshold. Families may surface it or ignore it.
        """
        ...

    def make_handler(self, engine_name: str) -> Any:
        """Build a per-engine async handler whose explicit signature FastMCP
        will introspect for the tool schema."""
        ...
