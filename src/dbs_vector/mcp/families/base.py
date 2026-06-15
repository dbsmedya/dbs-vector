"""SearchFamily Protocol: contract that each search family implements."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dbs_vector.services.search import SearchService

if TYPE_CHECKING:
    from dbs_vector.config import EngineConfig

# MCP response-size cap shared by every family formatter. A transport-level
# property, not a per-family choice — defined once, beside the helper.
RESPONSE_BUDGET_BYTES = 1_000_000


def _byte_len(value: str) -> int:
    return len(value.encode("utf-8"))


def embeddings_phrase(model: str) -> str:
    """Human label for an embedding model key, shared across families."""
    return {"granite-r2": "Granite embeddings", "gemma-bf16": "Gemma embeddings"}.get(
        model, f"{model} embeddings"
    )


def render_with_budget(
    header: str,
    blocks: Iterable[str],
    budget_bytes: int = RESPONSE_BUDGET_BYTES,
    total: int | None = None,
) -> str:
    """Join `header` + `blocks` under a UTF-8 byte budget, appending an
    elision footer when trailing blocks must be dropped to fit.

    Each block is appended whole; if the next block would push the joined
    payload over `budget_bytes`, emission stops and a footer reporting how
    many of the total blocks were elided is appended (popping already-added
    blocks if even the footer would overflow). Output is byte-identical to the
    prior inline SqlFamily implementation, so both families — and the future
    ImageFamily — can share one transport-ceiling guard.

    `blocks` may be a LAZY iterable so callers skip formatting blocks past the
    cap; pass `total` (the result count) in that case, because the footer
    reports "[X of TOTAL results elided]" and a consumed iterator cannot be
    counted after the early exit. Byte accounting is incremental — one join at
    return — but decision-for-decision identical to re-joining per block.
    """
    if total is None:
        blocks = list(blocks)
        total = len(blocks)

    output = [header]
    used = _byte_len(header)  # bytes of "\n".join(output)

    def _footer(omitted: int) -> tuple[str, int]:
        text = f"[{omitted} of {total} results elided due to MCP response size cap]"
        return text, 1 + _byte_len(text)  # +1 for the "\n" join separator

    emitted = 0
    for block in blocks:
        cost = 1 + _byte_len(block)
        if used + cost > budget_bytes:
            omitted = total - emitted
            footer, fcost = _footer(omitted)
            while len(output) > 1 and used + fcost > budget_bytes:
                used -= 1 + _byte_len(output.pop())
                omitted += 1
                footer, fcost = _footer(omitted)
            if used + fcost <= budget_bytes:
                output.append(footer)
            break
        output.append(block)
        used += cost
        emitted += 1
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

    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        """Compose the LLM-facing description for this engine's search tool
        from inert config facts (chunker_type, model) on the passed `engine`.
        Takes `engine` directly (not via the global settings) so it composes
        purely from its arguments — no hidden global read, trivially testable.
        Families own this prose; config.yaml holds only a short human summary."""
        ...

    def make_handler(self, engine_name: str, allow_raw_queries: bool = False) -> Any:
        """Build a per-engine async handler whose explicit signature FastMCP
        will introspect for the tool schema.

        `allow_raw_queries` is the server-level egress flag (from
        --allow-raw-queries). SQL families honour it to gate verbatim
        `raw_query`; families with no raw payload accept and ignore it.
        Defaults to False so a forgotten wiring fails closed."""
        ...


@runtime_checkable
class BrowseFamily(Protocol):
    """A family that additionally exposes a non-semantic `browse` MCP tool.

    Only SQL-style families implement this; document families do not. Declared
    separately from SearchFamily so the browse registrar can narrow the
    FamilyRegistry's SearchFamily to a browse-capable family with isinstance().
    """

    def make_browse_handler(self, engine_name: str, allow_raw_queries: bool) -> Any:
        """Build a per-engine async browse handler for FastMCP."""
        ...

    def browse_description(
        self, engine_name: str, engine: "EngineConfig", allow_raw_queries: bool
    ) -> str:
        """Compose the LLM-facing description for this engine's browse tool."""
        ...
