"""SearchFamily Protocol: contract that each search family implements."""

from typing import Any, Protocol

from dbs_vector.services.search import SearchService


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

    def format_results(self, results: list[Any], query: str) -> str:
        """Render results for an MCP tool's stdout."""
        ...

    def make_handler(self, engine_name: str) -> Any:
        """Build a per-engine async handler whose explicit signature FastMCP
        will introspect for the tool schema."""
        ...
