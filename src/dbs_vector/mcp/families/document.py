"""DocumentFamily — search engines whose results are document chunks."""

import asyncio
from typing import TYPE_CHECKING, Any

from dbs_vector.mcp.families.base import (
    RESPONSE_BUDGET_BYTES,
    embeddings_phrase,
    render_with_budget,
)
from dbs_vector.services.search import SearchService, retrieved_by_label

if TYPE_CHECKING:
    from dbs_vector.config import EngineConfig


class DocumentFamily:
    """SearchFamily implementation for document-style engines (markdown,
    prose, etc.)."""

    name: str = "document"

    def run_search(
        self,
        service: SearchService,
        query: str,
        limit: int,
        source_filter: str | None,
        **family_kwargs: Any,
    ) -> list[Any]:
        return service.execute_query(query, source_filter, limit, extra_filters={})

    def format_results(self, results: list[Any], query: str, total_matching: int = 0) -> str:
        if not results:
            return f"No results found for query: '{query}'"

        header = f"Found {len(results)} results for '{query}':\n"

        def _block(res: Any) -> str:
            chunk = res.chunk
            return (
                f"--- Result (similarity {res.similarity:.2f}, "
                f"retrieved by: {retrieved_by_label(res.retrieved_by)}) ---\n"
                f"Source: {chunk.source}\n"
                f"Content:\n{chunk.text}\n"
            )

        return render_with_budget(
            header,
            (_block(res) for res in results),
            RESPONSE_BUDGET_BYTES,
            total=len(results),
        )

    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        emb = embeddings_phrase(engine.model)
        return (
            f"Semantic search over Markdown documentation chunks ({emb}). "
            f"Returns the top-K most similar passages, ranked by cosine "
            f"similarity — not by recency or size."
        )

    def make_handler(self, engine_name: str, allow_raw_queries: bool = False) -> Any:
        # documents have no raw_query; the egress flag is accepted for
        # SearchFamily Protocol parity and intentionally ignored.
        family = self  # closure capture

        async def handler(
            query: str,
            limit: int = 5,
            source_filter: str | None = None,
        ) -> str:
            from dbs_vector.mcp.state import _services  # lazy import

            service = _services.get(engine_name)
            if service is None:
                return f"Error: search service '{engine_name}' is not initialized."
            try:
                results = await asyncio.to_thread(
                    family.run_search,
                    service,
                    query,
                    limit,
                    source_filter,
                )
                return family.format_results(results, query)
            except Exception as e:
                return f"Search execution failed: {e}"

        return handler
