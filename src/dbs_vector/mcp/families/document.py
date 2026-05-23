"""DocumentFamily — search engines whose results are document chunks."""

import asyncio
from typing import Any

from dbs_vector.services.search import SearchService


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

        output = [f"Found {len(results)} results for '{query}':\n"]
        for res in results:
            if res.distance is not None:
                dist_str = f"{res.distance:.4f}"
            elif res.score is not None:
                dist_str = f"{res.score:.4f}"
            else:
                dist_str = "N/A (FTS)"
            chunk = res.chunk
            output.append(
                f"--- Result (Score: {dist_str}) ---\n"
                f"Source: {chunk.source}\n"
                f"Content:\n{chunk.text}\n"
            )
        return "\n".join(output)

    def make_handler(self, engine_name: str) -> Any:
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
