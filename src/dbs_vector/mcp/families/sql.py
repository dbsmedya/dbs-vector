"""SqlFamily — search engines whose results are SQL query log entries."""

import asyncio
from typing import Any

from dbs_vector.services.search import SearchService


class SqlFamily:
    """SearchFamily implementation for SQL-log-style engines.

    Adds three family-specific filters on top of query/limit/source_filter:
      - `min_time`       — minimum cumulative execution_time_ms
      - `min_lock_time`  — minimum cumulative lock_time_sec
      - `table_filter`   — restrict to queries that touch a specific table
                           (matches against the `tables` list column)
    """

    name: str = "sql"

    def run_search(
        self,
        service: SearchService,
        query: str,
        limit: int,
        source_filter: str | None,
        **family_kwargs: Any,
    ) -> list[Any]:
        extra_filters: dict[str, Any] = {}
        for key in ("min_time", "min_lock_time", "table_filter"):
            value = family_kwargs.get(key)
            if value is not None:
                extra_filters[key] = value
        return service.execute_query(query, source_filter, limit, extra_filters=extra_filters)

    def format_results(self, results: list[Any], query: str) -> str:
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
                f"Source Database: {chunk.source}\n"
                f"Execution Time: {chunk.execution_time_ms}ms (Calls: {chunk.calls})\n"
                f"SQL Query:\n{chunk.raw_query}\n"
            )
        return "\n".join(output)

    def make_handler(self, engine_name: str) -> Any:
        family = self  # closure capture

        async def handler(
            query: str,
            limit: int = 5,
            source_filter: str | None = None,
            min_time: float | None = None,
            min_lock_time: float | None = None,
            table_filter: str | None = None,
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
                    min_time=min_time,
                    min_lock_time=min_lock_time,
                    table_filter=table_filter,
                )
                return family.format_results(results, query)
            except Exception as e:
                return f"Search execution failed: {e}"

        return handler
