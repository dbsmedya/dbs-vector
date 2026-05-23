"""SqlFamily — search engines whose results are SQL query log entries."""

import asyncio
from datetime import datetime
from typing import Any

from dbs_vector.services.search import SearchService

_RAW_QUERY_DISPLAY_LIMIT = 2_000
_RESPONSE_BUDGET_BYTES = 1_000_000


def _byte_len(value: str) -> int:
    return len(value.encode("utf-8"))


def _truncate_raw_query(raw_query: str) -> str:
    if len(raw_query) <= _RAW_QUERY_DISPLAY_LIMIT:
        return raw_query
    elided = len(raw_query) - _RAW_QUERY_DISPLAY_LIMIT
    return f"{raw_query[:_RAW_QUERY_DISPLAY_LIMIT]}\n... ({elided:,} more chars elided)"


def _fmt_int(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "n/a"


def _fmt_float(value: Any, suffix: str = "") -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:,.3f}{suffix}"
    return "n/a"


def _fmt_ts(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else "n/a"


def _fmt_selectivity(rows_examined: Any, rows_sent: Any) -> str:
    if (
        isinstance(rows_examined, int)
        and isinstance(rows_sent, int)
        and not isinstance(rows_sent, bool)
        and rows_sent > 0
    ):
        return f"{rows_examined / rows_sent:,.0f}:1"
    return "n/a"


def _fmt_tables(tables: Any) -> str:
    if isinstance(tables, list) and tables:
        return ", ".join(tables)
    return "n/a"


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

    def format_results(
        self,
        results: list[Any],
        query: str,
        total_matching: int = 0,
        include_raw: bool = False,
    ) -> str:
        if not results:
            if total_matching > 0:
                return (
                    f"No results found for query: '{query}'. "
                    f"({total_matching} rows matched your filters but "
                    f"none ranked above the similarity/FTS threshold - "
                    f"try broadening the query or relaxing filters.)"
                )
            return f"No results found for query: '{query}'"

        if len(results) > total_matching:
            from loguru import logger

            logger.warning(
                "format_results: len(results)={} exceeds total_matching={} "
                "for query={!r} — count_matching and search disagreed on the "
                "filter set. Investigate the filter pipeline.",
                len(results),
                total_matching,
                query,
            )
            header = (
                f"Showing {len(results)} results for '{query}' "
                f"(ranked by similarity). WARNING: count_matching reported "
                f"only {total_matching} rows would match the same filters — "
                f"this is a filter/count mismatch; see server logs.\n"
            )
        else:
            header = (
                f"Showing {len(results)} of {total_matching} results "
                f"that matched your filters for '{query}' "
                f"(ranked by similarity):\n"
            )
        output = [header]

        def _append_elision_footer(omitted: int) -> None:
            footer = f"[{omitted} of {len(results)} results elided due to MCP response size cap]"
            while (
                len(output) > 1 and _byte_len("\n".join([*output, footer])) > _RESPONSE_BUDGET_BYTES
            ):
                output.pop()
                omitted += 1
                footer = (
                    f"[{omitted} of {len(results)} results elided due to MCP response size cap]"
                )
            if _byte_len("\n".join([*output, footer])) <= _RESPONSE_BUDGET_BYTES:
                output.append(footer)

        for idx, res in enumerate(results):
            if res.distance is not None:
                dist_str = f"{res.distance:.4f}"
            elif res.score is not None:
                dist_str = f"{res.score:.4f}"
            else:
                dist_str = "N/A (FTS)"
            chunk = res.chunk

            normalized_sql = _truncate_raw_query(chunk.text or "")
            block_parts = [
                f"--- Result (Score: {dist_str}) ---",
                f"Fingerprint ID: {chunk.id}",
                f"Source Database: {chunk.source or 'n/a'}",
                f"Tables: {_fmt_tables(chunk.tables)}",
                f"Host: {chunk.host or 'n/a'}  User: {chunk.user or 'n/a'}",
                f"Last Seen: {_fmt_ts(chunk.latest_ts)}",
                f"Execution Time: {_fmt_float(chunk.execution_time_ms, 'ms')} "
                f"(Calls: {_fmt_int(chunk.calls)})",
                f"Rows Examined / Sent: {_fmt_int(chunk.rows_examined)} / "
                f"{_fmt_int(chunk.rows_sent)} "
                f"(selectivity {_fmt_selectivity(chunk.rows_examined, chunk.rows_sent)})",
                f"Lock Time: {_fmt_float(chunk.lock_time_sec, 's')}",
                f"Normalized SQL:\n{normalized_sql}",
            ]
            if include_raw:
                block_parts.append(f"Raw SQL:\n{_truncate_raw_query(chunk.raw_query or '')}")
            block = "\n".join(block_parts) + "\n"

            candidate = "\n".join([*output, block])
            if _byte_len(candidate) > _RESPONSE_BUDGET_BYTES:
                _append_elision_footer(len(results) - idx)
                break
            output.append(block)
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
            include_raw: bool = False,
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
                extra_filters: dict[str, Any] = {}
                for key, value in (
                    ("min_time", min_time),
                    ("min_lock_time", min_lock_time),
                    ("table_filter", table_filter),
                ):
                    if value is not None:
                        extra_filters[key] = value
                total = await asyncio.to_thread(
                    service.count_matching,
                    source_filter,
                    extra_filters,
                )
                return family.format_results(
                    results, query, total_matching=total, include_raw=include_raw
                )
            except Exception as e:
                return f"Search execution failed: {e}"

        return handler
