"""SqlFamily — search engines whose results are SQL query log entries."""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from dbs_vector.core.naming import normalize_tool_name
from dbs_vector.mcp.families.base import (
    RESPONSE_BUDGET_BYTES,
    embeddings_phrase,
    render_with_budget,
)
from dbs_vector.services.browse import (
    BrowseResult,
    BrowseService,
    BrowseValidationError,
)
from dbs_vector.services.search import SearchService

if TYPE_CHECKING:
    from dbs_vector.config import EngineConfig

_RAW_QUERY_DISPLAY_LIMIT = 2_000
_TRIAGE_SELECT = (
    "id,tables,calls,execution_time_ms,impact_score,avg_ms_per_call,"
    "lock_time_sec,rows_examined,rows_sent,selectivity,latest_ts"
)
_TRIAGE_ORDER_ALLOWLIST = (
    "impact_score",
    "execution_time_ms",
    "calls",
    "lock_time_sec",
    "avg_ms_per_call",
    "selectivity",
)


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


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, list):
        return _truncate_raw_query(", ".join(str(v) for v in value)) if value else "n/a"
    return _truncate_raw_query(str(value))


def format_browse(result: BrowseResult) -> str:
    """Render a BrowseResult under the MCP byte budget. One compact block per
    row ('col=val | col=val'); None cells render as 'n/a'."""
    if not result.rows:
        return "0 rows matched."
    noun = "groups" if result.grouped else "rows"
    header = f"Showing {len(result.rows)} of {result.total_matching} {noun}:\n"

    def _block(row: dict[str, Any]) -> str:
        return " | ".join(f"{col}={_fmt_cell(row.get(col))}" for col in result.columns)

    return render_with_budget(
        header,
        (_block(r) for r in result.rows),
        RESPONSE_BUDGET_BYTES,
        total=len(result.rows),
    )


def format_triage(result: BrowseResult) -> str:
    """Render curated triage rows, with raw_query on a paste-friendly block."""
    if not result.rows:
        return "0 rows matched."
    header = f"Showing {len(result.rows)} of {result.total_matching} fingerprints:\n"
    scalar_cols = [c for c in result.columns if c != "raw_query"]
    has_raw = "raw_query" in result.columns

    def _block(row: dict[str, Any]) -> str:
        line = " | ".join(f"{c}={_fmt_cell(row.get(c))}" for c in scalar_cols)
        if has_raw:
            raw = _truncate_raw_query(str(row.get("raw_query") or ""))
            return f"{line}\nRaw SQL:\n{raw}"
        return line

    return render_with_budget(
        header,
        (_block(r) for r in result.rows),
        RESPONSE_BUDGET_BYTES,
        total=len(result.rows),
    )


def _sql_source_phrase(chunker_type: str) -> str:
    return {"api": "a remote slow-log API", "duckdb": "a local DuckDB slow-query log"}.get(
        chunker_type, "a SQL slow-query log"
    )


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

        def _block(res: Any) -> str:
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
            return "\n".join(block_parts) + "\n"

        # Lazy generator + explicit total: blocks past the byte cap are never
        # formatted (with include_raw a block can be ~4 KB of string work).
        return render_with_budget(
            header,
            (_block(res) for res in results),
            RESPONSE_BUDGET_BYTES,
            total=len(results),
        )

    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        source = _sql_source_phrase(engine.chunker_type)
        emb = embeddings_phrase(engine.model)
        browse_tool = normalize_tool_name(engine_name, verb="browse")
        return (
            f"Semantic search over slow-query fingerprints from {source} "
            f"({emb}). Returns up to `limit` results ranked by cosine "
            f"similarity to the query string — NOT by execution_time_ms or "
            f"calls. Filters (optional, AND prefilters applied before "
            f"ranking): `min_time` — minimum cumulative execution_time_ms in "
            f"ms; `min_lock_time` — minimum cumulative lock_time_sec in "
            f"seconds; `table_filter` — restrict to fingerprints whose "
            f"`tables` list contains the given table. The header reports "
            f"'Showing N of M results that matched your filters' so callers "
            f"can tell when results are similarity-truncated. For ranking by "
            f"a scalar column, aggregation, or point lookup (no query string) "
            f"use the sibling `{browse_tool}` tool."
        )

    def browse_description(
        self, engine_name: str, engine: "EngineConfig", allow_raw_queries: bool
    ) -> str:
        source = _sql_source_phrase(engine.chunker_type)
        cols = (
            "id, content_hash, user, host, source, tables, calls, "
            "execution_time_ms, lock_time_sec, rows_examined, rows_sent, "
            "latest_ts, text"
        )
        if allow_raw_queries:
            cols += ", raw_query (verbatim production SQL with literal values)"
        return (
            f"Analytical (non-semantic) access to slow-query fingerprints from "
            f"{source}. Ranks by the column you choose, NOT by similarity — no "
            f"query string. Parameters: filters `id`, `content_hash`, `user`, "
            f"`host`, `source`, `table` (matches the `tables` list), "
            f"`min_calls`, `min_execution_time_ms`, `min_lock_time_sec`; "
            f"`group_by` (comma-separated columns — set to `tables` to group "
            f"by table); `order_by` ('<col>[:asc|:desc]', default "
            f"execution_time_ms:desc); `select` (comma-separated output "
            f"columns); `limit` (default 10). Columns: {cols}. Non-grouped mode "
            f"adds two derived columns (selectable and orderable): impact_score "
            f"(calls*execution_time_ms) and selectivity (rows_examined/rows_sent); "
            f"avg_ms_per_call (per-fingerprint execution_time_ms/calls) is available "
            f"in both modes. Grouping yields "
            f"fingerprints (COUNT), calls/execution_time_ms/lock_time_sec/"
            f"rows_examined/rows_sent (SUM), latest_ts (MAX), "
            f"avg_ms_per_fingerprint and avg_ms_per_call (the per-execution "
            f"average a DBA usually reads)."
        )

    def make_handler(self, engine_name: str, allow_raw_queries: bool = False) -> Any:
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
                extra_filters: dict[str, Any] = {}
                for key, value in (
                    ("min_time", min_time),
                    ("min_lock_time", min_lock_time),
                    ("table_filter", table_filter),
                ):
                    if value is not None:
                        extra_filters[key] = value

                # Search and count are independent, but both ultimately call
                # `self.table.checkout_latest()` on the SAME LanceDB Table handle,
                # which is a documented in-place mutation of the handle's version
                # pointer. `asyncio.to_thread` dispatches each call to a separate
                # ThreadPoolExecutor worker thread, so
                # `asyncio.gather(asyncio.to_thread(run_search...), asyncio.to_thread(count...))`
                # would run them on TWO concurrent threads — both mutating the same
                # handle simultaneously. lancedb 0.30.2 provides no guarantee that
                # concurrent checkout_latest+read sequences on one shared handle are
                # safe. Do NOT gather them. Running both sequentially inside ONE
                # `to_thread` closure frees the event loop without ever placing two
                # concurrent reads on the same handle.
                def _search_then_count() -> tuple[list[Any], int]:
                    r = family.run_search(
                        service,
                        query,
                        limit,
                        source_filter,
                        min_time=min_time,
                        min_lock_time=min_lock_time,
                        table_filter=table_filter,
                    )
                    t = service.count_matching(source_filter, extra_filters)
                    return r, t

                results, total = await asyncio.to_thread(_search_then_count)
                # Verbatim raw_query leaves the process ONLY when the server was
                # started with --allow-raw-queries. include_raw=True is silently
                # downgraded otherwise — the same lock browse already fits.
                effective_include_raw = include_raw and allow_raw_queries
                return family.format_results(
                    results, query, total_matching=total, include_raw=effective_include_raw
                )
            except Exception as e:
                return f"Search execution failed: {e}"

        return handler

    def make_browse_handler(self, engine_name: str, allow_raw_queries: bool) -> Any:
        async def handler(
            id: str | None = None,
            content_hash: str | None = None,
            user: str | None = None,
            host: str | None = None,
            source: str | None = None,
            table: str | None = None,
            min_calls: int | None = None,
            min_execution_time_ms: float | None = None,
            min_lock_time_sec: float | None = None,
            group_by: str | None = None,
            order_by: str = "execution_time_ms:desc",
            select: str | None = None,
            limit: int = 10,
        ) -> str:
            from loguru import logger

            from dbs_vector.mcp.state import _services

            service = _services.get(engine_name)
            if service is None:
                return f"Error: search service '{engine_name}' is not initialized."

            frame_alias = engine_name.replace("-", "_")
            browse = BrowseService(service.vector_store, frame_alias)
            filters = {
                "id": id,
                "content_hash": content_hash,
                "user": user,
                "host": host,
                "source": source,
                "table": table,
                "min_calls": min_calls,
                "min_execution_time_ms": min_execution_time_ms,
                "min_lock_time_sec": min_lock_time_sec,
            }

            def _run() -> BrowseResult:
                return browse.build_and_run(
                    filters=filters,
                    group_by=group_by,
                    order_by=order_by,
                    select=select,
                    limit=limit,
                    allow_raw_queries=allow_raw_queries,
                )

            try:
                result = await asyncio.to_thread(_run)
                return format_browse(result)
            except BrowseValidationError as e:
                return str(e)  # safe, author-controlled
            except Exception as e:  # infra: log full, return generic
                logger.warning("browse '{}' failed: {}", engine_name, e)
                return "browse execution failed (see server logs)."

        return handler

    def triage_description(
        self, engine_name: str, engine: "EngineConfig", allow_raw_queries: bool
    ) -> str:
        source = _sql_source_phrase(engine.chunker_type)
        raw = (
            " When the server was started with --allow-raw-queries AND "
            "include_raw=true, a truncated verbatim raw_query exemplar (ready to "
            "paste into a MySQL EXPLAIN) is appended."
            if allow_raw_queries
            else ""
        )
        return (
            f"Triage the highest-impact slow-query fingerprints from {source}. "
            f"Returns the top `limit` (default 10) ranked by impact_score = "
            f"calls * execution_time_ms (frequency-weighted 'what is hammering the "
            f"database'). Columns: id, tables, calls, execution_time_ms, "
            f"impact_score, avg_ms_per_call, lock_time_sec, rows_examined, "
            f"rows_sent, selectivity, latest_ts. NOTE: rows_examined/rows_sent are "
            f"the most-recent call's values (not averages). Optional params: "
            f"`table` (scope to a table, lowercased match), `min_calls`, "
            f"`order_by` ('<col>[:asc|:desc]', default impact_score:desc; col one "
            f"of {', '.join(_TRIAGE_ORDER_ALLOWLIST)}), `include_raw`.{raw}"
        )

    def make_triage_handler(self, engine_name: str, allow_raw_queries: bool = False) -> Any:
        async def handler(
            limit: int = 10,
            table: str | None = None,
            order_by: str = "impact_score:desc",
            min_calls: int | None = None,
            include_raw: bool = False,
        ) -> str:
            from loguru import logger

            from dbs_vector.mcp.state import _services

            service = _services.get(engine_name)
            if service is None:
                return f"Error: search service '{engine_name}' is not initialized."

            col = order_by.partition(":")[0].strip()
            if col not in _TRIAGE_ORDER_ALLOWLIST:
                return (
                    f"order_by must be one of {', '.join(_TRIAGE_ORDER_ALLOWLIST)}; "
                    f"got '{col}'."
                )

            select = _TRIAGE_SELECT
            # Silent downgrade (search-style): raw exemplar only under the flag.
            if include_raw and allow_raw_queries:
                select += ",raw_query"

            frame_alias = engine_name.replace("-", "_")
            browse = BrowseService(service.vector_store, frame_alias)
            filters = {"table": table, "min_calls": min_calls}

            def _run() -> BrowseResult:
                return browse.build_and_run(
                    filters=filters,
                    group_by=None,
                    order_by=order_by,
                    select=select,
                    limit=limit,
                    allow_raw_queries=allow_raw_queries,
                )

            try:
                result = await asyncio.to_thread(_run)
                return format_triage(result)
            except BrowseValidationError as e:
                return str(e)  # safe, author-controlled
            except Exception as e:  # infra: log full, return generic
                logger.warning("triage '{}' failed: {}", engine_name, e)
                return "triage execution failed (see server logs)."

        return handler
