"""BrowseService — read-only analytical SQL over a SQL engine's table.

Execution core shared by the CLI (raw SQL passthrough, uncapped) and the MCP
(structured params compiled to SQL, capped). polars.SQLContext runs the SQL
over the Arrow read; filter VALUES on the MCP path are applied as bound polars
expressions, never interpolated (see build_and_run). No embedder, no semantic
ranking, no read-only guard (polars frames are in-memory; mutation is impossible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from dbs_vector.core.ports import IVectorStore

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

# Scalar columns selectable on the MCP path (raw_query gated separately;
# vector/workflow never present in the frame).
_SCALAR_COLUMNS = (
    "id", "content_hash", "user", "host", "source", "tables",
    "calls", "execution_time_ms", "lock_time_sec",
    "rows_examined", "rows_sent", "latest_ts", "text",
)
# Aggregate output names produced in grouped mode.
_GROUP_AGGREGATES = (
    "fingerprints", "calls", "execution_time_ms", "lock_time_sec",
    "rows_examined", "rows_sent", "latest_ts",
    "avg_ms_per_fingerprint", "avg_ms_per_call",
)
# Default non-grouped projection (raw_query appended only when allowed).
_DEFAULT_SELECT = (
    "id", "user", "host", "source", "tables", "calls",
    "execution_time_ms", "lock_time_sec", "rows_examined",
    "rows_sent", "latest_ts",
)
# filter param name -> column
_EQ_FILTERS = {"id": "id", "content_hash": "content_hash", "user": "user",
               "host": "host", "source": "source"}
_GE_FILTERS = {"min_calls": "calls", "min_execution_time_ms": "execution_time_ms",
               "min_lock_time_sec": "lock_time_sec"}


def _q(ident: str) -> str:
    """Quote an allowlisted identifier. Idents are validated against the
    vocabulary before reaching here, so this only adds quotes."""
    return '"' + ident.replace('"', "") + '"'


class BrowseError(Exception):
    """Raised when a browse SQL statement fails to execute."""


class BrowseValidationError(BrowseError):
    """Raised by the MCP structured builder on bad params. Safe to disclose:
    carries no infrastructure detail, so it is returned to the LLM verbatim."""


@dataclass
class BrowseResult:
    rows: list[dict[str, Any]]
    columns: list[str]
    total_matching: int
    grouped: bool
    limit_applied: bool


class BrowseService:
    def __init__(self, store: IVectorStore, frame_alias: str) -> None:
        self.store = store
        self.frame_alias = frame_alias

    # --- shared executor -------------------------------------------------
    def _frames(self) -> dict[str, pl.DataFrame]:
        df = pl.from_arrow(self.store.scan())
        if isinstance(df, pl.Series):  # pragma: no cover - from_arrow returns DataFrame for tables
            df = df.to_frame()
        return {self.frame_alias: df, "t": df, "t_by_table": df.explode("tables")}

    @staticmethod
    def _execute(sql: str, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
        try:
            return pl.SQLContext(frames=frames).execute(sql, eager=True)
        except Exception as e:  # surface polars message; caller decides disclosure
            raise BrowseError(str(e)) from e

    # --- CLI path: raw SQL, uncapped ------------------------------------
    def run_sql(self, sql: str) -> BrowseResult:
        result = self._execute(sql, self._frames())
        return BrowseResult(
            rows=result.to_dicts(),
            columns=result.columns,
            total_matching=result.height,
            grouped=False,
            limit_applied=False,
        )

    # --- MCP path: structured builder, bound-value filters, capped ------
    def build_and_run(
        self,
        *,
        filters: dict[str, Any],
        group_by: str | None,
        order_by: str,
        select: str | None,
        limit: int,
        allow_raw_queries: bool = False,
    ) -> BrowseResult:
        grouped = group_by is not None
        group_cols = self._parse_group_by(group_by)
        sql = self._build_sql(
            filters=filters, group_by=group_by, order_by=order_by,
            select=select, allow_raw_queries=allow_raw_queries,
        )
        frame = self._filtered_frame(filters, group_cols)
        result = self._execute(sql, {"data": frame})
        capped = result.head(limit)
        return BrowseResult(
            rows=capped.to_dicts(),
            columns=result.columns,
            total_matching=result.height,
            grouped=grouped,
            limit_applied=result.height > limit,
        )

    # --- builder helpers -------------------------------------------------
    @staticmethod
    def _parse_group_by(group_by: str | None) -> list[str]:
        if group_by is None:
            return []
        cols = [c.strip() for c in group_by.split(",") if c.strip()]
        for c in cols:
            if c not in _SCALAR_COLUMNS:
                raise BrowseValidationError(
                    f"Unknown group_by column '{c}'. Known: {', '.join(_SCALAR_COLUMNS)}."
                )
        return cols

    @staticmethod
    def _parse_order_by(order_by: str, valid: set[str]) -> tuple[str, str]:
        col, _, direction = order_by.partition(":")
        col = col.strip()
        direction = (direction.strip() or "desc").lower()
        if direction not in ("asc", "desc"):
            raise BrowseValidationError(
                f"order_by direction must be 'asc' or 'desc', got '{direction}'."
            )
        if col == "tables":
            raise BrowseValidationError(
                "Cannot order_by the list column 'tables'; group by it via group_by=tables."
            )
        if col not in valid:
            raise BrowseValidationError(
                f"Unknown order_by column '{col}'. Valid: {', '.join(sorted(valid))}."
            )
        return col, direction.upper()

    def _selectable(self, allow_raw_queries: bool) -> tuple[str, ...]:
        return _SCALAR_COLUMNS + (("raw_query",) if allow_raw_queries else ())

    def _build_sql(
        self,
        *,
        filters: dict[str, Any],
        group_by: str | None,
        order_by: str,
        select: str | None,
        allow_raw_queries: bool,
    ) -> str:
        group_cols = self._parse_group_by(group_by)
        if group_cols:
            return self._build_grouped_sql(group_cols, order_by, select)
        return self._build_flat_sql(order_by, select, allow_raw_queries)

    def _build_grouped_sql(
        self, group_cols: list[str], order_by: str, select: str | None
    ) -> str:
        agg_sql = {
            "fingerprints": "COUNT(*) AS fingerprints",
            "calls": 'SUM("calls") AS calls',
            "execution_time_ms": 'SUM("execution_time_ms") AS execution_time_ms',
            "lock_time_sec": 'SUM("lock_time_sec") AS lock_time_sec',
            "rows_examined": 'SUM("rows_examined") AS rows_examined',
            "rows_sent": 'SUM("rows_sent") AS rows_sent',
            "latest_ts": 'MAX("latest_ts") AS latest_ts',
            "avg_ms_per_fingerprint":
                'SUM("execution_time_ms")/NULLIF(COUNT(*),0) AS avg_ms_per_fingerprint',
            "avg_ms_per_call":
                'SUM("execution_time_ms")/NULLIF(SUM("calls"),0) AS avg_ms_per_call',
        }
        available = set(group_cols) | set(_GROUP_AGGREGATES)
        if select is not None:
            chosen = [c.strip() for c in select.split(",") if c.strip()]
            for c in chosen:
                if c not in available:
                    raise BrowseValidationError(
                        f"Grouped select column '{c}' is not a group column or "
                        f"aggregate. Available: {', '.join(sorted(available))}."
                    )
            out_group = [c for c in chosen if c in group_cols]
            out_agg = [c for c in chosen if c in _GROUP_AGGREGATES]
        else:
            out_group = list(group_cols)
            out_agg = list(_GROUP_AGGREGATES)
        select_terms = [_q(c) for c in out_group] + [agg_sql[c] for c in out_agg]
        col, direction = self._parse_order_by(order_by, available)
        frame = "data"
        return (
            f"SELECT {', '.join(select_terms)} FROM {frame} "
            f"GROUP BY {', '.join(_q(c) for c in group_cols)} "
            f"ORDER BY {_q(col)} {direction} NULLS LAST"
        )

    def _build_flat_sql(
        self, order_by: str, select: str | None, allow_raw_queries: bool
    ) -> str:
        selectable = self._selectable(allow_raw_queries)
        if select is not None:
            chosen = [c.strip() for c in select.split(",") if c.strip()]
            for c in chosen:
                if c == "raw_query" and not allow_raw_queries:
                    raise BrowseValidationError(
                        "raw query text is not exposed on this engine; start the "
                        "server with --allow-raw-queries to enable it."
                    )
                if c not in selectable:
                    raise BrowseValidationError(
                        f"Unknown select column '{c}'. Known: {', '.join(selectable)}."
                    )
        else:
            chosen = list(_DEFAULT_SELECT)
        col, direction = self._parse_order_by(order_by, set(selectable))
        return (
            f"SELECT {', '.join(_q(c) for c in chosen)} FROM data "
            f"ORDER BY {_q(col)} {direction} NULLS LAST"
        )

    def _filtered_frame(
        self, filters: dict[str, Any], group_cols: list[str]
    ) -> pl.DataFrame:
        from dbs_vector.core.models import _normalize_table_name

        df = pl.from_arrow(self.store.scan())
        if isinstance(df, pl.Series):  # pragma: no cover
            df = df.to_frame()
        explode = "tables" in group_cols
        frame = df.explode("tables") if explode else df
        exprs: list[pl.Expr] = []
        for param, column in _EQ_FILTERS.items():
            value = filters.get(param)
            if value is not None:
                exprs.append(pl.col(column) == value)
        for param, column in _GE_FILTERS.items():
            value = filters.get(param)
            if value is not None:
                exprs.append(pl.col(column) >= value)
        table = filters.get("table")
        if table is not None:
            normalized = _normalize_table_name(str(table))
            if explode:
                exprs.append(pl.col("tables") == normalized)
            else:
                exprs.append(pl.col("tables").list.contains(normalized))
        if exprs:
            frame = frame.filter(*exprs)
        return frame


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def result_to_json(result: BrowseResult) -> str:
    """Serialize all rows as JSON; datetimes → ISO 8601."""
    return json.dumps(result.rows, indent=2, ensure_ascii=False, default=_json_default)
