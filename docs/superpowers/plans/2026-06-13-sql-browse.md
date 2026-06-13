# `browse` — Analytical SQL Access for SQL Engines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only analytical `browse` verb to SQL engines — raw SQL on the CLI, structured params on the MCP — executed by polars over the LanceDB table, with no embedder and no semantic ranking.

**Architecture:** One execution core (`services/browse.py:BrowseService`) reads the table as Arrow via a new `IVectorStore.scan()` port, loads it into a polars frame, and runs SQL through `polars.SQLContext`. Two front-ends: the CLI passes raw SQL straight through (`run_sql`, uncapped, full column access); the MCP compiles **typed params** to a builder-controlled SQL string with filter *values* applied as bound polars expressions (never interpolated), so file-reads / injection are structurally impossible (`build_and_run`, capped via `head(limit)`). A server-level `--allow-raw-queries` flag gates only the verbatim `raw_query` PII column on the MCP. Tool descriptions move from `config.yaml` into the families.

**Tech Stack:** Python 3.12, polars 1.40 (`SQLContext`, expression API), pyarrow, LanceDB 0.30.2, FastMCP, Typer, pytest. No new dependencies (sqlglot is NOT used by browse).

**Spec:** `docs/superpowers/specs/2026-06-13-sql-browse-design.md`

**Verified polars behavior (tests must match these, not the spec's prose):**
- `NULLIF(SUM(x),0)` divide-by-zero → `null` (reliable).
- Grouped `SUM` of an all-null group → `0.0` (NOT null — the spec's "sums to NULL" line came from an ungrouped probe; assert `0.0`).
- `df.explode("tables")` turns an empty list `[]` into one row with `tables = null`.
- `df.filter(pl.col("user") == "x' OR 1=1--")` matches 0 rows — bound values are opaque data.
- `table.search().select(cols).to_arrow()` returns ALL rows (no default cap).

---

## File Structure

**New files:**
- `src/dbs_vector/services/browse.py` — `BrowseService`, `BrowseResult`, `BrowseError`/`BrowseValidationError`, column vocabulary constants, JSON/table renderers. The whole execution core.
- `tests/unit/test_browse_service.py` — `run_sql`, `build_and_run`, injection regression, NULL/agg behavior.
- `tests/unit/test_browse_descriptions.py` — `search_description` / `browse_description` composition.
- `tests/unit/test_browse_mcp_handler.py` — `make_browse_handler` formatting + error sanitization.
- `tests/integration/test_browse_integration.py` — real tmpdir LanceDB end-to-end.

**Modified files:**
- `src/dbs_vector/core/ports.py` — add `scan()` to `IVectorStore`.
- `src/dbs_vector/infrastructure/storage/lancedb_engine.py` — implement `scan()`.
- `src/dbs_vector/services/bootstrap.py` — add `build_store()` (store-only; no embedder/chunker).
- `src/dbs_vector/core/naming.py` — add `verb` param to `normalize_tool_name`.
- `src/dbs_vector/mcp/families/base.py` — add `search_description` to `SearchFamily` Protocol.
- `src/dbs_vector/mcp/families/document.py` — implement `search_description`.
- `src/dbs_vector/mcp/families/sql.py` — `search_description`, `browse_description`, `make_browse_handler`, `format_browse`.
- `src/dbs_vector/mcp/dynamic_tools.py` — `register_browse_tools`; switch `register_search_tools` description to family.
- `src/dbs_vector/mcp/server.py` — `start_stdio_server(allow_raw_queries)` + call `register_browse_tools`.
- `src/dbs_vector/cli.py` — new `browse` command; `--allow-raw-queries` on `mcp`.
- `config.yaml` — shorten the six engine `description:` fields.

---

## Task 1: `scan()` port + LanceDBStore implementation

**Files:**
- Modify: `src/dbs_vector/core/ports.py` (add to `IVectorStore`, after `count_matching`, ~line 97)
- Modify: `src/dbs_vector/infrastructure/storage/lancedb_engine.py` (add method after `count_matching`, ~line 286)
- Test: `tests/integration/test_browse_integration.py` (new; this task adds the scan test only)

- [ ] **Step 1: Add `scan` to the `IVectorStore` Protocol**

In `src/dbs_vector/core/ports.py`, add this method to the `IVectorStore` class (after `count_matching`):

```python
    def scan(self, columns: list[str] | None = None) -> Any:
        """Read ALL rows as a pyarrow.Table for in-process analytical SQL.

        `columns=None` projects every column EXCEPT the embedding `vector`
        and `workflow`. Implementations MUST call checkout_latest() first
        (like search()/count_matching()) so the read sees the latest version.
        No vector query, no row cap.
        """
        ...
```

- [ ] **Step 2: Write the failing integration test for `scan`**

Create `tests/integration/test_browse_integration.py`:

```python
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pytest

from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import SqlMapper

VECTOR_DIM = 4


def _make_store(tmp_path) -> LanceDBStore:
    mapper = SqlMapper(VECTOR_DIM)
    return LanceDBStore(
        db_path=str(tmp_path / "db"),
        table_name="query_vault",
        vector_dimension=VECTOR_DIM,
        mapper=mapper,
    )


def _seed(store: LanceDBStore) -> None:
    from dbs_vector.core.models import SqlChunk

    chunks = [
        SqlChunk(
            id="A", text="select 1", raw_query="SELECT 1 WHERE email='a@x.com'",
            source="db1", execution_time_ms=100.0, calls=10, content_hash="h1",
            tables=["orders", "items"], latest_ts=datetime(2026, 1, 1, tzinfo=UTC),
            user="alice", host="h1", rows_sent=5, rows_examined=50, lock_time_sec=1.0,
        ),
        SqlChunk(
            id="B", text="select 2", raw_query="SELECT 2",
            source="db1", execution_time_ms=50.0, calls=5, content_hash="h2",
            tables=["orders"], latest_ts=datetime(2026, 1, 2, tzinfo=UTC),
            user="bob", host="h2", rows_sent=1, rows_examined=2, lock_time_sec=None,
        ),
    ]
    vectors = np.ones((len(chunks), VECTOR_DIM), dtype=np.float32)
    store.ingest_chunks(chunks, vectors, workflow="sql_clustering")


def test_scan_returns_all_rows_without_vector_or_workflow(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)

    table = store.scan()

    assert table.num_rows == 2
    assert "vector" not in table.schema.names
    assert "workflow" not in table.schema.names
    assert "raw_query" in table.schema.names
    assert "id" in table.schema.names
    assert set(table.column("id").to_pylist()) == {"A", "B"}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/integration/test_browse_integration.py::test_scan_returns_all_rows_without_vector_or_workflow -v`
Expected: FAIL — `AttributeError`/`TypeError` (scan not implemented; Protocol method is `...`).

- [ ] **Step 4: Implement `scan` on `LanceDBStore`**

In `src/dbs_vector/infrastructure/storage/lancedb_engine.py`, add after `count_matching`:

```python
    def scan(self, columns: list[str] | None = None) -> Any:
        """Read all rows as a pyarrow.Table, projecting out vector/workflow.

        Mirrors get_existing_hashes' column-projected query (avoids
        materialising the embedding vector) but selects the full analytical
        column set. checkout_latest() first so a long-lived MCP server sees
        rows committed by a separate ingest process.
        """
        self.table.checkout_latest()
        if columns is None:
            columns = [c for c in self.schema.names if c not in ("vector", "workflow")]
        if len(self.table) == 0:
            return self.table.search().select(columns).limit(1).to_arrow().slice(0, 0)
        return self.table.search().select(columns).to_arrow()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/integration/test_browse_integration.py::test_scan_returns_all_rows_without_vector_or_workflow -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/core/ports.py src/dbs_vector/infrastructure/storage/lancedb_engine.py tests/integration/test_browse_integration.py
git commit -m "feat(browse): add IVectorStore.scan() reading table as projected Arrow"
```

---

## Task 2: `BrowseService` core — `BrowseResult`, errors, `_execute`, `run_sql` (CLI path)

**Files:**
- Create: `src/dbs_vector/services/browse.py`
- Test: `tests/unit/test_browse_service.py`

- [ ] **Step 1: Write failing tests for `run_sql`**

Create `tests/unit/test_browse_service.py`:

```python
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from dbs_vector.services.browse import (
    BrowseResult,
    BrowseService,
    BrowseValidationError,
)


class FakeStore:
    """Minimal IVectorStore.scan stand-in returning a fixed Arrow table."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table
        self.scanned = 0

    def scan(self, columns=None):
        self.scanned += 1
        return self._table


def _table() -> pa.Table:
    return pa.table(
        {
            "id": ["A", "B", "C"],
            "content_hash": ["h1", "h2", "h3"],
            "text": ["select 1", "select 2", "select 3"],
            "raw_query": ["SELECT 1 /*a@x.com*/", "SELECT 2", "SELECT 3"],
            "source": ["db1", "db1", "db2"],
            "user": ["alice", "bob", None],
            "host": ["h1", "h2", None],
            "tables": [["orders", "items"], ["orders"], []],
            "calls": [10, 5, 1],
            "execution_time_ms": [100.0, 50.0, 25.0],
            "lock_time_sec": [None, None, None],
            "rows_examined": [50, 20, 5],
            "rows_sent": [5, 2, 1],
            "latest_ts": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 3, tzinfo=UTC),
            ],
        }
    )


def _svc() -> BrowseService:
    return BrowseService(FakeStore(_table()), frame_alias="sql_api")


def test_run_sql_returns_all_rows_uncapped():
    result = _svc().run_sql("SELECT id, calls FROM t ORDER BY calls DESC")
    assert isinstance(result, BrowseResult)
    assert result.total_matching == 3
    assert result.limit_applied is False
    assert result.grouped is False
    assert [r["id"] for r in result.rows] == ["A", "B", "C"]


def test_run_sql_respects_authors_own_limit():
    result = _svc().run_sql("SELECT id FROM t ORDER BY calls DESC LIMIT 1")
    assert result.total_matching == 1
    assert [r["id"] for r in result.rows] == ["A"]


def test_run_sql_quoted_user_and_alias_frame():
    result = _svc().run_sql('SELECT "user" FROM sql_api WHERE "user" = \'alice\'')
    assert [r["user"] for r in result.rows] == ["alice"]


def test_run_sql_t_by_table_explodes():
    result = _svc().run_sql("SELECT id FROM t_by_table WHERE tables = 'orders'")
    assert sorted(r["id"] for r in result.rows) == ["A", "B"]


def test_run_sql_bad_column_raises_browse_error():
    with pytest.raises(Exception):  # BrowseError; polars message surfaced
        _svc().run_sql("SELECT nonexistent_col FROM t")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_browse_service.py -v`
Expected: FAIL — `ModuleNotFoundError: dbs_vector.services.browse`.

- [ ] **Step 3: Create `services/browse.py` with the core + `run_sql`**

Create `src/dbs_vector/services/browse.py`:

```python
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


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def result_to_json(result: BrowseResult) -> str:
    """Serialize all rows as JSON; datetimes → ISO 8601."""
    return json.dumps(result.rows, indent=2, ensure_ascii=False, default=_json_default)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_browse_service.py -v`
Expected: PASS (all five). The bad-column test passes because `_execute` wraps the polars error in `BrowseError`.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/services/browse.py tests/unit/test_browse_service.py
git commit -m "feat(browse): BrowseService core + run_sql (CLI raw SQL path)"
```

---

## Task 3: `build_and_run` — MCP structured builder (bound-value filters, aggregates, validation)

**Files:**
- Modify: `src/dbs_vector/services/browse.py`
- Test: `tests/unit/test_browse_service.py`

- [ ] **Step 1: Write failing tests for `build_and_run`**

Append to `tests/unit/test_browse_service.py`:

```python
def test_build_and_run_grouped_aggregates_by_user():
    result = _svc().build_and_run(
        filters={}, group_by="user", order_by="execution_time_ms:desc",
        select=None, limit=10,
    )
    assert result.grouped is True
    # one row per distinct user incl. the null group
    assert result.total_matching == 3
    cols = result.columns
    for expected in ["fingerprints", "calls", "execution_time_ms",
                     "avg_ms_per_fingerprint", "avg_ms_per_call", "latest_ts"]:
        assert expected in cols
    top = result.rows[0]
    assert top["user"] == "alice"
    assert top["fingerprints"] == 1
    assert top["execution_time_ms"] == 100.0


def test_build_and_run_caps_to_limit_but_reports_total():
    result = _svc().build_and_run(
        filters={}, group_by="user", order_by="execution_time_ms:desc",
        select=None, limit=1,
    )
    assert result.total_matching == 3
    assert len(result.rows) == 1
    assert result.limit_applied is True


def test_build_and_run_filter_values_are_bound_not_interpolated():
    svc = _svc()
    sql = svc._build_sql(
        filters={"user": "alice"}, group_by=None,
        order_by="execution_time_ms:desc", select=None, allow_raw_queries=False,
    )
    # the value 'alice' must NOT appear in the generated SQL string
    assert "alice" not in sql


def test_build_and_run_injection_value_is_inert():
    svc = _svc()
    payload = "x') UNION SELECT 1 FROM read_csv('/etc/passwd')--"
    result = svc.build_and_run(
        filters={"source": payload}, group_by=None,
        order_by="execution_time_ms:desc", select=None, limit=10,
    )
    assert result.rows == []                 # matches no row; never executes
    sql = svc._build_sql(
        filters={"source": payload}, group_by=None,
        order_by="execution_time_ms:desc", select=None, allow_raw_queries=False,
    )
    assert "read_csv" not in sql and "passwd" not in sql


def test_build_and_run_div_by_zero_average_is_null():
    # craft a frame whose only group has SUM(calls)=0 → NULLIF → null
    tbl = pa.table({
        "id": ["A"], "content_hash": ["h"], "text": ["x"], "raw_query": ["x"],
        "source": ["db1"], "user": ["alice"], "host": ["h"],
        "tables": [["orders"]], "calls": [0], "execution_time_ms": [10.0],
        "lock_time_sec": [None], "rows_examined": [1], "rows_sent": [1],
        "latest_ts": [datetime(2026, 1, 1, tzinfo=UTC)],
    })
    svc = BrowseService(FakeStore(tbl), frame_alias="sql_api")
    result = svc.build_and_run(
        filters={}, group_by="user", order_by="execution_time_ms:desc",
        select=None, limit=10,
    )
    assert result.rows[0]["avg_ms_per_call"] is None    # rendered n/a downstream


def test_build_and_run_group_by_tables_uses_exploded_frame():
    result = _svc().build_and_run(
        filters={}, group_by="tables", order_by="fingerprints:desc",
        select=None, limit=10,
    )
    counts = {r["tables"]: r["fingerprints"] for r in result.rows}
    assert counts.get("orders") == 2


def test_build_and_run_table_filter_uses_list_contains():
    result = _svc().build_and_run(
        filters={"table": "orders"}, group_by=None,
        order_by="calls:desc", select=None, limit=10,
    )
    assert sorted(r["id"] for r in result.rows) == ["A", "B"]


def test_build_and_run_rejects_unknown_column():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={}, group_by="nonsense", order_by="execution_time_ms:desc",
            select=None, limit=10,
        )


def test_build_and_run_rejects_order_by_tables_scalar():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={}, group_by=None, order_by="tables:desc",
            select=None, limit=10,
        )


def test_build_and_run_rejects_grouped_select_of_raw_field():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={}, group_by="user", order_by="execution_time_ms:desc",
            select="id", limit=10,
        )


def test_build_and_run_raw_query_gated_off():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={}, group_by=None, order_by="execution_time_ms:desc",
            select="id,raw_query", limit=10, allow_raw_queries=False,
        )


def test_build_and_run_raw_query_allowed_on():
    result = _svc().build_and_run(
        filters={}, group_by=None, order_by="execution_time_ms:desc",
        select="id,raw_query", limit=10, allow_raw_queries=True,
    )
    assert "raw_query" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_browse_service.py -k build_and_run -v`
Expected: FAIL — `AttributeError: 'BrowseService' has no attribute 'build_and_run'`.

- [ ] **Step 3: Implement the vocabulary constants + `build_and_run` + `_build_sql`**

In `src/dbs_vector/services/browse.py`, add the constants below the imports (after the dataclass is fine):

```python
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
# filter param name -> (column, comparison)
_EQ_FILTERS = {"id": "id", "content_hash": "content_hash", "user": "user",
               "host": "host", "source": "source"}
_GE_FILTERS = {"min_calls": "calls", "min_execution_time_ms": "execution_time_ms",
               "min_lock_time_sec": "lock_time_sec"}


def _q(ident: str) -> str:
    """Quote an allowlisted identifier. Idents are validated against the
    vocabulary before reaching here, so this only adds quotes."""
    return '"' + ident.replace('"', "") + '"'
```

Then add these methods to `BrowseService` (after `run_sql`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_browse_service.py -v`
Expected: PASS (all `run_sql` and `build_and_run` tests, including the injection and div-by-zero cases).

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/services/browse.py tests/unit/test_browse_service.py
git commit -m "feat(browse): build_and_run — bound-value filters, curated aggregates, validation"
```

---

## Task 4: Generalize `normalize_tool_name` with a verb

**Files:**
- Modify: `src/dbs_vector/core/naming.py:15-17`
- Test: `tests/unit/test_naming_verb.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_naming_verb.py`:

```python
from dbs_vector.core.naming import normalize_tool_name


def test_default_verb_is_search():
    assert normalize_tool_name("sql-api") == "search_sql_api"


def test_browse_verb():
    assert normalize_tool_name("sql-api", verb="browse") == "browse_sql_api"
    assert normalize_tool_name("sql", verb="browse") == "browse_sql"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_naming_verb.py -v`
Expected: FAIL — `TypeError: normalize_tool_name() got an unexpected keyword argument 'verb'`.

- [ ] **Step 3: Add the `verb` parameter**

Replace `normalize_tool_name` in `src/dbs_vector/core/naming.py`:

```python
def normalize_tool_name(engine_name: str, verb: str = "search") -> str:
    """Convert an engine name to its MCP tool name (dashes → underscores).

    `verb` selects the tool family prefix: "search" (default) or "browse".
    """
    return f"{verb}_{engine_name.replace('-', '_')}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_naming_verb.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/core/naming.py tests/unit/test_naming_verb.py
git commit -m "feat(browse): parameterize normalize_tool_name with verb"
```

---

## Task 5: Family descriptions — `search_description` + `browse_description`

**Files:**
- Modify: `src/dbs_vector/mcp/families/base.py` (add to Protocol, ~after line 103)
- Modify: `src/dbs_vector/mcp/families/document.py` (add method to `DocumentFamily`)
- Modify: `src/dbs_vector/mcp/families/sql.py` (add methods to `SqlFamily`)
- Test: `tests/unit/test_browse_descriptions.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_browse_descriptions.py`:

```python
from dbs_vector.config import EngineConfig
from dbs_vector.mcp.families.document import DocumentFamily
from dbs_vector.mcp.families.sql import SqlFamily


def _engine(**over) -> EngineConfig:
    base = dict(
        description="x", model="gemma-bf16", mapper_type="sql",
        chunker_type="api", table_name="query_vault",
        workflow="sql_clustering", tuning_profile="gemma-sql-atomic",
    )
    base.update(over)
    return EngineConfig(**base)


def test_sql_search_description_keeps_filter_docs_and_source_phrase():
    d = SqlFamily().search_description("sql-api", _engine(chunker_type="api", model="gemma-bf16"))
    assert "min_time" in d and "min_lock_time" in d and "table_filter" in d
    assert "API" in d            # source phrase from chunker_type="api"
    assert "Gemma" in d          # embeddings phrase from model
    assert "Showing" in d        # N-of-M note


def test_sql_search_description_duckdb_granite_phrases():
    d = SqlFamily().search_description(
        "sql-granite", _engine(chunker_type="duckdb", model="granite-r2")
    )
    assert "DuckDB" in d
    assert "Granite" in d


def test_document_search_description_similarity_clause():
    d = DocumentFamily().search_description(
        "md", _engine(mapper_type="document", chunker_type="document",
                      model="gemma-bf16", tuning_profile="gemma-md")
    )
    assert "similarity" in d.lower()


def test_browse_description_off_omits_raw_query():
    d = SqlFamily().browse_description(
        "sql-api", _engine(chunker_type="api"), allow_raw_queries=False
    )
    assert "raw_query" not in d
    assert "group_by" in d and "order_by" in d
    assert "execution_time_ms" in d


def test_browse_description_on_includes_raw_query():
    d = SqlFamily().browse_description(
        "sql-api", _engine(chunker_type="api"), allow_raw_queries=True
    )
    assert "raw_query" in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_browse_descriptions.py -v`
Expected: FAIL — `AttributeError: 'SqlFamily' object has no attribute 'search_description'`.

- [ ] **Step 3: Add `search_description` to the `SearchFamily` Protocol**

In `src/dbs_vector/mcp/families/base.py`, add a `TYPE_CHECKING` import at the top (after the existing imports):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbs_vector.config import EngineConfig
```

Then add to the `SearchFamily` Protocol (after `format_results`, before `make_handler`):

```python
    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        """Compose the LLM-facing description for this engine's search tool
        from inert config facts (chunker_type, model) on the passed `engine`.
        Takes `engine` directly (not via the global settings) so it composes
        purely from its arguments — no hidden global read, trivially testable.
        Families own this prose; config.yaml holds only a short human summary."""
        ...
```

- [ ] **Step 4: Implement composition helpers + `search_description` in `DocumentFamily`**

In `src/dbs_vector/mcp/families/document.py`, add a module-level helper and the method.

At the top, add a `TYPE_CHECKING` import (alongside the existing imports):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbs_vector.config import EngineConfig
```

At module level (after imports):

```python
def _embeddings_phrase(model: str) -> str:
    return {"granite-r2": "Granite embeddings",
            "gemma-bf16": "Gemma embeddings"}.get(model, f"{model} embeddings")
```

Inside `class DocumentFamily`, add:

```python
    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        emb = _embeddings_phrase(engine.model)
        return (
            f"Semantic search over Markdown documentation chunks ({emb}). "
            f"Returns the top-K most similar passages, ranked by cosine "
            f"similarity — not by recency or size."
        )
```

- [ ] **Step 5: Implement `search_description` + `browse_description` in `SqlFamily`**

In `src/dbs_vector/mcp/families/sql.py`, add module-level helpers (after the existing `_fmt_*` helpers) and the two methods inside `class SqlFamily`.

Module level:

```python
def _sql_source_phrase(chunker_type: str) -> str:
    return {"api": "a remote slow-log API",
            "duckdb": "a local DuckDB slow-query log"}.get(
        chunker_type, "a SQL slow-query log")


def _sql_embeddings_phrase(model: str) -> str:
    return {"granite-r2": "Granite embeddings",
            "gemma-bf16": "Gemma embeddings"}.get(model, f"{model} embeddings")
```

Add a `TYPE_CHECKING` import at the top (alongside the existing imports):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbs_vector.config import EngineConfig
```

Inside `class SqlFamily` (anywhere among the methods):

```python
    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        source = _sql_source_phrase(engine.chunker_type)
        emb = _sql_embeddings_phrase(engine.model)
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
            f"use the sibling `browse_{engine_name.replace('-', '_')}` tool."
        )

    def browse_description(
        self, engine_name: str, engine: "EngineConfig", allow_raw_queries: bool
    ) -> str:
        source = _sql_source_phrase(engine.chunker_type)
        cols = ("id, content_hash, user, host, source, tables, calls, "
                "execution_time_ms, lock_time_sec, rows_examined, rows_sent, "
                "latest_ts, text")
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
            f"columns); `limit` (default 10). Columns: {cols}. Grouping yields "
            f"fingerprints (COUNT), calls/execution_time_ms/lock_time_sec/"
            f"rows_examined/rows_sent (SUM), latest_ts (MAX), "
            f"avg_ms_per_fingerprint and avg_ms_per_call (the per-execution "
            f"average a DBA usually reads)."
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_browse_descriptions.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/mcp/families/base.py src/dbs_vector/mcp/families/document.py src/dbs_vector/mcp/families/sql.py tests/unit/test_browse_descriptions.py
git commit -m "feat(browse): family-owned search_description + browse_description templates"
```

---

## Task 6: `register_browse_tools` + switch search registrar to family descriptions

**Files:**
- Modify: `src/dbs_vector/mcp/dynamic_tools.py`
- Test: `tests/unit/test_register_browse_tools.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_register_browse_tools.py`:

```python
import pytest
from mcp.server.fastmcp import FastMCP

import dbs_vector.mcp.dynamic_tools as dyn
from dbs_vector.config import EngineConfig


def _engine(**over) -> EngineConfig:
    base = dict(
        description="short summary", model="gemma-bf16", mapper_type="sql",
        chunker_type="api", table_name="query_vault",
        workflow="sql_clustering", tuning_profile="gemma-sql-atomic",
    )
    base.update(over)
    return EngineConfig(**base)


@pytest.fixture
def patched(monkeypatch):
    engines = {
        "sql-api": _engine(chunker_type="api"),
        "md": _engine(mapper_type="document", chunker_type="document",
                      tuning_profile="gemma-md"),
    }

    class _S:
        pass
    s = _S()
    s.engines = engines
    monkeypatch.setattr(dyn, "settings", s)
    return engines


@pytest.mark.asyncio
async def test_search_tools_use_family_description(patched):
    mcp = FastMCP("t")
    dyn.register_search_tools(mcp)
    tool = next(t for t in await mcp.list_tools() if t.name == "search_sql_api")
    assert "min_time" in tool.description          # family prose, not config's "short summary"
    assert tool.description != "short summary"
```

The browse *registration* tests live in Task 7 (they require `make_browse_handler`,
implemented there), keeping every Task-6 commit green. Note `patched` monkeypatches
`dyn.settings`; the description methods take `engine` as an argument (Task 5), so
they compose from the patched engine objects with no separate global read.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_register_browse_tools.py::test_search_tools_use_family_description -v`
Expected: FAIL — the description still comes from `engine.description` ("short summary"), so the `min_time` assertion fails.

- [ ] **Step 3: Switch `register_search_tools` to the family description**

In `src/dbs_vector/mcp/dynamic_tools.py`, in `register_search_tools`, change the `mcp.add_tool` call (lines ~81-85) from:

```python
        engine = settings.engines[engine_name]
        handler = family.make_handler(engine_name)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=engine.description,
        )
```

to:

```python
        engine = settings.engines[engine_name]
        handler = family.make_handler(engine_name)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=family.search_description(engine_name, engine),
        )
```

- [ ] **Step 4: Add `register_browse_tools`**

Append to `src/dbs_vector/mcp/dynamic_tools.py`:

```python
def register_browse_tools(mcp: FastMCP, allow_raw_queries: bool) -> None:
    """Register one browse_<engine> tool per SQL-family engine.

    Mirrors register_search_tools' pre-flight (name pattern, collision,
    family resolution, idempotency) but registers ONLY engines whose
    resolved_family == "sql", uses verb="browse" tool names, and sources the
    description from family.browse_description(engine, allow_raw_queries).
    """
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}
    registrations: dict[str, tuple[str, str]] = mcp_any._dbs_vector_registrations

    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str, Any]] = []
    for engine_name, engine in settings.engines.items():
        if engine.resolved_family != "sql":
            continue
        if not ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {ENGINE_NAME_PATTERN.pattern}."
            )
        tool_name = normalize_tool_name(engine_name, verb="browse")
        if tool_name in seen:
            raise ValueError(
                f"MCP tool name collision: '{seen[tool_name]}' and '{engine_name}' "
                f"both normalize to '{tool_name}'."
            )
        seen[tool_name] = engine_name
        family_key = engine.resolved_family
        FamilyRegistry.get(family_key)
        resolved.append((engine_name, tool_name, family_key, engine))

    for engine_name, tool_name, family_key, engine in resolved:
        family = FamilyRegistry.get(family_key)
        prior = registrations.get(tool_name)
        if prior is not None:
            if prior == (engine_name, family_key):
                continue
            raise RuntimeError(
                f"Stale tool registration for '{tool_name}': was {prior}, "
                f"now engine={engine_name} family={family_key}."
            )
        handler = family.make_browse_handler(engine_name, allow_raw_queries)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=family.browse_description(engine_name, engine, allow_raw_queries),
        )
        registrations[tool_name] = (engine_name, family_key)
```

Note: `register_browse_tools` calls `family.make_browse_handler`, implemented in Task 7. That's fine — no test committed in *this* task exercises that path (the browse-registration tests are in Task 7). The code is committed here; its tests run green in Task 7. **Run Task 7 immediately after Task 6** (they are paired).

- [ ] **Step 5: Run the committed test to verify it passes**

Run: `uv run pytest tests/unit/test_register_browse_tools.py::test_search_tools_use_family_description -v`
Expected: PASS — the only test in this file so far; it asserts the search tool description now comes from the family.

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/mcp/dynamic_tools.py tests/unit/test_register_browse_tools.py
git commit -m "feat(browse): register_browse_tools + family-sourced search descriptions"
```

---

## Task 7: `SqlFamily.make_browse_handler` + `format_browse` (MCP handler, error sanitization)

**Files:**
- Modify: `src/dbs_vector/mcp/families/sql.py`
- Test: `tests/unit/test_browse_mcp_handler.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_browse_mcp_handler.py`:

```python
from datetime import UTC, datetime

import pyarrow as pa
import pytest

import dbs_vector.mcp.state as state
from dbs_vector.mcp.families.sql import SqlFamily
from dbs_vector.services.browse import BrowseService


class _FakeStore:
    def __init__(self, table, raise_exc=None):
        self._t = table
        self._raise = raise_exc

    def scan(self, columns=None):
        if self._raise is not None:
            raise self._raise
        return self._t


class _FakeService:
    def __init__(self, store):
        self.vector_store = store


def _table() -> pa.Table:
    return pa.table({
        "id": ["A", "B"], "content_hash": ["h1", "h2"],
        "text": ["s1", "s2"], "raw_query": ["RAW-A-SECRET", "RAW-B"],
        "source": ["db1", "db1"], "user": ["alice", "bob"], "host": ["h1", "h2"],
        "tables": [["orders"], ["items"]], "calls": [10, 5],
        "execution_time_ms": [100.0, 50.0], "lock_time_sec": [1.0, None],
        "rows_examined": [50, 20], "rows_sent": [5, 2],
        "latest_ts": [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)],
    })


@pytest.fixture
def wired(monkeypatch):
    def _wire(store):
        monkeypatch.setitem(state._services, "sql-api", _FakeService(store))
    return _wire


@pytest.mark.asyncio
async def test_handler_groups_and_formats(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(group_by="user", order_by="execution_time_ms:desc")
    assert "alice" in out
    assert "Showing" in out


@pytest.mark.asyncio
async def test_handler_validation_error_returned_verbatim(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(select="raw_query")          # gated off
    assert "not exposed" in out
    assert "raw query text" in out.lower()


@pytest.mark.asyncio
async def test_handler_infra_exception_is_sanitized(wired):
    secret_path = "/Users/op/.ssh/id_rsa"
    wired(_FakeStore(_table(), raise_exc=RuntimeError(f"lance IO error at {secret_path}")))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(order_by="execution_time_ms:desc")
    assert secret_path not in out
    assert "browse execution failed" in out.lower()


@pytest.mark.asyncio
async def test_handler_raw_query_gated_off_not_in_output(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(order_by="execution_time_ms:desc")
    assert "RAW-A-SECRET" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_browse_mcp_handler.py -v`
Expected: FAIL — `AttributeError: 'SqlFamily' object has no attribute 'make_browse_handler'`.

- [ ] **Step 3: Implement `format_browse` + `make_browse_handler`**

In `src/dbs_vector/mcp/families/sql.py`, add the imports at the top:

```python
from dbs_vector.services.browse import (
    BrowseResult,
    BrowseService,
    BrowseValidationError,
)
```

Add a module-level formatter (after the `_fmt_*` helpers):

```python
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
        return ", ".join(str(v) for v in value) if value else "n/a"
    return str(value)


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
```

Add the handler factory inside `class SqlFamily`:

```python
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
                "id": id, "content_hash": content_hash, "user": user,
                "host": host, "source": source, "table": table,
                "min_calls": min_calls,
                "min_execution_time_ms": min_execution_time_ms,
                "min_lock_time_sec": min_lock_time_sec,
            }

            def _run() -> BrowseResult:
                return browse.build_and_run(
                    filters=filters, group_by=group_by, order_by=order_by,
                    select=select, limit=limit, allow_raw_queries=allow_raw_queries,
                )

            try:
                result = await asyncio.to_thread(_run)
                return format_browse(result)
            except BrowseValidationError as e:
                return str(e)                       # safe, author-controlled
            except Exception as e:                  # infra: log full, return generic
                logger.warning("browse '{}' failed: {}", engine_name, e)
                return "browse execution failed (see server logs)."

        return handler
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_browse_mcp_handler.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Add the browse-registration tests (now that `make_browse_handler` exists)**

Append to `tests/unit/test_register_browse_tools.py`:

```python
@pytest.mark.asyncio
async def test_register_browse_tools_only_sql_engines(patched):
    mcp = FastMCP("t")
    dyn.register_browse_tools(mcp, allow_raw_queries=False)
    tools = {t.name for t in await mcp.list_tools()}
    assert "browse_sql_api" in tools
    assert "browse_md" not in tools          # md is not the sql family


@pytest.mark.asyncio
async def test_register_browse_tools_idempotent(patched):
    mcp = FastMCP("t")
    dyn.register_browse_tools(mcp, allow_raw_queries=False)
    dyn.register_browse_tools(mcp, allow_raw_queries=False)   # no raise
    tools = {t.name for t in await mcp.list_tools()}
    assert "browse_sql_api" in tools
```

- [ ] **Step 6: Run the browse handler + registration tests**

Run: `uv run pytest tests/unit/test_browse_mcp_handler.py tests/unit/test_register_browse_tools.py -v`
Expected: PASS (handler tests + both registration tests now that `make_browse_handler` exists).

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/mcp/families/sql.py tests/unit/test_browse_mcp_handler.py tests/unit/test_register_browse_tools.py
git commit -m "feat(browse): SqlFamily.make_browse_handler + format_browse with error sanitization"
```

---

## Task 8: Store-only bootstrap + CLI `browse` command + server flag

`browse` needs only the store — never the embedder or chunker. The existing
`build_dependencies` unconditionally constructs `MLXEmbedder` (`bootstrap.py:45`)
and a chunker (`:55`), so reusing it for the CLI would pay the full model-load
cost and could fail for MLX/model reasons on a browse-only invocation. This task
adds a store-only factory first, then wires the CLI command to it. (The MCP path
already reuses the pre-loaded `_services[engine].vector_store`, so it needs no
embedder either.)

**Files:**
- Modify: `src/dbs_vector/services/bootstrap.py` (add `build_store`)
- Modify: `src/dbs_vector/mcp/server.py`
- Modify: `src/dbs_vector/cli.py` (a `_build_store` wrapper, a new `browse` command after `search` ~206, and `--allow-raw-queries` on the `mcp` command ~209-237)
- Test: `tests/unit/test_build_store.py` (new), `tests/unit/test_cli_browse.py` (new)

- [ ] **Step 1: Write a failing test that `build_store` constructs NO embedder**

Create `tests/unit/test_build_store.py`:

```python
import dbs_vector.services.bootstrap as boot


def test_build_store_does_not_construct_embedder(monkeypatch, tmp_path):
    calls = {"embedder": 0, "store": 0}

    def _fake_embedder(*a, **k):
        calls["embedder"] += 1
        raise AssertionError("build_store must NOT construct an embedder")

    captured = {}

    class _FakeStore:
        def __init__(self, **kwargs):
            calls["store"] += 1
            captured.update(kwargs)

    monkeypatch.setattr(boot, "MLXEmbedder", _fake_embedder)
    monkeypatch.setattr(boot, "LanceDBStore", _FakeStore)

    store = boot.build_store("sql-api")     # sql-api must exist in config.yaml

    assert calls["embedder"] == 0
    assert calls["store"] == 1
    assert captured["table_name"] == "query_vault"
    assert isinstance(store, _FakeStore)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_build_store.py -v`
Expected: FAIL — `AttributeError: module 'dbs_vector.services.bootstrap' has no attribute 'build_store'`.

- [ ] **Step 3: Implement `build_store`**

In `src/dbs_vector/services/bootstrap.py`, add after `build_dependencies`:

```python
def build_store(engine_name: str) -> LanceDBStore:
    """Resolve ONLY the vector store for an engine — no embedder, no chunker.

    For read paths (browse) that never embed. Avoids MLX model load and the
    cost/failure modes of constructing a chunker. The mapper (hence the table
    schema) and vector_dimension come from the engine's model contract.
    """
    if engine_name not in settings.engines:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Check {os.environ.get('DBS_CONFIG_FILE', 'config.yaml')}."
        )
    engine = settings.engines[engine_name]
    contract = ModelRegistry.get(engine.model)
    MapperClass = ComponentRegistry.get_mapper(engine.mapper_type)
    mapper = MapperClass(vector_dimension=contract.vector_dimension)
    return LanceDBStore(
        db_path=settings.db_path,
        table_name=engine.table_name,
        vector_dimension=contract.vector_dimension,
        mapper=mapper,
        nprobes=settings.nprobes,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_build_store.py -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for the CLI `browse` command**

Create `tests/unit/test_cli_browse.py`:

```python
import json

import pyarrow as pa
import pytest
from typer.testing import CliRunner

import dbs_vector.cli as cli_mod
from dbs_vector.cli import app

runner = CliRunner()


class _FakeStore:
    def scan(self, columns=None):
        return pa.table({"id": ["A", "B"], "calls": [10, 5]})


@pytest.fixture
def patched(monkeypatch):
    class _E:
        def __init__(self, fam):
            self.resolved_family = fam
    monkeypatch.setattr(cli_mod.settings, "engines",
                        {"sql-api": _E("sql"), "md": _E("document")}, raising=False)
    monkeypatch.setattr(cli_mod, "_build_store", lambda name: _FakeStore())


def test_browse_table_output(patched):
    res = runner.invoke(app, ["browse", "--type", "sql-api",
                              "--sql", "SELECT id, calls FROM t ORDER BY calls DESC"])
    assert res.exit_code == 0
    assert "A" in res.stdout and "B" in res.stdout


def test_browse_json_output(patched):
    res = runner.invoke(app, ["browse", "--type", "sql-api",
                              "--sql", "SELECT id FROM t", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert {r["id"] for r in data} == {"A", "B"}


def test_browse_rejects_non_sql_engine(patched):
    res = runner.invoke(app, ["browse", "--type", "md", "--sql", "SELECT 1"])
    assert res.exit_code == 1
    assert "sql" in res.stdout.lower()
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_browse.py -v`
Expected: FAIL — no `browse` command (`exit_code == 2`, "No such command").

- [ ] **Step 7: Add the CLI `browse` command + a table renderer**

In `src/dbs_vector/services/browse.py`, add a plain-text table renderer (used by the CLI):

```python
def result_to_table(result: BrowseResult) -> str:
    """Aligned text table for CLI display (no byte budget; operator-facing)."""
    if not result.rows:
        return "0 rows."
    cols = result.columns

    def cell(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, list):
            return ",".join(str(x) for x in v) if v else "n/a"
        return str(v)

    widths = {c: max(len(c), *(len(cell(r.get(c))) for r in result.rows)) for c in cols}
    head = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(
        " | ".join(cell(r.get(c)).ljust(widths[c]) for c in cols) for r in result.rows
    )
    return f"{head}\n{sep}\n{body}\n({result.total_matching} rows)"
```

In `src/dbs_vector/cli.py`, add the imports near the top (with the other service imports):

```python
from dbs_vector.services.bootstrap import build_store
from dbs_vector.services.browse import BrowseService, result_to_json, result_to_table
```

Add a `_build_store` CLI wrapper next to `_build_dependencies` (mirrors its
schema-mismatch → typer.Exit handling):

```python
def _build_store(engine_name: str) -> Any:
    """CLI-facing store-only builder: converts schema-mismatch to a typer exit."""
    try:
        return build_store(engine_name)
    except ValueError as e:
        if "Schema mismatch" in str(e):
            typer.echo(f"\n[!] Database Error: {e}", err=True)
            raise typer.Exit(code=1) from e
        raise
```

(Add `from typing import Any` to the imports if not already present — `cli.py`
currently imports only `Annotated` from `typing`.)

Add the new command after the `search` command (after line ~206):

```python
@app.command()
def browse(
    sql: Annotated[
        str, typer.Option("--sql", help="A read-only SELECT (polars SQL dialect).")
    ],
    engine_name: Annotated[
        str, typer.Option("--type", "-t", help="SQL engine to browse (sql, sql-api, ...).")
    ] = "sql-api",
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit rows as JSON instead of a table.")
    ] = False,
) -> None:
    """Analytical SQL over a SQL engine's table (no embedder, no ranking).

    Frames: `t` (one row per fingerprint), `t_by_table` (exploded on `tables`),
    and the engine name with dashes→underscores. Quote "user" (SQL keyword).
    Unbounded — use LIMIT in your SQL; `SELECT * FROM t` is a full export.
    """
    if engine_name not in settings.engines:
        typer.echo(
            f"Error: Unknown engine type '{engine_name}'. Available: "
            f"{list(settings.engines.keys())}"
        )
        raise typer.Exit(code=1)
    if settings.engines[engine_name].resolved_family != "sql":
        sql_engines = [
            n for n, e in settings.engines.items() if e.resolved_family == "sql"
        ]
        typer.echo(
            f"Error: browse is only available for SQL engines. "
            f"'{engine_name}' is not one. Available SQL engines: {sql_engines}"
        )
        raise typer.Exit(code=1)

    store = _build_store(engine_name)
    frame_alias = engine_name.replace("-", "_")
    service = BrowseService(store, frame_alias)
    try:
        result = service.run_sql(sql)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(result_to_json(result))
    else:
        typer.echo(result_to_table(result))
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_browse.py -v`
Expected: PASS

- [ ] **Step 9: Wire `--allow-raw-queries` into the server**

In `src/dbs_vector/mcp/server.py`, change the imports and `start_stdio_server`:

```python
from dbs_vector.mcp.dynamic_tools import register_browse_tools, register_search_tools
```

```python
def start_stdio_server(allow_raw_queries: bool = False) -> None:
    """Initialize services, register all tools, and run stdio MCP."""
    initialize_services()
    register_search_tools(mcp)
    register_browse_tools(mcp, allow_raw_queries=allow_raw_queries)
    register_discovery_tool(mcp)
    mcp.run()
```

In `src/dbs_vector/cli.py`, add the flag to the `mcp` command signature (after `config_file`, ~line 217) and pass it through:

```python
    allow_raw_queries: Annotated[
        bool,
        typer.Option(
            "--allow-raw-queries",
            help="Expose the verbatim raw_query column (literal PII values) to "
            "browse MCP tools. Default off — enable only for a trusted local model.",
        ),
    ] = False,
```

and change the call (line ~234) from `start_stdio_server()` to:

```python
        start_stdio_server(allow_raw_queries=allow_raw_queries)
```

- [ ] **Step 10: Verify server wiring imports cleanly**

Run: `uv run python -c "from dbs_vector.mcp.server import start_stdio_server; from dbs_vector.cli import app; print('ok')"`
Expected: prints `ok` (no import/syntax errors).

- [ ] **Step 11: Commit**

```bash
git add src/dbs_vector/services/bootstrap.py src/dbs_vector/mcp/server.py src/dbs_vector/cli.py src/dbs_vector/services/browse.py tests/unit/test_build_store.py tests/unit/test_cli_browse.py
git commit -m "feat(browse): store-only bootstrap + CLI browse command + --allow-raw-queries flag"
```

---

## Task 9: Shorten `config.yaml` engine descriptions

**Files:**
- Modify: `config.yaml` (six `description:` fields)
- Test: manual config-load check

- [ ] **Step 1: Replace each engine `description:` with a one-line human summary**

Edit `config.yaml`, replacing only the `description:` value on each engine (leave all other fields untouched):

```yaml
  md:
    description: "Markdown documentation chunks (Gemma embeddings)."
```
```yaml
  sql:
    description: "Slow-query fingerprints from a local DuckDB log (Gemma embeddings)."
```
```yaml
  sql-api:
    description: "Slow-query fingerprints from a remote slow-log API (Gemma embeddings)."
```
```yaml
  md-granite:
    description: "Markdown documentation chunks (Granite embeddings)."
```
```yaml
  sql-granite:
    description: "Slow-query fingerprints from a local DuckDB log (Granite embeddings)."
```
```yaml
  sql-api-granite:
    description: "Slow-query fingerprints from a remote slow-log API (Granite embeddings)."
```

- [ ] **Step 2: Verify config still loads and validates**

Run: `uv run dbs-vector list-engines 2>/dev/null || uv run python -c "from dbs_vector.config import load_settings; s=load_settings('config.yaml', validate=True); print(sorted(s.engines)); print(s.engines['sql-api'].description)"`
Expected: prints the engine list and the new one-line `sql-api` description (no validation error).

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "refactor(browse): shorten config.yaml descriptions (prose now in families)"
```

---

## Task 10: End-to-end integration tests (real LanceDB)

**Files:**
- Modify: `tests/integration/test_browse_integration.py` (extend with `BrowseService` end-to-end cases)

- [ ] **Step 1: Write failing end-to-end tests**

Append to `tests/integration/test_browse_integration.py`:

```python
def test_browse_run_sql_point_lookup_end_to_end(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    result = svc.run_sql("SELECT id, calls FROM t WHERE id = 'A'")
    assert result.total_matching == 1
    assert result.rows[0]["id"] == "A"
    assert result.rows[0]["calls"] == 10


def test_browse_grouped_by_user_end_to_end(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    result = svc.build_and_run(
        filters={}, group_by="user", order_by="execution_time_ms:desc",
        select=None, limit=10,
    )
    assert result.grouped is True
    assert result.rows[0]["user"] == "alice"
    assert result.rows[0]["execution_time_ms"] == 100.0


def test_browse_grouped_by_table_end_to_end(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    result = svc.build_and_run(
        filters={}, group_by="tables", order_by="fingerprints:desc",
        select=None, limit=10,
    )
    counts = {r["tables"]: r["fingerprints"] for r in result.rows}
    assert counts["orders"] == 2          # A and B both touch orders


def test_browse_sees_checkout_latest_updates(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    assert svc.run_sql("SELECT id FROM t").total_matching == 2

    # add a row via a second store handle on the same path/table
    store2 = _make_store(tmp_path)
    _seed(store2)  # adds A,B again (dup ids ok for count) → table grows
    assert svc.run_sql("SELECT id FROM t").total_matching == 4
```

- [ ] **Step 2: Run tests to verify they fail, then pass**

Run: `uv run pytest tests/integration/test_browse_integration.py -v`
Expected: All pass (the code from Tasks 1-3 already implements this; if a test fails it reveals a real integration gap — fix the implementation, not the test).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_browse_integration.py
git commit -m "test(browse): end-to-end integration over real LanceDB"
```

---

## Task 11: Full validation sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite + lint + types**

Run: `uv run poe check`
Expected: format clean, lint clean, mypy/pyright 0 new errors in `src/`, all tests pass.

- [ ] **Step 2: Fix any failures**

Common spots to check if something breaks:
- An existing test asserting a `search_*` tool description equals `engine.description` — update it to expect the family prose (the description now comes from `family.search_description`).
- `tests/` pyright baseline: keep `src/` at 0 errors; the 13 known intentional `tests/` errors are not regressions.

- [ ] **Step 3: Manual smoke test of the CLI**

Run: `uv run dbs-vector browse --type sql-api --sql "SELECT \"user\", COUNT(*) AS n FROM t GROUP BY \"user\" ORDER BY n DESC LIMIT 5"`
Expected: a small aligned table (requires an ingested `sql-api` table; if empty, prints "0 rows.").

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test(browse): fix description-regression assertions; green suite"
```

---

## Self-Review (completed during planning)

**Spec coverage:** scan port (T1), polars execution core + run_sql (T2), build_and_run with bound-value filters + curated aggregates + validation + injection guard (T3), verb naming (T4), family-owned descriptions incl. raw_query gating (T5), browse registrar + search-description switch (T6), MCP handler with error sanitization (T7), store-only bootstrap + CLI command + uncapped export + server flag (T8), config shortening (T9), integration incl. checkout_latest (T10), validation sweep (T11). All spec sections map to a task.

**Review fixes applied:**
- **Embedder-free CLI:** Task 8 adds `build_store()` (store only — no `MLXEmbedder`, no chunker) and a `_build_store` CLI wrapper, with a test asserting no embedder is constructed. CLI `browse` no longer pays model-load cost or risks MLX failures. (MCP already reuses the pre-loaded store.)
- **Descriptions take `engine`:** `search_description(engine_name, engine)` and `browse_description(engine_name, engine, allow_raw_queries)` receive the `EngineConfig` from the registrar (which already holds it) instead of re-reading the global `settings`. Tests pass a constructed `EngineConfig` — no global-state monkeypatch mismatch.
- **Green commits:** the browse-*registration* tests moved to Task 7 (where `make_browse_handler` exists); Task 6 commits only the search-description test, which passes immediately.

**Deviations from spec prose (intentional, verified):**
- Grouped all-null `SUM` → `0.0` (not `NULL`); only `NULLIF` div-by-zero → `NULL`. Tests assert the real behavior. The formatter still renders `None` → `n/a`.
- `browse_description` documents the **structured MCP params** (filters/group_by/order_by/select/limit), not the CLI `t`/`t_by_table` frames — the MCP never takes raw SQL, so frame names are a CLI-only concept (the CLI command's docstring documents them).
- MCP builder registers the pre-filtered frame under the internal name `data` (not `t`), since values are filtered out via polars expressions before SQL runs — keeps the emitted SQL free of any user value.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `BrowseService(store, frame_alias)`, `run_sql(sql)`, `build_and_run(*, filters, group_by, order_by, select, limit, allow_raw_queries)`, `BrowseResult(rows, columns, total_matching, grouped, limit_applied)`, `build_store(engine_name)`, `make_browse_handler(engine_name, allow_raw_queries)`, `search_description(engine_name, engine)`, `browse_description(engine_name, engine, allow_raw_queries)`, `register_browse_tools(mcp, allow_raw_queries)`, `start_stdio_server(allow_raw_queries)`, `normalize_tool_name(engine_name, verb)` — consistent across all tasks.
