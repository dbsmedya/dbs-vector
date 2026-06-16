# MCP Impact-Triage Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dbs-vector MCP self-sufficient for the find-impacting-queries workflow — frequency-weighted impact ranking, per-call selectivity signals, transport-safe exemplars, and a one-call `top_impacting_<engine>` tool.

**Architecture:** Three additive changes, no LanceDB schema change, no `--rebuild`, no GoFast dependency. (A) Add derived columns (`impact_score`, `avg_ms_per_call`, `selectivity`) to flat `browse` in `services/browse.py`. (B) Truncate long string cells in the MCP browse formatter (`mcp/families/sql.py`) to fix the transport ceiling. (C) Add a thin `top_impacting_<engine>` triage tool (new `TriageFamily` Protocol + `register_triage_tools`) that wraps `BrowseService` with curated presets.

**Tech Stack:** Python 3.12, polars `SQLContext` (in-memory SQL over Arrow), FastMCP, pyarrow, pytest / pytest-asyncio, `uv run poe` task runner.

**Spec:** `docs/superpowers/specs/2026-06-16-mcp-impact-triage-design.md`

**Conventions for every commit step:**
- Stage ONLY the exact files listed — never `git add -A`, `git add .`, or `git commit -a`. The branch has unrelated in-flight work that must not be committed.
- End every commit message with this trailer line (matches the branch history):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- Run `uv run pytest <file> -v` for the named tests; run `uv run poe check` before the final commit.

---

### Task 1: Transport-safe truncation in the MCP browse formatter (Component B)

Fixes the live bug where a single multi-MB `raw_query` cell in a `browse` response produces an oversized JSON-RPC frame and drops the connection (`search` already truncates; `browse` does not). The fix reuses the existing `_truncate_raw_query` helper inside `_fmt_cell` so any long string/list cell is capped. CLI formatters stay uncapped.

**Files:**
- Modify: `src/dbs_vector/mcp/families/sql.py` (`_fmt_cell`, lines 64-75)
- Test: `tests/unit/test_browse_mcp_handler.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/unit/test_browse_mcp_handler.py`:

```python
from dbs_vector.mcp.families.sql import _RAW_QUERY_DISPLAY_LIMIT, _fmt_cell, format_browse
from dbs_vector.services.browse import BrowseResult


def test_fmt_cell_truncates_long_string():
    long = "x" * (_RAW_QUERY_DISPLAY_LIMIT + 500)
    out = _fmt_cell(long)
    assert len(out) < len(long)
    assert "more chars elided" in out


def test_fmt_cell_short_string_unchanged():
    assert _fmt_cell("SELECT 1") == "SELECT 1"


def test_format_browse_truncates_raw_query_cell():
    big = "SELECT " + "a," * 3000  # > 2000 chars
    result = BrowseResult(
        rows=[{"id": "A", "raw_query": big}],
        columns=["id", "raw_query"],
        total_matching=1,
        grouped=False,
        limit_applied=False,
    )
    out = format_browse(result)
    assert "more chars elided" in out
    assert len(out.encode("utf-8")) < 1_000_000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_browse_mcp_handler.py -k "fmt_cell or format_browse_truncates" -v`
Expected: `test_fmt_cell_truncates_long_string` and `test_format_browse_truncates_raw_query_cell` FAIL (the long string is returned untruncated, so "more chars elided" is absent). `test_fmt_cell_short_string_unchanged` PASSES already.

- [ ] **Step 3: Implement the truncation**

In `src/dbs_vector/mcp/families/sql.py`, replace the body of `_fmt_cell` (lines 64-75) with:

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
        return _truncate_raw_query(", ".join(str(v) for v in value)) if value else "n/a"
    return _truncate_raw_query(str(value))
```

(`_truncate_raw_query` is defined above at line 26 and is a no-op for strings ≤ `_RAW_QUERY_DISPLAY_LIMIT`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_browse_mcp_handler.py -v`
Expected: PASS (all, including the pre-existing handler tests).

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/families/sql.py tests/unit/test_browse_mcp_handler.py
git commit -m "fix(mcp): truncate long browse cells to clear the raw_query transport ceiling

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Derived columns in flat browse (Component A)

Add `impact_score` (`calls * execution_time_ms`), `avg_ms_per_call`, and `selectivity` as selectable + orderable columns in non-grouped browse. Defined as bare SQL expressions so they can be used in `ORDER BY` even when not selected. `_DEFAULT_SELECT` is unchanged (backwards-compatible).

**Files:**
- Modify: `src/dbs_vector/services/browse.py` (`_FLAT_DERIVED_EXPR` new; `_selectable` line 217; `_build_flat_sql` lines 256-276)
- Test: `tests/unit/test_browse_service.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/unit/test_browse_service.py`:

```python
def test_flat_select_derived_columns():
    result = _svc().build_and_run(
        filters={},
        group_by=None,
        order_by="impact_score:desc",
        select="id,impact_score,avg_ms_per_call,selectivity",
        limit=10,
    )
    assert "impact_score" in result.columns
    top = result.rows[0]
    # row A: calls=10, exec=100 -> impact=1000, avg=10; examined=50,sent=5 -> sel=10
    assert top["id"] == "A"
    assert top["impact_score"] == 1000.0
    assert top["avg_ms_per_call"] == 10.0
    assert top["selectivity"] == 10.0


def test_flat_order_by_impact_score_without_selecting_it():
    result = _svc().build_and_run(
        filters={},
        group_by=None,
        order_by="impact_score:desc",
        select="id",  # impact_score NOT in the projection
        limit=10,
    )
    # impacts: A=1000, B=250, C=25 -> descending order A,B,C
    assert [r["id"] for r in result.rows] == ["A", "B", "C"]


def test_flat_select_rejects_unknown_derived():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by=None,
            order_by="execution_time_ms:desc",
            select="id,bogus_score",
            limit=10,
        )


def test_flat_selectivity_div_by_zero_is_null():
    tbl = pa.table(
        {
            "id": ["A"],
            "content_hash": ["h"],
            "text": ["x"],
            "raw_query": ["x"],
            "source": ["db1"],
            "user": ["alice"],
            "host": ["h"],
            "tables": [["orders"]],
            "calls": [10],
            "execution_time_ms": [100.0],
            "lock_time_sec": [None],
            "rows_examined": [50],
            "rows_sent": [0],  # division by zero -> NULLIF -> null
            "latest_ts": [datetime(2026, 1, 1, tzinfo=UTC)],
        }
    )
    svc = BrowseService(FakeStore(tbl), frame_alias="sql_api")
    result = svc.build_and_run(
        filters={},
        group_by=None,
        order_by="execution_time_ms:desc",
        select="id,selectivity",
        limit=10,
    )
    assert result.rows[0]["selectivity"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_browse_service.py -k "flat_select_derived or flat_order_by_impact or flat_select_rejects_unknown or flat_selectivity" -v`
Expected: FAIL — `test_flat_select_derived_columns` raises `BrowseValidationError` ("Unknown select column 'impact_score'") because the derived names aren't selectable yet.

- [ ] **Step 3: Add the derived-expression map**

In `src/dbs_vector/services/browse.py`, immediately after the `_AGG_SQL` block and its assert (after line 68), add:

```python
# Flat-mode derived columns: bare SQL expressions over per-fingerprint rows.
# Selectable AND orderable in non-grouped browse. Keys are the output names;
# values omit the "AS name" so the same expression works in ORDER BY.
_FLAT_DERIVED_EXPR = {
    "impact_score": '"calls" * "execution_time_ms"',
    "avg_ms_per_call": '"execution_time_ms" / NULLIF("calls", 0)',
    "selectivity": '"rows_examined" / NULLIF("rows_sent", 0)',
}
```

- [ ] **Step 4: Make the derived columns selectable**

In `src/dbs_vector/services/browse.py`, replace `_selectable` (lines 217-218) with:

```python
    def _selectable(self, allow_raw_queries: bool) -> tuple[str, ...]:
        return (
            _SCALAR_COLUMNS
            + tuple(_FLAT_DERIVED_EXPR)
            + (("raw_query",) if allow_raw_queries else ())
        )
```

- [ ] **Step 5: Emit derived expressions in `_build_flat_sql`**

In `src/dbs_vector/services/browse.py`, replace `_build_flat_sql` (lines 256-276) with:

```python
    def _build_flat_sql(self, order_by: str, select: str | None, allow_raw_queries: bool) -> str:
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
        select_terms = [
            f"{_FLAT_DERIVED_EXPR[c]} AS {_q(c)}" if c in _FLAT_DERIVED_EXPR else _q(c)
            for c in chosen
        ]
        order_term = _FLAT_DERIVED_EXPR[col] if col in _FLAT_DERIVED_EXPR else _q(col)
        return (
            f"SELECT {', '.join(select_terms)} FROM data "
            f"ORDER BY {order_term} {direction} NULLS LAST"
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_browse_service.py -v`
Expected: PASS (all, including the pre-existing browse-service tests). Column-level
`NULLIF` is confirmed working in the current polars `SQLContext` (verified directly), so
the `selectivity`/`avg_ms_per_call` divide-by-zero guards return null as intended — no
`CASE WHEN` fallback is needed.

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/services/browse.py tests/unit/test_browse_service.py
git commit -m "feat(browse): add impact_score/avg_ms_per_call/selectivity flat columns

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Surface the derived columns in `browse_description`

Tell the LLM the derived columns exist so it can select/order by them.

**Files:**
- Modify: `src/dbs_vector/mcp/families/sql.py` (`browse_description`, lines 228-253)
- Test: `tests/unit/test_browse_descriptions.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_browse_descriptions.py`:

```python
def test_browse_description_mentions_derived_columns():
    from dbs_vector.config import EngineConfig
    from dbs_vector.mcp.families.sql import SqlFamily

    engine = EngineConfig(
        description="short summary",
        model="gemma-bf16",
        mapper_type="sql",
        chunker_type="api",
        table_name="query_vault",
        workflow="sql_clustering",
        tuning_profile="gemma-sql-atomic",
    )
    desc = SqlFamily().browse_description("sql-api", engine, allow_raw_queries=False)
    assert "impact_score" in desc
    assert "selectivity" in desc
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_browse_descriptions.py::test_browse_description_mentions_derived_columns -v`
Expected: FAIL — "impact_score" not in the description string.

- [ ] **Step 3: Add the derived columns to the description**

In `src/dbs_vector/mcp/families/sql.py` `browse_description`, leave the `cols` literal and the `raw_query` conditional unchanged. Insert a clause into the returned description: replace the single line (line 248)

```python
            f"columns); `limit` (default 10). Columns: {cols}. Grouping yields "
```

with

```python
            f"columns); `limit` (default 10). Columns: {cols}. Non-grouped mode "
            f"adds two derived columns (selectable and orderable): impact_score "
            f"(calls*execution_time_ms) and selectivity (rows_examined/rows_sent); "
            f"avg_ms_per_call (per-fingerprint execution_time_ms/calls) is available "
            f"in both modes. Grouping yields "
```

This keeps `raw_query` in the base column list (it is not a derived column) and adds the derived-columns clause cleanly before the grouping sentence.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_browse_descriptions.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/families/sql.py tests/unit/test_browse_descriptions.py
git commit -m "docs(mcp): note flat browse derived columns in browse_description

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `TriageFamily` Protocol + `SqlFamily` triage handler & description (Component C)

Add a narrow `TriageFamily` Protocol (so the registrar can `isinstance`-narrow) and implement it on `SqlFamily`. The handler wraps `BrowseService.build_and_run` with curated presets — no new ranking logic.

**Files:**
- Modify: `src/dbs_vector/mcp/families/base.py` (add `TriageFamily` after `BrowseFamily`, line 152)
- Modify: `src/dbs_vector/mcp/families/sql.py` (add constants + module-level `format_triage` + `make_triage_handler` + `triage_description` on `SqlFamily`)
- Test: `tests/unit/test_triage_mcp_handler.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_triage_mcp_handler.py`:

```python
from datetime import UTC, datetime

import pyarrow as pa
import pytest

import dbs_vector.mcp.state as state
from dbs_vector.mcp.families.base import TriageFamily
from dbs_vector.mcp.families.sql import SqlFamily


class _FakeStore:
    def __init__(self, table):
        self._t = table

    def scan(self, columns=None):
        return self._t


class _FakeService:
    def __init__(self, store):
        self.vector_store = store


def _table() -> pa.Table:
    return pa.table(
        {
            "id": ["A", "B"],
            "content_hash": ["h1", "h2"],
            "text": ["s1", "s2"],
            "raw_query": ["RAW-A-SECRET", "RAW-B"],
            "source": ["db1", "db1"],
            "user": ["alice", "bob"],
            "host": ["h1", "h2"],
            "tables": [["orders"], ["items"]],
            "calls": [10, 5],
            "execution_time_ms": [100.0, 50.0],
            "lock_time_sec": [1.0, None],
            "rows_examined": [50, 20],
            "rows_sent": [5, 2],
            "latest_ts": [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)],
        }
    )


@pytest.fixture
def wired(monkeypatch):
    def _wire(store):
        monkeypatch.setitem(state._services, "sql-api", _FakeService(store))

    return _wire


def test_sqlfamily_is_triage_family():
    assert isinstance(SqlFamily(), TriageFamily)


@pytest.mark.asyncio
async def test_triage_default_ranks_by_impact_score(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler()
    assert "impact_score" in out
    assert out.index("A") < out.index("B")  # A (impact 1000) before B (250)


@pytest.mark.asyncio
async def test_triage_rejects_bad_order_by(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(order_by="user:desc")
    assert "order_by must be one of" in out


@pytest.mark.asyncio
async def test_triage_raw_query_silently_downgraded(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(include_raw=True)  # flag off -> no raw_query
    assert "RAW-A-SECRET" not in out


@pytest.mark.asyncio
async def test_triage_raw_query_present_when_allowed(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=True)
    out = await handler(include_raw=True)
    assert "RAW-A-SECRET" in out
    assert "Raw SQL:" in out  # exemplar rendered on its own block, not an inline cell


@pytest.mark.asyncio
async def test_triage_table_filter(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(table="items")
    assert "Showing 1 of 1 fingerprints" in out
    assert "items" in out


@pytest.mark.asyncio
async def test_triage_min_calls_filter(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(min_calls=8)  # only A (calls=10)
    assert "Showing 1 of 1 fingerprints" in out


@pytest.mark.asyncio
async def test_triage_uninitialized_service():
    handler = SqlFamily().make_triage_handler("missing-engine", allow_raw_queries=False)
    out = await handler()
    assert "not initialized" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_triage_mcp_handler.py -v`
Expected: FAIL — `ImportError: cannot import name 'TriageFamily'` (Protocol not defined yet), and `SqlFamily` has no `make_triage_handler`.

- [ ] **Step 3: Add the `TriageFamily` Protocol**

In `src/dbs_vector/mcp/families/base.py`, append after the `BrowseFamily` Protocol (after line 151):

```python


@runtime_checkable
class TriageFamily(Protocol):
    """A family that additionally exposes a top_impacting triage MCP tool.

    Only SQL-style families implement this. Declared separately so the triage
    registrar can narrow a SearchFamily to a triage-capable family with
    isinstance(), exactly as BrowseFamily does for browse.
    """

    def make_triage_handler(self, engine_name: str, allow_raw_queries: bool = False) -> Any:
        """Build a per-engine async triage handler for FastMCP."""
        ...

    def triage_description(
        self, engine_name: str, engine: "EngineConfig", allow_raw_queries: bool
    ) -> str:
        """Compose the LLM-facing description for this engine's triage tool."""
        ...
```

(`runtime_checkable`, `Protocol`, `Any`, and the `EngineConfig` TYPE_CHECKING import are already present at the top of the file.)

- [ ] **Step 4: Add triage constants + methods to `SqlFamily`**

In `src/dbs_vector/mcp/families/sql.py`, add these module-level constants after `_RAW_QUERY_DISPLAY_LIMIT` (line 23):

```python
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
```

Then add these two methods inside `class SqlFamily` (e.g. after `make_browse_handler`, at the end of the class, line 377):

```python
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
```

Also add this module-level function near `format_browse` (after line 94 in `families/sql.py`). Unlike browse's compact `col=val | col=val` row, the triage formatter puts the `raw_query` exemplar on its **own block** so multi-line SQL is paste-ready for `EXPLAIN`:

```python
def format_triage(result: BrowseResult) -> str:
    """Render a triage result under the MCP byte budget: curated scalar columns on
    a header line per fingerprint, and the raw_query exemplar (when present) on its
    OWN block for clean EXPLAIN paste. Long cells are truncated via _fmt_cell /
    _truncate_raw_query.

    raw_query is in result.columns ONLY when gated in upstream (--allow-raw-queries
    + include_raw); this formatter never re-checks the flag (matches format_browse).
    """
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
```

(`format_triage` is the new function added here. `asyncio`, `BrowseService`, `BrowseResult`, `BrowseValidationError`, `format_browse`, `_fmt_cell`, `_truncate_raw_query`, `render_with_budget`, `RESPONSE_BUDGET_BYTES`, `_sql_source_phrase`, `Any`, and the `EngineConfig` TYPE_CHECKING import are all already present at the top of `families/sql.py`.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_triage_mcp_handler.py -v`
Expected: PASS (all 8 tests).

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/mcp/families/base.py src/dbs_vector/mcp/families/sql.py tests/unit/test_triage_mcp_handler.py
git commit -m "feat(mcp): add TriageFamily protocol and SqlFamily top_impacting handler

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Register `top_impacting_<engine>` tools + wire into the server (Component C)

Mirror `register_browse_tools` for the triage tool: SQL-family-only, `verb="top_impacting"`, `isinstance(family, TriageFamily)` guard, shared idempotency tuple. Then call it from `start_stdio_server`.

**Files:**
- Modify: `src/dbs_vector/mcp/dynamic_tools.py` (add `register_triage_tools`; import `TriageFamily`)
- Modify: `src/dbs_vector/mcp/server.py` (import + call `register_triage_tools`)
- Test: `tests/unit/test_register_triage_tools.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_register_triage_tools.py`:

```python
import pytest
from mcp.server.fastmcp import FastMCP

import dbs_vector.mcp.dynamic_tools as dyn
from dbs_vector.config import EngineConfig


def _engine(**over) -> EngineConfig:
    base = dict(
        description="short summary",
        model="gemma-bf16",
        mapper_type="sql",
        chunker_type="api",
        table_name="query_vault",
        workflow="sql_clustering",
        tuning_profile="gemma-sql-atomic",
    )
    base.update(over)
    return EngineConfig(**base)


@pytest.fixture
def patched(monkeypatch):
    engines = {
        "sql-api": _engine(chunker_type="api"),
        "md": _engine(mapper_type="document", chunker_type="document", tuning_profile="gemma-md"),
    }

    class _S:
        pass

    s = _S()
    s.engines = engines
    monkeypatch.setattr(dyn, "settings", s)
    return engines


@pytest.mark.asyncio
async def test_register_triage_tools_only_sql_engines(patched):
    mcp = FastMCP("t")
    dyn.register_triage_tools(mcp, allow_raw_queries=False)
    tools = {t.name for t in await mcp.list_tools()}
    assert "top_impacting_sql_api" in tools
    assert "top_impacting_md" not in tools  # md is not the sql family


@pytest.mark.asyncio
async def test_register_triage_tools_idempotent(patched):
    mcp = FastMCP("t")
    dyn.register_triage_tools(mcp, allow_raw_queries=False)
    dyn.register_triage_tools(mcp, allow_raw_queries=False)  # no raise
    tools = {t.name for t in await mcp.list_tools()}
    assert "top_impacting_sql_api" in tools


@pytest.mark.asyncio
async def test_register_triage_tools_flag_change_raises(patched):
    mcp = FastMCP("t")
    dyn.register_triage_tools(mcp, allow_raw_queries=False)
    with pytest.raises(RuntimeError):
        dyn.register_triage_tools(mcp, allow_raw_queries=True)


@pytest.mark.asyncio
async def test_triage_tool_uses_family_description(patched):
    mcp = FastMCP("t")
    dyn.register_triage_tools(mcp, allow_raw_queries=False)
    tool = next(t for t in await mcp.list_tools() if t.name == "top_impacting_sql_api")
    assert "impact_score" in tool.description
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_register_triage_tools.py -v`
Expected: FAIL — `AttributeError: module 'dbs_vector.mcp.dynamic_tools' has no attribute 'register_triage_tools'`.

- [ ] **Step 3: Add `register_triage_tools`**

In `src/dbs_vector/mcp/dynamic_tools.py`, change the families import (line 13) to also import `TriageFamily`:

```python
from dbs_vector.mcp.families.base import BrowseFamily, TriageFamily
```

Append this function to the end of the file:

```python
def register_triage_tools(mcp: FastMCP, allow_raw_queries: bool) -> None:
    """Register one top_impacting_<engine> tool per SQL-family engine.

    Mirrors register_browse_tools' pre-flight (name pattern, collision, family
    resolution, idempotency) but registers ONLY engines whose
    resolved_family == "sql", uses verb="top_impacting" tool names, and sources
    the description from family.triage_description(engine, allow_raw_queries).
    """
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}
    registrations: dict[str, tuple] = mcp_any._dbs_vector_registrations

    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str, Any]] = []
    for engine_name, engine in settings.engines.items():
        if engine.resolved_family != "sql":
            continue
        if not ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {ENGINE_NAME_PATTERN.pattern}."
            )
        tool_name = normalize_tool_name(engine_name, verb="top_impacting")
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
        if not isinstance(family, TriageFamily):
            raise RuntimeError(
                f"Family '{family_key}' for engine '{engine_name}' does not "
                f"support triage (missing make_triage_handler/triage_description)."
            )
        prior = registrations.get(tool_name)
        current = (engine_name, family_key, allow_raw_queries)
        if prior is not None:
            if prior == current:
                continue
            raise RuntimeError(
                f"Stale triage tool registration for '{tool_name}': previously "
                f"{prior}, now {current}. Reset the FastMCP instance instead of "
                f"re-registering with different settings."
            )
        handler = family.make_triage_handler(engine_name, allow_raw_queries)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=family.triage_description(engine_name, engine, allow_raw_queries),
        )
        registrations[tool_name] = current
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_register_triage_tools.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Wire it into the server**

In `src/dbs_vector/mcp/server.py`, update the import (line 12):

```python
from dbs_vector.mcp.dynamic_tools import (
    register_browse_tools,
    register_search_tools,
    register_triage_tools,
)
```

and add the call after `register_browse_tools(...)` in `start_stdio_server` (after line 34):

```python
    register_triage_tools(mcp, allow_raw_queries=allow_raw_queries)
```

- [ ] **Step 6: Verify nothing regressed**

Run: `uv run pytest tests/unit/test_register_triage_tools.py tests/unit/test_register_browse_tools.py tests/unit/test_dynamic_tools.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/mcp/dynamic_tools.py src/dbs_vector/mcp/server.py tests/unit/test_register_triage_tools.py
git commit -m "feat(mcp): register top_impacting_<engine> triage tools and wire into stdio server

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Integration test — triage over real LanceDB

Prove the triage handler ranks correctly end-to-end against a real `LanceDBStore`, and that the derived `impact_score` ordering matches `calls * execution_time_ms`.

**Files:**
- Test: `tests/integration/test_browse_integration.py` (add one test, reusing `_make_store`/`_seed`)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_browse_integration.py`:

```python
@pytest.mark.asyncio
async def test_triage_handler_end_to_end(tmp_path, monkeypatch):
    import dbs_vector.mcp.state as state
    from dbs_vector.mcp.families.sql import SqlFamily

    store = _make_store(tmp_path)
    _seed(store)  # A: calls=10 exec=100 -> impact 1000; B: calls=5 exec=50 -> impact 250

    class _Svc:
        def __init__(self, s):
            self.vector_store = s

    monkeypatch.setitem(state._services, "sql-api", _Svc(store))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler()

    assert "impact_score" in out
    assert out.index("A") < out.index("B")  # A outranks B by impact_score
    assert "RAW-A" not in out  # raw_query gated off by default
```

Add `import pytest` to the top of the file if it is not already imported.

- [ ] **Step 2: Run the test (it passes immediately — this is a regression guard, not red-green TDD)**

This task adds no production code; Tasks 2 and 4 already provide everything it exercises, so the test passes as soon as it is written. Run it to confirm the end-to-end path over the real store.

Run: `uv run pytest tests/integration/test_browse_integration.py::test_triage_handler_end_to_end -v`
Expected: PASS.

- [ ] **Step 3: Run the full integration suite**

Run: `uv run pytest tests/integration/test_browse_integration.py -v`
Expected: PASS (all).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_browse_integration.py
git commit -m "test(integration): triage handler ranks by impact_score over real LanceDB

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Full validation suite

**Files:** none (verification only)

- [ ] **Step 1: Run the full check suite**

Run: `uv run poe check`
Expected: format clean, lint clean, typecheck clean, all tests PASS. (Pyright baseline: `src/` must stay 0 errors; the `tests/` tree has 13 known-intentional errors — do not treat those as regressions. If a NEW `src/` error appears, fix it before proceeding.)

- [ ] **Step 2: If `poe check` reformatted or fixed anything, commit it**

```bash
git add -p   # stage ONLY the files this plan touched; never git add -A
git commit -m "chore: ruff/format fixups for impact-triage tools

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

(Skip this commit if `poe check` made no changes.)

---

## Notes for the executor

- **No `--rebuild` is needed** — there is no LanceDB schema change in this plan.
- **The MCP server must be restarted** to pick up the new `top_impacting_<engine>` tools (registration happens at `start_stdio_server`).
- **Deferred / out of scope** (do NOT implement here): `tables_original`, any upstream rows aggregate, grouped-mode derived columns, the `list_engines` discoverability addition (optional — only if trivial and requested), and the `find-impacting-queries` skill refinement (separate follow-on, sequenced after these tools exist).
- **Staging discipline:** the branch carries unrelated in-flight work. Every commit stages only the files named in that task.
