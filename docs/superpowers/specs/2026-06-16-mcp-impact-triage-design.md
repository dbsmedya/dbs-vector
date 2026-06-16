# dbs-vector MCP impact-triage enhancements

**Date:** 2026-06-16
**Branch:** `feat/sql-browse`
**Status:** Design complete — next is writing-plans.

## Goal

Make the dbs-vector MCP self-sufficient for the **find-impacting-queries** workflow:
rank slow-query fingerprints by impact, expose the selectivity signals a DBA needs, and
return replayable exemplars safely — so the workflow never reaches around the MCP to read
LanceDB directly (LanceDB is debug-only; see memory `mcp-transport-ceiling`).

## Background & motivation

A live run of the find-impacting-queries skill against the odeal slow-log (`sql-api`
engine) surfaced three gaps in the MCP surface:

1. **Ranking is manual.** `browse` exposes `calls` and `execution_time_ms` separately, and
   `avg_ms_per_call` / selectivity exist only in *grouped* mode or in *search* output —
   not as flat per-fingerprint columns. Finding "what's hammering the DB" meant eyeballing
   columns and doing arithmetic by hand.
2. **`browse` can drop the connection.** `search` truncates long `raw_query` bodies
   (`_truncate_raw_query`, `families/sql.py:26`) but the **browse formatter does not**
   (`_fmt_cell`, `families/sql.py:64` returns `str(value)` untruncated). A single multi-MB
   exemplar in a `select=raw_query` browse produces an oversized JSON-RPC frame and drops
   the batch — the live bug tracked in memory `mcp-transport-ceiling`.
3. **No one-call triage.** Getting the top-N offenders with their stats + replayable SQL
   took a ranking call followed by N per-id fetches.

This spec closes all three, MCP-side only.

### The impact metric (decided with the domain owner)

Rank by **`impact_score = calls × execution_time_ms`**. Because `execution_time_ms` is
*already cumulative* (a true SUM across all calls — see the data-model note below),
`impact_score` expands to `calls² × avg_ms_per_call`: it weights call **frequency**
quadratically. This is deliberate — a "frequently hammering" query has externalities raw
total time misses (lock acquisitions, connection churn, buffer-pool thrash). Plain
`execution_time_ms` (the classic pt-query-digest total-time rank) stays available as an
`order_by` override.

Worked example (odeal top-5), `impact_score` re-ranks to match operator intuition — the
two highest-frequency offenders rise to the top, and a low-frequency point-lookup that was
3rd by total time correctly sinks to last:

| rank by impact_score | id | calls | total exec (ms) | impact_score |
|---|---|---:|---:|---:|
| 1 | `4D7F5F80…` | 41,025 | 268,181,513 | 1.10×10¹³ |
| 2 | `EF973A49…` | 30,276 | 85,156,956 | 2.58×10¹² |
| 3 | `B8865E12…` | 11,606 | 87,258,758 | 1.01×10¹² |
| 4 | `AE66E4D9…` | 2,680 | 181,839,134 | 4.87×10¹¹ |
| 5 | `85AD43A4…` | 957 | 109,236,876 | 1.05×10¹¹ |

### Data-model truth (grounds the column design)

Per the API contract (`scripts/claude_api_contract.md` §5.2) and the DuckDB chunker
(`infrastructure/chunking/duckdb.py:27`, `arg_max(rows_examined, ts)`):

- `execution_time_ms` is **cumulative** (SUM across all calls). `calls` is the COUNT.
- `rows_examined`, `rows_sent`, `lock_time_sec` are **`arg_max` values — the single
  most-recent call**, NOT sums and NOT averages.

Consequences for derived columns:

- `avg_ms_per_call = execution_time_ms / calls` is **valid** (summed time ÷ count).
- `selectivity = rows_examined / rows_sent` is a valid **per-call ratio** (NULL when
  `rows_sent = 0`, i.e. writes).
- A true `rows_per_call` average is **NOT derivable** in the MCP today — the corpus carries
  no sum/average of rows to divide. `rows_examined` / `rows_sent` are *already* per-call
  (one specific call's) figures; surface them as-is. (A true average/worst-case needs an
  upstream `AVG`/`MAX(rows_examined)` aggregate — deferred; see §"Forward-compat".)

## Scope

In scope (MCP-only, **no LanceDB schema change, no `--rebuild`, no GoFast dependency**):

- **A.** Derived columns in flat `browse`: `impact_score`, `avg_ms_per_call`,
  `selectivity` — selectable and orderable per fingerprint.
- **B.** Transport-safety: truncate long string cells (`raw_query`, `text`) in the MCP
  browse formatter.
- **C.** A thin `top_impacting_<engine>` tool (SQL family only) — top-N by `impact_score`
  with curated columns + optional truncated exemplar, in one call.

Out of scope:

- `tables_original` and any upstream rows aggregate (see Forward-compat; deferred — under
  heavy development upstream).
- Grouped-mode derived columns (flat-only now; grouped keeps its existing aggregates).
- The CLI raw-SQL browse path (`BrowseService.run_sql`) — a user already writes
  `calls*execution_time_ms` themselves there; no change.
- The `content_hash` cross-fingerprint conflation (documented elsewhere as known).

---

## A. Derived columns in flat `browse` — `services/browse.py`

Today flat (non-grouped) browse selects only base scalar columns (`_SCALAR_COLUMNS`,
`browse.py:27`). Add three computed columns, mirroring the existing grouped `_AGG_SQL`
pattern (`browse.py:57`). Define them as **bare SQL expressions** (no `AS` alias) so they
can be used in `ORDER BY` even when not selected:

```python
# browse.py — near _AGG_SQL. Flat-mode derived columns: bare expressions over
# per-fingerprint rows. Selectable AND orderable in non-grouped browse.
_FLAT_DERIVED_EXPR = {
    "impact_score":   '"calls" * "execution_time_ms"',
    "avg_ms_per_call": '"execution_time_ms" / NULLIF("calls", 0)',
    "selectivity":     '"rows_examined" / NULLIF("rows_sent", 0)',
}
```

> **Verification point for the plan:** the grouped builder already uses `NULLIF(SUM(...),0)`
> in `pl.SQLContext`, so `NULLIF` is supported. The TDD divide-by-zero test confirms
> column-level `NULLIF` works; if polars rejects it, fall back to
> `CASE WHEN "calls" = 0 THEN NULL ELSE "execution_time_ms" / "calls" END`.

### A.1 Make them selectable

`_selectable` (`browse.py:217`) currently returns `_SCALAR_COLUMNS + (raw_query?)`. Extend:

```python
def _selectable(self, allow_raw_queries: bool) -> tuple[str, ...]:
    return (
        _SCALAR_COLUMNS
        + tuple(_FLAT_DERIVED_EXPR)
        + (("raw_query",) if allow_raw_queries else ())
    )
```

`_DEFAULT_SELECT` (`browse.py:70`) is **unchanged** — derived columns appear only when
explicitly selected (backwards-compatible for existing browse callers).

### A.2 Emit them in `_build_flat_sql`

In `_build_flat_sql` (`browse.py:256`):

- Validation against `selectable` already covers the new names (raw_query gate unchanged).
- Build SELECT terms with a helper that emits the aliased expression for derived columns
  and a quoted identifier for base columns:

```python
def _flat_select_term(c: str) -> str:
    if c in _FLAT_DERIVED_EXPR:
        return f"{_FLAT_DERIVED_EXPR[c]} AS {_q(c)}"
    return _q(c)
# ... SELECT {', '.join(_flat_select_term(c) for c in chosen)} FROM data
```

- `order_by`: `_parse_order_by(order_by, set(selectable))` already validates the column
  (it rejects only `tables` and unknown names — derived names are now in `selectable`).
  Build the ORDER BY term from the **bare expression** for derived columns so ordering
  works even when the column is not in the projection:

```python
order_term = _FLAT_DERIVED_EXPR[col] if col in _FLAT_DERIVED_EXPR else _q(col)
# ... ORDER BY {order_term} {direction} NULLS LAST
```

### A.3 Surface in `browse_description`

Append to the `cols` string in `SqlFamily.browse_description` (`families/sql.py:232`):
note `impact_score (calls×execution_time_ms)`, `avg_ms_per_call`, `selectivity
(rows_examined/rows_sent)` as additional selectable/orderable columns available in
non-grouped mode.

---

## B. Transport-safety — `mcp/families/sql.py`

Truncate long string cells in the **MCP** browse formatter so no single row can exceed the
frame budget. The CLI formatters (`result_to_table`, `result_to_json` in `browse.py`) stay
**uncapped** (operator wants full SQL). Modify `_fmt_cell` (`families/sql.py:64`) to reuse
the existing `_truncate_raw_query` helper (which is a no-op below `_RAW_QUERY_DISPLAY_LIMIT`,
`families/sql.py:23` = 2,000):

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

This protects both `raw_query` and `text` (and any future long string column). Combined
with the existing `render_with_budget` total-response cap (`families/base.py:27`), the
browse path can no longer drop the connection. This is the server-side fix memory
`mcp-transport-ceiling` has tracked.

---

## C. `top_impacting_<engine>` tool — SQL family only

A thin convenience that wraps `BrowseService.build_and_run` with curated presets — **no new
ranking logic** (DRY). Registered only for `resolved_family == "sql"` engines, alongside
`search_`/`browse_`.

### C.1 New Protocol — `mcp/families/base.py`

Mirror `BrowseFamily` (`base.py:134`) with a narrow Protocol so the registrar can
`isinstance`-narrow:

```python
@runtime_checkable
class TriageFamily(Protocol):
    """A family that exposes a top_impacting triage tool. SQL families only."""
    def make_triage_handler(self, engine_name: str, allow_raw_queries: bool = False) -> Any: ...
    def triage_description(
        self, engine_name: str, engine: "EngineConfig", allow_raw_queries: bool
    ) -> str: ...
```

### C.2 Handler — `SqlFamily.make_triage_handler` (`mcp/families/sql.py`)

```python
_TRIAGE_SELECT = (
    "id,tables,calls,execution_time_ms,impact_score,avg_ms_per_call,"
    "lock_time_sec,rows_examined,rows_sent,selectivity,latest_ts"
)
_TRIAGE_ORDER_ALLOWLIST = (
    "impact_score", "execution_time_ms", "calls",
    "lock_time_sec", "avg_ms_per_call", "selectivity",
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
        # Silent downgrade (search-style): include_raw honoured only under the flag.
        if include_raw and allow_raw_queries:
            select += ",raw_query"

        frame_alias = engine_name.replace("-", "_")
        browse = BrowseService(service.vector_store, frame_alias)
        filters = {"table": table, "min_calls": min_calls}

        def _run() -> BrowseResult:
            return browse.build_and_run(
                filters=filters, group_by=None, order_by=order_by,
                select=select, limit=limit, allow_raw_queries=allow_raw_queries,
            )

        try:
            result = await asyncio.to_thread(_run)
            return format_triage(result)  # dedicated formatter (see below)
        except BrowseValidationError as e:
            return str(e)
        except Exception as e:
            logger.warning("triage '{}' failed: {}", engine_name, e)
            return "triage execution failed (see server logs)."
    return handler
```

Notes:
- `filters` passes only `table`/`min_calls`; `_filtered_frame` (`browse.py:278`) reads the
  rest via `.get(...)` → `None`, so omitted filters are inert.
- `table` matches the normalized (lowercased) `tables` list — same semantics as `browse`.
- `include_raw` with the flag off is **silently downgraded** (search posture), not an error
  — friendlier for a convenience tool.
- `order_by` is restricted to the impact-relevant allowlist; anything else returns a clear
  message rather than ordering by an odd column.
- **Dedicated `format_triage` formatter** (module-level in `families/sql.py`, beside
  `format_browse`): renders the curated scalar columns as a compact header line per
  fingerprint, and the `raw_query` exemplar — when present — on its **own block**
  (`Raw SQL:\n…`, like `search` does) rather than as an inline `raw_query=<blob>` cell, so a
  multi-line exemplar is paste-ready for `EXPLAIN`. Long cells are truncated via `_fmt_cell` /
  `_truncate_raw_query` (Component B) and the whole response stays under the byte budget.

### C.3 Description — `SqlFamily.triage_description`

Explain: returns the top-`limit` fingerprints ranked by `impact_score`
(`calls × execution_time_ms`, frequency-weighted "what's hammering the DB"); columns
include `avg_ms_per_call`, `selectivity`, per-call `rows_examined`/`rows_sent`; `order_by`
override ∈ the allowlist; `table`/`min_calls` filters; and that the verbatim `raw_query`
exemplar (ready to paste into a MySQL `EXPLAIN`) appears only when the server has
`--allow-raw-queries` **and** `include_raw=true`. State that `rows_examined`/`rows_sent` are
most-recent-call values, not averages.

### C.4 Registration — `mcp/dynamic_tools.py` + `mcp/server.py`

Add `register_triage_tools(mcp, allow_raw_queries)` mirroring `register_browse_tools`
(`dynamic_tools.py:97`): same pre-flight (name pattern, collision, family resolution),
same `_dbs_vector_registrations` idempotency tuple `(engine_name, family_key,
allow_raw_queries)`, filtered to `resolved_family == "sql"`, `verb="top_impacting"` (→
`normalize_tool_name` yields `top_impacting_<engine>`), and an
`isinstance(family, TriageFamily)` guard. Tool names don't collide with `search_`/`browse_`
keys, so they share the registrations dict safely.

In `start_stdio_server` (`server.py:18`) add, after the browse registration:

```python
register_triage_tools(mcp, allow_raw_queries=allow_raw_queries)
```

### C.5 Discoverability (optional, low-priority)

`list_engines` (`mcp/discovery.py`) currently advertises only `mcp_tool: search_<engine>`.
Optionally add `browse_tool` and `top_impacting_tool` names to SQL-engine entries so an LLM
discovers the triage path. Not required for the workflow; include only if cheap.

---

## Forward-compat (deferred — not built here)

The triage curated select and the browse selectable set are designed so these slot in
additively, with no breaking change, when the upstream work lands:

- **`tables_original`** (original-case, schema-qualified table names) — already specced in
  `docs/superpowers/specs/2026-06-14-sql-tables-original-followups-design.md` (item A).
  When GoFast emits it and the LanceDB schema is rebuilt, add `tables_original` to
  `_TRIAGE_SELECT` and the browse selectable set. Makes fingerprints replayable against a
  `lower_case_table_names=0` server without needing `raw_query`.
- **True per-call rows aggregate** — an upstream `AVG`/`MAX(rows_examined)` (and rows_sent)
  per fingerprint would enable a genuine average/worst-case `rows_per_call`. Bundle with
  the `tables_original` finalization (both are GoFast `/sql/queries` aggregation changes).

---

## Testing (TDD order)

Unit-first; all three components are unit-testable without I/O. Integration covers the
derived columns + triage over real LanceDB.

**A — derived columns (`tests/unit/test_browse_service.py`):**
- flat `select=impact_score,avg_ms_per_call,selectivity` returns correct values for a known
  row (e.g. calls=10, execution_time_ms=1000 → impact_score=10000, avg=100).
- divide-by-zero: `calls=0` → `avg_ms_per_call` is null; `rows_sent=0` → `selectivity` null.
- `order_by=impact_score:desc` orders rows correctly **when `impact_score` is not in
  `select`** (proves the bare-expression ORDER BY path).
- `order_by=selectivity:desc` works; unknown derived name in `select` → `BrowseValidationError`.

**B — transport-safety (`tests/unit/test_browse_mcp_handler.py` or sibling):**
- `_fmt_cell` truncates a string > 2,000 chars with the elision marker; a short string is
  unchanged; a long joined list is truncated.
- `format_browse` over a `BrowseResult` whose `raw_query` cell is ~5,000 chars produces
  output containing the elision marker and stays under `RESPONSE_BUDGET_BYTES`.

**C — triage handler + registration:**
- handler (new `tests/unit/test_triage_mcp_handler.py`, mirroring the browse handler test):
  default `order_by=impact_score`; curated columns present; `order_by` not in allowlist →
  the friendly message; `include_raw=True, allow_raw_queries=False` → **no** `raw_query`
  column (silent downgrade); `include_raw=True, allow_raw_queries=True` → present;
  `table`/`min_calls` filters applied; uninitialized service → error string.
- registration (`tests/unit/` alongside the browse-registration test): `register_triage_tools`
  registers `top_impacting_<engine>` only for SQL engines, threads `allow_raw_queries`, is
  idempotent, and raises on stale/colliding registration (mirror the browse tests).

**Integration (`tests/integration/test_browse_integration.py`):**
- ingest fingerprints over real LanceDB; triage returns ranked rows; assert
  `impact_score` ordering equals `calls × execution_time_ms` ordering and the derived
  columns are present.

## Sequencing & independence

- **A, B, C are independent of GoFast and of `tables_original`/`--rebuild`.** They can all
  land on `feat/sql-browse` without any upstream or schema change.
- **B** is self-contained (one formatter function) and can land first.
- **A** is a prerequisite for **C** (the triage tool selects `impact_score`/`selectivity`).
- The branch's unrelated in-flight work is untouched.

## Skill refinement (`find-impacting-queries`) — follow-on

After the MCP changes land, update the skill's reference notes (not its live steps):
- Use `top_impacting_<engine>` as the step-2 entry point; rank by `impact_score`.
- Use the derived `selectivity`/`avg_ms_per_call` columns instead of manual math.
- Record the **`--super-read-only` blocks `EXPLAIN UPDATE/DELETE`** workaround (rewrite the
  WHERE into a `SELECT`).
- Record the **no-direct-LanceDB** policy (debug-only), matching memory
  `mcp-transport-ceiling`.
- Note the per-call `rows_examined` caveat: it can exceed table size (a full-scan signal
  the replica EXPLAIN may miss, as seen on the `terminalversion` UPDATE).
