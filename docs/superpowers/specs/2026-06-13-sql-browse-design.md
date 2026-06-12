# `browse` — Analytical Scalar Access for SQL Engines

**Date:** 2026-06-13
**Status:** Approved (design)
**Branch:** `feat/sql-browse`

## Problem

The public search surface (`dbs-vector search` + `search_<engine>` MCP tools)
only does **hybrid semantic retrieval**: it embeds a query string and ranks by
cosine similarity, exposing exactly three SQL prefilters (`min_time`,
`min_lock_time`, `table_filter`). There is no way through the public surface to:

- **Point-lookup** a fingerprint by `id` or `content_hash`.
- **Rank** fingerprints by a scalar column (`calls`, `execution_time_ms`, …)
  without a query string.
- **Aggregate** ("which user / table / service burns the most DB time").
- **Project** a compact column subset.

Today these require dropping into a Python REPL and reading the LanceDB table as
Arrow by hand. This work exposes that scalar/analytical access as a first-class
verb on both the CLI and MCP, **without touching the semantic path**.

## Goals

- A new `browse` operation: filter + group-by + order-by + projection over a
  SQL engine's table, reading scalar columns directly (no embedder).
- Available on the CLI (`dbs-vector browse`) and MCP (`browse_<engine>`).
- Available for **all SQL engines** (`sql`, `sql-granite`, `sql-api`,
  `sql-api-granite`) via the shared `SqlFamily`.
- Zero changes to the existing semantic `search` path — new code only.

## Non-Goals (YAGNI)

- No `browse` for document/markdown engines (different columns, not the use case).
- No write/mutation operations — `browse` is strictly read-only.
- No structured per-predicate flags (`--user`, `--min-calls`, …). Filtering is a
  single `--where` expression passed through to LanceDB.
- No joins across tables, no cross-engine queries.
- No new ranking inside `search`. `search` stays purely semantic.

## Surface

### CLI

```bash
# Rank: heaviest users by total execution time
dbs-vector browse sql-api \
  --where 'execution_time_ms>1000000 AND user="app"' \
  --group-by user \
  --order-by execution_time_ms:desc \
  --select id,calls,execution_time_ms \
  --limit 10

# Point lookup
dbs-vector browse sql-api --where 'id="93FEDEB240C723E3"'

# Top fingerprints by call count (raw, no grouping)
dbs-vector browse sql-api --order-by calls:desc --limit 10
```

Command: `dbs-vector browse <engine> [options]`. **Decision: engine is a
positional argument.** `search` uses its positional slot for the query string
and takes the engine via `--type`; `browse` has no query string, so its
positional slot is free for the engine — which reads naturally
(`browse sql-api ...`) and matches the sketched syntax. A non-SQL engine is
rejected with the list of available SQL engines.

Options:

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `<engine>` (positional) | str | required | Engine name; rejected if not a SQL-family engine. |
| `--where` | str \| None | None | LanceDB filter expression (DataFusion SQL predicate). |
| `--group-by` | str \| None | None | Column to aggregate by. Presence switches to grouped output. |
| `--order-by` | str | `execution_time_ms:desc` | `<column>[:asc\|:desc]`; default direction `desc`. |
| `--select` | str \| None | None (shape default) | Comma-separated output columns. |
| `--limit/-l` | int | 10 | Max rows returned. |
| `--json` | bool | False | Emit rows as JSON instead of a table. |

### MCP

One tool per SQL engine: `browse_sql`, `browse_sql_granite`, `browse_sql_api`,
`browse_sql_api_granite`. Parameters mirror the CLI:

```
browse_<engine>(
  where:    str | None = None,
  group_by: str | None = None,
  order_by: str        = "execution_time_ms:desc",
  select:   str | None = None,
  limit:    int        = 10,
)
```

The tool description enumerates the available columns and the `--where`
predicate syntax (LanceDB/DataFusion expression, string literals double-quoted,
operators `= != > >= < <= AND OR`, `IN (...)`) so the calling LLM forms valid
filters. It states explicitly: **`browse` ranks by the chosen scalar column, not
by similarity; there is no query string.**

## Available Columns

From `SqlChunk` (`core/models.py`):

| Column | Type | Groupable | Aggregatable |
|--------|------|-----------|--------------|
| `id` | str | yes | — |
| `content_hash` | str | yes | — |
| `user` | str \| None | yes | — |
| `host` | str \| None | yes | — |
| `source` | str (db name) | yes | — |
| `tables` | list[str] | yes (exploded) | — |
| `calls` | int | — | sum, avg |
| `execution_time_ms` | float | — | sum, avg |
| `lock_time_sec` | float \| None | — | sum |
| `rows_examined` | int \| None | — | sum |
| `rows_sent` | int \| None | — | sum |
| `latest_ts` | datetime | — | max |

`text` / `raw_query` are large and excluded from default projections (but
selectable explicitly).

## Output Shapes

### Raw (no `--group-by`)

One row per fingerprint. Default columns (when `--select` omitted):
`id, calls, execution_time_ms, lock_time_sec, user, tables`.

### Grouped (`--group-by <col>`)

One row per distinct group key. Aggregate set (**Standard + avg variants**):

| Output column | Definition |
|---------------|------------|
| `<group-key>` | the group value |
| `fingerprints` | `count(*)` rows in the group |
| `calls` | `sum(calls)` |
| `execution_time_ms` | `sum(execution_time_ms)` |
| `lock_time_sec` | `sum(lock_time_sec)` |
| `rows_examined` | `sum(rows_examined)` |
| `avg_execution_time_ms` | `sum(execution_time_ms) / fingerprints` |
| `avg_calls` | `sum(calls) / fingerprints` |

`--order-by` may target the group key or any aggregate column above.
`--select` may name the key or any aggregate column.

Grouping by `tables` (a list column) **explodes** each fingerprint into one row
per table first, so a fingerprint touching N tables contributes to N groups.
This is documented as a known semantic (a fingerprint's `calls` count toward
every table it touches — touches, not exclusive attribution).

`NULL` group keys (e.g. fingerprints with no `user`) collapse into a single
`(none)` group rather than being dropped.

## Architecture

Follows the existing flow: **CLI/MCP → Service → Port → Infrastructure**.

### Port — `core/ports.py`

Add one method to `IVectorStore`:

```python
def scan(
    self,
    where: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> Any:  # pyarrow.RecordBatch
    """Filtered, projected scalar read. No vector search."""
    ...
```

### Infrastructure — `LanceDBStore.scan()`

Pushes `where` down to LanceDB's native filter and projects `columns`. No vector
query is issued. A malformed `where` raises; the service catches it and surfaces
the underlying message verbatim (see Error Handling). `limit` here is an
optional pre-aggregation cap; when grouping, the service does **not** pass a
`limit` to `scan` (it must see all matching rows to aggregate correctly) and
applies the limit after aggregation.

### Service — `services/browse.py` `BrowseService`

Pure orchestration, unit-testable, depends only on `IVectorStore`:

```python
class BrowseService:
    def __init__(self, store: IVectorStore) -> None: ...

    def browse(
        self,
        where: str | None,
        group_by: str | None,
        order_by: str,        # "<col>[:asc|:desc]"
        select: list[str] | None,
        limit: int,
    ) -> BrowseResult: ...
```

Pipeline:

1. Validate `group_by` / `order_by` column / `select` columns against the known
   column + aggregate vocabulary → friendly error on unknown names.
2. `store.scan(where, columns=<needed>)` → Arrow.
3. If `group_by`: explode `tables` if needed → aggregate (Standard + avg).
4. Apply `order_by` (parse `:asc`/`:desc`, default `desc`).
5. Truncate to `limit`.
6. Project to `select` (or shape default).

Returns a `BrowseResult` (rows as `list[dict]` + `total_matching` count +
`grouped: bool`) so the formatter and `--json` path share one structure.
All steps 3–6 are pure pandas/Arrow compute with no I/O → directly unit-tested.

### MCP — `SqlFamily`

Add to `SqlFamily` (`mcp/families/sql.py`):

- `make_browse_handler(engine_name) -> handler` — async handler with the five
  browse params; builds a `BrowseService` from the engine's store and renders
  via a `format_browse` method (reuses `render_with_budget` for the MCP byte
  budget; compact table output).
- `register_search_tools` (or a sibling `register_browse_tools`) registers
  `browse_<engine>` for every engine whose `resolved_family == "sql"`. Same
  pre-flight atomic / idempotent / collision-safe discipline as the existing
  search-tool registration.

### CLI — `cli.py`

New `@app.command() def browse(...)` mirroring `search`: resolves the engine,
rejects non-SQL engines with the available-SQL-engines list, builds
`BrowseService` (no embedder needed), calls `browse(...)`, prints a table or
`--json`.

## Error Handling

- **Malformed `--where`** → catch the LanceDB/DataFusion exception, return a
  single-line error that includes the engine's available columns and the
  offending message. MCP: returned as the tool result text (not an exception)
  so the LLM can self-correct. CLI: `typer.echo` + `Exit(code=1)`.
- **Unknown `--group-by` / `--order-by` / `--select` column** → validated before
  any I/O; friendly message listing valid columns / aggregates.
- **Non-SQL engine** → rejected up front (browse is SQL-family only).
- **Empty result** → "0 rows matched" message, not an error.

## Testing Strategy

- **Unit (`tests/unit/test_browse_service.py`)** — pure aggregation/sort/limit
  logic against an in-memory Arrow table via a fake `IVectorStore.scan`. Covers:
  raw vs grouped, `tables` explode, NULL group collapse, avg computation,
  order-by direction, select projection, unknown-column validation.
- **Unit (MCP)** — `make_browse_handler` formatting, byte-budget truncation,
  malformed-`where` → error-string-not-exception.
- **Integration (`tests/integration/`)** — real tmpdir LanceDB seeded with a
  few `SqlChunk`s, exercise `LanceDBStore.scan` with real `where` pushdown,
  point lookup by id, group-by user end-to-end.
- **CLI smoke** — `browse` command wiring (engine resolution, non-SQL
  rejection) with a mocked service.

## Defaults Summary

- `--order-by` default: `execution_time_ms:desc`.
- `--limit` default: `10` (analytical top-N; larger than `search`'s `5`).
- Raw default columns: `id, calls, execution_time_ms, lock_time_sec, user, tables`.
- Grouped default columns: group key + all Standard+avg aggregates.
