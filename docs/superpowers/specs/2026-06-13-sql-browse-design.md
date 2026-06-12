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
- Zero changes to the existing semantic `search` **behavior** — new code only.
- **Description ownership refactor** (bundled, see dedicated section): move the
  verbose LLM-facing tool descriptions out of `config.yaml` and into the
  families, composed from inert engine facts. `config.yaml` keeps only a short
  one-line summary. This avoids `config.yaml` carrying *two* paragraphs per
  engine once browse adds a second description.

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
dbs-vector browse --type sql-api \
  --where 'execution_time_ms>1000000 AND user="app"' \
  --group-by user \
  --order-by execution_time_ms:desc \
  --select id,calls,execution_time_ms \
  --limit 10

# Point lookup
dbs-vector browse --type sql-api --where 'id="93FEDEB240C723E3"'

# Top fingerprints by call count (raw, no grouping)
dbs-vector browse --type sql-api-granite --order-by calls:desc --limit 10
```

Command: `dbs-vector browse --type <engine> [options]`. **Decision: engine is
selected with `--type/-t`, consistent with `ingest` and `search`** (engines are
ingested with `--type sql-api`, so they are browsed the same way). A non-SQL
engine is rejected with the list of available SQL engines.

Options:

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--type/-t` | str | required (must be a SQL engine) | Engine name; rejected if not a SQL-family engine. |
| `--where` | str \| None | None | LanceDB filter expression (DataFusion SQL predicate). |
| `--group-by` | str \| None | None | Column to aggregate by. Presence switches to grouped output. |
| `--order-by` | str | `execution_time_ms:desc` | `<column>[:asc\|:desc]`; default direction `desc`. |
| `--select` | str \| None | None (shape default) | Comma-separated output columns. |
| `--limit/-l` | int | 10 | Max rows returned. |
| `--json` | bool | False | Emit rows as JSON instead of a table. |

### MCP

One tool per SQL engine: `browse_sql`, `browse_sql_granite`, `browse_sql_api`,
`browse_sql_api_granite`. Registered by a **sibling registrar**
`register_browse_tools(mcp)` (see Architecture → MCP), not by modifying
`register_search_tools`.

The tool's **input schema is derived by FastMCP from the handler's function
signature** (type hints + defaults), so the parameters are literally:

```python
async def handler(
    where:    str | None = None,
    group_by: str | None = None,
    order_by: str        = "execution_time_ms:desc",
    select:   str | None = None,
    limit:    int        = 10,
) -> str: ...
```

The tool **description** is browse-specific (NOT `engine.description`, which is
semantic-flavored). It is a family-level template, with the engine's columns
injected, that enumerates the available columns and the `--where` predicate
syntax (LanceDB/DataFusion expression: string literals double-quoted, operators
`= != > >= < <= AND OR`, `IN (...)`) so the calling LLM forms valid filters. It
states explicitly: **`browse` ranks by the chosen scalar column, not by
similarity; there is no query string.**

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

### MCP — naming, registrar, family handler

**Naming (`core/naming.py`).** Generalize the tool-name helper to take a verb:

```python
def normalize_tool_name(engine_name: str, verb: str = "search") -> str:
    return f"{verb}_{engine_name.replace('-', '_')}"
```

`search` callers are unaffected (default verb); browse passes `verb="browse"`
→ `browse_sql_api`. Browse and search tool names share the
`_dbs_vector_registrations` namespace but cannot collide (`browse_*` vs
`search_*`).

**Registrar (`mcp/dynamic_tools.py`).** Add a sibling
`register_browse_tools(mcp)` mirroring `register_search_tools`'s pre-flight
(name-pattern check, collision check, family resolution, idempotency via
`_dbs_vector_registrations`), with two differences:

- It **only registers engines whose `resolved_family == "sql"`** — document
  engines are skipped entirely.
- It passes a **browse-specific description** (family template + injected
  columns), not `engine.description`.

It is invoked in `start_stdio_server()` (`mcp/server.py`) between the existing
`register_search_tools(mcp)` and `register_discovery_tool(mcp)` calls.
`register_search_tools` is also touched: its `description=` argument switches
from `engine.description` to `family.search_description(engine_name)` (see
**Description Ownership**).

**Family handler (`mcp/families/sql.py`).** Add to `SqlFamily`:

- `make_browse_handler(engine_name) -> handler` — async handler whose signature
  IS the five-param browse schema above. It:
  1. Looks up the already-initialized store via `_services[engine_name]`
     (browse needs **no embedder** — zero extra model load).
  2. Runs the blocking `BrowseService.browse(...)` inside a **single
     `asyncio.to_thread` closure** — same shared-`checkout_latest`-handle
     discipline as the search handler, but simpler (one read, no separate
     count call; the total is computed in-process from the scanned Arrow).
  3. Renders via a `format_browse(BrowseResult)` method that reuses
     `render_with_budget` for the MCP byte budget (compact table output).
  4. Catches exceptions (incl. malformed `where`) and returns the message as
     the tool result string so the LLM can self-correct — never raises.
- `browse_description(engine_name) -> str` — builds the family-level tool
  description with the engine's columns injected.

### CLI — `cli.py`

New `@app.command() def browse(...)` mirroring `search`: resolves the engine,
rejects non-SQL engines with the available-SQL-engines list, builds
`BrowseService` (no embedder needed), calls `browse(...)`, prints a table or
`--json`.

## Description Ownership (config cleanup)

**Problem.** The verbose, LLM-facing tool prose currently lives in each engine's
`config.yaml` `description:` field. It is long (a full paragraph per SQL engine)
and would *double* once browse adds a second description. config is the wrong
home for prose tuned for an LLM tool schema.

**Rule after this change.** *Families own the MCP tool descriptions; `config.yaml`
owns a short human summary.*

**Protocol.** `SearchFamily` (`mcp/families/base.py`) gains:

```python
def search_description(self, engine_name: str) -> str: ...
```

`SqlFamily` additionally has `browse_description(engine_name)` (already listed
above). Both `DocumentFamily` and `SqlFamily` implement `search_description`.

**Composition — template from inert engine facts (no per-engine code map).**
Each family holds ONE hardcoded template per verb and fills the variable bits
from config fields that already exist, so a new A/B engine variant still needs
**only a `config.yaml` edit** (preserving the contract in CLAUDE.md):

- *source phrase* ← `chunker_type`: `"api"` → "a remote slow-log API";
  `"duckdb"` → "a local DuckDB slow-query log".
- *embeddings phrase* ← `model`: `"granite-r2"` → "Granite embeddings";
  `"gemma-bf16"` → "Gemma embeddings". Unknown model → fall back to the model
  key (non-breaking).

The composed `search_description` reproduces the current verbose semantic prose
(filters `min_time` / `min_lock_time` / `table_filter`, the "Showing N of M"
truncation note) so there is **no LLM-facing regression** — the truncation note
is now applied uniformly to all four SQL engines (today `sql` / `sql-granite`
omit it; gaining it is an improvement). `browse_description` is the analytic
template (columns, `--where` grammar, group-by aggregates, "ranks by scalar, no
query string").

**Registrars source descriptions from the family, not config:**

- `register_search_tools`: `description=family.search_description(engine_name)`
  (was `engine.description`).
- `register_browse_tools`: `description=family.browse_description(engine_name)`.

**`config.yaml` change.** `description:` is shortened to a one-line summary and
**remains a required field**, now consumed only by the `list_engines` discovery
tool (`discovery.py:47`) and human readers. Example before/after for `sql-api`:

```yaml
# before (paragraph) → after:
description: "Slow-query fingerprints from a remote slow-log API (Gemma embeddings)."
```

All six engine `description:` fields are shortened the same way. No change to
`EngineConfig` (`config.py`) — the field stays `description: str`.

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
- **Unit (descriptions)** — `search_description` / `browse_description` compose
  the right source phrase per `chunker_type` and embeddings phrase per `model`
  across all four SQL engines; the "Showing N of M" truncation note is present;
  unknown model falls back to the model key. Guards against LLM-facing
  regression vs the current config prose.

## Defaults Summary

- `--order-by` default: `execution_time_ms:desc`.
- `--limit` default: `10` (analytical top-N; larger than `search`'s `5`).
- Raw default columns: `id, calls, execution_time_ms, lock_time_sec, user, tables`.
- Grouped default columns: group key + all Standard+avg aggregates.
