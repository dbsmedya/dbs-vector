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
# Rank: heaviest users by total execution time (grouped: select = key + aggregates only)
dbs-vector browse --type sql-api \
  --where 'execution_time_ms>1000000 AND user="app"' \
  --group-by user \
  --order-by execution_time_ms:desc \
  --select user,calls,execution_time_ms,avg_ms_per_call \
  --limit 10

# Point lookup
dbs-vector browse --type sql-api --where 'id="93FEDEB240C723E3"'

# All fingerprints touching a given table (list-membership predicate)
dbs-vector browse --type sql-api --where "array_has(tables, 'orders')" --order-by calls:desc

# Top fingerprints by call count (raw, no grouping)
dbs-vector browse --type sql-api-granite --order-by calls:desc --limit 10
```

Command: `dbs-vector browse --type <engine> [options]`. **Decision: engine is
selected with `--type/-t`, consistent with `ingest` and `search`** (engines are
ingested with `--type sql-api`, so they are browsed the same way). A non-SQL
engine is rejected with the list of available SQL engines.

**`--where` grammar.** The string is passed to LanceDB's native
(DataFusion-backed) filter — expression-only, read-only. Supported:

- Comparison `= != > >= < <=`, boolean `AND OR NOT`, `IN (...)`.
- **List membership** `array_has(tables, 'orders')` — the ONLY way to filter the
  `tables` list column (the headline "queries touching table X" use case).
  This is exactly what the existing `table_filter` compiles to
  (`lancedb_engine.py:152`).
- **Null tests** `lock_time_sec IS NULL`, `user IS NOT NULL` — useful given the
  nullable columns.
- **String literals:** both single (`user='app'`) and double (`user="app"`)
  quotes parse on lancedb 0.30.x. Shell-facing examples use double quotes
  *inside* a single-quoted `--where '...'` so the shell doesn't eat them; the
  programmatic codebase filters use single quotes (no shell involved). Either is
  valid — the MCP description shows double-quoted literals.

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
`= != > >= < <= AND OR NOT`, `IN (...)`, list membership
`array_has(tables, 'x')`, null tests `IS NULL` / `IS NOT NULL`) so the calling
LLM forms valid filters. It states explicitly: **`browse` ranks by the chosen
scalar column, not by similarity; there is no query string**, and that `tables`
can only be filtered via `array_has`, never `tables = 'x'`.

## Available Columns

From `SqlChunk` (`core/models.py`):

| Column | Type | Group-by | Order-by | Aggregate (grouped) |
|--------|------|----------|----------|---------------------|
| `id` | str | yes | yes (lexical) | — |
| `content_hash` | str | yes | yes (lexical) | — |
| `user` | str \| None | yes | yes (lexical) | — |
| `host` | str \| None | yes | yes (lexical) | — |
| `source` | str (db name) | yes | yes (lexical) | — |
| `tables` | list[str] | yes (exploded) | **no** (list order undefined) | — |
| `calls` | int | — | yes | sum |
| `execution_time_ms` | float | — | yes | sum |
| `lock_time_sec` | float \| None | — | yes | sum |
| `rows_examined` | int \| None | — | yes | sum |
| `rows_sent` | int \| None | — | yes | sum |
| `latest_ts` | datetime | — | yes (chronological) | max |

**Order-by vocabulary (#9):** any scalar column above is a valid `--order-by`
target (numeric, string→lexical, timestamp→chronological); `tables` is **not**
(ordering a list is undefined → rejected). In grouped mode `--order-by` /
`--select` may also name any aggregate output column (see Grouped table).

**Excluded internal columns (#5):** `vector` and `workflow` are physical Arrow
columns but are **not** part of the browse vocabulary — not selectable,
groupable, orderable, or in the `--where` autocomplete (a raw `--where` string
can still reference them, but the description never advertises them). `text` /
`raw_query` are large embedding/display fields: excluded from default
projections, but `raw_query` is selectable explicitly; `text` is not.

**Table-sharing semantics (#5).** `browse` scans an engine's `table_name`
directly and does **not** filter on `workflow`. Several engines share a physical
table: `sql` and `sql-api` both point at `query_vault` with the same `workflow`
string (`config.yaml:32,43`), so `browse_sql` and `browse_sql_api` scan the same
rows, and aggregates merge both corpora if both have been ingested. This is the
**same behavior as the existing `search` tools** (they don't filter on `workflow`
either) — browse introduces no new isolation and no regression. Engines with
distinct tables (`*_granite`, `*_granite_api`) are naturally isolated.

## Output Shapes

### Raw (no `--group-by`)

One row per fingerprint. Default columns (when `--select` omitted):
`id, calls, execution_time_ms, lock_time_sec, user, tables`.

### Grouped (`--group-by <col>`)

One row per distinct group key. Aggregate set:

| Output column | Definition |
|---------------|------------|
| `<group-key>` | the group value |
| `fingerprints` | `count(*)` rows in the group |
| `calls` | `sum(calls)` |
| `execution_time_ms` | `sum(execution_time_ms)` |
| `lock_time_sec` | `sum(lock_time_sec)` |
| `rows_examined` | `sum(rows_examined)` |
| `rows_sent` | `sum(rows_sent)` |
| `latest_ts` | `max(latest_ts)` — most recent occurrence in the group |
| `avg_ms_per_fingerprint` | `sum(execution_time_ms) / fingerprints` |
| `avg_ms_per_call` | `sum(execution_time_ms) / sum(calls)` — per-execution cost |

The two averages are named to be unambiguous (#10): `avg_ms_per_fingerprint` is
the average cumulative time per fingerprint in the group; `avg_ms_per_call` is
the average time of a single query execution — the number a DBA usually reads.

`--order-by` may target the group key or any aggregate column above.
`--select` may name the key or any aggregate column (it may **not** name a raw
per-fingerprint field like `id`, which is undefined under grouping).

Grouping by `tables` (a list column) **explodes** each fingerprint into one row
per table first, so a fingerprint touching N tables contributes to N groups.
This is documented as a known semantic (a fingerprint's `calls` count toward
every table it touches — touches, not exclusive attribution).

**NULL semantics (#7):**

- *Group keys:* `NULL` keys (e.g. fingerprints with no `user`) collapse into a
  single `(none)` group rather than being dropped.
- *Sums:* NULLs are skipped; a group whose summed column is entirely NULL yields
  `0.0`, never NULL. `avg_ms_per_call` with `sum(calls) == 0` yields `0.0`
  (guard against divide-by-zero).
- *Sorting:* when `--order-by` targets a nullable column, NULLs sort **last**
  regardless of `:asc` / `:desc` (applies to both raw and grouped modes).

**Counts / `total_matching` (#8):**

- *Raw mode:* `total_matching` = number of fingerprint rows matching `--where`
  (pre-limit).
- *Grouped mode:* `total_matching` = number of distinct groups after
  aggregation (pre-limit) — so the header reads "Showing 10 of 47 users",
  matching the unit that was actually limited (groups, not rows).

## Architecture

Follows the existing flow: **CLI/MCP → Service → Port → Infrastructure**.

### Port — `core/ports.py`

Add one method to `IVectorStore`:

```python
def scan(
    self,
    where: str | None = None,
    columns: list[str] | None = None,
) -> Any:  # pyarrow.RecordBatch
    """Filtered, projected scalar read of ALL matching rows. No vector search,
    no ordering, no row cap. Implementations MUST call checkout_latest() before
    reading, exactly as search()/count_matching() do, so the scan sees the
    current table version."""
    ...
```

**No `limit` parameter (#1).** `scan` has no ordering, so a pushed-down row cap
would return an *arbitrary* subset, not the global top-N — and would make raw
mode's default `order_by` and the `total_matching` count wrong. Ordering and the
limit are always applied **in-process by the service** after the full filtered
read. The corpus is ~6,200 rows, so reading all matching rows is cheap.

### Infrastructure — `LanceDBStore.scan()`

Calls `checkout_latest()` (same as the search path, `lancedb_engine.py:118`),
pushes `where` down to LanceDB's native filter, projects `columns`, and returns
**all** matching rows as Arrow. No vector query, no limit, no ordering. A
malformed `where` raises; the service catches it and surfaces the underlying
message verbatim (see Error Handling).

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

1. Validate `group_by` / `order_by` / `select` against the column + aggregate
   vocabulary (Available Columns) → friendly error on unknown names, on
   `order_by tables` (list), or on a grouped `select` naming a raw field.
2. `store.scan(where, columns=<needed>)` → Arrow (ALL matching rows).
3. If `group_by`: explode `tables` if grouping by it → aggregate (skip-null
   sums, `0.0` for all-null/zero-denominator). Set `total_matching` = group
   count. Else `total_matching` = row count.
4. Apply `order_by` (parse `:asc`/`:desc`, default `desc`; NULLs last).
5. Truncate to `limit`.
6. Project to `select` (or shape default). `latest_ts` serializes to an ISO
   8601 string in the `--json` path.

Returns a `BrowseResult` (rows as `list[dict]` + `total_matching` count +
`grouped: bool`) so the formatter and `--json` path share one structure.
All steps 3–6 are pure pandas/Arrow compute with no I/O → directly unit-tested.
`BrowseService` is constructed per-call from an `IVectorStore` (the MCP handler
passes `_services[engine_name].vector_store`; see below) — no new startup state.

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
  1. Builds `BrowseService(_services[engine_name].vector_store)` — reuses the
     already-initialized store (`_services[engine_name]` is a `SearchService`,
     `state.py:8`; its store is `.vector_store`). Browse needs **no embedder** —
     zero extra model load.
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
  `"duckdb"` → "a local DuckDB slow-query log"; **unknown → "a SQL slow-query
  log"** (generic fallback, symmetric with the model fallback, #10).
- *embeddings phrase* ← `model`: `"granite-r2"` → "Granite embeddings";
  `"gemma-bf16"` → "Gemma embeddings"; **unknown → the model key** (non-breaking).

The composed `search_description` **reproduces-or-improves** the current verbose
semantic prose (#10): the SQL template carries the filter docs (`min_time` /
`min_lock_time` / `table_filter`) and the "Showing N of M" truncation note —
applied uniformly to all four SQL engines (today `sql` / `sql-granite` omit it;
gaining it is an improvement). The **document** template carries the "ranked by
cosine similarity, not recency or size" clause for **both** md engines (today
only the gemma `md` config has it; `md-granite` gains it — an improvement, not a
verbatim copy). So the guarantee is "reproduces or strictly improves," not
"byte-identical," and it holds for every family, not just SQL. `browse_description`
is the analytic template (columns, `--where` grammar incl. `array_has` and
`IS NULL`, group-by aggregates, "ranks by scalar, no query string").

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
- **`--order-by tables`** (a list column) → rejected with "cannot order by a list
  column" (#9).
- **Grouped `--select` naming a raw field** (e.g. `id` under `--group-by user`)
  → rejected with the valid grouped columns (#3).
- **Non-SQL engine** → rejected up front (browse is SQL-family only).
- **Empty result** → "0 rows matched" message, not an error.

## Testing Strategy

- **Unit (`tests/unit/test_browse_service.py`)** — pure aggregation/sort/limit
  logic against an in-memory Arrow table via a fake `IVectorStore.scan`. Covers:
  raw vs grouped, `tables` explode, NULL group collapse → `(none)`, skip-null
  sums (all-null group → `0.0`), `avg_ms_per_call` with `sum(calls)==0` → `0.0`,
  NULLs-last sort in both directions, `total_matching` = groups (grouped) vs
  rows (raw), order-by direction, select projection, and the rejections:
  unknown column, `order_by tables`, grouped `select id`.
- **Unit (MCP)** — `make_browse_handler` formatting, byte-budget truncation,
  malformed-`where` → error-string-not-exception, store sourced from
  `_services[engine].vector_store`.
- **Integration (`tests/integration/`)** — real tmpdir LanceDB seeded with a
  few `SqlChunk`s, exercise `LanceDBStore.scan` with real `where` pushdown
  (incl. `array_has(tables, 'x')` and `IS NULL`), `checkout_latest` visibility,
  point lookup by id, group-by user end-to-end.
- **CLI smoke** — `browse` command wiring (engine resolution, non-SQL
  rejection) with a mocked service.
- **Unit (descriptions)** — `search_description` / `browse_description` compose
  the right source phrase per `chunker_type` and embeddings phrase per `model`
  across all four SQL engines; the "Showing N of M" truncation note is present;
  unknown model falls back to the model key. Guards against LLM-facing
  regression vs the current config prose.

## Defaults Summary

- `--order-by` default: `execution_time_ms:desc`; NULLs always sort last.
- `--limit` default: `10` (analytical top-N; larger than `search`'s `5`).
- `limit` is never pushed to `scan` — applied in-process after order-by.
- Raw default columns: `id, calls, execution_time_ms, lock_time_sec, user, tables`.
- Grouped default columns: group key, `fingerprints`, `calls`,
  `execution_time_ms`, `lock_time_sec`, `rows_examined`, `rows_sent`,
  `latest_ts`, `avg_ms_per_fingerprint`, `avg_ms_per_call`.
