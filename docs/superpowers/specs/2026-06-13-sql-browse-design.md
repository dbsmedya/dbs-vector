# `browse` — Analytical SQL Access for SQL Engines

**Date:** 2026-06-13
**Status:** Approved (design)
**Branch:** `feat/sql-browse`

## Problem

The public search surface (`dbs-vector search` + `search_<engine>` MCP tools)
only does **hybrid semantic retrieval**: it embeds a query string and ranks by
cosine similarity, exposing three SQL prefilters (`min_time`, `min_lock_time`,
`table_filter`). There is no way through the public surface to point-look-up a
fingerprint by `id`, rank by a scalar column (`calls`, `execution_time_ms`)
without a query string, aggregate ("which user / table burns the most DB time"),
or project a column subset. Today that means dropping into a Python REPL and
reading the LanceDB table as Arrow by hand.

This work exposes scalar/analytical access as a first-class `browse` verb on the
CLI and MCP, **without touching the semantic path**.

## Core idea: SQL over the table, executed by polars

`browse` runs a read-only **SQL `SELECT`** over an engine's table. Execution uses
**`polars.SQLContext`** (already a core dep, `polars==1.40.0`) over the Arrow data
read from LanceDB, with **`sqlglot`** (core dep, `sqlglot>=27.0.0`) guarding that
the statement is a single read-only `SELECT` (so a typo cannot mutate the table).
No new dependency.

This collapses the would-be `--select/--group-by/--order-by` machinery into "you
write the `SELECT`, polars does the grouping/sorting/aggregation."

The one genuine exposure concern is **data egress, not query injection**: browse
output can contain `raw_query` (verbatim production SQL with real literal values),
and an MCP client may be a remote model — so raw SQL would leave the operator's
network. That boundary is handled by an explicit, default-off config gate (see
**Raw query exposure**). On the CLI there is nothing to gate: the operator runs
SQL in their own terminal under their own privileges, exactly as they would
`cat` a file.

**Two front-ends, one execution core:**

- **CLI** — raw SQL passthrough. Deliberately minimal (the user writes SQL).
- **MCP** — structured params (`where/group_by/order_by/select/limit`) that the
  handler **compiles to SQL** and runs through the same core. The structured,
  validated, well-described shape is what an LLM consumer benefits from; the
  curated aggregate semantics live in the builder.

## Goals

- A `browse` operation: read-only analytical SQL over a SQL engine's table, no
  embedder.
- CLI (`dbs-vector browse --sql "…"`) + MCP (`browse_<engine>(structured params)`).
- All SQL engines (`sql`, `sql-granite`, `sql-api`, `sql-api-granite`) via the
  shared `SqlFamily`.
- One execution core (polars + sqlglot); CLI and MCP are thin front-ends over it.
- Zero changes to existing semantic `search` **behavior** — new code only.
- **Description ownership refactor** (bundled, see dedicated section): move the
  verbose LLM-facing tool descriptions out of `config.yaml` into the families.

## Non-Goals (YAGNI)

- No `browse` for document/markdown engines (different columns, not the use case).
- No write/mutation — `browse` rejects anything but a single `SELECT`.
- No structured filter flags on the **CLI** (`--group-by`, `--order-by`, …). The
  CLI is raw SQL; only the MCP tool has structured params.
- No DuckDB dependency. polars is the engine.
- No joins across engine tables; one frame (+ its exploded variant) per call.
- No new ranking inside `search`. `search` stays purely semantic.

## Surface

### CLI — raw SQL

```bash
# Heaviest users by total execution time  (note: "user" must be quoted)
dbs-vector browse --type sql-api \
  --sql 'SELECT "user", host, COUNT(*) AS fingerprints, SUM(execution_time_ms) AS total_ms
         FROM t GROUP BY "user", host ORDER BY total_ms DESC LIMIT 10'

# Point lookup
dbs-vector browse --type sql-api --sql "SELECT * FROM t WHERE id = '93FEDEB240C723E3'"

# Everything touching a table — use the exploded frame
dbs-vector browse --type sql-api \
  --sql "SELECT id, calls FROM t_by_table WHERE tables = 'orders' ORDER BY calls DESC"
```

Options (intentionally small):

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--type/-t` | str | required (must be a SQL engine) | Engine; rejected if not SQL-family. |
| `--sql` | str | required | A single read-only `SELECT`, polars SQL dialect. |
| `--json` | bool | False | Emit rows as JSON instead of a table. |

`--type/-t` matches `ingest` / `search`. A non-SQL engine is rejected with the
list of available SQL engines. If the SQL omits a `LIMIT`, a safety
`LIMIT` (default 1000) is appended (see Architecture) and noted in the output.

**Frames available in `FROM`:**

- `t` — the engine's table, one row per fingerprint.
- `t_by_table` — `t` exploded on the `tables` list (one row per (fingerprint,
  table)); use it to filter or group by table.
- The engine name with dashes→underscores (e.g. `sql_api`) is also registered as
  an alias for `t`, so `FROM sql_api` works too.

**polars SQL dialect — key gotchas (documented for CLI authors and in the MCP
description):**

- **Quote `"user"`** — it collides with the SQL `USER` keyword. Other
  identifiers are safe unquoted; the MCP builder quotes everything defensively.
- Filter/group by a **table** via `t_by_table` (the `tables` list is exploded to
  a scalar there), e.g. `WHERE tables = 'orders'`.
- `NULLS LAST`, `NULLIF(...)`, `IS NULL` / `IS NOT NULL` are supported (verified
  on polars 1.40).

### MCP — structured params compiled to SQL

One tool per SQL engine (`browse_sql`, `browse_sql_granite`, `browse_sql_api`,
`browse_sql_api_granite`), registered by a sibling `register_browse_tools(mcp)`.
The handler signature IS the schema (FastMCP derives it from type hints):

```python
async def handler(
    where:    str | None = None,   # raw polars-SQL predicate fragment, optional
    group_by: str | None = None,   # comma-separated column(s); presence → grouped
    order_by: str        = "execution_time_ms:desc",  # "<col>[:asc|:desc]"
    select:   str | None = None,   # comma-separated output columns
    limit:    int        = 10,
) -> str: ...
```

The handler **builds a SQL string** from these params (quoting identifiers,
choosing `t` vs `t_by_table`, emitting the curated aggregate set for grouped
mode) and runs it through the shared executor. Because the builder only ever
emits a `SELECT`, the read-only guard always passes. The tool **description**
(family template, see Description Ownership) documents the columns, the
group-by-table-via-`t_by_table` rule, the `"user"` quoting gotcha, and that
`browse` ranks by the chosen column, not similarity — no query string.

## Available Columns

From `SqlChunk` (`core/models.py`). These are the columns in frame `t`:

| Column | Type | Notes |
|--------|------|-------|
| `id` | str | point-lookup key |
| `content_hash` | str | point-lookup key |
| `user` | str \| None | **quote as `"user"`** in SQL |
| `host` | str \| None | |
| `source` | str (db name) | |
| `tables` | list[str] | list; filter/group via `t_by_table` |
| `calls` | int | |
| `execution_time_ms` | float | cumulative across all calls of a fingerprint |
| `lock_time_sec` | float \| None | |
| `rows_examined` | int \| None | |
| `rows_sent` | int \| None | |
| `latest_ts` | datetime | ISO 8601 string in `--json` output |

**Excluded from frames (#5):** the embedding `vector` column is **never** read
into the frame (large; projected out at scan time). `workflow` is also excluded
from the frame. `text` (normalized fingerprint) is freely available for display.
`raw_query` (verbatim production SQL, real literals) is in the frame but
**gated on the MCP path** — see **Raw query exposure**.

**Table-sharing semantics (#5).** `browse` reads an engine's `table_name`
directly and does not filter on `workflow`. `sql` and `sql-api` share the physical
table `query_vault` with the same `workflow` (`config.yaml:32,43`), so
`browse_sql` and `browse_sql_api` see the same rows — identical to existing
`search` behavior (it doesn't filter `workflow` either). `*_granite` /
`*_granite_api` have distinct tables and are isolated.

## Raw query exposure (data egress, #1)

`raw_query` is the **verbatim production SQL**, literal values included (emails,
ids, tokens that appeared in the captured statement). `text` is the **normalized
fingerprint** — literals stripped — which the `search` path already surfaces to
the calling LLM today. The egress boundary is therefore *normalized-yes, raw-no*,
identical to `search`'s existing `include_raw=False` default
(`SqlFamily.make_handler`). browse must not silently widen it.

The MCP server cannot know whether its client is a remote model (raw SQL leaves
the network) or a local one, so exposure is an **explicit operator choice, not
auto-detected**:

- **New per-engine config field `expose_raw_query: bool`, default `false`**,
  added to `EngineConfig` (`config.py`).
- **MCP, gate OFF (default):** `raw_query` is **not** in the selectable column
  vocabulary. A `select` that names it is rejected before any scan with a
  friendly message ("raw query text is not exposed on this engine; set
  `expose_raw_query: true` to enable"), and `browse_description` omits
  `raw_query` from the advertised columns. `text` (normalized) stays freely
  selectable — consistent with `search`.
- **MCP, gate ON:** `raw_query` joins the selectable vocabulary and the
  description advertises it. Operators set this only when driving the engine with
  a local model.
- **CLI, always unrestricted:** `SELECT raw_query FROM t` works regardless of the
  flag. The CLI prints to the operator's own terminal — no network egress — so
  the gate lives only in the MCP builder's column-vocabulary validation
  (`build_and_run`), which the CLI's `run_sql` path never invokes.

This reuses the established `normalized-yes / raw-no` contract rather than
inventing a new mechanism, and keeps raw production literals off a remote model
by default.

## MCP grouped-mode output (what the builder emits)

When `group_by` is set, the builder emits this curated aggregate set (raw CLI
authors can of course write any aggregates they like):

| Output column | SQL the builder emits |
|---------------|------------------------|
| `<group cols>` | the `group_by` columns (quoted) |
| `fingerprints` | `COUNT(*)` |
| `calls` | `SUM(calls)` |
| `execution_time_ms` | `SUM(execution_time_ms)` |
| `lock_time_sec` | `SUM(lock_time_sec)` |
| `rows_examined` | `SUM(rows_examined)` |
| `rows_sent` | `SUM(rows_sent)` |
| `latest_ts` | `MAX(latest_ts)` |
| `avg_ms_per_fingerprint` | `SUM(execution_time_ms)/NULLIF(COUNT(*),0)` |
| `avg_ms_per_call` | `SUM(execution_time_ms)/NULLIF(SUM(calls),0)` |

Two explicitly-named averages (#10): per-fingerprint vs per-execution — the
latter (`avg_ms_per_call`) is the number a DBA usually reads. `NULLIF` guards
**divide-by-zero by producing `NULL`, not `0.0`** — when `SUM(calls)` is 0 the
ratio is genuinely undefined, and `NULL` is the honest answer (verified on polars
1.40.0). Likewise polars `SUM` skips NULLs but an **all-NULL group sums to
`NULL`, not `0.0`** (verified, #7) — e.g. a group whose every `lock_time_sec` is
NULL. The builder does **not** wrap these in `COALESCE`; instead `NULL` is
rendered as `n/a` in the table formatter and `null` in `--json` (#2, #7),
matching the existing `SqlFamily` `_fmt_*` "n/a" convention. When `group_by` is
`tables`, the builder targets `t_by_table`. `select`, if given, restricts the
emitted columns to a validated subset of {group cols} ∪ {aggregate names};
naming a raw per-fingerprint field (e.g. `id`) under grouping is rejected (#3).
`order_by` appends `ORDER BY <col> <dir> NULLS LAST` (#7); `<col>` is validated
against the available columns / aggregate names, and the list column `tables` is
not a valid `order_by` target except via `t_by_table` (#9).

`total_matching` for the "Showing N of M" header (#8): the builder runs its query
once **without** the `LIMIT` and counts — raw rows in raw mode, **groups** in
grouped mode ("Showing 10 of 47 users"). Cheap on a ~6,200-row corpus.

## Architecture

Flow: **CLI/MCP → BrowseService (core) → Port → Infrastructure**.

### Port — `core/ports.py`

```python
def scan(self, columns: list[str] | None = None) -> Any:  # pyarrow.RecordBatch
    """Read ALL rows of the table as Arrow for in-process SQL. `columns=None`
    means every column EXCEPT the embedding `vector` (and `workflow`).
    Implementations MUST call checkout_latest() first, like search()/
    count_matching(), so the read sees the current table version."""
    ...
```

No `where`/`limit`/ordering on the port — all of that is expressed in the SQL and
applied by polars. The corpus is ~6,200 rows, so a full projected read is cheap.

### Infrastructure — `LanceDBStore.scan()`

`checkout_latest()` (as the search path does, `lancedb_engine.py:118`), then read
all rows projecting out `vector` (and `workflow`), return Arrow. No vector query.

### Service — `services/browse.py` `BrowseService`

The single execution core. Constructed per-call from an `IVectorStore`:

```python
class BrowseService:
    def __init__(self, store: IVectorStore, frame_alias: str) -> None: ...
        # frame_alias = engine name, dashes→underscores (e.g. "sql_api")

    def run_sql(self, sql: str) -> BrowseResult: ...        # raw path (CLI)
    def build_and_run(self, *, where, group_by, order_by,   # structured path (MCP)
                      select, limit,
                      expose_raw_query: bool = False) -> BrowseResult: ...
```

`run_sql`:

1. **Guard** with sqlglot: parse with **`dialect="postgres"`** (the dialect that
   round-trips polars' `NULLS LAST` and double-quoted identifiers — see step 2);
   require exactly one statement and that it is a `SELECT` (reject any
   DDL/DML/`PRAGMA`/`COPY`/`ATTACH`/multiple statements) → `BrowseError` with a
   friendly message on failure.
2. If the parsed statement has no `LIMIT` on the outer query, inject the safety
   `LIMIT` (default 1000) on the AST and regenerate with **`.sql(dialect=
   "postgres")`**, then flag `limit_injected`. **The dialect pin is load-bearing
   (#3):** sqlglot's *default* dialect drops the `NULLS LAST` clause on
   regeneration, and polars sorts NULLs *first* under a bare `DESC` — so a
   default-dialect round-trip would silently reorder exactly the ranked queries
   browse exists to run. Postgres preserves `NULLS LAST` and the `"user"`
   quoting (verified on sqlglot 30.7 + polars 1.40.0). Parsing and regenerating
   under the **same** dialect avoids any translation artifact.
3. `arrow = store.scan()` → `pl.from_arrow(arrow)` as `df`.
4. Register frames: `{frame_alias: df, "t": df, "t_by_table": df.explode("tables")}`.
5. `pl.SQLContext(frames).execute(sql, eager=True)` → rows.
6. Return `BrowseResult(rows, columns, total_matching=len(rows), grouped=False,
   limit_injected=…)`.

`build_and_run` (MCP): validate the structured params against the column /
aggregate vocabulary (friendly errors: unknown column, `order_by tables`,
grouped `select` naming a raw field, **`select raw_query` when
`expose_raw_query` is false** — see Raw query exposure), **build the SQL string**
(quoting all
identifiers, choosing `t`/`t_by_table`, emitting the grouped aggregate set,
`NULLS LAST`, `NULLIF`), compute `total_matching` via the same query minus
`LIMIT`, then reuse `run_sql`'s executor. All of step-3-onward is pure
polars/Arrow compute → directly unit-tested with a fake `scan`.

`BrowseResult`: `rows: list[dict]`, `columns: list[str]`, `total_matching: int`,
`grouped: bool`, `limit_injected: bool`. `latest_ts` serializes to ISO 8601 in
the `--json` path. `None` cell values (NULL aggregates, missing scalars) render
as `n/a` in the table formatter and `null` in `--json` — no `COALESCE` rewriting
(#2, #7).

### MCP — naming, registrar, family handler

**Naming (`core/naming.py`).** Generalize the helper with a verb:

```python
def normalize_tool_name(engine_name: str, verb: str = "search") -> str:
    return f"{verb}_{engine_name.replace('-', '_')}"
```

`search` callers unaffected; browse passes `verb="browse"` → `browse_sql_api`.
`browse_*` and `search_*` share the `_dbs_vector_registrations` namespace but
cannot collide.

**Registrar (`mcp/dynamic_tools.py`).** Sibling `register_browse_tools(mcp)`
mirrors `register_search_tools`'s pre-flight (name pattern, collision, family
resolution, idempotency) with two differences: it registers **only engines whose
`resolved_family == "sql"`**, and passes `family.browse_description(engine_name)`.
Invoked in `start_stdio_server()` (`mcp/server.py`) between
`register_search_tools(mcp)` and `register_discovery_tool(mcp)`.
`register_search_tools` is also touched: its `description=` switches from
`engine.description` to `family.search_description(engine_name)` (see Description
Ownership).

**Family handler (`mcp/families/sql.py`).** Add to `SqlFamily`:

- `make_browse_handler(engine_name)` — async handler with the five structured
  params. It:
  1. Builds `BrowseService(_services[engine_name].vector_store, frame_alias)` —
     reuses the already-initialized store (`_services[engine_name]` is a
     `SearchService`, `state.py:8`; store is `.vector_store`). **No embedder** —
     zero extra model load.
  2. Runs `BrowseService.build_and_run(..., expose_raw_query=engine.expose_raw_query)`
     (the per-engine gate, default false) inside a **single
     `asyncio.to_thread` closure** — same shared-`checkout_latest`-handle
     discipline as the search handler.
  3. Renders via `format_browse(BrowseResult)` reusing `render_with_budget` for
     the MCP byte budget (compact table).
  4. Catches exceptions (sqlglot guard failure, polars error, bad params) and
     returns the message as the tool result string so the LLM self-corrects —
     never raises.
- `browse_description(engine_name)` — family template + injected columns.

### CLI — `cli.py`

New `@app.command() def browse(...)`: resolve engine, reject non-SQL with the
available-SQL-engines list, build `BrowseService` (no embedder), call
`run_sql(sql)`, print a table or `--json`.

## Description Ownership (config cleanup)

**Problem.** The verbose LLM-facing tool prose lives in each engine's
`config.yaml` `description:` — long, and it would double once browse adds a second
description. config is the wrong home for prose tuned for an LLM tool schema.

**Rule.** *Families own the MCP tool descriptions; `config.yaml` owns a short
human summary.*

**Protocol.** `SearchFamily` (`mcp/families/base.py`) gains
`search_description(self, engine_name) -> str`. `SqlFamily` also has
`browse_description(engine_name)`. Both `DocumentFamily` and `SqlFamily`
implement `search_description`.

**Composition — template from inert engine facts (no per-engine code map).** One
hardcoded template per verb, variable bits filled from existing config fields, so
a new A/B variant stays **config-only** (preserving the CLAUDE.md contract):

- *source phrase* ← `chunker_type`: `"api"` → "a remote slow-log API"; `"duckdb"`
  → "a local DuckDB slow-query log"; unknown → "a SQL slow-query log".
- *embeddings phrase* ← `model`: `"granite-r2"` → "Granite embeddings";
  `"gemma-bf16"` → "Gemma embeddings"; unknown → the model key.

The composed `search_description` **reproduces-or-improves** today's prose: the
SQL template keeps the `min_time`/`min_lock_time`/`table_filter` filter docs and
the "Showing N of M" note (now uniform across all four SQL engines — `sql` /
`sql-granite` gain it). The document template carries the "ranked by similarity,
not recency or size" clause for both md engines (`md-granite` gains it). So the
guarantee is "reproduces or strictly improves," for every family.
`browse_description` is the analytic template (frames `t` / `t_by_table`,
columns, the `"user"` quoting note, grouped aggregate names, "ranks by the chosen
column, no query string").

**Registrars source descriptions from the family, not config:**
`register_search_tools` → `family.search_description(engine_name)`;
`register_browse_tools` → `family.browse_description(engine_name)`.

**`config.yaml` change.** `description:` shortens to a one-line summary, stays a
required field, now consumed only by `list_engines` (`discovery.py:47`) + humans:

```yaml
description: "Slow-query fingerprints from a remote slow-log API (Gemma embeddings)."
```

All six engine `description:` fields shorten the same way. The only
`EngineConfig` (`config.py`) change is the new `expose_raw_query: bool = False`
field (see **Raw query exposure**); `description` stays `description: str`.

## Error Handling

- **Non-SELECT / multi-statement / DDL/DML** → rejected by the sqlglot guard with
  a friendly "browse is read-only; use a single SELECT" message. MCP: returned as
  tool text; CLI: `typer.echo` + `Exit(1)`.
- **polars execution error** (bad column, syntax, unquoted `user`) → caught, the
  polars message returned verbatim so the caller can fix it.
- **Bad MCP structured params** (unknown column, `order_by tables`, grouped
  `select` naming a raw field) → validated before building SQL; friendly message.
- **MCP `select raw_query` with `expose_raw_query` false** → rejected before any
  scan with the "raw query text is not exposed on this engine" message (see Raw
  query exposure). CLI is unaffected.
- **Non-SQL engine** → rejected up front (browse is SQL-family only).
- **Empty result** → "0 rows" message, not an error.

## Testing Strategy

- **Unit (`tests/unit/test_browse_service.py`)** — against a fake `IVectorStore.scan`
  returning an in-memory Arrow table:
  - `run_sql` guard: rejects `INSERT`/`UPDATE`/`DROP`/`PRAGMA`/`COPY`/`ATTACH`,
    multi-statement; accepts a `SELECT`; injects safety `LIMIT` when absent **and
    the injection preserves a `NULLS LAST` clause** (regression guard for #3 — a
    default-dialect round-trip would drop it).
  - `build_and_run`: params → expected SQL; grouped aggregate set incl.
    `avg_ms_per_call` = `SUM(exec)/NULLIF(SUM(calls),0)`; **all-NULL group and
    zero-denominator average yield `NULL`, rendered `n/a` in table / `null` in
    JSON** (#2, #7); NULLs-last ordering, `total_matching` = groups (grouped) vs
    rows (raw),
    `t_by_table` selected for `group_by=tables`, rejections (unknown column,
    `order_by tables`, grouped `select id`), `"user"` quoted.
  - raw-query gate: `select=raw_query` rejected when `expose_raw_query=False`,
    accepted when `True`; `browse_description` omits `raw_query` from advertised
    columns when off and includes it when on; CLI `run_sql("SELECT raw_query …")`
    is unaffected by the flag.
- **Unit (MCP)** — `make_browse_handler` formatting, byte-budget truncation,
  guard/polars errors → error-string-not-exception, store sourced from
  `_services[engine].vector_store`.
- **Integration (`tests/integration/`)** — real tmpdir LanceDB seeded with a few
  `SqlChunk`s; `LanceDBStore.scan` projects out `vector`, sees `checkout_latest`
  updates; raw `run_sql` point lookup; grouped `build_and_run` by user and by
  table (via `t_by_table`) end-to-end.
- **Unit (descriptions)** — `search_description` / `browse_description` compose
  the right source/embeddings phrases per engine; truncation note present;
  unknown model/chunker fall back. Guards LLM-facing regression.

## Defaults Summary

- CLI: `--sql` required; safety `LIMIT 1000` appended if the SQL has none.
- MCP: `order_by` default `execution_time_ms:desc` (NULLS LAST); `limit` default
  `10`.
- Frames: `t`, `t_by_table`, and the engine-name alias for `t`.
- `vector` (and `workflow`) never read into the frame; `latest_ts` → ISO 8601 in
  `--json`.
- `raw_query` exposure on MCP is gated by per-engine `expose_raw_query` (default
  `false`); CLI is always unrestricted.
- MCP grouped default columns: group cols, `fingerprints`, `calls`,
  `execution_time_ms`, `lock_time_sec`, `rows_examined`, `rows_sent`,
  `latest_ts`, `avg_ms_per_fingerprint`, `avg_ms_per_call`.
