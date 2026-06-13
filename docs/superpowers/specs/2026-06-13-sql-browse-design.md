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
read from LanceDB. No new dependency.

This collapses the would-be `--select/--group-by/--order-by` machinery into "you
write the `SELECT` (CLI), or you send structured params the handler compiles to a
`SELECT` (MCP), and polars does the grouping/sorting/aggregation."

### What polars `SQLContext` can and cannot do (verified)

Two facts, established empirically on `polars==1.40.0`, drive the trust model:

- **It cannot mutate the store.** Frames registered in a `SQLContext` are
  in-memory copies of the Arrow read; there is **no write-back path to LanceDB**.
  `INSERT`/`COPY` are rejected by polars itself; `DROP TABLE t` only drops the
  in-memory frame (discarded when the call returns). So a malformed `browse`
  statement **cannot corrupt or delete data** — no read-only guard is needed to
  prevent that, and the project deliberately omits one.
- **It is NOT a filesystem sandbox.** `SELECT * FROM read_csv('/etc/passwd')`
  *succeeds* — `read_csv` / `read_parquet` are reachable inside polars SQL and can
  read any file the server process can. On the **CLI** this is irrelevant (the
  operator already has full file and data access — `read_csv` is no escalation
  over `cat`). On the **MCP** it is closed structurally: the MCP never accepts raw
  SQL, only structured params the builder compiles, and the builder owns `FROM`
  and emits no function calls — so `read_csv(...)` is *unconstructable* there (see
  Trust & exposure model).

`sqlglot` (used elsewhere in the repo by `services/sql_parser.py`) is **not** used
by `browse`: there is no statement to guard for mutation (impossible) and no
`LIMIT` to inject (handled post-execution, see Architecture).

**Two front-ends, one execution core:**

- **CLI** — raw SQL passthrough. Deliberately minimal (the operator writes SQL).
  Always full power: every column (incl. `raw_query`), `read_csv`, no restrictions.
- **MCP** — **structured params only** (typed filters + `group_by/order_by/select/
  limit`) that the handler **compiles to SQL** and runs through the same core. The
  MCP never accepts raw SQL. The structured, validated, well-described shape is
  what an LLM consumer benefits from; the curated aggregate semantics live in the
  builder. One server flag (`--allow-raw-queries`) controls whether the verbatim
  `raw_query` column may be returned to the model.

## Trust & exposure model

`browse` is a single-operator tool, not a network database server. The CLI runs
under the operator's own privileges — full power, no restrictions. The only place
exposure matters is the **MCP server**, whose client may be a *remote* model.
There are **two independent concerns**, handled by two independent mechanisms:

**1. Arbitrary SQL (file reads, mutation attempts) — closed unconditionally by the
structured-only MCP surface.** The MCP never accepts raw SQL or a raw predicate
string; it accepts typed filters + shape params that the builder compiles. The
builder hard-codes `FROM t` / `FROM t_by_table`, emits no function calls, and
validates every identifier against a fixed column vocabulary. So `read_csv(...)`
and any non-`SELECT` are *unconstructable* on the MCP — not policed, structurally
absent. (This is precisely why the MCP filter surface is **typed params** and not
a raw `where` string: a raw predicate fragment would let `WHERE id IN (SELECT x
FROM read_csv('/secret'))` through, reopening the hole.)

**2. Verbatim query *values* (PII) — gated by `--allow-raw-queries`.** The
`raw_query` column is the captured production SQL *with real literal values*:
emails, credit-card numbers, GDPR personal data, PHI. The normalized `text`
fingerprint has those literals stripped. So the egress boundary is
*normalized-yes, raw-no* by default, and a single server flag flips it:

```
dbs-vector mcp --allow-raw-queries     # default: OFF
```

| Capability | CLI (operator) | MCP, flag **off** (default) | MCP, flag **on** |
|---|---|---|---|
| Raw SQL passthrough | ✅ always | ❌ never (structured params only) | ❌ never (structured params only) |
| `read_csv` / arbitrary file reads | ✅ (no escalation) | ❌ unconstructable (builder owns `FROM`) | ❌ unconstructable |
| `raw_query` column (literal values) | ✅ always | ❌ not in select vocabulary | ✅ selectable |
| normalized `text`, scalar cols, aggregates | ✅ | ✅ | ✅ |

`--allow-raw-queries` is an explicit operator opt-in: "I am driving this MCP with
a trusted (local) model, so it may see verbatim query values." It admits
`raw_query` to the select vocabulary and to the advertised columns; nothing else
about the surface changes. It is **server-level** (one flag on `dbs-vector mcp`),
applies to all SQL engines, and so `EngineConfig` gains **no** new field.

## Goals

- A `browse` operation: read-only analytical SQL over a SQL engine's table, no
  embedder.
- CLI (`dbs-vector browse --sql "…"`) + MCP (`browse_<engine>(structured params)`).
- All SQL engines (`sql`, `sql-granite`, `sql-api`, `sql-api-granite`) via the
  shared `SqlFamily`.
- One execution core (polars); CLI and MCP are thin front-ends over it.
- Zero changes to existing semantic `search` **behavior** — new code only.
- MCP is structured-only (file-read / arbitrary-SQL unconstructable); a single
  `--allow-raw-queries` server flag (default off) gates the verbatim `raw_query`
  PII column.
- **Description ownership refactor** (bundled, see dedicated section): move the
  verbose LLM-facing tool descriptions out of `config.yaml` into the families.

## Non-Goals (YAGNI)

- No `browse` for document/markdown engines (different columns, not the use case).
- No mutation: structurally impossible (in-memory frames), so no guard is written.
- No raw SQL / raw `where` string on the **MCP** at all (any state of the flag).
  Raw SQL is a CLI-only capability. The flag only toggles the `raw_query` column.
- No `sqlglot` in the browse path. No SQL rewriting; `LIMIT` is a display cap.
- No DuckDB dependency. polars is the engine.
- No joins across engine tables; one frame (+ its exploded variant) per call.
- No new ranking inside `search`. `search` stays purely semantic.

## Surface

### CLI — raw SQL (always full power)

```bash
# Heaviest users by total execution time  (note: "user" must be quoted)
dbs-vector browse --type sql-api \
  --sql 'SELECT "user", host, COUNT(*) AS fingerprints, SUM(execution_time_ms) AS total_ms
         FROM t GROUP BY "user", host ORDER BY total_ms DESC' --limit 10

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
| `--sql` | str | required | A SQL `SELECT`, polars SQL dialect. |
| `--limit/-n` | int | 10 | Display cap (post-execution head); see Architecture. |
| `--json` | bool | False | Emit rows as JSON instead of a table. |

`--type/-t` matches `ingest` / `search`. A non-SQL engine is rejected with the
list of available SQL engines. The CLI imposes no read-only guard (there is
nothing to guard — see Trust & exposure model) and no column restrictions. If a
result exceeds `--limit`, only the first `--limit` rows are printed with a
"Showing N of M" note; the operator's own `LIMIT`, if any, is applied first by
polars (see Architecture).

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
  on polars 1.40); the builder emits them directly to polars (no rewrite), so
  ordering semantics are exactly what polars executes.

### MCP — structured params compiled to SQL

One tool per SQL engine (`browse_sql`, `browse_sql_granite`, `browse_sql_api`,
`browse_sql_api_granite`), registered by a sibling `register_browse_tools(mcp,
allow_raw_queries)`. The handler signature IS the schema (FastMCP derives it from
type hints). There is **one** handler shape regardless of the flag — the flag only
changes whether `raw_query` is an accepted `select` value and whether the
description advertises it:

```python
async def handler(
    # --- typed filters (compiled to a safe WHERE; all optional) ---
    id:                    str | None   = None,  # exact match (point lookup)
    content_hash:          str | None   = None,  # exact match (point lookup)
    user:                  str | None   = None,  # exact match (quoted "user")
    host:                  str | None   = None,
    source:                str | None   = None,  # db name
    table:                 str | None   = None,  # uses t_by_table
    min_calls:             int | None   = None,  # calls >=
    min_execution_time_ms: float | None = None,  # execution_time_ms >=
    min_lock_time_sec:     float | None = None,  # lock_time_sec >=
    # --- shape ---
    group_by: str | None = None,   # comma-separated column(s); presence → grouped
    order_by: str        = "execution_time_ms:desc",  # "<col>[:asc|:desc]"
    select:   str | None = None,   # comma-separated output columns
    limit:    int        = 10,      # display cap (post-execution head)
) -> str: ...
```

The handler **builds a SQL string** from these params (quoting identifiers, ANDing
the filters into a `WHERE`, choosing `t` vs `t_by_table`, emitting the curated
aggregate set for grouped mode) and runs it through the shared executor. Because
the builder only ever emits a builder-controlled `SELECT`, neither `read_csv` nor
any non-`SELECT` can appear. When `--allow-raw-queries` is off, `select=raw_query`
is rejected and `raw_query` is omitted from the advertised columns; when on, it is
accepted and advertised. The tool **description** (family template, see
Description Ownership) documents the columns, the group-by-table-via-`t_by_table`
rule, the `"user"` quoting gotcha, and that `browse` ranks by the chosen column,
not similarity — no query string.

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

**Excluded from frames:** the embedding `vector` column is **never** read into the
frame (large; projected out at scan time). `workflow` is also excluded. `text`
(normalized fingerprint, literals stripped) is freely available everywhere.
`raw_query` (verbatim production SQL with **real literal values** — PII) is in the
frame and freely available on the CLI, but on the MCP it is gated by
`--allow-raw-queries` (off → not in the select vocabulary; on → selectable) — see
Trust & exposure model.

**Table-sharing semantics.** `browse` reads an engine's `table_name` directly and
does not filter on `workflow`. `sql` and `sql-api` share the physical table
`query_vault` with the same `workflow` (`config.yaml:32,43`), so `browse_sql` and
`browse_sql_api` see the same rows — identical to existing `search` behavior (it
doesn't filter `workflow` either). `*_granite` / `*_granite_api` have distinct
tables and are isolated.

## MCP grouped-mode output (what the builder emits)

When `group_by` is set, the builder emits this curated aggregate set (raw-SQL
authors on the CLI can of course write any aggregates they like):

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

Two explicitly-named averages: per-fingerprint vs per-execution — the latter
(`avg_ms_per_call`) is the number a DBA usually reads. `NULLIF` guards
**divide-by-zero by producing `NULL`, not `0.0`** — when `SUM(calls)` is 0 the
ratio is genuinely undefined, and `NULL` is the honest answer (verified on polars
1.40.0). Likewise polars `SUM` skips NULLs but an **all-NULL group sums to
`NULL`, not `0.0`** (verified) — e.g. a group whose every `lock_time_sec` is
NULL. The builder does **not** wrap these in `COALESCE`; instead `NULL` is
rendered as `n/a` in the table formatter and `null` in `--json`, matching the
existing `SqlFamily` `_fmt_*` "n/a" convention. When `group_by` is `tables`, the
builder targets `t_by_table`. `select`, if given, restricts the emitted columns to
a validated subset of {group cols} ∪ {aggregate names}; naming a raw
per-fingerprint field (e.g. `id`, or `raw_query`) under grouping is rejected.
`order_by` appends `ORDER BY <col> <dir> NULLS LAST`; `<col>` is validated against
the available columns / aggregate names, and the list column `tables` is not a
valid `order_by` target except via `t_by_table`.

`total_matching` for the "Showing N of M" header: because no `LIMIT` is ever
emitted into the SQL (see Architecture), the executed result already contains
**all** matching rows — raw rows in raw mode, **groups** in grouped mode. So
`total_matching = result.height` directly, and the displayed slice is
`result.head(limit)` ("Showing 10 of 47 users"). One execution, no count query.

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

    def run_sql(self, sql: str, *, display_cap: int) -> BrowseResult: ...
        # raw path — CLI only
    def build_and_run(self, *, filters: dict, group_by, order_by, select, limit,
                      allow_raw_queries: bool = False) -> BrowseResult: ...
        # structured path — MCP
```

`run_sql` (CLI):

1. `arrow = store.scan()` → `pl.from_arrow(arrow)` as `df`.
2. Register frames: `{frame_alias: df, "t": df, "t_by_table": df.explode("tables")}`.
3. `result = pl.SQLContext(frames).execute(sql, eager=True)` — polars raises on
   bad SQL (unknown column, unquoted `user`, etc.); the message is caught upstream
   and surfaced to the caller verbatim. **No read-only guard** — mutation is
   impossible (in-memory frames) and the CLI is the operator's own shell.
4. `total = result.height`; `rows = result.head(display_cap)`;
   `limit_applied = total > display_cap`.
5. Return `BrowseResult(rows, columns, total_matching=total, grouped=False,
   limit_applied)`. The raw path does not parse the SQL, so it reports `grouped=
   False` and the header reads "Showing N of M rows" regardless of any `GROUP BY`
   the author wrote; only `build_and_run` sets `grouped=True`.

**`LIMIT` is never injected into SQL.** The result is materialized in full (cheap
at corpus scale), `total_matching = result.height`, and `display_cap` is applied
as a post-execution `.head()`. This removes all SQL rewriting (and thus any need
for `sqlglot` or dialect round-tripping). A raw query's own `LIMIT` is honored by
polars and simply lowers `height`.

`build_and_run` (MCP structured): validate the typed filters and shape params
against the column / aggregate vocabulary (friendly errors: unknown column,
`order_by tables`, grouped `select` naming a raw field, **`select raw_query` when
`allow_raw_queries` is false**), **build the SQL string** (quoting all
identifiers, ANDing filters into `WHERE`, choosing `t`/`t_by_table`, emitting the
grouped aggregate set, `NULLS LAST`, `NULLIF`), then call the same executor as
`run_sql` with `display_cap=limit`, setting `grouped` from whether `group_by` was
given. The builder emits **no `LIMIT`** and **no raw SQL** — it is the only thing
that ever feeds the MCP path. All of step-1-onward is pure polars/Arrow compute →
directly unit-tested with a fake `scan`.

`BrowseResult`: `rows: list[dict]`, `columns: list[str]`, `total_matching: int`,
`grouped: bool`, `limit_applied: bool`. `latest_ts` serializes to ISO 8601 in the
`--json` path. `None` cell values (NULL aggregates, missing scalars) render as
`n/a` in the table formatter and `null` in `--json` — no `COALESCE` rewriting.

### MCP — naming, registrar, family handler

**Naming (`core/naming.py`).** Generalize the helper with a verb:

```python
def normalize_tool_name(engine_name: str, verb: str = "search") -> str:
    return f"{verb}_{engine_name.replace('-', '_')}"
```

`search` callers unaffected; browse passes `verb="browse"` → `browse_sql_api`.
`browse_*` and `search_*` share the `_dbs_vector_registrations` namespace but
cannot collide.

**Registrar (`mcp/dynamic_tools.py`).** Sibling `register_browse_tools(mcp,
allow_raw_queries)` mirrors `register_search_tools`'s pre-flight (name pattern,
collision, family resolution, idempotency) with two differences: it registers
**only engines whose `resolved_family == "sql"`**, and passes
`family.browse_description(engine_name, allow_raw_queries)`. The
`allow_raw_queries` flag is forwarded into the handler (gating `raw_query`) and
into the description text. Invoked in `start_stdio_server(allow_raw_queries)`
(`mcp/server.py`) between `register_search_tools(mcp)` and
`register_discovery_tool(mcp)`. `register_search_tools` is also touched: its
`description=` switches from `engine.description` to
`family.search_description(engine_name)` (see Description Ownership).

**Flag plumbing.** `dbs-vector mcp` (`cli.py`) gains `--allow-raw-queries`
(`bool`, default `False`), passed to `start_stdio_server(allow_raw_queries=...)`,
which forwards it to `register_browse_tools`. No global state — the flag is
captured in the handler closure and the registrar argument.

**Family handler (`mcp/families/sql.py`).** Add to `SqlFamily`:

- `make_browse_handler(engine_name, allow_raw_queries)` — async handler with the
  structured params. It:
  1. Builds `BrowseService(_services[engine_name].vector_store, frame_alias)` —
     reuses the already-initialized store (`_services[engine_name]` is a
     `SearchService`, `state.py:8`; store is `.vector_store`). **No embedder** —
     zero extra model load.
  2. Runs `BrowseService.build_and_run(..., allow_raw_queries=allow_raw_queries)`
     inside a **single `asyncio.to_thread` closure** (same
     shared-`checkout_latest`-handle discipline as the search handler).
  3. Renders via `format_browse(BrowseResult)` reusing `render_with_budget` for
     the MCP byte budget (compact table).
  4. Catches exceptions (polars error, bad params, disallowed column) and returns
     the message as the tool result string so the LLM self-corrects — never
     raises.
- `browse_description(engine_name, allow_raw_queries)` — family template +
  injected columns; advertises `raw_query` **only when the flag is on**.

### CLI — `cli.py`

New `@app.command() def browse(...)`: resolve engine, reject non-SQL with the
available-SQL-engines list, build `BrowseService` (no embedder), call
`run_sql(sql, display_cap=limit)`, print a table or `--json`. No column
restrictions and no flag — the CLI is always full power; `--limit` (default 10)
controls the display cap only.

## Description Ownership (config cleanup)

**Problem.** The verbose LLM-facing tool prose lives in each engine's
`config.yaml` `description:` — long, and it would double once browse adds a second
description. config is the wrong home for prose tuned for an LLM tool schema.

**Rule.** *Families own the MCP tool descriptions; `config.yaml` owns a short
human summary.*

**Protocol.** `SearchFamily` (`mcp/families/base.py`) gains
`search_description(self, engine_name) -> str`. `SqlFamily` also has
`browse_description(engine_name, allow_raw_queries)`. Both `DocumentFamily` and
`SqlFamily` implement `search_description`.

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
column, no query string"); it lists `raw_query` among the columns **only when
`allow_raw_queries` is on**.

**Registrars source descriptions from the family, not config:**
`register_search_tools` → `family.search_description(engine_name)`;
`register_browse_tools` → `family.browse_description(engine_name, allow_raw_queries)`.

**`config.yaml` change.** `description:` shortens to a one-line summary, stays a
required field, now consumed only by `list_engines` (`discovery.py:47`) + humans:

```yaml
description: "Slow-query fingerprints from a remote slow-log API (Gemma embeddings)."
```

All six engine `description:` fields shorten the same way. `EngineConfig`
(`config.py`) is **unchanged** (`description` stays `description: str`; the
raw-query gate is a server CLI flag, not config).

## Error Handling

- **polars execution error** (bad column, syntax, unquoted `user`, etc.) →
  caught, the polars message returned verbatim so the caller can fix it. MCP:
  returned as tool text; CLI: `typer.echo` + `Exit(1)`. (No read-only guard
  exists — mutation is impossible and the CLI is the operator's own shell.)
- **Bad MCP structured params** (unknown column/filter, `order_by tables`, grouped
  `select` naming a raw field) → validated before building SQL; friendly message.
- **MCP `select raw_query` when `--allow-raw-queries` is off** → the `raw_query`
  column is absent from the vocabulary; rejected with "raw query text is not
  exposed; start the server with --allow-raw-queries". CLI is unaffected.
- **Non-SQL engine** → rejected up front (browse is SQL-family only).
- **Empty result** → "0 rows" message, not an error.

## Testing Strategy

- **Unit (`tests/unit/test_browse_service.py`)** — against a fake `IVectorStore.scan`
  returning an in-memory Arrow table:
  - `run_sql`: a `SELECT` executes; `total_matching = result.height`;
    `result.head(display_cap)` truncates with `limit_applied` set; a query's own
    `LIMIT` lowers `height`; mutation is a non-issue (frames are in-memory —
    assert `DROP TABLE t` does not affect a subsequent `store.scan`).
  - `build_and_run`: typed filters → expected `WHERE`; params → expected SQL; no
    `LIMIT` emitted; grouped aggregate set incl. `avg_ms_per_call` =
    `SUM(exec)/NULLIF(SUM(calls),0)`; **all-NULL group and zero-denominator
    average yield `NULL`, rendered `n/a` in table / `null` in JSON**; NULLs-last
    ordering; `total_matching` = groups (grouped) vs rows (raw);
    `t_by_table` selected for `group_by=tables` / `table=` filter; rejections
    (unknown column, `order_by tables`, grouped `select id`); `"user"` quoted.
  - raw-query gate: with `allow_raw_queries=False`, `select=raw_query` rejected and
    `browse_description` omits `raw_query`; with `True`, `raw_query` selectable and
    advertised; CLI `run_sql("SELECT raw_query …")` works regardless.
- **Unit (MCP)** — `make_browse_handler` formatting, byte-budget truncation,
  polars/param errors → error-string-not-exception, store sourced from
  `_services[engine].vector_store`.
- **Integration (`tests/integration/`)** — real tmpdir LanceDB seeded with a few
  `SqlChunk`s; `LanceDBStore.scan` projects out `vector`, sees `checkout_latest`
  updates; raw `run_sql` point lookup; grouped `build_and_run` by user and by
  table (via `t_by_table`) end-to-end.
- **Unit (descriptions)** — `search_description` / `browse_description` compose
  the right source/embeddings phrases per engine; truncation note present;
  unknown model/chunker fall back; `browse_description` flag-on vs flag-off
  difference. Guards LLM-facing regression.

## Defaults Summary

- CLI: `--sql` required; full power; `--limit` display cap default **10**
  (configurable; head, with a "Showing N of M" note).
- MCP: `order_by` default `execution_time_ms:desc` (NULLS LAST); `limit` default
  `10` (display head).
- MCP is structured-only (no raw SQL / raw `where` in any flag state) → `read_csv`
  and non-`SELECT` unconstructable.
- `--allow-raw-queries` default **off**: gates the verbatim `raw_query` PII column
  only. Off → `raw_query` not in vocabulary / not advertised. On → selectable and
  advertised, for a trusted local model. Server-level; no per-engine config field.
- Frames: `t`, `t_by_table`, and the engine-name alias for `t`.
- `vector` (and `workflow`) never read into the frame; `latest_ts` → ISO 8601 in
  `--json`.
- No `LIMIT` ever injected into SQL; `total_matching = result.height`, display via
  `.head()`. No `sqlglot` in the browse path.
- MCP grouped default columns: group cols, `fingerprints`, `calls`,
  `execution_time_ms`, `lock_time_sec`, `rows_examined`, `rows_sent`,
  `latest_ts`, `avg_ms_per_fingerprint`, `avg_ms_per_call`.
