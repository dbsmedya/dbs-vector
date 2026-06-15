# SQL Vector Engine

The SQL engine in `dbs-vector` is specifically designed for analyzing and clustering database queries. It enables finding "semantically similar" slow queries, allowing developers to identify patterns in database performance bottlenecks.

## Overview
Unlike standard prose search, the SQL engine uses specialized code-native models (like `bigcode/starencoder`) to understand SQL structure. It relies on a separate schema in LanceDB that includes execution metrics like duration and call counts.

---

## Ingestion Format (JSON)
The SQL engine expects a JSON file containing an array of query records. This is typically exported from `pg_stat_statements` or slow query logs.

### Required JSON Fields:
| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The original, raw SQL query. |
| `normalized_query` | `string` | The structural version of the query (literals stripped). Used for embedding. |
| `database` | `string` | The name of the source database. |
| `duration` | `float` | Execution time in milliseconds. |
| `calls` | `integer` | Number of times the query was executed. |

### Example Input File (`queries.json`):
```json
[
  {
    "query": "SELECT * FROM users WHERE id = 42",
    "normalized_query": "SELECT * FROM users WHERE id = ?",
    "query_hash": "abc123hash",
    "database": "production_db",
    "duration": 1250.5,
    "calls": 500
  },
  {
    "query": "SELECT * FROM users WHERE id = 101",
    "normalized_query": "SELECT * FROM users WHERE id = ?",
    "query_hash": "abc123hash",
    "database": "production_db",
    "duration": 1100.2,
    "calls": 30
  }
]
```

---

## Command Line Usage

### 1. Ingesting SQL Logs
To index your JSON log file into the `query_vault` table:
```bash
uv run dbs-vector ingest "path/to/slow_queries.json" --type sql
```

### 2. Searching for Similar Slow Queries
You can search using a raw SQL string to find clusters of similar slow queries:
```bash
uv run dbs-vector search "SELECT * FROM users" --type sql --min-time 1000
```
*   `--min-time`: Filter results to only include queries that took longer than the specified milliseconds.

---

## Analytical access with `browse` *(shipped in v1.0.0)*

> **Status:** shipped. Designed in
> [`docs/superpowers/specs/2026-06-13-sql-browse-design.md`](superpowers/specs/2026-06-13-sql-browse-design.md).

`search` is purely semantic — it embeds a query string and ranks by cosine
similarity. It deliberately cannot point-look-up a fingerprint by `id`, rank by
a scalar column (`calls`, `execution_time_ms`) without a query string, or
aggregate ("which user / table burns the most DB time"). Those need scalar,
analytical access — which is what `browse` adds, **without touching the semantic
path**.

`browse` runs a **read-only SQL `SELECT`** over a SQL engine's table (no
embedder). Two front-ends share one execution core:

- **CLI — raw SQL.** You write the `SELECT`; polars executes it. A `sqlglot`
  guard rejects anything but a single read-only `SELECT` (so a typo can't mutate
  the table), and a safety `LIMIT 1000` is appended if you omit one.
- **MCP — structured params** (`where` / `group_by` / `order_by` / `select` /
  `limit`) that an LLM fills in; the handler compiles them to the same SQL.

### CLI examples

```bash
# Heaviest users by total execution time  (note: "user" must be quoted)
dbs-vector browse --type sql-api \
  --sql 'SELECT "user", COUNT(*) AS fingerprints, SUM(execution_time_ms) AS total_ms
         FROM t GROUP BY "user" ORDER BY total_ms DESC LIMIT 10'

# Point lookup by fingerprint id
dbs-vector browse --type sql-api --sql "SELECT * FROM t WHERE id = '93FEDEB240C723E3'"

# Everything touching a table — use the exploded frame t_by_table
dbs-vector browse --type sql-api \
  --sql "SELECT id, calls FROM t_by_table WHERE tables = 'rental' ORDER BY calls DESC"
```

**Frames available in `FROM`:** `t` (one row per fingerprint), `t_by_table`
(`t` exploded on the `tables` list — one row per table, for filtering/grouping
by table), and the engine name with dashes→underscores (e.g. `sql_api`) as an
alias for `t`.

**Engine selection** uses `--type/-t`, consistent with `ingest` and `search`.
Only SQL engines (`sql`, `sql-granite`, `sql-api`, `sql-api-granite`) are
browsable; a non-SQL engine is rejected with the list of valid ones.

### Privacy: `raw_query` and the `--allow-raw-queries` flag

`raw_query` is the **verbatim production SQL with real literal values** (PII).
Embeddings are computed only on the normalized fingerprint (`text`), never on
`raw_query`, so exposing it is a pure egress decision — controlled by a single
**server-level** flag on the MCP server:

```bash
uv run dbs-vector mcp                       # default: raw_query is NEVER exposed
uv run dbs-vector mcp --allow-raw-queries   # opt-in: raw_query exposable to the model
```

- **Initial state: OFF (fail-closed).** Without the flag, no MCP tool emits
  `raw_query`. The normalized `text` is always available regardless.
- **`browse_<engine>` (MCP):** `select=raw_query` is rejected with a validation
  error unless the server was started with `--allow-raw-queries`.
- **`search_<engine>` (MCP):** the `include_raw=true` argument adds a `Raw SQL:`
  block only under `--allow-raw-queries`; otherwise it is silently downgraded
  (the block is omitted and the call still succeeds). Both surfaces share one
  lock — verbatim `raw_query` leaves the process only under the flag.
- **CLI `browse --sql` is unrestricted** — it runs on your own terminal, so you
  can `SELECT raw_query` freely; the flag applies only to the MCP server.

---

## Why use Vector Search for SQL?
Traditional structural analysis often relies on exact hash matching of normalized queries. While useful, it misses queries that are **logically identical but structurally different** (e.g., a `JOIN` written in a different order, or different aliasing). 

Vector similarity captures the **semantic intent** of the SQL, grouping together queries that touch the same tables and indices even if the syntax varies slightly.

---

## Granite alternative: `sql-granite`

A second SQL engine, `sql-granite`, uses IBM Granite 311m R2 instead of embeddinggemma. It writes to a separate table (`query_vault_granite`) so you can A/B compare against the existing `sql` engine on the same DuckDB slow-query fixture without index mixing. The `--min-time` filter applies identically to both engines.

```bash
uv run dbs-vector ingest "slow_log.duckdb" --type sql-granite
uv run dbs-vector search "find users by email" --type sql-granite --min-time 100
```

See [README_EMBEDDINGS.md](README_EMBEDDINGS.md) for model details.

---

## Investigation skills — deferred to a later phase

Earlier versions of this README documented a bundled Claude Skill
(`slow-query-investigator`, later split into `slow-query-triage` and
`locking-query-triage`) that answered questions like "lock contention on
`rental`?", "slowest queries on `payment`?", and "what indexes should I add to
`rental`?".

**Those skills have been removed.** They were built on semantic-search
*workarounds* — e.g. calling `search_sql_api(query="lock contention row write",
min_lock_time=…)` purely to coax a ranking out of a verb that only ranks by
cosine similarity. That is exactly the gap `browse` (above) closes: ranking by a
scalar column, aggregation, and point-lookup, done directly and correctly.

New triage and index-recommendation skills will be **rebuilt on top of `browse`**
in a later phase. Until then, run the analytical queries directly through
`browse` (CLI), which shipped in v1.0.0.

> The companion Phase 2 workflow (live-schema index recommendation via
> [`askdba/mysql-mcp-server`](https://github.com/askdba/mysql-mcp-server)) is
> also deferred and will return with the rebuilt skills.

---

### Further reading

- [`docs/superpowers/specs/2026-06-13-sql-browse-design.md`](superpowers/specs/2026-06-13-sql-browse-design.md) — `browse` design spec
- [`README_MCP.md`](README_MCP.md) — MCP installation and tool naming
- [`README_REMOTE_SQL_API.md`](README_REMOTE_SQL_API.md) — ingestion via remote slow-log API
- [`README_EMBEDDINGS.md`](README_EMBEDDINGS.md) — embedding model details
