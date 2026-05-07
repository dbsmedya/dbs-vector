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

## Investigating Slow Queries with the Bundled Claude Skill

`dbs-vector` ships a Claude Skill at
[`skills/slow-query-investigator/SKILL.md`](../skills/slow-query-investigator/SKILL.md)
that turns the ingested slow-query corpus into an interactive
investigation tool. Once installed, you ask Claude natural-language
questions about lock contention, slow queries on a particular table,
or "what indexes should I add to TABLE X" — and the skill picks the
right MCP tool calls automatically.

The examples below use the
[Sakila sample database](https://dev.mysql.com/doc/sakila/en/) so they
read clearly without exposing real production schemas. Substitute your
own table names directly.

### Two phases

| Phase | What you can ask | Required MCP servers |
|---|---|---|
| **1 — Investigation** | "Lock contention on `rental`?" / "Slowest queries on `payment`?" / "What's hammering `inventory`?" | `dbs-vector` only |
| **2 — Index recommendation** | "What indexes should I add to `rental`?" / "Missing indexes on `payment`?" | `dbs-vector` + `mysql-mcp-server` |

### Setup

#### 1. Install `dbs-vector` MCP server (Phase 1 + 2)

The CLI already includes the MCP server entry point — you just need to
register it with your Claude client. See
[README_MCP.md](README_MCP.md#integrating-with-claude-desktop) for full
instructions; the short version:

```jsonc
// Claude Desktop:  ~/Library/Application Support/Claude/claude_desktop_config.json
// Claude Code:     ~/.claude.json  (or use `claude mcp add` CLI)
{
  "mcpServers": {
    "dbs-vector-stdio": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/dbs-vector", "dbs-vector", "mcp"]
    }
  }
}
```

You'll also need to ingest at least one slow-query log first
(`uv run dbs-vector ingest <source> --type sql` etc.) — see the
[Command Line Usage](#command-line-usage) section above.

#### 2. Install `askdba/mysql-mcp-server` (Phase 2 only)

For index recommendation, the skill needs schema introspection (live
indexes, table size, EXPLAIN, foreign keys) — that comes from
[`askdba/mysql-mcp-server`](https://github.com/askdba/mysql-mcp-server):

```bash
brew install askdba/tap/mysql-mcp-server
```

Configure connection details and **enable the extended toolset**
(it's gated behind an env flag because it adds 13 schema-introspection
tools beyond the basic listing/query set):

```bash
export MYSQL_DSN="user:pass@tcp(localhost:3306)/sakila?parseTime=true"
export MYSQL_MCP_EXTENDED=1
```

Register it with Claude alongside dbs-vector:

```jsonc
{
  "mcpServers": {
    "dbs-vector-stdio": { /* as above */ },
    "mysql": {
      "command": "mysql-mcp-server",
      "env": {
        "MYSQL_DSN": "user:pass@tcp(localhost:3306)/sakila?parseTime=true",
        "MYSQL_MCP_EXTENDED": "1"
      }
    }
  }
}
```

The skill is RDBMS-agnostic at the architecture level (its abstract
operation contract maps to MySQL today; PostgreSQL adapter planned).

#### 3. Activate the skill

Skills in this repo are discoverable as `skills/<name>/SKILL.md`. To
make them available globally to your Claude Code workspace, symlink:

```bash
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/slow-query-investigator" ~/.claude/skills/slow-query-investigator
```

After this, the trigger phrases below activate the skill automatically.

---

### Worked examples

#### Example 1 — All queries that lock a specific table  *(Phase 1)*

**You ask:** *"Show me all queries that lock `rental` rows."*

**The skill calls:**
```
mcp__dbs-vector-stdio__search_sql_api(
    query="lock contention row write",
    table_filter="rental",
    min_lock_time=0.001,
    limit=100,
)
```

**Why these parameters:**
- `table_filter="rental"` returns only queries that touch the `rental`
  table. Internally this triggers exact flat-scan over the slow-log
  table so the IVF approximate index doesn't drop matches.
- `min_lock_time=0.001` excludes queries with zero lock contribution
  (pure SELECTs, EXPLAINs, monitoring tools).
- `query=...` doesn't filter — it ranks within the filtered set,
  biasing toward write-locking patterns.

**You get back** (sorted by `lock_time_sec` DESC after Claude
post-processes):

```
─── 23 queries on `rental`, total lock_time 142.7s, total calls 8,412 ───
1. lock_sec=24.3  exec_ms=24,280  calls=1
   INSERT INTO rental(rental_date, inventory_id, customer_id, return_date,
   staff_id, last_update) VALUES (?, ?, ?, NULL, ?, NOW())
   ON DUPLICATE KEY UPDATE last_update = NOW(), staff_id = VALUES(staff_id)

2. lock_sec=17.5  exec_ms=8,964  calls=835
   UPDATE rental SET return_date = NOW() WHERE rental_id = ?

3. lock_sec=15.9  exec_ms=1,342,476  calls=131
   INSERT INTO rental(rental_date, inventory_id, customer_id, return_date,
   staff_id, last_update) VALUES (?, ?, ?, ?, ?, ?)
…
```

The high-volume INSERT and the `return_date` UPDATE are the lock-time
floor — they're the ones to optimize first.

#### Example 2 — Where is lock contention coming from?  *(Phase 1, no specific table)*

**You ask:** *"Where is our lock contention coming from?"*

**The skill calls:**
```
mcp__dbs-vector-stdio__search_sql_api(
    query="lock contention waiting blocking writes",
    min_lock_time=10.0,
    limit=50,
)
```

…then post-aggregates the `tables` field across the result set in
Claude's reply, ranking tables by attributed lock-time:

```
─── Top 5 tables by attributed lock_time (50 queries with lock ≥ 10s) ───
rental                  342.1s  ── 12 queries
payment                 178.5s  ──  8 queries
inventory                94.2s  ──  6 queries
customer                 41.8s  ──  4 queries
staff                    18.3s  ──  2 queries
```

Use this to pick which table to drill into next (Example 1 or 4 with
the worst offender).

#### Example 3 — Slowest queries hitting a specific table  *(Phase 1)*

**You ask:** *"What are the slowest queries on `inventory`?"*

**The skill calls:**
```
mcp__dbs-vector-stdio__search_sql_api(
    query="slow query large execution time",
    table_filter="inventory",
    min_time=1000.0,
    limit=50,
)
```

Same shape as Example 1 but ranks by `execution_time_ms` DESC instead
of `lock_time_sec`. Use when the bottleneck is wall-clock rather than
lock contention — typically table scans, missing indexes, or N+1
patterns hidden in batch jobs.

#### Example 4 — Recommend indexes for a specific table  *(Phase 2)*

This is the full end-to-end Phase 2 workflow: the skill chains
`dbs-vector` (slow-log corpus) + `mysql-mcp-server` (live schema +
EXPLAIN) into a concrete CREATE INDEX recommendation.

**You ask:** *"What indexes should I add to `rental`?"*

**The skill walks 10 steps:**

```
Step 1: Confirm scope
  ─ table_size("rental") → 16,044 rows | 1.6 MB data | 4.0 MB indexes
  ─ "I'll analyze 80% of the slow-query call volume against
     sakila.rental. Continue?"

Step 2: Fetch slow-query corpus filtered to `rental`
  ─ search_sql_api(query="*", table_filter="rental", limit=10000)
  ─ 47 distinct queries returned

Step 3: 80% cumulant
  ─ Top 6 queries cover 80% of call volume (13% of fingerprints)

Step 4: Fetch existing schema
  ─ list_indexes("rental"):
      PRIMARY              (rental_id)                   UNIQUE
      rental_date          (rental_date,inventory_id,customer_id)  UNIQUE
      idx_fk_inventory_id  (inventory_id)
      idx_fk_customer_id   (customer_id)
      idx_fk_staff_id      (staff_id)
  ─ foreign_keys: 3 (inventory_id, customer_id, staff_id)

Step 5: Extract predicate signatures (sqlglot + regex fallback)
  Q1 [4,210 calls]:  WHERE rental_id eq             (kind=UPDATE)
  Q2 [2,816 calls]:  WHERE customer_id eq, return_date eq (NULL)
  Q3 [1,944 calls]:  WHERE staff_id eq, rental_date range
  Q4 [1,508 calls]:  pure INSERT (no WHERE)
  Q5 [1,022 calls]:  JOIN inventory_id, WHERE rental_date range
  Q6 [  698 calls]:  WHERE customer_id eq

Step 6: Synthesize index candidates
  C1: (customer_id, return_date)        ← Q2, Q6 — 3,514 calls
  C2: (staff_id, rental_date)           ← Q3 — 1,944 calls
  C3: (rental_date, inventory_id)       ← Q5 — 1,022 calls

Step 7: Eliminate redundancy
  ─ C1 leading column = customer_id; existing idx_fk_customer_id is
    a strict prefix → C1 strictly dominates and replaces it.
  ─ C2 leading column = staff_id; existing idx_fk_staff_id is a strict
    prefix → C2 strictly dominates and replaces it.
  ─ C3 has no overlap with existing indexes.

Step 8: EXPLAIN before/after
  Q2 before: type=ref, key=idx_fk_customer_id, rows=14, Extra=Using where
  Q2 after:  type=ref, key=<C1>,               rows=2,  Extra=(none)
  Q3 before: type=ref, key=idx_fk_staff_id,    rows=8,  Extra=Using where; Using filesort
  Q3 after:  type=range, key=<C2>,             rows=3,  Extra=(none)
  Q5 before: type=ALL,  key=NULL,              rows=16,044
  Q5 after:  type=range, key=<C3>,             rows=210

Step 9: Cost analysis
  C1: ~0.5 MB  | replaces idx_fk_customer_id; net +0 indexes; +1 col/INSERT
  C2: ~0.5 MB  | replaces idx_fk_staff_id;    net +0 indexes; +1 col/INSERT
  C3: ~0.5 MB  | net +1 index;                              ; +1 col/INSERT
  Write rate (slow-log): 1,508 INSERTs/period → +3 col-writes per INSERT.

Step 10: Recommendation

  -- Drop the dominated single-column FK indexes:
  ALTER TABLE sakila.rental DROP INDEX `idx_fk_customer_id`;
  ALTER TABLE sakila.rental DROP INDEX `idx_fk_staff_id`;

  -- Add the composite replacements (covers FK + hot query patterns):
  ALTER TABLE sakila.rental
    ADD INDEX `idx_customer_returns` (customer_id, return_date);
  ALTER TABLE sakila.rental
    ADD INDEX `idx_staff_date` (staff_id, rental_date);
  ALTER TABLE sakila.rental
    ADD INDEX `idx_date_inventory` (rental_date, inventory_id);

Coverage: 3 indexes cover 6,480 of the top-80% calls.
Net change: +1 secondary index (3 added, 2 dropped).
EXPLAIN-verified row reduction across Q2/Q3/Q5: 16,044 → 215 (-99%).
```

The skill **does not run the DDL** — `mysql-mcp-server` is read-only.
You apply the changes manually after reviewing.

---

### Quick reference card

| Question shape | Phase | Key parameters |
|---|---|---|
| Locks on TABLE X | 1 | `table_filter`, `min_lock_time` |
| Slow queries on TABLE X | 1 | `table_filter`, `min_time` |
| Lock contention (any table) | 1 | `min_lock_time`, post-aggregate |
| Similar to QUERY | 1 | `query` (free text), no filter |
| Rank tables by metric | 1 | bypass MCP entirely (Python + LanceDB) |
| Recommend indexes for TABLE X | 2 | dbs-vector + mysql-mcp-server (10-step) |

### Limitations

The skill exposes its own caveats prominently in every Phase 2 report,
but worth knowing up front:

- **EXPLAIN with synthetic params** doesn't reflect production row
  estimates. The plans use `1` for int columns, `'x'` for strings,
  `'2024-01-01'` for dates. Real workloads may pick different paths.
- **Slow-log under-represents fast queries.** A query that finishes
  in <1ms is usually missing from the corpus. The skill optimizes
  what's recorded; it can't help with what isn't.
- **No FORCE INDEX recommendations.** The skill suggests indexes; it
  doesn't override the optimizer.
- **JSON / fulltext / spatial** indexes are out of scope.
  Functional indexes get noted in the rejected list with a manual-
  design hint.
- **Read-only.** `mysql-mcp-server` cannot run the DDL — you apply
  recommendations yourself in staging first.
- **Drop indexes only after re-running EXPLAIN** on production-shaped
  queries with real parameter values. The skill's redundancy logic
  is correct on B-tree leftmost-prefix grounds, but pathological
  histograms or stale ANALYZE statistics can produce surprising
  optimizer choices.

### Further reading

- [`skills/slow-query-investigator/SKILL.md`](../skills/slow-query-investigator/SKILL.md) — full skill body with Phase 2 algorithm
- [`README_MCP.md`](README_MCP.md) — MCP installation and tool naming
- [`README_REMOTE_SQL_API.md`](README_REMOTE_SQL_API.md) — ingestion via remote slow-log API
- [`askdba/mysql-mcp-server`](https://github.com/askdba/mysql-mcp-server) — schema introspection MCP server (Phase 2)
- [Sakila DB documentation](https://dev.mysql.com/doc/sakila/en/) — sample database used in examples
