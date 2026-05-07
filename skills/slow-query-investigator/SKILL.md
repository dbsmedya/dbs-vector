---
name: slow-query-investigator
description: Use when the user asks investigative questions about slow SQL queries OR asks for index recommendations on a specific table. Phase 1 (dbs-vector MCP only) — pinpointing lock contention, finding worst-offending queries by execution time or lock time, identifying which queries hit a table, comparing query patterns. Triggers like "show me queries that lock TABLE X", "what's hammering table Y", "find all queries hitting Z", "slowest queries on T", "lock contention on T". Phase 2 (dbs-vector + mysql-mcp-server / future postgres equivalent) — analyzing slow-query traffic against a target table, fetching its current schema and indexes, running EXPLAIN, and proposing the minimum sufficient set of new indexes that covers ~80% of the query traffic. Triggers like "what indexes should I add to T", "missing indexes on T", "optimize queries on T", "suggest indexes for T", "recommend indexes for T". Routes both phases to the right combination of MCP tools so the LLM client doesn't have to reinvent the workflow each time.
---

# Slow Query Investigator

A skill for routing slow-query investigation questions to dbs-vector's
filter-aware SQL search tools (`search_sql`, `search_sql_granite`,
`search_sql_api`, `search_sql_api_granite`).

## When this skill applies

User asks an **investigative** question about SQL slow-log data — typically:

- "Show me all queries that lock `<table>` rows"
- "Which queries are causing lock contention on `<table>`?"
- "What are the slowest queries hitting `<table>`?"
- "Find queries similar to `UPDATE foo SET …`"
- "What's the lock-time damage on the `tx_*` shard tables?"

If the question is purely analytical (e.g. "rank databases by total lock_time"
with no reference to a specific table or query family), prefer the **direct
aggregation fallback** at the bottom of this skill. The MCP search tools
add no value when there's no semantic similarity component.

## Prerequisites — verify before invoking tools

1. Call `mcp__dbs-vector-stdio__list_engines` once at the start of an
   investigation. Confirm at least one SQL-family engine has `loaded: true`.
2. Pick the engine by data source:
   - Local DuckDB slow-log file → `search_sql` or `search_sql_granite`
   - Remote HTTP slow-log API → `search_sql_api` or `search_sql_api_granite`
3. Default to the **Gemma** variant (`search_sql`, `search_sql_api`). Gemma
   matches Granite on SQL clustering quality and uses ~1/3 the GPU memory.
   Switch to Granite only if the user explicitly asks or the slow-log
   contains substantial non-English content.

If `list_engines` returns 0 SQL engines, abort with: "dbs-vector has no SQL
engine loaded. Add `sql` or `sql-api` (or their Granite siblings) under
`engines:` in `config.yaml` and restart `dbs-vector mcp`."

## SQL family handler signature

Every SQL-family tool accepts the same parameters:

| Parameter | Type | Default | What it does |
|---|---|---:|---|
| `query` | string | required | Natural-language or partial-SQL string used for **semantic ranking** (does NOT filter) |
| `limit` | int | 5 | Max results to return |
| `source_filter` | string | None | Restrict to one **database** name (e.g. `"biotekno_biz"`) |
| `min_time` | float | None | Minimum cumulative `execution_time_ms` |
| `min_lock_time` | float | None | Minimum cumulative `lock_time_sec` |
| `table_filter` | string | None | Restrict to queries that touch a specific **table** (uses array_has predicate, exact flat scan) |

**Mental model:** filters narrow the candidate set; `query` ranks within it.
For investigation questions, the **filters carry most of the load**; the
`query` string is mostly a tie-breaker on ordering.

## Pattern 1 — All queries that lock a specific table  *(world-class case)*

User asks: *"Show me all queries that lock dt_customer_performance_report rows."*

```
mcp__dbs-vector-stdio__search_sql_api(
    query="lock contention row write",
    table_filter="dt_customer_performance_report",
    min_lock_time=0.001,
    limit=100,
)
```

Why these parameters:

- **`table_filter`** returns ONLY queries that touch the table. Internally
  this triggers `bypass_vector_index()` so the IVF approximate index
  doesn't silently drop matches living in unscanned partitions.
- **`min_lock_time=0.001`** excludes queries with zero lock contribution
  — pure SELECTs, EXPLAINs, monitoring queries — that would otherwise
  pollute the result set. Use a larger threshold (e.g. `1.0`) if the
  user wants only the heavy hitters.
- **`query="lock contention row write"`** is for ranking, not filtering.
  It biases toward queries that look like writes (INSERT / UPDATE /
  DELETE / ON DUPLICATE KEY UPDATE) rather than reads. Pass a different
  hint if the user is investigating read-locking specifically.
- **`limit=100`** captures the whole universe for a typical hot table.
  Bump higher for tables with hundreds of distinct query fingerprints.

After receiving results, **sort the returned set by `lock_time_sec` DESC**
before reporting — the tool returns results ranked by hybrid score, not
lock impact, and the user wants worst-offenders first.

Always end the report with the aggregate:
- Total queries returned
- Total `lock_time_sec` summed across them
- Total `calls` summed across them

This lets the user immediately verify completeness (cross-check against
ground truth from the database itself if available) and gauge severity.

## Pattern 2 — Slowest queries on a specific table

User asks: *"What are the slowest queries on `tx_process`?"*

```
mcp__dbs-vector-stdio__search_sql_api(
    query="slow query large execution time",
    table_filter="tx_process",
    min_time=1000.0,
    limit=50,
)
```

Sort returned set by `execution_time_ms` DESC. `min_time=1000.0` filters
out fast/cheap queries; raise the threshold for systems with a heavy
slow-query baseline.

## Pattern 3 — Lock contention without naming a table

User asks: *"Where is our lock contention coming from?"*

When the user has no specific table in mind, run a lock-time-only scan:

```
mcp__dbs-vector-stdio__search_sql_api(
    query="lock contention waiting blocking writes",
    min_lock_time=10.0,
    limit=50,
)
```

Then post-aggregate the `tables` field across results in your reply:

```python
from collections import defaultdict
table_lock = defaultdict(float)
for r in results:
    for tbl in (r.chunk.tables or []):
        table_lock[tbl.strip('"')] += r.chunk.lock_time_sec
top = sorted(table_lock.items(), key=lambda x: x[1], reverse=True)[:10]
```

Report the top 10 tables ranked by attributed lock time, with a sample
query for each. Note the strip — the chunker stores names with literal
`"` chars due to a SQLGlot artifact.

## Pattern 4 — Find queries semantically similar to a known pattern

User asks: *"Find all queries similar to `UPDATE tx_process SET STATUS = ?`"*

```
mcp__dbs-vector-stdio__search_sql(
    query="UPDATE tx_process SET STATUS = ? WHERE PROCESS_ID = ?",
    limit=10,
)
```

For pure similarity search, **don't** apply `table_filter` — the query
text already constrains via FTS + vector hybrid scoring. Filter only
when the user wants a specific table; otherwise the engine surfaces
similar query patterns even on lookalike tables (e.g., shard variants).

## Reporting format

For every reported result, format with:

```
Source DB: {chunk.source}
Lock time: {chunk.lock_time_sec:.2f}s    Exec time: {chunk.execution_time_ms:,.0f}ms    Calls: {chunk.calls:,}
Tables touched: {sorted({t.strip('"') for t in chunk.tables})}
SQL:
{chunk.raw_query[:300]}{'…' if len(chunk.raw_query) > 300 else ''}
```

For tables-touched, **always strip the surrounding `"` characters** when
displaying to the user. The quote chars are an internal data-quality
artifact (SQLGlot qualified-name representation); the user shouldn't
see them.

End the report with a **summary block**:

```
─── Summary ───────────────
Returned:        N queries
Total lock_time: X.Xs
Total calls:     Y
─────────────────────────
```

## Direct aggregation fallback (skip the embedding layer)

For pure top-N-by-aggregate questions ("rank tables by lock_time", "top 10
databases by call volume"), embeddings add nothing — and the IVF index
will actively hurt completeness. Drop to a direct LanceDB read:

```python
import lancedb
from collections import defaultdict

db = lancedb.connect('./lancedb_dbs_vector')
# Choose table by which engine ingested:
#   query_vault            — sql / sql-api (Gemma)
#   query_vault_granite    — sql-granite
#   query_vault_granite_api — sql-api-granite
t = db.open_table('query_vault')
df = t.to_pandas()

lock_per_table = defaultdict(float)
calls_per_table = defaultdict(int)
for _, row in df.iterrows():
    tables = row['tables']
    if tables is None or len(tables) == 0:
        continue
    lt = float(row['lock_time_sec'] or 0)
    n = int(row['calls'] or 0)
    for tbl in tables:
        clean = tbl.strip('"')
        lock_per_table[clean] += lt
        calls_per_table[clean] += n

top_lock = sorted(lock_per_table.items(), key=lambda x: x[1], reverse=True)[:10]
```

Run this only when a tool call would be the wrong shape — never for
"queries on this specific table", which the MCP tools handle natively.

## Known data quality artifact

The chunker stores the `tables` column with literal `"` chars around
each entry (e.g., `'"dt_customer_performance_report"'`). Both the
`table_filter` parameter and the result-formatting pattern in this
skill compensate for this transparently. **Never expose the quoted
form to the user.** When reading raw `tables` arrays from LanceDB,
always `.strip('"')` before display or aggregation.

A future re-ingest will fix the chunker output; at that point the
`strip('"')` calls become no-ops (safe to keep, just redundant).

## Phase 2 — Index recommendation for a target table

Combines the slow-log corpus (dbs-vector MCP) with live schema
introspection and EXPLAIN (mysql-mcp-server / future postgres adapter)
to recommend the **minimum sufficient set** of indexes covering the
top-80% of call volume against a named table.

### Required MCP servers

| Capability | Server | Tool prefix |
|---|---|---|
| Slow-log retrieval (Phase 1) | dbs-vector | `mcp__dbs-vector-stdio__*` |
| MySQL schema + EXPLAIN | [askdba/mysql-mcp-server](https://github.com/askdba/mysql-mcp-server) with `MYSQL_MCP_EXTENDED=1` | `mcp__mysql__*` (or whatever alias the user configured) |
| PostgreSQL schema + EXPLAIN | TBD — postgres adapter | `mcp__postgres__*` (planned) |

If the mysql/postgres adapter is not connected, abort Phase 2 with:
*"I need a database-introspection MCP server to recommend indexes. Set
up `askdba/mysql-mcp-server` (or a postgres equivalent) and reconnect."*

### Abstract operation contract

Each Phase 2 step uses one **abstract operation**. The skill maps each
operation to the active RDBMS adapter's tool name. When postgres lands,
only this table changes — the workflow stays the same.

| Abstract op | mysql-mcp-server tool | What it returns |
|---|---|---|
| `list_indexes(T)` | `list_indexes` | Existing indexes on T: name, columns, uniqueness, type |
| `show_create_table(T)` | `show_create_table` | Full CREATE TABLE — covers PK + FKs + collation + storage |
| `table_size(T)` | `table_size` | Rows, data_length, index_length (cost basis for new indexes) |
| `foreign_keys(T)` | `foreign_keys` | FKs that should also have a covering index |
| `explain_query(SQL)` | `explain_query` | Execution plan (whether MySQL would use the index in question) |
| `run_query(SQL)` | `run_query` | Read-only SELECT — used for cardinality probes |
| `describe_table(T)` | `describe_table` | Column types (used to synthesize sane EXPLAIN parameter values) |

### Workflow

#### Step 1 — Confirm scope and connection

1. Call `mcp__dbs-vector-stdio__list_engines` and pick a SQL family engine
   that contains data for the target database.
2. Call the mysql adapter's `list_databases` and confirm the target
   database is reachable. If the user has multiple connections configured,
   call `use_connection(...)` to pick the right one.
3. Confirm with the user: *"I'll analyze the slow-query corpus against
   the live schema of `<db>.<table>` (rows: N, data_length: X MB).
   Top-80% of call volume covered by Y queries. Continue?"* Quote the
   table_size output verbatim to set expectations.

#### Step 2 — Fetch the slow-query corpus for T

```
mcp__dbs-vector-stdio__search_sql_api(
    query="* * * * *",
    table_filter="<T>",
    min_time=0,
    limit=10000,
)
```

The vague `query` string just satisfies the required parameter; the
filter is doing the work. `limit=10000` is intentionally generous —
any real table has at most ~hundreds of distinct query fingerprints.

If the engine is a non-API one (e.g., `search_sql`, `search_sql_granite`),
substitute its name. Honor the engine the user picked in Step 1.

#### Step 3 — Compute the 80% cumulant

Sort the returned queries by `calls` DESC. Cumulative-sum
the call counts. Cut at 80% of the total. The resulting set is the
**covered** corpus — the queries that any index recommendation must
serve. Queries beyond the cut are ignored.

If the user explicitly says they care about total execution time
instead of call volume (e.g., "we have a few rare-but-expensive
batch jobs"), sort by `execution_time_ms` DESC instead and cut at
80% of total time. State which metric you used in the final report.

#### Step 4 — Fetch existing schema

Call once per target table:
- `show_create_table(T)` — get current PK + secondary indexes + FKs + storage
- `list_indexes(T)` — structured form of the same (easier to diff against)
- `foreign_keys(T)` — FKs that need covering indexes (any column that's an
  FK referent should have an index — flag missing ones)
- `describe_table(T)` — column types (needed for synthesizing EXPLAIN params)
- `table_size(T)` — for cost estimation

Persist these as in-memory state for the rest of the workflow.

#### Step 5 — Extract predicate signatures

Use the project's bundled extractor — it sanitizes slow-log placeholders
(`?+`, `?,?,?`, bare `?`), tries sqlglot first, falls back to regex for
queries sqlglot can't parse (commonly `INSERT … ON DUPLICATE KEY UPDATE`
with nested IF/values expressions):

```python
from dbs_vector.services.sql_parser import extract_predicate_signature

for q in covered_queries:
    sig = extract_predicate_signature(q["raw_query"])
    # sig keys: eq_cols, range_cols, join_keys, order_by, is_write, kind, parser
```

The returned dict guarantees these keys for every input — no parse
failures bubble up; pathological SQL still returns an empty-but-shaped
signature. Aggregate the `parser` field across the corpus to gauge
confidence: if more than ~5% fall to the regex path, the recommendation
is built on partial structural information and you should say so in the
final report.

The fields:

| Field | Meaning |
|---|---|
| `eq_cols` | Equality predicates: `WHERE col = ?` or `WHERE col IN (...)` |
| `range_cols` | Range predicates: `>`, `>=`, `<`, `<=`, `BETWEEN` |
| `join_keys` | JOIN ON columns from any side of the join condition |
| `order_by` | List of `(column, "ASC" \| "DESC")` tuples |
| `is_write` | True for INSERT/UPDATE/DELETE |
| `kind` | Statement kind: `SELECT` / `INSERT` / `UPDATE` / `DELETE` / `?` |
| `parser` | Which path produced this — `"sqlglot"` or `"regex"` |

Aggregate across covered queries:
```
[
    {"query_id": "...", "calls": 8478,
     "eq_cols": ["process_id"], "range_cols": [], "join_keys": [],
     "order_by": [], "is_write": False, "parser": "sqlglot"},
    ...
]
```

The parse-success rate against real slow-log corpora is verified by
`tests/integration/test_skill_predicate_extraction.py` (99% yield rate
against the project's reference corpus). If you encounter a query
shape that yields empty signatures even though it has a real WHERE
clause, that's a parser regression — add it as a unit test to
`tests/unit/test_sql_parser.py` so it doesn't recur.

#### Step 6 — Synthesize index candidates

For each unique `(eq_cols, range_cols, order_by)` signature, generate
one composite index candidate. Column ordering rule (critical for MySQL
B-tree efficiency):

1. **Equality columns first**, in any order (B-tree leftmost-prefix rule
   matches any prefix of equality columns).
2. **Range columns next**, in source order. Only the first range column
   uses the index for range scan; subsequent ranges are filtered.
3. **ORDER BY columns last**, only if they aren't already in eq/range.
   Match the direction of the ORDER BY for MySQL 8+ descending indexes.

Tag each candidate with:
- The set of query IDs it covers
- Combined call volume (sum of `calls` across covered queries)

#### Step 7 — Eliminate redundancy

Three passes:

1. **Existing-index containment**: For each candidate, check if its
   leading columns match any **prefix** of an existing index from
   Step 4. If yes, drop the candidate (existing index already serves
   the queries).
2. **Inter-candidate prefix containment**: If candidate A's columns
   are a strict prefix of candidate B, drop A and reassign A's
   covered queries to B (B is strictly more powerful).
3. **Coverage de-duplication**: After (1) and (2), if two candidates
   cover overlapping query sets, prefer the one with more covered
   queries. Drop the other.

#### Step 8 — Validate with EXPLAIN

For a representative query from each remaining candidate's covered set,
call `explain_query(<sql>)`. Synthesize parameter values from
`describe_table(T)` column types — `1` for int columns, `'x'` for
strings, `'2024-01-01'` for dates, `0.0` for floats. EXPLAIN with
synthetic params gives row-estimate + key-usage info, not real timings.

What to look for:
- `key` column: is it `NULL` (table scan) or one of the existing indexes?
- `rows` estimate: high values (>10% of table) suggest the proposed
  index would meaningfully improve scan size.
- `Extra` column: `Using filesort`, `Using temporary` — both can be
  resolved by adding the right index.

Tag each candidate with its before-state EXPLAIN summary. Flag any
candidate where existing indexes already give `key=<index_name>` —
those proposals are weak and should drop in priority.

#### Step 9 — Cost analysis

For each surviving candidate:

- **Build cost**: roughly proportional to `data_length × (column_byte_size_sum / row_byte_size)`.
  Pull `data_length` from `table_size(T)`. Don't be precise; an order-of-magnitude
  estimate is sufficient ("~50 MB", "~5 GB", etc.).
- **Write amplification**: every new index adds I/O on every INSERT/UPDATE
  to the table that touches the indexed columns. Estimate write-rate
  from the slow log: count INSERT + UPDATE + DELETE queries against T,
  sum their `calls`. Compare to the read query count covered by the
  proposed index. If writes dominate by more than 2× reads, flag the
  index as potentially regressive.
- **Selectivity sanity check**: optionally run `run_query("SELECT COUNT(DISTINCT col) FROM T")`
  for the index's leading column. If `count_distinct < 100` (very low
  cardinality), warn that the index may not give meaningful scan
  reduction.

#### Step 10 — Produce the recommendation

Emit a structured report:

```
Target: <db>.<table>
Table size: <rows> rows, <data_length> data, <index_length> indexes
Slow-log corpus: <N> queries spanning <total_calls> calls
Coverage cut: top <K> queries cover 80% of call volume

Existing indexes (preserved, no change recommended):
  - PRIMARY (id)
  - idx_customer_id (customer_id)
  ...

Proposed indexes (in apply order):

  1. CREATE INDEX idx_customer_send_date ON <table> (customer_id, process_send_type, d_sent_start_date);
     Covers:        12 queries / 12,806 calls (62% of corpus)
     Build cost:    ~80 MB
     Write amp:     1 added per INSERT (current write rate: 4,200 /day)
     EXPLAIN before: type=ALL, rows=2.1M
     EXPLAIN after:  type=ref, rows~~50 (estimated)

  2. CREATE INDEX idx_status_process ON <table> (process_status_id, process_id);
     Covers:        4 queries / 2,140 calls (10% of corpus)
     ...

Rejected candidates:
  - (process_id) — already covered by PRIMARY (process_id)
  - (customer_id) — already covered by idx_customer_id
  - (force_report) — cardinality < 10, would not improve scans

Unparsed queries: 3 (skipped). See appendix.

Summary: 2 new indexes cover 72% of slow-query call volume.
Net write amp: +2 secondary index updates per write to <table>.
```

DDL is presented as MySQL syntax. For postgres targets the recommender
will swap `CREATE INDEX` syntax variants (e.g., `INCLUDE` columns,
`USING btree`, partial-index `WHERE`) once the postgres adapter is wired.

### Caveats

1. **EXPLAIN with synthetic params** doesn't reflect real workload row
   estimates. The plan's `rows` column is an upper bound based on the
   index's structure; actual production performance may differ. Always
   verify by running EXPLAIN against a real captured parameter set
   when one is available (e.g., from MySQL's general log or the
   slow log's parameter values column).
2. **The slow log under-represents fast queries**. A "high-volume"
   query in the slow log is one that hit the slow-log threshold AND
   ran often. Queries that finish in <1 ms are usually missing from
   the corpus entirely, so the recommender can't tell whether they'd
   benefit from indexes too. Note this limitation in the report.
3. **No FORCE INDEX recommendations**. The skill suggests indexes;
   it doesn't second-guess the optimizer. If a candidate index ISN'T
   used after creation, that's a separate investigation (statistics
   stale? optimizer cost model misconfigured?).
4. **JSON / fulltext / spatial** indexes are out of scope. The skill
   only synthesizes B-tree composite indexes. Functional indexes
   (e.g., `JSON_EXTRACT(col, '$.foo')`) get noted in the rejected list
   with: *"consider a functional index — manual design required."*
5. **The recommender does not modify the database**. mysql-mcp-server
   is read-only; it cannot run `CREATE INDEX`. The user must apply
   the DDL themselves and re-run the workflow afterward to verify.

### Postgres adapter (planned, not yet wired)

When a postgres MCP server is connected, the abstract-operation table
gets a postgres column. Likely candidates:
- [crystaldba/postgres-mcp](https://github.com/crystaldba/postgres-mcp)
- A future `askdba/postgres-mcp-server` mirroring the mysql server's API

The Step 2 → 9 workflow stays identical; only Step 6 (column-ordering rule)
and Step 10 (DDL syntax) need RDBMS-specific tweaks. Postgres-specific
considerations to fold in when that adapter lands:
- `INCLUDE` columns (covering indexes without storing in B-tree key)
- Partial indexes (`WHERE`-clause filters on the index itself)
- BRIN / GIN / GiST index types for non-B-tree workloads
- `ANALYZE` recommendation alongside the index DDL

## Quick reference card

| Question shape | Skill section | Key tools / parameters |
|---|---|---|
| Locks on TABLE | Pattern 1 | dbs-vector `table_filter` + `min_lock_time` |
| Slow queries on TABLE | Pattern 2 | dbs-vector `table_filter` + `min_time` |
| Lock contention (any table) | Pattern 3 | dbs-vector `min_lock_time` + post-aggregate |
| Similar to QUERY | Pattern 4 | dbs-vector `query` free text, no filter |
| Rank tables by metric | Direct aggregation | bypass MCP entirely (Python + LanceDB) |
| **Recommend indexes for TABLE** | **Phase 2** | dbs-vector + mysql-mcp-server (10-step workflow) |
