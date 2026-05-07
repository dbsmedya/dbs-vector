---
name: slow-query-investigator
description: Use when the user asks investigative questions about slow SQL queries — pinpointing lock contention on a specific table, finding the worst-offending queries by execution time or lock time, identifying which queries hit a table, or comparing query patterns across the slow-log corpus. Triggers include phrasings like "show me queries that lock TABLE X", "what's hammering table Y", "find all queries hitting Z", "slowest queries on T", "lock contention on T", "queries blocking writes to T". Routes the question to dbs-vector's MCP search tools with the right filter combination so the LLM client doesn't have to reinvent the parameter mapping every time.
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

## Phase 2 — Index recommendation (planned, not yet wired)

This skill will be extended to recommend the **minimum sufficient set
of indexes** that covers the top-80% of slow-query traffic against a
named table. Pending design questions before implementation:

1. **Coverage target** — currently ~80% of the *call volume* on the
   table. Confirm with user whether they prefer call-volume coverage or
   total-execution-time coverage (these can rank queries very
   differently for tables with a few rare-but-expensive queries).
2. **RDBMS dialect** — MySQL vs PostgreSQL vs MariaDB matters for
   index features (functional indexes, partial indexes, descending
   indexes, INCLUDE columns). The skill needs to know the target
   engine before recommending.
3. **Composite vs single-column** — composite indexes on (a, b) cover
   many (a, b) and (a) workloads but not (b)-only. The recommender
   needs WHERE-column / JOIN-column / ORDER BY-column extraction from
   each query's SQL text.
4. **Write amplification** — every new index slows down writes on the
   table. The recommender must report the existing write rate
   alongside the read benefit.

Phase 2 algorithm sketch (for when implementation begins):

```
1. Fetch all queries on T:
     search_sql_api(query="*", table_filter=T, min_time=0, limit=10_000)
2. For each query, extract:
     - column-equality predicates from WHERE
     - column-range predicates from WHERE
     - JOIN keys
     - ORDER BY columns
3. Sort queries by `calls` desc; cumsum until coverage reaches 80%.
4. For the top-cumulative set, group by (equality_columns, range_columns,
   order_columns) tuple. Each unique tuple is one index candidate.
5. Sort candidates by combined coverage; emit each as
   `CREATE INDEX <suggested_name> ON T (<col>, <col>, ...)`.
6. Eliminate redundancy: if one candidate's columns are a strict prefix
   of another, drop the shorter one.
7. Report: index DDL, queries it covers, % cumulative coverage,
   estimated write-amplification cost.
```

When Phase 2 is implemented, replace this section with the full workflow
and add concrete tool-call patterns for the index-extraction step.

Until then, route index-recommendation requests to: *"I can list all the
queries hitting `<table>` ranked by call volume — you'll need to design
the indexes manually until the recommender is wired up."*

## Quick reference card

| Question shape | Skill section | Key parameters |
|---|---|---|
| Locks on TABLE | Pattern 1 | `table_filter`, `min_lock_time` |
| Slow queries on TABLE | Pattern 2 | `table_filter`, `min_time` |
| Lock contention (any table) | Pattern 3 | `min_lock_time`, post-aggregate |
| Similar to QUERY | Pattern 4 | `query` (free text), no filter |
| Rank tables by metric | Direct aggregation fallback | bypass MCP entirely |
| Index recommendation for TABLE | Phase 2 (deferred) | not yet implemented |
