---
name: slow-query-triage
description: Use when asked to find the most impacted slow query, "what's burning DB time", "top slow queries", "find the worst query", or to triage SQL performance via dbs-vector-stdio + a MySQL MCP server. Scope is SINGLE-QUERY triage — Phase 1 finds the one most-impacted fingerprint, Phase 2 validates its root cause against live MySQL schema. Do NOT use for aggregate-across-table index recommendations ("what indexes should I add to table X") — that requires per-query predicate attribution this skill deliberately does not attempt.
---

# Slow Query Triage (token-efficient, single-query scope)

A two-phase workflow that finds the worst slow-log fingerprint in **one MCP
call**, then validates its root cause with three or four MySQL calls.
Designed to spend ~10–15× fewer tokens than broad-probing the corpus.

## Scope

- **In scope:** "find the most impacted query", "what's the worst slow query",
  "what's burning DB time", "triage the slow log".
- **Out of scope:** "recommend indexes for table X", "what should I add to T",
  aggregate-across-many-queries analysis. Those need predicate attribution
  (alias→table mapping, JOIN-target filtering) that this skill does not
  perform.

## Prerequisites

This skill assumes two MCP servers are reachable:

- **`dbs-vector-stdio`** — vector-indexed slow-log fingerprints. Exposes the
  `search_sql*` family (one tool per configured engine — gemma, granite,
  local file or remote API) plus `list_engines`.
- **A generic MySQL MCP** — exposes `list_indexes`, `show_create_table`,
  `explain_query`, `search_schema`, `list_tables`, `list_databases`,
  `table_size`. The skill uses unprefixed call names below; adjust to the
  prefix your MCP host uses.

The example calls in this skill use `search_sql_api` (the remote-API gemma
variant) for Phase 1. Substitute whichever `search_sql_*` engine your
`config.yaml` defines.

## Tool contract — memorize, don't relearn

### `search_sql_*` family

| Param | Type | Unit / semantics |
|---|---|---|
| `query` | str (required) | Used for **cosine similarity ranking only**, not impact. |
| `limit` | int (default 5) | Max results. Stay ≤500; combined with `table_filter` flakes above that on some transports. |
| `min_time` | float \| None | **Milliseconds.** Cumulative `execution_time_ms` across ALL calls of a fingerprint. NOT per-call avg. |
| `min_lock_time` | float \| None | **Seconds.** Cumulative `lock_time_sec`. Note: different unit from `min_time`. |
| `table_filter` | str \| None | Matches any fingerprint whose `tables` list contains the table. Includes pure JOIN targets — see "trap" below. |
| `source_filter` | str \| None | Restrict to one database name (only useful when the slow-log agent tags `source_database`; on many corpora the field is `n/a` and the filter is inert). |
| `include_raw` | bool (default False) | `True` adds the original (un-normalized) SQL alongside `Normalized SQL:`. Default False saves tokens; turn on only when you need to see literal values. |

**Mental model:** filters narrow the candidate set; `query` ranks within it.
For impact-finding the **filter does the heavy lifting** — the `query` string
is almost a no-op when `min_time` is high.

### Engine choice

With `min_time` set, **engines converge on the same top-impact set** —
verified across multiple embedding backends. Default to whichever variant
has the best recall on low-signal generic queries in your deployment
(typically the gemma-based engine, exposed as `search_sql_api`). Switch
engines only if the user explicitly asks or you suspect tokenization
differences (rare).

## Phase 1 — find the worst query (ONE MCP call)

```
search_sql_api(
    query="select from where",          # placeholder — min_time does the work
    limit=20,
    min_time=999999,                    # ≈ 1000 s cumulative; pre-filters to heavy tail
)
```

Typical effect on a real corpus: thousands of fingerprints → ~100 candidates
→ top 20 returned.

**Pick the winner by `Calls` descending; break ties on cumulative
`execution_time_ms`.** Frequency = production hot-path. A 3-second query
running 4,000 times/day beats a 1-shot 50-minute batch — the batch is a
one-time cost; the 3-second query compounds across every user request. The
cumulative `execution_time_ms` field is informative for total-burn
estimation but is NOT the primary ranking key.

No further searches needed in 90% of cases.

If 20 isn't enough breadth, escalate to `limit=50` (still one call). If the
heavy tail is even narrower, raise `min_time` to `3600000` (1 hr cumulative)
to focus on the absolute top.

### Triage filter — skip these in Phase 1 results

| Skip when | Detection | Why |
|---|---|---|
| Ad-hoc DBA / analyst queries | `User` is a named human account, or `Host` matches a bastion / VPN / cloud-SQL-proxy pattern | Human GUI/IDE sessions, not production load. Often `Calls=1` with multi-minute exec time. |
| Single-shot reports | `Calls=1` AND `User` is NOT a service account | One-time analytical runs; fix is "don't run it" not an index. |
| Monitoring queries | `Tables: n/a` or `SHOW ENGINE INNODB STATUS` etc. | Infrastructure noise. |

**Production signal:** `Host` is an internal service IP (RFC1918), `User`
follows the shop's service-account convention (common patterns: `s_*`,
`svc_*`, `app_*`, `worker_*`). These represent real recurring load.

### When the top candidate has no clean B-tree fix

Some winners can't be solved with `ALTER TABLE ... ADD INDEX ...`. Detect
these BEFORE running Phase 2 by scanning the normalized SQL for:

| Signal | Why it blocks indexes |
|---|---|
| `FORCE INDEX (PRIMARY)` | App has deliberately chosen PK ordering; adding an index won't be used. |
| `lower(col) LIKE concat(?, ?, ?)` or `LIKE concat('%', ?, '%')` | Function on column + leading wildcard. B-tree useless. Needs FULLTEXT / trigram / functional index. |
| `ORDER BY length(col)`, `ORDER BY DATE(col)`, any `ORDER BY f(col)` | Function on column defeats any ORDER BY index. |
| Range scan with `id <= ? AND id > ?` + status post-filter | PK-pagination pattern. Selectivity issue, not index issue. |

**When you detect one of these:** still run Phase 2 against this fingerprint
and **report it with the verdict "query / service rewrite required, not an
index."** Do NOT silently skip to the next candidate — the user needs to
know what's burning the time, even if the answer is "fix the application."

You may *additionally* offer the next-highest-impact candidate that DOES fit
the textbook pattern as a "while you're refactoring, here's a cheap win."

## Phase 2 — validate root cause (four MySQL calls)

Once you have ONE candidate fingerprint with a clear `Tables:` value:

### Step 0 — canonicalize the table name (REQUIRED)

The slow-log normalizes table names to lowercase. The live MySQL schema may
use CamelCase or other casing verbatim. On case-sensitive deployments
(Linux MySQL by default), `list_indexes(database=<db>, table='orderitem')`
will fail with `Error 1146 Table doesn't exist` if the table is actually
`OrderItem`.

**Always canonicalize first:**

```
search_schema(database=<db>, pattern='%<lowercase_table_name>%')
```

Read the returned `table` field with `type: TABLE` and use that exact
casing for all subsequent calls. Skipping this step costs one round-trip
and one error message every time.

### Step 1–3 — the three plan calls

```
list_indexes(database=<db>, table=<CanonicalTable>)
show_create_table(database=<db>, table=<CanonicalTable>)
explain_query(sql=<query with real values>, format="traditional", database=<db>)
```

`explain_query` returns the plan as a **JSON array of row objects**, one
per step (`{table, type, key, key_len, rows, filtered, Extra, ...}`). The
`Extra` column is where filesort / temporary-table / index-merge hints
appear. Read it yourself — auto-suggested fix narratives are NOT returned
by generic MySQL MCPs. Diagnose from the columns directly.

### MySQL MCP quirks

- **Active connection may be a local dev mirror, not prod.** Compare
  `table_size.rows` against `AUTO_INCREMENT` from `show_create_table`. If
  `AUTO_INCREMENT` is 50×+ the row count, you're on a small mirror.
  **Schema diagnosis is still valid** — DDL is replicated. Row estimates are
  not — annotate as "local-only" in the report.
- **`format="json"` returns a stub** (`{"plan":{"query_cost":N}}` only on
  many MCP implementations). Always use `format="traditional"` for the
  full plan. Note: despite the name, `traditional` does NOT return MySQL's
  ASCII tabular `EXPLAIN` — it returns a JSON array of step objects. Same
  content, different framing.
- **Do not propose `ALTER TABLE` execution.** This skill diagnoses; the
  user decides when to apply changes. Show the recommended DDL as a
  fenced SQL block, never run it.

## The textbook fix pattern

Most heavy-tail slow queries match `WHERE col_a = ? ORDER BY col_b DESC LIMIT N`
with a single-column index on `col_a` only. Plan signature:

```
type: ref   key: <single-col index on col_a>   Extra: Using filesort
```

Fix:

```sql
ALTER TABLE <table> ADD INDEX idx_a_b (col_a, col_b);
```

Converts a filesort over N rows into a backward index seek (MySQL can scan
the second column DESC). Expected speedup: 100×–10,000× depending on
per-tenant row count.

## Trap: `table_filter` matches JOIN targets too

`table_filter="<table>"` returns any query whose `tables` array contains
that table — **including queries where it's only a LEFT JOIN target with
zero predicates on it**. Do not aggregate predicates across these hits and
recommend an index on the JOIN target — most "hits" won't benefit. Stay
single-query: validate the predicate yourself before recommending.

## What good output looks like

A one-screen report:

1. **Winner:** Fingerprint ID, normalized SQL, cumulative ms × calls, user/host.
2. **Plan:** EXPLAIN output (traditional) + `Extra` column verbatim.
3. **Diagnosis:** One sentence — usually "missing composite index on (X, Y)".
4. **Fix:** One DDL statement in a code fence. No commentary on rollout.
5. **Caveat (if applicable):** "Validated against local mirror (N rows;
   AUTO_INCREMENT M); prod row estimates will differ."

## Anti-patterns (token wasters)

- ❌ Running 5+ probes with different `query` strings hoping to surface heavy
  hitters. Use `min_time=999999` once.
- ❌ Ranking by cumulative `execution_time_ms` alone. The right key is
  **`Calls` descending, tiebreak on cumulative ms** — frequency dominates.
- ❌ Running multiple engine variants when filters are active — they converge.
- ❌ Calling `explain_query` with `format="json"` — returns a stub on most
  MCPs.
- ❌ Recommending an index for a table that appears in the slow-log only as a
  JOIN target. Confirm it's in the `WHERE` clause first.
- ❌ Calling `list_indexes` / `show_create_table` / `explain_query` with the
  lowercase table name straight from the slow log on a case-sensitive
  MySQL host. Canonicalize via `search_schema` first.
- ❌ Silently skipping a top winner because it doesn't fit the textbook
  composite-index pattern. Report it; tell the user it needs a rewrite.
- ❌ Triaging single-`Calls=1` queries from human DBA / analyst accounts as
  if they were production impact. They are not.
