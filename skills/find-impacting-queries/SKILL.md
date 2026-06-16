---
name: find-impacting-queries
description: >-
  Find and diagnose the highest-impact SQL queries hitting a live MySQL database by
  combining dbs-vector's slow-query corpus (top_impacting_/browse_/search_ MCP tools)
  with live schema validation through a MySQL MCP such as mysql-mcp-server (explain_query,
  list_indexes, table_size). Use this whenever the user wants to find slow queries, the
  most expensive / impacting / heavy / costly queries, "top offenders", what's hammering
  or overloading the database, missing-index opportunities, query optimization or tuning
  candidates, lock contention, or example SQL for a given table and why it's slow — even
  if they never say "dbs-vector", "slow log", or "EXPLAIN". Triggers include "which
  queries are slow", "what's killing the DB", "find expensive queries on <table>",
  "index opportunities", "why is <table> slow", "query tuning candidates".
---

# Find Impacting Queries

Diagnose the SQL that costs a live MySQL the most, using two MCP servers:

- **dbs-vector** indexes a slow-query log and ranks fingerprints.
  `top_impacting_<engine>` is the front door; `browse_`/`search_` are follow-ups.
- **a live MySQL MCP** (e.g. `mysql-mcp-server`) validates each candidate against the real
  schema (`explain_query`, `list_indexes`, `table_size`, `run_query`).

Output is a **read-only** ranked diagnosis: which queries hurt, why, a suggested fix per
query. Only `SELECT`/`SHOW`/`DESCRIBE`/`EXPLAIN` ever touch the live server — never DDL
or DML.

## Key concepts (read once)

- A **fingerprint** = one normalized query pattern (deduped by hash). It carries
  aggregate stats (`calls`, `execution_time_ms`, `lock_time_sec`) plus one **exemplar
  `raw_query`** with real literals. `rows_examined`/`rows_sent` are the **most-recent
  call's** values, not averages.
- **Rank by `impact_score` = calls × execution_time_ms** — frequency-weighted "what's
  hammering the DB". This is what `top_impacting` returns by default (`execution_time_ms`
  is already cumulative). Use `avg_ms_per_call` and `selectivity` to *explain*, not rank.
- **Aggregate stats say WHICH queries hurt; the exemplar SQL says WHY.** EXPLAIN the
  exemplar; read selectivity for the rest.

## Three outcomes — every query lands in one

1. **Index fix** — the schema lacks the right index or the optimizer picks a poor one.
   EXPLAIN + `list_indexes` + `table_size` → a concrete `CREATE INDEX` (as text). **This
   is the main deliverable; no business knowledge needed.**
2. **Already optimal / contention-bound** — the right index is already used (covering,
   `Using index`) and the query shape is fine, so no index or rewrite helps. The lever is
   app-side: call-rate, caching/approximation, or — when `avg_ms_per_call` is high
   *despite* a tight plan (a 1-row `const`/`eq_ref`/`ref`) and `lock_time_sec` is
   nonzero — **lock/row contention**. Name which; don't invent schema changes.
3. **Rewrite candidate** — no index helps (full aggregate, `SELECT *` + deep pagination,
   ORM `CASE`/over-fetch). State *why* an index won't help, then hand off to the
   **query-rewrite** skill / domain owner. Do NOT author the rewrite from the SQL alone —
   it needs business meaning the SQL text doesn't carry (what a status code means,
   whether a column is effectively nullable, which columns the caller truly needs).

## Workflow

### 1. Triage — one call
`top_impacting_<engine>(include_raw=true, limit=N)` returns the top-N fingerprints
ranked by `impact_score`, each with its **paste-ready exemplar SQL** (original identifier
case + real literals) plus `avg_ms_per_call` and `selectivity` inline. No re-casing, no
second call to fetch SQL.
- **Scope to a table:** `table="<lowercased-name>"` — the corpus lowercases identifiers,
  so pass `orders`, not `Orders`.
- **Other lenses:** `order_by=lock_time_sec:desc` (contention), `min_calls=N`. Order-by
  column must be one of: `impact_score, execution_time_ms, calls, lock_time_sec,
  avg_ms_per_call, selectivity`.
- **Engines:** `sql` / `sql-api` (Gemma) and `sql-granite` / `sql-api-granite` (Granite)
  may hold different corpora. Run `list_engines` if unsure which has the data; it reports
  `loaded` but not row counts, so a `top_impacting limit:1` confirms data.
- **Follow-up tools (rarely needed):** `browse_<engine>` for arbitrary ranking/grouping
  (`group_by=tables` for hot tables); `search_<engine>` for *concept* similarity
  ("queries like this join"). See `references/dbs-vector-notes.md`.

### 2. Validate against the live database
Per candidate, EXPLAIN the exemplar:
- **SELECT** → `explain_query(sql, format="traditional")` (or `"tree"` for cost/order).
- **UPDATE/DELETE** → `explain_query` is SELECT-only, so use
  `run_query(sql="EXPLAIN <statement>")`. **But a `--super-read-only` replica refuses
  `EXPLAIN UPDATE/DELETE` (Error 1290)** — there, skip that dead call and EXPLAIN the
  WHERE as a SELECT-probe instead: `explain_query("SELECT <pk> FROM <t> WHERE <...>")`.
  When unsure which the connection is, go straight to the SELECT-probe — it always works.
- `list_indexes(db, table)` for what exists; `table_size(db, table)` for scale — a scan
  of 350M rows ≠ a scan of 1k.

Read the plan: `type` (`const`/`eq_ref` best → `ref`/`range` ok → `index`/`ALL` = scan),
`key` (chosen) vs `possible_keys` (a better unused candidate = the smoking gun), `Extra`
(`Using filesort`/`temporary` = sort/aggregate cost; `Using index` = covering). Full
field guide: `references/live-validation.md`.

**Fidelity caveat:** the exemplar carries *one* real literal set, not necessarily the
slow one. A point-lookup exemplar can EXPLAIN clean while the aggregate is slow on *other*
parameters (wide date ranges, big `IN` lists). Trust **impact_score** to rank; lean on
**selectivity** (`rows_examined : rows_sent`; high = examining far more than it returns =
index opportunity) for the truth.

### 3. Report
```
# Impacting queries on <db> (top N by impact_score)

## 1. <fingerprint id> — <tables> (<table size>)
- Impact: impact_score=<>, <total ms> over <calls> calls (~<avg ms>/call), selectivity <examined:sent>
- Exemplar SQL (real literals): <raw SQL>
- Plan: type=<>, key=<chosen>, possible_keys not chosen=<>, Extra=<>
- Outcome: [ INDEX FIX | ALREADY OPTIMAL/CONTENTION | REWRITE CANDIDATE ]
- Recommendation:
    INDEX FIX        -> CREATE INDEX ...           (text only, not executed)
    ALREADY OPTIMAL  -> right index already used; lever is call-rate/caching, or lock
                        contention if avg_ms_per_call high despite a 1-row plan (app)
    REWRITE CANDIDATE-> why no index helps + hand off to query-rewrite skill / domain owner
```
Surface everything as **suggestions** for a human DBA.

## Friction & gotchas (current server behavior)

- **Raw SQL is gated** behind `--allow-raw-queries` on **every** tool (top_impacting,
  search, browse). If the flag is off, `include_raw` is silently dropped (no raw block)
  and `browse select=raw_query` errors. Fallback: use the normalized `text` and re-case
  table names (recipe in `references/live-validation.md`).
- **Raw SQL is truncated at ~2,000 chars** (`... N more chars elided`) on every tool — a
  transport-safety cap, not a bug. Counts, paginated selects, and joins paste cleanly. A
  long **mega-projection ORM `SELECT`** gets elided — that elision is itself the tell of
  an **over-fetch rewrite candidate**. To EXPLAIN its access path, rebuild a minimal
  `SELECT 1 FROM <tables/joins> WHERE <predicates>` probe from the visible head; there is
  no MCP path to >2,000-char raw SQL.
- **`selectivity=n/a`** when `rows_sent=0` (UPDATE/DELETE, or a count that returned
  nothing). Use `rows_examined` vs `table_size` instead.
- **Drift:** a fingerprint may name a table absent on the connected server (different
  environment / dropped table). Verify with `search_schema`/`list_tables`; flag and
  skip — never fabricate a table name.
- **Do NOT read LanceDB directly.** Everything needed comes from the MCP tools; direct
  store reads are debug-only, never a workflow fallback.

## References
- `references/dbs-vector-notes.md` — engines, top_impacting vs browse vs search, the
  unified raw-query gate + 2,000-char truncation ceiling, NL-search weakness, and what to
  do when the MCP isn't exposing its tools.
- `references/live-validation.md` — re-casing recipe (flag-off fallback), EXPLAIN field
  guide, non-SELECT handling, selectivity heuristics, corpus↔live drift.
