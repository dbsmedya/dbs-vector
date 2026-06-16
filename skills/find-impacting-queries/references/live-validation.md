# Live MySQL validation — field guide

How to validate a corpus fingerprint against the live database (via a MySQL MCP such as
`mysql-mcp-server`).

## Re-casing corpus identifiers → live schema (flag-off fallback)
With `--allow-raw-queries` on, the exemplar `raw_query` already carries original case —
EXPLAIN it directly and skip this whole section. Re-casing is only needed when raw SQL is
unavailable (flag off) and you must work from the normalized `text`.

The corpus lowercases identifiers. MySQL table names are case-sensitive when
`SHOW VARIABLES LIKE 'lower_case_table_names'` returns `0` (Linux default), so a
lowercased table name fails to resolve (`Error 1146 ... doesn't exist`).

- If you have **raw SQL**, the original case is already there — EXPLAIN it directly.
- If you only have **normalized text**, build a one-time map:
  ```sql
  SELECT table_schema, table_name
  FROM information_schema.tables
  WHERE table_schema = '<db>';
  ```
  Match each lowercased fingerprint table to its real name via `lower(table_name)`.
  Only **table** names need re-casing — MySQL **column** names are case-insensitive.
- Can't find the table at all? Use `search_schema(pattern='%name%')` across databases.
  If it lives in another schema, qualify it; if it exists nowhere, it's **drift** (the
  slow log came from a different environment / a dropped table) — flag and skip.

## EXPLAIN
- SELECT: `explain_query(sql, format="traditional")` for the index/rows table, or
  `format="tree"` for cost + access order.
- Non-SELECT (UPDATE/DELETE/REPLACE): `explain_query` is SELECT-only. Use
  `run_query(sql="EXPLAIN <statement>")` (a leading `EXPLAIN` is allowed), or rewrite the
  statement's WHERE into a `SELECT COUNT(*)` / `SELECT <pk>` to inspect the access path.
  On a `--super-read-only` replica even `EXPLAIN UPDATE/DELETE` is refused (Error 1290) —
  the WHERE→SELECT rewrite is then the only path.
- Literals only need to be **type-correct** to produce a plan (EXPLAIN does not execute).
  Get types from `information_schema.columns` when synthesizing. But prefer the
  exemplar's real literals — synthetic values distort row estimates (see caveat below).

### Reading the plan
| Field | What it tells you |
|---|---|
| `type` | Access method. `const`/`eq_ref` best → `ref`/`range` ok → `index`/`ALL` = scan, usually bad. |
| `key` | Index actually chosen. |
| `possible_keys` | Indexes the optimizer considered but may have rejected — a better one here is the smoking gun. |
| `rows` | Estimated rows examined (depends heavily on the literal — see caveat). |
| `filtered` | % of rows kept after the WHERE that the index didn't cover. |
| `Extra` | `Using where` = post-filter after a too-broad index; `Using filesort`, `Using temporary` = sort/aggregate cost; `Using index` = covering index (good). |

### Fidelity caveat (important)
The exemplar carries **one** real literal set, not necessarily the slow one. Example:
`select id from orders where order_id = '12345'` EXPLAINs as a clean single-row point
lookup — yet a different fingerprint on the same table is the aggregate offender because
its parameters (wide `created_at` ranges, multi-value `IN` lists) hit a worse plan. And a
synthetic literal (e.g. a made-up `customer_id`) can make EXPLAIN estimate `rows=1` and
look healthy while production examined tens of thousands.

So: rank by **aggregate `execution_time_ms`**, corroborate with **selectivity**
(`rows_examined / rows_sent` — high ratio = examining far more than it returns = index
opportunity), and treat the exemplar plan as evidence of *an* access path, not proof the
query is fine.

## Useful live calls
- `list_indexes(db, table)` — existing indexes (column order matters for composites).
- `table_size(db, table)` — rows + data/index MB; scale changes severity.
- `describe_table` / `show_create_table` — column types, constraints.
- `server_info(detailed=true)` — buffer-pool hit rate, slow-query count, threads.

## Read-only discipline
Only `SELECT`/`SHOW`/`DESCRIBE`/`EXPLAIN`. Emit `CREATE INDEX`/rewrites as text for a
human to review and apply. Never run DDL or DML against the live server.
