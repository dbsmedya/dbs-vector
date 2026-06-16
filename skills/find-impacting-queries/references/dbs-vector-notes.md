# dbs-vector MCP — field notes

Practical notes for driving the dbs-vector slow-query tools, learned by exercising them.

## Tools per SQL engine
For each SQL engine in `config.yaml` the server exposes three tools:
- `top_impacting_<engine>` — **triage front door.** Top-`limit` fingerprints ranked by
  `impact_score = calls * execution_time_ms`, each row pre-formatted with curated scalar
  columns and (under the raw-query flag) a paste-ready exemplar. Params: `limit`,
  `table` (lowercased scope), `min_calls`, `order_by`, `include_raw`. Use this first.
- `browse_<engine>` — **analytical**, ranked by a column you choose (no query string).
  Filters: `id`, `content_hash`, `user`, `host`, `source`, `table`, `min_calls`,
  `min_execution_time_ms`, `min_lock_time_sec`; plus `group_by`, `order_by`, `select`,
  `limit`. Reach for it when you need grouping (`group_by=tables`) or columns/orderings
  `top_impacting` doesn't expose.
- `search_<engine>` — **semantic** retrieval, ranked by cosine similarity to a query
  string. Filters: `min_time`, `min_lock_time`, `table_filter`, `include_raw`. Use it
  for *concept* discovery, not ranking.
Plus one `list_engines` discovery tool.

Engines are named `sql`, `sql-api`, `sql-granite`, `sql-api-granite` → tools
`top_impacting_sql`, `browse_sql`, `search_sql`, etc.

## top_impacting columns
`id, tables, calls, execution_time_ms, impact_score, avg_ms_per_call, lock_time_sec,
rows_examined, rows_sent, selectivity, latest_ts` — plus a `Raw SQL:` block per row when
`include_raw=true` and the flag is on. `order_by` column must be one of `impact_score,
execution_time_ms, calls, lock_time_sec, avg_ms_per_call, selectivity` (default
`impact_score:desc`). This is the one-call replacement for the old browse-then-search
dance.

## browse columns
`id, content_hash, user, host, source, tables, calls, execution_time_ms, lock_time_sec,
rows_examined, rows_sent, latest_ts, text` (and `raw_query` only when the flag is on).
Non-grouped mode also exposes derived, selectable+orderable `impact_score`
(calls*execution_time_ms) and `selectivity` (rows_examined/rows_sent); `avg_ms_per_call`
is available in both modes. Grouped mode (`group_by`) yields `fingerprints` (COUNT), SUMs
of `calls/execution_time_ms/lock_time_sec/rows_examined/rows_sent`, `MAX(latest_ts)`, and
the per-execution averages `avg_ms_per_fingerprint`, `avg_ms_per_call`.

`order_by` syntax: `<col>[:asc|:desc]`, default `execution_time_ms:desc`. You cannot
`order_by` the `tables` list column — order by an aggregate or scalar instead.

## raw_query is gated — uniformly — and truncated
- **Gate:** verbatim `raw_query` leaves the process only when the server was started with
  `--allow-raw-queries`. This is now consistent across **all three** tools:
  `top_impacting`/`search` silently drop `include_raw` when the flag is off; `browse`
  rejects `select=raw_query` with "raw query text is not exposed on this engine; start
  the server with --allow-raw-queries to enable it."
- **Truncation:** every tool truncates raw SQL (and any long cell) at **2,000 chars**,
  appending `... (N more chars elided)`. There is no MCP path to longer raw SQL — it is a
  deliberate transport-safety cap (multi-MB `IN (...)` lists / multi-row INSERTs once
  dropped the stdio frame). Short/medium queries paste cleanly into EXPLAIN; a long
  mega-projection ORM SELECT will be elided (see SKILL.md "Friction & gotchas").

Raw SQL preserves **original identifier case** and **real literal values** — both of
which the normalized `text` column loses (it is lowercased with `?` placeholders), so
prefer it for EXPLAIN.

> `--allow-raw-queries` exposes verbatim production SQL (literal values, possibly PII).
> Enable only for a trusted local model. It is set in `.mcp.json` args and needs a
> server restart to take effect.

## Semantic search is tuned for clustering, not NL retrieval
The SQL engines embed with a symmetric `task: clustering | query:` prefix, so a natural-
language query string ("queries to orders") scores very low (~0.016) and can rank
unrelated queries highly. For "queries touching table X", scope with `table` on
`top_impacting`/`browse` (exact, against the lowercased `tables` list) — not the NL
string. Use the NL string only for *concept* discovery ("queries similar to a
subscription/invoice join").

## Other gotchas
- **`list_engines` reports `loaded` but no row count / emptiness.** Probe an engine with
  `top_impacting_<engine> limit:1` to learn whether it actually holds data.
- **MCP not exposing its tools?** If the dbs-vector (or MySQL) tools don't appear in the
  session even though the server is up, this is environment-specific (a stale/duplicate
  registration, a disabled server, or a restart needed) — **consult the user** rather
  than guessing or working around it.
