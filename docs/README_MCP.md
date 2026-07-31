# Model Context Protocol (MCP) Server

`dbs-vector` includes a built-in MCP server that exposes semantic search over
your vector database as tools for any MCP-compatible AI assistant.

## Prerequisites

Before using the MCP server, ensure you have:

1. **Ingested data** into the vector store:
   ```bash
   # For documents
   uv run dbs-vector ingest "./docs/"

   # For SQL logs
   uv run dbs-vector ingest "slow_queries.json" --type sql
   ```

2. **macOS with Apple Silicon** (M1/M2/M3) — the MLX embedder requires Apple Silicon.

> **Note**: On first startup the MLX model (`embeddinggemma-300m-bf16`) is
> downloaded from HuggingFace (~600 MB). This happens once and is then cached.

---

## Transport

`dbs-vector` ships an MCP **stdio** transport only. The AI assistant
spawns `dbs-vector mcp` as a subprocess and communicates over its
standard input/output. No network ports are opened. Each client process
loads its own copy of the MLX models (~1.2 GB GPU memory each).

Streamable-HTTP MCP transport is not currently shipped.

---

## Standard I/O (stdio)

The AI assistant spawns `dbs-vector mcp` as a subprocess and communicates
over its standard input/output. No network ports are opened. Each client
process loads its own copy of the MLX models (~1.2 GB GPU memory each).

### Start manually (optional — for log inspection)

```bash
uv run dbs-vector mcp
```

Logs go to stderr; the MCP JSON-RPC stream goes to stdout.

---

## Integrating with Claude Desktop

### Configuration

Open the Claude Desktop config file:

```bash
# macOS
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

Add the `dbs-vector` entry:

```json
{
  "mcpServers": {
    "dbs-vector": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/dbs-vector",
        "run",
        "dbs-vector",
        "mcp",
        "--config-file",
        "/ABSOLUTE/PATH/TO/dbs-vector/config.yaml"
      ]
    }
  }
}
```

Replace `/ABSOLUTE/PATH/TO/dbs-vector` with the real path. Both
`--directory` and `--config-file` must be absolute.

### Verification

1. Restart Claude Desktop.
2. Look for the tools icon (bottom-right of the input box).
3. Confirm the per-engine tools (e.g., `search_md`, `search_sql`) appear.
4. Try: *"Search for how the ingestion pipeline works."*

---

## Integrating with Claude Code (CLI)

Claude Code reads MCP server config from three locations, selected by scope:

| Scope | File | Shared with team? |
|-------|------|-------------------|
| `local` (default) | `~/.claude.json` | No — machine-specific |
| `project` | `.mcp.json` in project root | Yes — check it in |
| `user` | `~/.claude.json` (home) | No — across all projects |

### Add via stdio

```bash
claude mcp add --transport stdio dbs-vector -- \
  uv --directory /ABSOLUTE/PATH/TO/dbs-vector \
     run dbs-vector mcp \
     --config-file /ABSOLUTE/PATH/TO/dbs-vector/config.yaml
```

To share with the team (adds to `.mcp.json`):

```bash
claude mcp add --scope project --transport stdio dbs-vector -- \
  uv --directory /ABSOLUTE/PATH/TO/dbs-vector \
     run dbs-vector mcp
```

### Manage servers

```bash
# List configured servers
claude mcp list

# Show details for one server
claude mcp get dbs-vector

# Remove a server
claude mcp remove dbs-vector
```

---

## Integrating with Cursor

Cursor's MCP integration currently expects an HTTP endpoint. Since
`dbs-vector` ships only stdio, Cursor cannot connect directly. If your
team needs Cursor support, you can wrap stdio with an external bridge
(e.g., `mcp-proxy`) — but this is not officially supported.

---

## Tools Provided

`dbs-vector` registers tools dynamically from `config.yaml`, plus one
`list_engines` discovery tool. Tool names follow `<verb>_<engine_name>`
with dashes (`-`) replaced by underscores. Each **document** engine gets
`search_` and `read_` tools; each **SQL** engine gets three — `search_` (semantic),
`browse_` (analytical), and `top_impacting_` (triage).

For the default `config.yaml` shipped with the project:

| Engine | Family | Tools |
|--------|--------|-------|
| `md` | document | `search_md`, `read_md` |
| `md-granite` | document | `search_md_granite`, `read_md_granite` |
| `sql` | sql | `search_sql`, `browse_sql`, `top_impacting_sql` |
| `sql-api` | sql | `search_sql_api`, `browse_sql_api`, `top_impacting_sql_api` |
| `sql-granite` | sql | `search_sql_granite`, `browse_sql_granite`, `top_impacting_sql_granite` |
| `sql-api-granite` | sql | `search_sql_api_granite`, `browse_sql_api_granite`, `top_impacting_sql_api_granite` |
| — | — | `list_engines` |

The three SQL verbs are complementary: **`search_`** ranks by semantic
similarity to a query string, **`browse_`** ranks analytically by any column
you choose (see [README_SQL.md](README_SQL.md)), and **`top_impacting_`** is
a one-call triage returning the highest-impact fingerprints ready for
diagnosis (below).

### Search tools (per family)

**Document family** (`search_md`, `search_md_granite`, etc.) takes:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Semantic search query |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to part of the corpus. Resolved in precedence order: the full stored path, then a trailing path fragment (`specs/api.md`, `api.md`), then a directory to scope to (`specs`, `docs/specs`) which matches every source beneath it. Still not a glob — no `*` or `%`. A value that resolves to nothing returns an explicit "matched no indexed source" message with the closest candidates, **never** a silent empty result. |
| `min_similarity` | float | no | Per-call admission floor override (range `[-1, 1]`). Takes precedence over the engine's configured `similarity_floor`. See [Similarity, ranking, and admission](#similarity-ranking-and-admission). |
| `disable_similarity_floor` | bool | no | Bypass admission filtering entirely — the exact unfloored baseline (no floor **and** the original, non-oversampled candidate pool). `min_similarity=0` is **not** equivalent. |

Every document result includes a `Chunk cursor`. The corresponding `read_`
tool performs an exact stored-text read when the caller needs surrounding
context:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `chunk_id` | string | yes | A `Chunk cursor` returned by the matching `search_` or prior `read_` tool. |
| `direction` | `previous` or `next` | yes | Which side of the anchor chunk to read. Navigation never crosses into another source document. |
| `count` | int | no | Number of adjacent chunks (default 1, maximum 3). |

The read tool does not embed text, invoke vector/FTS search, or include the
anchor again. It returns chunks in natural document order plus
`has_more`, `continuation_cursor`, and boundary metadata. Use it only when
the initially retrieved chunk needs more context; otherwise it adds no token
cost beyond the short cursor shown in each search result.

**SQL family** (`search_sql`, `search_sql_granite`, `search_sql_api_granite`) takes:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Natural language or partial SQL |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to one database by its stored name, matched case-sensitively. Unlike `table_filter`, no case/schema normalization is applied. An unresolvable name returns an explicit message naming the closest known databases, never a silent empty result. |
| `min_time` | float | no | Minimum cumulative execution time in ms |
| `min_lock_time` | float | no | Minimum cumulative lock time in seconds |
| `table_filter` | string | no | Restrict to queries that touch a specific table (case/schema-insensitive, whole-name exact match) |
| `include_raw` | bool | no | Add a `Raw SQL:` block (verbatim query with literal values). Honoured **only** when the server was started with `--allow-raw-queries`; otherwise silently downgraded. See [Raw query exposure](#raw-query-exposure---allow-raw-queries). |
| `min_similarity` | float | no | Per-call admission floor override (range `[-1, 1]`). Takes precedence over the engine's configured `similarity_floor`. See [Similarity, ranking, and admission](#similarity-ranking-and-admission). |
| `disable_similarity_floor` | bool | no | Bypass admission filtering entirely — the exact unfloored baseline (no floor **and** the original, non-oversampled candidate pool). `min_similarity=0` is **not** equivalent. |

When `table_filter` **or** `source_filter` is set the search bypasses the IVF
approximate index in favor of an exact flat scan, ensuring no candidate rows
are missed from unscanned IVF partitions. This matters because an exact-match
filter decides which rows may be *returned*, not which partitions are
*opened* — those are chosen by the query vector alone, so without the bypass
matching rows in unprobed partitions are never scored and vanish silently.
Combined with `min_lock_time > 0` this answers focused investigation questions
like "show me all queries that lock `dt_customer_performance_report` rows" —
the filter narrows the universe; the embedding only ranks within it.

### Similarity, ranking, and admission

Every search result — document or SQL family — carries three fields:

| Field | Meaning |
|-------|---------|
| `similarity` | Exact cosine similarity between the query and chunk embedding, computed in NumPy at search time. Range `[-1, 1]`. It is a consistent **geometric** scale, **not** a calibrated probability of relevance — comparisons are meaningful only within the same engine/configuration. |
| `retrieved_by` | Which retrieval channel(s) returned the row, rendered as `vector+fts`, `vector-only`, or `fts-only`. This is channel membership only — it is **not** evidence that the match is semantically or lexically correct. |
| `rrf_score` | The fused Reciprocal Rank Fusion (K=60) value that drove ranking. JSON/debug output only; never rendered in text surfaces. |

Result blocks render as:

```
--- Result (similarity 0.78, retrieved by: vector+fts) ---
```

**Ranking stays hybrid RRF fusion, not a similarity sort.** Results are ordered
by rank fusion, so display order can disagree with a plain sort by
`similarity` — `similarity` is exposed as evidence for the reader, not as the
sort key.

**Admission floor.** An engine may configure a `similarity_floor` in
`config.yaml` (see [README_PROFILES.md](README_PROFILES.md#engines-block));
any call can override it with `min_similarity` (range `[-1, 1]`, takes
precedence over the engine floor) or bypass it entirely with
`disable_similarity_floor=true`. When a floor is active, a candidate is
admitted when **either**:

1. `similarity >= floor` (semantic channel), **or**
2. every eligible query term appears verbatim in the chunk text — word
   boundary, case-insensitive, no stemming (an **all-terms verbatim** match,
   not phrase equality: token order/adjacency is not checked). Eligible
   tokens exclude a frozen 33-word English stopword set and tokens under 3
   characters; the lexical channel only fires on FTS-channel rows
(`retrieved_by` is `fts` or `both`).

No engine ships a floor: a safe value is a property of one corpus, not of the
software, so every `similarity_floor` starts unset and stays unset until
measured. See
[`README_CALIBRATE_CORPUS.md`](README_CALIBRATE_CORPUS.md) to calibrate your
own; the measurements stay with your deployment. A caller may still use
`min_similarity` deliberately for one attempt; that override is not a
calibrated corpus default.

**Known limitation, stated openly:** a single-common-token query (e.g.
`lock`) can still pass the gate against an unrelated file like `uv.lock` —
the gate trades that residual noise for exact-identifier recall. And because
FTS indexes only the `text` column, the gate protects filename/path recall
only when the filename also appears inside the chunk text.

`disable_similarity_floor=true` is the **exact unfloored baseline**: no
admission filtering **and** the original (non-oversampled) candidate-pool
size. `min_similarity=0` is **not** the same thing — it still drops
negative-similarity rows and still triggers the oversampled fetch described
next. When a floor is active (engine or per-call), the search fetches
`limit * 3` candidates per retrieval leg before admission filtering and
truncation, so filtering doesn't starve the requested `limit`. This enlarges
the RRF fusion inputs relative to the no-floor path — a deliberate,
spec-stated trade, not a bug.

**Empty responses.** An empty response after admission filtering means *no
inspected candidate passed the floor for this attempt* — a low-confidence
signal, **not** proof the corpus lacks relevant content (only the inspected
pool is known). Schematically:

```
No inspected candidate passed admission (similarity >= <floor> or all query
terms verbatim) for '<query>'. Retrieval confidence for this attempt is low;
this does not establish corpus-level absence. Retry with different terms or a
lower min_similarity if you expected a match.
```

**CLI JSON envelope.** `dbs-vector search --json` emits the full envelope,
not a bare result array:

```json
{
  "floor": "<configured or per-call floor>",
  "inspected": "<candidate count>",
  "best_rejected": {"similarity": "<exact cosine>", "source": "...", "retrieved_by": "fts"},
  "results": [ ... ]
}
```

`rrf_score` appears on each entry inside `results` — JSON/debug output only,
never in text output.

**Migration note.** Any consumer that used to parse `Score:` lines from
search output (e.g. the `find-impacting-queries` skill) must read
`similarity` from the new result block instead. `Score:` was LanceDB's RRF
fusion value; `similarity` is exact cosine.

### `top_impacting_<engine>` — impact triage (SQL family)

A one-call triage that returns the highest-impact slow-query fingerprints
ranked by **`impact_score = calls × execution_time_ms`** (frequency-weighted
"what is hammering the database"). Unlike `search_` it needs no query string;
unlike `browse_` it pre-selects the columns a tuning investigation needs and
(under the raw-query flag) appends a paste-ready exemplar for `EXPLAIN`.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `limit` | int | no | Top-N to return (default 10) |
| `table` | string | no | Scope to one table (case/schema-insensitive, whole-name exact match) |
| `order_by` | string | no | `<col>[:asc\|:desc]`, default `impact_score:desc`; col ∈ `impact_score, execution_time_ms, calls, lock_time_sec, avg_ms_per_call, selectivity` |
| `min_calls` | int | no | Drop fingerprints below this call count |
| `include_raw` | bool | no | Append a `Raw SQL:` exemplar block. Honoured **only** under `--allow-raw-queries`; otherwise silently downgraded. |

Each row carries: `id, tables, calls, execution_time_ms, impact_score,
avg_ms_per_call, lock_time_sec, rows_examined, rows_sent, selectivity,
latest_ts`. Note `rows_examined`/`rows_sent` are the **most-recent call's**
values (not averages or sums); `execution_time_ms` is cumulative. `raw_query`
and any long cell are truncated at ~2,000 chars for transport safety.

### `list_engines`

Returns a JSON-encoded array describing every configured engine: name,
family, model, description, table name, profile knobs
(`max_token_length`, `chunk_max_chars`, `batch_size`), MCP tool name,
and a `loaded` flag indicating whether the runtime service object is
currently registered. Useful for A/B-testing harnesses and for clients
that want to enumerate engines programmatically.

---

## Raw query exposure (`--allow-raw-queries`)

SQL fingerprints carry two SQL representations: the normalized `text`
(literals stripped — what embeddings are built on) and `raw_query`, the
**verbatim production SQL with real literal values** (PII). Because embeddings
never touch `raw_query`, whether it leaves the process is a pure egress
decision, controlled by one **server-level** flag on `dbs-vector mcp`:

```bash
uv run dbs-vector mcp                       # default: raw_query is NEVER exposed
uv run dbs-vector mcp --allow-raw-queries   # opt-in: raw_query exposable to the model
```

- **Initial state: OFF (fail-closed).** Without the flag, no MCP tool emits
  `raw_query`; the normalized `text` is always available.
- **`search_<engine>`** — the `include_raw=true` argument adds a `Raw SQL:`
  block only under `--allow-raw-queries`. With the flag off, `include_raw=true`
  is **silently downgraded**: the block is omitted and the call still succeeds
  (no error).
- **`browse_<engine>`** (one analytical tool per SQL engine; see
  [README_SQL.md](README_SQL.md)) — `select=raw_query` is **rejected with a
  validation error** unless the flag is on. The `raw_query` column also appears
  in the tool's self-description only when the flag is on.
- **`top_impacting_<engine>`** — `include_raw=true` appends the exemplar block
  only under the flag; otherwise **silently downgraded** (same contract as
  `search_`).

All three surfaces share one lock: verbatim `raw_query` leaves the process only
under `--allow-raw-queries`. The flag governs the **MCP server only** — the CLI
`dbs-vector browse --sql ...` path runs on your own terminal and is
unrestricted. Enable the flag only for a trusted, local model. This is an egress
setting, not a schema change, so it needs no re-ingest.

To pass it through Claude Desktop / Claude Code, append `"--allow-raw-queries"`
to the `args` array (Desktop) or the launch command (`claude mcp add ... -- uv
... run dbs-vector mcp --allow-raw-queries`).

---

## Diagnostic skills: find-impacting-queries + query-rewrite

Two bundled skills (under `skills/`) turn these tools into an end-to-end SQL
performance workflow. They pair `dbs-vector` — which says *which* queries hurt —
with a live **MySQL MCP** such as `mysql-mcp-server`, which says *why* (via
`EXPLAIN` / `list_indexes` / `table_size`). Everything is **read-only**: the
skills emit only `SELECT`/`SHOW`/`EXPLAIN` and hand index/rewrite suggestions to
a human.

### `find-impacting-queries`
Finds and diagnoses the costliest queries, index-first:

1. **Triage (one call):** `top_impacting_<engine>(include_raw=true)` →
   impact-ranked fingerprints with paste-ready exemplar SQL, `avg_ms_per_call`,
   and `selectivity` inline. Scope with `table=`.
2. **Validate live:** `EXPLAIN` each exemplar against the MySQL MCP, plus
   `list_indexes` / `table_size`.
3. **Classify:** every query lands in one of three outcomes —
   **INDEX FIX** (a concrete `CREATE INDEX`, the main deliverable),
   **ALREADY OPTIMAL / CONTENTION** (right index already used; the lever is
   call-rate, caching, or lock contention), or **REWRITE CANDIDATE** (no index
   helps → hand off).

### `query-rewrite`
Takes a **REWRITE CANDIDATE** and rewrites it for performance when an index
won't help (full-table aggregates, `SELECT *` + deep pagination, redundant
joins, ORM noise). Because a safe rewrite depends on business meaning the SQL
text doesn't carry, it **interviews the domain owner first**, then proposes
semantically-equivalent rewrites with a before/after plan and a
result-equivalence check for the human to run.

> Enable `--allow-raw-queries` for these skills so the exemplar SQL (with real
> literals) is available to `EXPLAIN`. Use a trusted, local model — `raw_query`
> may contain PII. See [Raw query exposure](#raw-query-exposure---allow-raw-queries).

The full workflow — the EXPLAIN field guide, the corpus↔live re-casing fallback,
and the friction notes — lives in `skills/find-impacting-queries/` and
`skills/query-rewrite/`.

---

## Migration from legacy tool names

`dbs-vector` previously exposed two hardcoded tools: `search_documents`
and `search_sql_logs`. Both are **removed** in this revision. Update
your MCP client config or LLM prompts:

- `search_documents` → `search_md`
- `search_sql_logs` → `search_sql`

The new naming convention covers every engine in `config.yaml`,
including the Granite variants which were previously unreachable.

## A/B testing tuning profiles

Adding an experimental engine variant requires only a config edit:

```yaml
profiles:
  granite-md-experimental: {max_token_length: 8192, chunk_max_chars: 3000, batch_size: 16}

engines:
  md-granite-experimental:
    description: "Granite, smaller chunks (A/B candidate vs md-granite)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite_exp"   # MUST differ from baseline
    workflow: "md_search_granite"
    tuning_profile: "granite-md-experimental"
```

After ingesting into the new engine and restarting `dbs-vector mcp`, a
new MCP tool `search_md_granite_experimental` becomes available. Use
`list_engines` to confirm both variants are loaded and to compare their
profile knobs in your evaluation report.

---

## Usage Examples

### Document search

Ask your assistant:

- *"Search for how the `IngestionService` is implemented."*
- *"Find documentation about MLXEmbedder configuration."*
- *"Search for Markdown files that explain the architecture."*

Internal tool call:

```json
{
  "name": "search_md",
  "arguments": {
    "query": "how to configure the MLX engine",
    "limit": 3
  }
}
```

### SQL log search

Ask your assistant:

- *"Find the SQL query used for calculating user retention."*
- *"Show queries that took longer than 500 ms and involve the orders table."*
- *"Find queries performing a JOIN between users and subscriptions."*

Internal tool call:

```json
{
  "name": "search_sql",
  "arguments": {
    "query": "join users and subscriptions",
    "min_time": 200
  }
}
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Document search service is not initialized" | No data ingested | Run `uv run dbs-vector ingest` first |
| "Failed to initialize search services" | Bad config or missing DB path | Check `config.yaml` and `db_path` |
| Slow first startup | Model downloading | Wait for MLX model download (~600 MB) |
| Tools not appearing in Claude Desktop | Config path error or JSON syntax | Verify absolute paths; validate JSON |

### Logs

**stdio mode** — logs go to stderr, not stdout (stdout is reserved for the
JSON-RPC stream):

```bash
# See logs in terminal
uv run dbs-vector mcp 2>&1 | head -50

# Claude Desktop logs
tail -f ~/Library/Logs/Claude/mcp.log
```

---

## Architecture Notes

- The MCP server is a `FastMCP` instance (`stateless_http=True`) created
  once in `src/dbs_vector/mcp/server.py`.
- Tool registration is dynamic: `register_search_tools(mcp, allow_raw_queries)`
  iterates `settings.engines` and registers one `search_<engine>` tool per
  engine via the family's `make_handler(engine_name, allow_raw_queries)`
  factory; `register_browse_tools(mcp, allow_raw_queries)` registers a
  `browse_<engine>` tool for each SQL-family engine;
  `register_triage_tools(mcp, allow_raw_queries)` registers a
  `top_impacting_<engine>` tool for each SQL-family engine; and
  `register_discovery_tool(mcp)` adds the `list_engines` tool.
- `register_read_tools(mcp)` registers a `read_<engine>` tool for each
  read-capable document family.
- All five registration helpers run inside
  `start_stdio_server(allow_raw_queries=...)` before `mcp.run()`. The
  `allow_raw_queries` flag (from the CLI `--allow-raw-queries` option) is
  threaded into the search, browse, and triage registrars, so `raw_query` egress
  is gated identically across all three surfaces. The helpers share an idempotency dict
  (`_dbs_vector_registrations`) attached to the FastMCP instance, keyed by tool
  name and recording `(engine, family, allow_raw_queries)`.
- All engines defined in `config.yaml` are loaded once at startup
  (transport-agnostic — `initialize_services()` is in
  `dbs_vector.mcp.state`). Each `dbs-vector mcp` process loads its own
  engine instances.
