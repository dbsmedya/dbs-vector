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

Streamable-HTTP MCP transport is not currently shipped — see the design
spec at `docs/superpowers/specs/2026-05-07-dynamic-engine-exposure-design.md`
for rationale and re-introduction notes.

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

`dbs-vector` registers one MCP tool per engine in `config.yaml`, plus
one `list_engines` discovery tool. Tool names follow the pattern
`search_<engine_name>` with dashes (`-`) replaced by underscores.

For the default `config.yaml` shipped with the project:

| Tool name | Engine | Family | Description |
|-----------|--------|--------|-------------|
| `search_md` | `md` | document | Markdown & Prose Document Engine (Gemma) |
| `search_sql` | `sql` | sql | SQL Slow Query Log Engine (Gemma) |
| `search_md_granite` | `md-granite` | document | Markdown & Prose (Granite, long context) |
| `search_sql_granite` | `sql-granite` | sql | SQL Slow Query Log (Granite) |
| `search_sql_api_granite` | `sql-api-granite` | sql | Remote slow query log API (Granite) |
| `list_engines` | — | — | Lists configured engines and tuning profiles |

### Search tools (per family)

**Document family** (`search_md`, `search_md_granite`, etc.) takes:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Semantic search query |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to a file path or pattern |

**SQL family** (`search_sql`, `search_sql_granite`, `search_sql_api_granite`) takes:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Natural language or partial SQL |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to a database name |
| `min_time` | float | no | Minimum cumulative execution time in ms |
| `min_lock_time` | float | no | Minimum cumulative lock time in seconds |
| `table_filter` | string | no | Restrict to queries that touch a specific table |
| `include_raw` | bool | no | Add a `Raw SQL:` block (verbatim query with literal values). Honoured **only** when the server was started with `--allow-raw-queries`; otherwise silently downgraded. See [Raw query exposure](#raw-query-exposure---allow-raw-queries). |

When `table_filter` is set the search bypasses the IVF approximate index
in favor of an exact flat scan, ensuring no candidate rows are missed
from unscanned IVF partitions. Combined with `min_lock_time > 0` this
answers focused investigation questions like "show me all queries that
lock `dt_customer_performance_report` rows" — the filter narrows the
universe; the embedding only ranks within it.

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

The two surfaces share one lock: verbatim `raw_query` leaves the process only
under `--allow-raw-queries`. The flag governs the **MCP server only** — the CLI
`dbs-vector browse --sql ...` path runs on your own terminal and is
unrestricted. Enable the flag only for a trusted, local model. This is an egress
setting, not a schema change, so it needs no re-ingest.

To pass it through Claude Desktop / Claude Code, append `"--allow-raw-queries"`
to the `args` array (Desktop) or the launch command (`claude mcp add ... -- uv
... run dbs-vector mcp --allow-raw-queries`).

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
  `browse_<engine>` tool for each SQL-family engine; and
  `register_discovery_tool(mcp)` adds the `list_engines` tool.
- All three registration helpers run inside
  `start_stdio_server(allow_raw_queries=...)` before `mcp.run()`. The
  `allow_raw_queries` flag (from the CLI `--allow-raw-queries` option) is
  threaded into both the search and browse registrars, so `raw_query` egress is
  gated identically on both surfaces. The helpers share an idempotency dict
  (`_dbs_vector_registrations`) attached to the FastMCP instance, keyed by tool
  name and recording `(engine, family, allow_raw_queries)`.
- All engines defined in `config.yaml` are loaded once at startup
  (transport-agnostic — `initialize_services()` is in
  `dbs_vector.mcp.state`). Each `dbs-vector mcp` process loads its own
  engine instances.
