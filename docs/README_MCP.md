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

## Transport Methods

`dbs-vector` supports two MCP transport methods:

| Method | Transport | Endpoint | Use case |
|--------|-----------|----------|----------|
| **stdio** | Process I/O | — | Single-user, no open ports, simplest setup |
| **Streamable HTTP** | HTTP POST | `http://127.0.0.1:8000/mcp` | Shared server, multiple clients, saves VRAM |

The old SSE transport (`/mcp/sse`) has been replaced by **Streamable HTTP**
(`/mcp`), which uses a single stateless POST endpoint instead of two
persistent connections. All modern MCP clients support this transport.

---

## Method 1: Standard I/O (stdio)

The AI assistant spawns `dbs-vector mcp` as a subprocess and communicates
over its standard input/output. No network ports are opened. Each client
process loads its own copy of the MLX models (~1.2 GB GPU memory each).

### Start manually (optional — for log inspection)

```bash
uv run dbs-vector mcp
```

Logs go to stderr; the MCP JSON-RPC stream goes to stdout.

---

## Method 2: Streamable HTTP

Start the FastAPI server once; all MCP clients connect to it and share the
same loaded MLX models.

```bash
uv run dbs-vector serve --port 8000
```

The MCP endpoint is mounted at:

```
http://127.0.0.1:8000/mcp
```

> Use `--host 0.0.0.0` to accept connections from other machines.
> The server sets `Access-Control-Allow-Origin: https://claude.ai` by default.

Verify the server is up:

```bash
curl http://127.0.0.1:8000/health
```

---

## Integrating with Claude Desktop

Claude Desktop supports both stdio and HTTP transport. Use stdio for a
self-contained local setup; use HTTP if you already have the server running.

### Option A — stdio (no server required)

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

### Option B — Streamable HTTP (server must be running)

For newer Claude Desktop builds that support the HTTP transport natively:

```json
{
  "mcpServers": {
    "dbs-vector": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

If your Claude Desktop version does not yet support `"type": "http"` directly,
use the `mcp-proxy` shim (no separate install needed with `uvx`):

```json
{
  "mcpServers": {
    "dbs-vector": {
      "command": "uvx",
      "args": [
        "mcp-proxy",
        "--transport", "streamablehttp",
        "http://127.0.0.1:8000/mcp"
      ]
    }
  }
}
```

### Verification

1. Restart Claude Desktop.
2. Look for the tools icon (bottom-right of the input box).
3. Confirm `search_documents` and `search_sql_logs` appear.
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

### Add via Streamable HTTP (server must be running)

```bash
claude mcp add --transport http dbs-vector http://127.0.0.1:8000/mcp
```

With explicit project scope:

```bash
claude mcp add --scope project --transport http dbs-vector http://127.0.0.1:8000/mcp
```

This writes the following to `.mcp.json`:

```json
{
  "mcpServers": {
    "dbs-vector": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
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

1. Start the API server: `uv run dbs-vector serve --port 8000`
2. Open **Cursor Settings → MCP → Add new MCP server**
3. Configure:
   - **Name**: `dbs-vector`
   - **Transport Type**: `HTTP`
   - **URL**: `http://127.0.0.1:8000/mcp`
4. Click **Connect**

---

## Tools Provided

### `search_documents`

Searches the document vector store (`md` engine) using hybrid
vector + full-text retrieval.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Semantic search query |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to a file path or pattern |

Returns formatted results with source path and content snippet.

### `search_sql_logs`

Searches the SQL query log vector store (`sql` engine).

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Natural language or partial SQL |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to a database name |
| `min_time` | float | no | Minimum execution time in ms |

Returns SQL queries with execution time, call count, and database source.

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
  "name": "search_documents",
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
  "name": "search_sql_logs",
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
| HTTP 404 at `/mcp` | Server not running | Start with `uv run dbs-vector serve` |
| HTTP 404 at `/mcp/sse` | Old SSE path | Update client URL to `/mcp` |
| Tools not appearing in Claude Desktop | Config path error or JSON syntax | Verify absolute paths; validate JSON |
| `mcp-proxy` not found | `uvx` not available | Install with `pip install uv` |

### Logs

**stdio mode** — logs go to stderr, not stdout (stdout is reserved for the
JSON-RPC stream):

```bash
# See logs in terminal
uv run dbs-vector mcp 2>&1 | head -50

# Claude Desktop logs
tail -f ~/Library/Logs/Claude/mcp.log
```

**HTTP mode** — logs appear in the terminal running `uv run dbs-vector serve`.

### Test the Streamable HTTP endpoint

```bash
# Start the server in one terminal
uv run dbs-vector serve --port 8000

# In another terminal — list available tools
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python3 -m json.tool
```

---

## Architecture Notes

- The MCP server is a `FastMCP` instance with `stateless_http=True`.
- In HTTP mode it is mounted at `/mcp` inside the FastAPI application via
  `app.mount("/mcp", mcp.streamable_http_app())`. A single POST endpoint
  handles all JSON-RPC calls; there are no persistent SSE connections.
- In stdio mode, `mcp_server.run()` is called directly from the `mcp` CLI
  command — no FastAPI or uvicorn is involved.
- Both modes share the same two tool implementations (`search_documents`,
  `search_sql_logs`) defined in `src/dbs_vector/api/mcp_server.py`.
- HTTP mode: all engines defined in `config.yaml` are loaded once at server
  startup and shared across all client connections (~1.2 GB VRAM total).
- stdio mode: each assistant process loads its own engine instances.
