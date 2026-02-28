# Model Context Protocol (MCP) Server

`dbs-vector` includes a built-in MCP server that allows AI assistants (like Claude Desktop, Cursor, and IDEs supporting MCP) to use your vector database as a tool for codebase and SQL query log search.

## Prerequisites

Before using the MCP server, ensure you have:

1. **Ingested data** into the vector store:
   ```bash
   # For documents
   uv run dbs-vector ingest "./docs/"

   # For SQL logs
   uv run dbs-vector ingest "slow_queries.json" --type sql
   ```

2. **macOS with Apple Silicon** (M1/M2/M3) - The current MLX embedder requires Apple Silicon.

> **Note**: On first startup, the MLX model (`embeddinggemma-300m-bf16`) will be downloaded from HuggingFace (~600MB). This may take a few minutes.

## Features

- **Semantic Document Search**: Search through indexed Markdown, Python, and other documents using natural language.
- **SQL Log Analysis**: Find historical SQL queries based on their intent or content.
- **Direct Integration**: No extra API server required; the MCP server communicates via standard input/output (stdio).

## Integration Methods

`dbs-vector` supports two different transport methods for connecting AI assistants: **Standard I/O (stdio)** and **Server-Sent Events (SSE)**.

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **stdio** | Claude Desktop, single user | Secure, no network ports | Each instance loads its own MLX model |
| **SSE** | Cursor, multiple clients | Shared instance, saves VRAM | Requires running API server |

### Method 1: Standard I/O (stdio)

In this method, the AI assistant runs the `dbs-vector` CLI as a hidden background process. This is the default and most secure method as it opens no network ports. However, each assistant instance will load its own copy of the MLX models.

#### Claude Desktop Configuration

Add the following to your Claude Desktop configuration file:

**macOS:**
```bash
# Open config file
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Configuration:**
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

> **Important**: Replace `/ABSOLUTE/PATH/TO/dbs-vector` with your actual installation path. Use absolute paths for both `--directory` and `--config-file`.

#### Verification

1. Restart Claude Desktop
2. Look for the 🔌 icon in the bottom-right corner
3. Click it to see available tools (`search_documents`, `search_sql_logs`)
4. Try asking: "Search for how the ingestion works"

### Method 2: Server-Sent Events (SSE)

If you are already running the API server (`uv run dbs-vector serve`), or if you want multiple AI assistants to connect to a single, shared instance (saving MLX VRAM), you can use the SSE endpoint.

#### Start the API Server

```bash
uv run dbs-vector serve --port 8000
```

The MCP endpoints are automatically mounted at `http://127.0.0.1:8000/mcp/sse`.

> The `--port` can be customized (default: 8000). Use `--host 0.0.0.0` to allow remote connections (not recommended for production).

#### Cursor Configuration

1. Open **Cursor Settings** → **MCP**
2. Click **Add new MCP server**
3. Configure:
   - **Name**: `dbs-vector`
   - **Transport Type**: `SSE`
   - **URL**: `http://127.0.0.1:8000/mcp/sse`
4. Click **Connect**

#### Other SSE Clients

For any MCP client supporting SSE transport, use:
- **Endpoint**: `http://127.0.0.1:8000/mcp/sse`
- **Session Messages**: `http://127.0.0.1:8000/mcp/messages/?session_id={SESSION_ID}`

## Tools Provided

### 1. `search_documents`

Searches the document vector store (`md` engine).

**Arguments:**
- `query` (string, **required**): The search query.
- `limit` (int, optional): Number of results (default: 5, max: 100).
- `source_filter` (string, optional): Filter by file path or pattern (e.g., `"*.py"` or `"docs/"`).

**Returns:** Formatted search results with source paths and content snippets.

### 2. `search_sql_logs`

Searches the SQL query log vector store (`sql` engine).

**Arguments:**
- `query` (string, **required**): The search query or partial SQL.
- `limit` (int, optional): Number of results (default: 5, max: 100).
- `source_filter` (string, optional): Filter by database name.
- `min_time` (float, optional): Filter by minimum execution time in milliseconds.

**Returns:** SQL queries with execution time, call count, and database source.

## Usage Examples

### Searching for Code or Documentation Context

You can ask your AI assistant questions like:
- "Search for how the `IngestionService` is implemented in our codebase."
- "Find any documentation about the `MLXEmbedder` configuration."
- "Search for Markdown files that explain the system architecture."

**Tool call example (internal):**
```json
{
  "name": "search_documents",
  "arguments": {
    "query": "how to configure MLX engine",
    "limit": 3
  }
}
```

### Finding Specific SQL Queries

You can ask your AI assistant:
- "Find the SQL query we used for calculating user retention."
- "Show me any queries that took longer than 500ms and involve the `orders` table."
- "Find queries that perform a `JOIN` between `users` and `subscriptions`."

**Tool call example (internal):**
```json
{
  "name": "search_sql_logs",
  "arguments": {
    "query": "join users and subscriptions",
    "min_time": 200
  }
}
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Document search service is not initialized" | No data ingested | Run `uv run dbs-vector ingest` first |
| "Failed to initialize search services" | Invalid config or missing DB path | Check `config.yaml` exists and `db_path` is valid |
| Slow first startup | Model downloading | Wait for MLX model download (~600MB) |
| Connection refused (SSE) | API server not running | Start with `uv run dbs-vector serve` |
| Tools not appearing in Claude | Config path error | Verify absolute paths in config |

### Logs

Since the MCP server uses `stdio` for communication, application logs and errors are redirected to `stderr`:

- **Claude Desktop**: `~/Library/Logs/Claude/mcp.log`
- **Terminal (stdio mode)**: Run manually to see logs:
  ```bash
  uv run dbs-vector mcp 2>&1
  ```
- **API Server**: Logs appear in the terminal running the server

### Testing Manually

Test the stdio server directly:
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | uv run dbs-vector mcp
```

Test the SSE endpoint:
```bash
# Terminal 1: Start server
uv run dbs-vector serve --port 8000

# Terminal 2: Test health
curl http://127.0.0.1:8000/health
```

## Architecture Notes

- The MCP server shares the same `SearchService` instances as the CLI and API
- Both `md` and `sql` engines are loaded at startup
- SSE mode allows multiple clients to share one model instance (saves ~1.2GB VRAM per client)
- stdio mode creates isolated instances per client
