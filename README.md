# ⚡️ dbs-vector

**A High-Performance, Arrow-Native Local Codebase Search Engine and MCP Stdio Server for Apple Silicon.**

`dbs-vector` is a optimized Retrieval-Augmented Generation (RAG) search engine designed specifically for macOS (M-Series chips). It bypasses traditional Python serialization bottlenecks by utilizing Apple's Unified Memory Architecture (UMA) and pure Apache Arrow data pipelines.

It enables lightning-fast, hybrid (Vector + Full-Text) search across your local codebase, entirely offline.

---

## ✨ Features

*   **Zero-Copy Memory Pipelines**: Uses **MLX** to compute embeddings on the Mac GPU, casting the resulting tensors instantly into NumPy arrays via Unified Memory without costly `float` object instantiation.
*   **Arrow-Native Storage**: Uses **LanceDB** to stream ingestion batches directly to disk via PyArrow, avoiding the massive memory overhead of JSON and dictionary comprehensions.
*   **Hybrid Retrieval**: Simultaneously executes Approximate Nearest Neighbor (ANN) cosine vector search and native **Tantivy** Full-Text Search (FTS).
*   **Code-Aware Chunking**: Intelligently splits documentation and code, respecting markdown fences so that code blocks are indexed as atomic units.
*   **Production Robustness**: Features dynamic `IVF_PQ` indexing, Rust-level predicate pushdown (metadata filtering), and dataset compaction for delta-updates.
*   **Remote SQL API Ingestion**: `ApiChunker` pulls pre-aggregated slow-query records from any networked backend over HTTP, replacing local files with a paginated REST API — no changes to the embedding or storage layers.
*   **Dynamic MCP Tools**: `dbs-vector mcp` exposes one stdio MCP tool per configured engine, so Gemma, Granite, SQL, and future engines become available from `config.yaml`.

## 🚀 Installation

This project is built using `uv`, an extremely fast Python package manager.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dbsmedya/dbs-vector.git
   cd dbs-vector
   ```

2. **Install the CLI package:**
   ```bash
   uv sync
   ```
   *This automatically sets up the environment and creates the `dbs-vector` executable in your path.*

   Optional extras unlock additional ingestion sources:

   ```bash
   uv sync --extra sql  # DuckDB ingestion
   uv sync --extra api  # Remote HTTP API ingestion
   ```

## 💻 Usage

The application is entirely configuration-driven via `config.yaml`. It supports multiple data types (Engines) such as Markdown and SQL.

### Global Options
*   `--config-file` / `-c`: Path to your custom `config.yaml` (Defaults to `./config.yaml`).

### Ingesting Documents
Index markdown files, JSON SQL logs, DuckDB analytical files, or a remote HTTP slow-query API into the local vector store.

```bash
# Ingest all markdown files (default)
uv run dbs-vector ingest "docs/"

# Ingest SQL slow query logs (JSON format)
uv run dbs-vector ingest "slow_queries.json" --type sql

# Ingest SQL slow queries from DuckDB (High-Performance Columnar)
uv run dbs-vector ingest "slow_queries.duckdb" --type sql --rebuild

# Ingest from a remote HTTP API (paginated GET) — uses api_base_url from config.yaml
uv run dbs-vector ingest --type sql-api

# Or override the URL on the fly (without editing config.yaml):
uv run dbs-vector ingest "https://slow-log-api.internal/api/v1" --type sql-api

# Ingest via a custom SELECT sent to the remote API
uv run dbs-vector ingest "https://slow-log-api.internal/api/v1" --type sql-api \
  --query "SELECT fingerprint_id AS id, sanitized_sql AS text, db AS source, ..."
```

### Searching the Codebase
Execute queries against your chosen engine.

```bash
# Semantic hybrid search across markdown
uv run dbs-vector search "What is MLX?"

# Find similar slow queries (SQL clustering)
uv run dbs-vector search "SELECT * FROM users" --type sql --min-time 1000
```

> **Indexes are built automatically at the end of every `ingest` run.** Two indexes are created:
> - **IVF_PQ** vector index (only when the table has > 256 rows)
> - **Tantivy FTS** inverted index (required for hybrid search)
>
> If you see a *"Cannot perform full text search unless an INVERTED index has been created"* error, it means the FTS index was never built for your table. Fix it by re-running ingestion — use `--rebuild` to wipe and re-index from scratch:
> ```bash
> uv run dbs-vector ingest "docs/" --rebuild
> uv run dbs-vector ingest "slow_queries.json" --type sql --rebuild
> ```

For detailed specifications on each ingestion source, see:
👉 **[SQL Engine Documentation](docs/README_SQL.md)**
👉 **[DuckDB Ingestion Documentation](docs/README_duckdb.md)**
👉 **[Remote SQL API Ingestion](docs/README_REMOTE_SQL_API.md)**

### Model Context Protocol (MCP) Server
`dbs-vector` includes a built-in FastMCP server compatible with stdio-based MCP clients such as Claude Desktop and Claude Code.

```bash
# stdio — each client spawns its own dbs-vector process
uv run dbs-vector mcp
```

Each configured engine registers a tool named `search_<engine_name>` with dashes replaced by underscores, for example `search_md`, `search_sql`, and `search_md_granite`. Use the `list_engines` tool to inspect loaded engines, model contracts, profiles, and table names.

For setup instructions, see:
👉 **[MCP Server Documentation](docs/README_MCP.md)**

### Bundled Claude Skill: `slow-query-investigator`

`dbs-vector` ships a Claude Skill at `skills/slow-query-investigator/SKILL.md`
that teaches Claude how to route slow-query investigation questions ("show
me all queries that lock `<table>` rows", "what are the slowest queries on
`<table>`?", "where is our lock contention coming from?") to the right
combination of `table_filter` / `min_lock_time` / `min_time` parameters
on the SQL family MCP tools.

Once the skill is installed in your Claude Code workspace, natural-language
investigation requests get the right MCP tool call automatically. A future
extension will add minimum-sufficient index recommendations based on the
top-80% query traffic against a named table.

## 🏗 Architecture & Roadmap

`dbs-vector` is built upon strict **Clean Architecture** and **SOLID** principles. It utilizes a **Configuration-Driven Registry Pattern**, allowing new data engines (e.g., LibCST, Logs) to be added by simply updating `config.yaml` and registering new mappers/chunkers without modifying core orchestration logic.

### Engines

| Type | Model | Notes |
|---|---|---|
| `md` | embeddinggemma-300m-bf16 | Markdown/prose, default |
| `sql` | embeddinggemma-300m-bf16 | DuckDB slow-query log |
| `sql-api` | embeddinggemma-300m-bf16 | Remote slow-query API |
| `md-granite` | granite-embedding-311m-multilingual-r2 | 32K context, multilingual |
| `sql-granite` | granite-embedding-311m-multilingual-r2 | DuckDB log, Granite |
| `sql-api-granite` | granite-embedding-311m-multilingual-r2 | Remote API, Granite |

See `docs/README_EMBEDDINGS.md` for model details.

### Gemma vs Granite — which to use

Gemma engines (`md`, `sql`, `sql-api`) are the recommended default for most workloads: instruction-tuned with asymmetric search/clustering prefixes, fast on Apple Silicon, and consistently the strongest on English documentation. Reach for Granite engines (`md-granite`, `sql-granite`, `sql-api-granite`) when your corpus contains substantial non-English content (Granite R2 supports 200+ languages, Gemma 100+), when individual documents exceed Gemma's 2K-token context (Granite handles up to 32K), or when you want to A/B test chunk-size profiles against the Gemma baseline. Granite is a symmetric bi-encoder trained without instruction prefixes — leave `passage_prefix` and `query_prefix` empty when wiring a Granite engine. See [`docs/README_granite.md`](docs/README_granite.md) for tuning recipes and the rationale.

### Specialized Gemma Workflows
The project is optimized for instruction-tuned models like `embeddinggemma`. It supports asymmetric task-based workflows defined in `config.yaml`:
*   **Markdown (Search Result)**: Uses the `task: search result` prefix for queries and `title: none | text: ` for documents, maximizing retrieval accuracy for RAG.
*   **SQL (Clustering)**: Uses the `task: clustering` prefix for both ingestion and search, enabling high-precision semantic grouping of logically similar slow queries.

### Future Hardware Support (CUDA/TPU)
Because the core RAG orchestration relies exclusively on the `IEmbedder` Protocol, the application is strictly hardware-agnostic at its core. While currently optimized for Apple Silicon via `MLXEmbedder`, future deployment to cloud GPUs or Linux environments simply requires implementing a new `CudaEmbedder` (using PyTorch/Transformers) that returns standard NumPy arrays. No changes to the ingestion, storage, CLI, or MCP layers are necessary to support new hardware accelerators. No access to a CUDA hardware at the moment.

For a deep dive into the engineering, the Apache Arrow ingestion lifecycle, and the blueprint for AST/LibCST integration, see the official documentation:

👉 **[Architecture & Engineering Documentation](docs/README.md)**

## 🛠 Development

To contribute to `dbs-vector`, the project utilizes `poethepoet` as a task runner and implements strict quality gates (Ruff & Mypy).

```bash
# Run the entire validation suite (Format, Lint, Typecheck, Pytest)
uv run poe check

# Run tests with coverage
uv run poe test-cov
```
