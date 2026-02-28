# ⚡️ dbs-vector

**A High-Performance, Arrow-Native Local Codebase Search Engine for Apple Silicon.**

`dbs-vector` is a optimized Retrieval-Augmented Generation (RAG) search engine designed specifically for macOS (M-Series chips). It bypasses traditional Python serialization bottlenecks by utilizing Apple's Unified Memory Architecture (UMA) and pure Apache Arrow data pipelines.

It enables lightning-fast, hybrid (Vector + Full-Text) search across your local codebase, entirely offline.

---

## ✨ Features

*   **Zero-Copy Memory Pipelines**: Uses **MLX** to compute embeddings on the Mac GPU, casting the resulting tensors instantly into NumPy arrays via Unified Memory without costly `float` object instantiation.
*   **Arrow-Native Storage**: Uses **LanceDB** to stream ingestion batches directly to disk via PyArrow, avoiding the massive memory overhead of JSON and dictionary comprehensions.
*   **Hybrid Retrieval**: Simultaneously executes Approximate Nearest Neighbor (ANN) cosine vector search and native **Tantivy** Full-Text Search (FTS).
*   **Code-Aware Chunking**: Intelligently splits documentation and code, respecting markdown fences so that code blocks are indexed as atomic units.
*   **Production Robustness**: Features dynamic `IVF_PQ` indexing, Rust-level predicate pushdown (metadata filtering), and dataset compaction for delta-updates.

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

## 💻 Usage

The application is entirely configuration-driven via `config.yaml`. It supports multiple data types (Engines) such as Markdown and SQL.

### Global Options
*   `--config-file` / `-c`: Path to your custom `config.yaml` (Defaults to `./config.yaml`).

### Ingesting Documents
Index your markdown files or SQL logs into the high-performance local vector store.

```bash
# Ingest all markdown files (default)
uv run dbs-vector ingest "docs/"

# Ingest SQL slow query logs (JSON format)
uv run dbs-vector ingest "slow_queries.json" --type sql

# Completely rebuild an index from scratch
uv run dbs-vector ingest "docs/" --rebuild
```

### Searching the Codebase
Execute queries against your chosen engine.

```bash
# Semantic hybrid search across markdown
uv run dbs-vector search "What is MLX?"

# Find similar slow queries (SQL clustering)
uv run dbs-vector search "SELECT * FROM users" --type sql --min-time 1000
```

For detailed specifications on the SQL engine and JSON format requirements, see:
👉 **[SQL Engine Documentation](docs/README_SQL.md)**

### Async API Server
The application includes a high-performance FastAPI server to expose the search engine over HTTP.

```bash
# Start the API server (loads all engines defined in config.yaml)
uv run dbs-vector serve
```

For full API specifications and swagger documentation, see:
👉 **[API Usage & Documentation](docs/README_API.md)**

### Model Context Protocol (MCP) Server
`dbs-vector` includes a built-in MCP server that allows AI assistants (like Claude Desktop and Cursor) to use your vector database as a tool for codebase and SQL query log search.

```bash
# Start the MCP stdio server
uv run dbs-vector mcp
```

For setup instructions and tool details, see:
👉 **[MCP Server Documentation](docs/README_MCP.md)**

## 🏗 Architecture & Roadmap

`dbs-vector` is built upon strict **Clean Architecture** and **SOLID** principles. It utilizes a **Configuration-Driven Registry Pattern**, allowing new data engines (e.g., LibCST, Logs) to be added by simply updating `config.yaml` and registering new mappers/chunkers without modifying core orchestration logic.

### Specialized Gemma Workflows
The project is optimized for instruction-tuned models like `embeddinggemma`. It supports asymmetric task-based workflows defined in `config.yaml`:
*   **Markdown (Search Result)**: Uses the `task: search result` prefix for queries and `title: none | text: ` for documents, maximizing retrieval accuracy for RAG.
*   **SQL (Clustering)**: Uses the `task: clustering` prefix for both ingestion and search, enabling high-precision semantic grouping of logically similar slow queries.

### Future Hardware Support (CUDA/TPU)
Because the core RAG orchestration relies exclusively on the `IEmbedder` Protocol, the application is strictly hardware-agnostic at its core. While currently optimized for Apple Silicon via `MLXEmbedder`, future deployment to cloud GPUs or Linux environments simply requires implementing a new `CudaEmbedder` (using PyTorch/Transformers) that returns standard NumPy arrays. No changes to the ingestion, storage, or API layers are necessary to support new hardware accelerators. No access to a CUDA hardware at the moment.

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

