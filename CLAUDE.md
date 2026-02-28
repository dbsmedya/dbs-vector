# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run full validation suite (format, lint, typecheck, test)
uv run poe check

# Run tests only
uv run poe test

# Run tests with coverage
uv run poe test-cov

# Run a single test file
uv run pytest tests/unit/test_chunker.py -v

# Run a single test by name
uv run pytest tests/unit/test_chunker.py::test_function_name -v

# Lint and format
uv run poe lint
uv run poe format

# Type checking
uv run poe typecheck

# CLI commands
uv run dbs-vector ingest "docs/" --type md
uv run dbs-vector ingest "queries.json" --type sql
uv run dbs-vector search "query text" --type md
uv run dbs-vector serve
uv run dbs-vector mcp
```

## Architecture

This is a Clean Architecture, configuration-driven RAG search engine for Apple Silicon (MLX). The dependency flow is: **CLI/API → Services → Core Protocols → Infrastructure**.

### Layers

**`core/`** — Pure domain layer with no external dependencies.
- `models.py`: Pydantic domain models (`Document`, `Chunk`, `SqlChunk`, `SearchResult`, `SqlSearchResult`).
- `ports.py`: Protocol interfaces (`IEmbedder`, `IChunker`, `IVectorStore`, `IStoreMapper`) that decouple infrastructure from services.
- `registry.py`: `ComponentRegistry` maps string names from `config.yaml` to concrete mapper/chunker classes.

**`infrastructure/`** — Concrete implementations of the core protocols.
- `embeddings/mlx_engine.py`: `MLXEmbedder` — runs models on Apple GPU via MLX, casts tensors to NumPy via Unified Memory. Includes a process-level `_MODEL_CACHE` dict to avoid reloading models.
- `storage/lancedb_engine.py`: `LanceDBStore` — Arrow-native storage; uses `IVF_PQ` vector index + Tantivy FTS. Schema mismatch on startup means `--rebuild --force` is needed.
- `storage/mappers.py`: `DocumentMapper` and `SqlMapper` convert domain chunks ↔ PyArrow `RecordBatch` for zero-copy ingestion and back to domain models on retrieval.
- `chunking/document.py`: `DocumentChunker` — uses `markdown-it-py` to parse `.md` semantically (code fences are kept atomic); falls back to naive splitting for `.txt`.
- `chunking/sql.py`: `SqlChunker` — parses JSON slow query log format.

**`services/`** — Orchestration, depend only on protocols.
- `ingestion.py`: `IngestionService` — reads files, chunks, deduplicates via SHA-256 content hashes, batches, embeds, and streams to `IVectorStore`.
- `search.py`: `SearchService` — embeds query and delegates hybrid search; also formats results for CLI output.

**`api/`** — FastAPI HTTP server + MCP stdio server.
- `main.py`: FastAPI app with lifespan startup (loads all engines from config), `/health`, `/search/md`, `/search/sql` endpoints. MCP is mounted at `/mcp` (SSE).
- `mcp_server.py`: `FastMCP` server exposing search as MCP tools.
- `state.py`: Shared `_services` dict (engine name → `SearchService`) used by both the FastAPI lifespan and the MCP command.

**`config.py`** — `Settings` (pydantic-settings) + `EngineConfig` per engine. Loaded from `config.yaml` at startup. Env prefix: `DBS_`. The path can be overridden with `--config-file` or `DBS_CONFIG_FILE` env var.

### Configuration-Driven Registry Pattern

Adding a new engine type requires:
1. Implement `IChunker` and `IStoreMapper` concrete classes.
2. Register them in `ComponentRegistry._chunkers` / `ComponentRegistry._mappers`.
3. Add the engine block to `config.yaml` with appropriate `mapper_type`, `chunker_type`, `model_name`, etc.

No changes to services, CLI, or API are needed.

### Key Design Details

- **Deduplication**: Content hashes (SHA-256 truncated to 16 chars) are computed at the file level and stored per chunk. Ingestion skips chunks whose hash already exists in the store.
- **Schema evolution**: If `LanceDBStore` detects a schema mismatch on startup, it raises a descriptive `ValueError` that the CLI surfaces with a `--rebuild --force` hint.
- **Asymmetric embeddings**: `MLXEmbedder` prepends different prefixes for passages (`passage_prefix`) vs queries (`query_prefix`), supporting instruction-tuned models like `embeddinggemma`.
- **Thread safety**: `MLXEmbedder` uses a per-model `threading.Lock`; FastAPI offloads synchronous search to `asyncio.to_thread`.
- **IVF_PQ indexing**: Only created when `total_rows > 256`; partitions scale as `sqrt(total_rows)` capped at 256.

### Test Structure

```
tests/
  unit/         # Mock-based, no I/O — fast
  integration/  # Uses tmpdir LanceDB + real chunkers/mappers
```

Mypy ignores `lancedb`, `pyarrow`, and `mlx_embeddings` (no stubs). Ruff enforces pycodestyle, pyflakes, bugbear, pyupgrade, and isort at line length 100.
