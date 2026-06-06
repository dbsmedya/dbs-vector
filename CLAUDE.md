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
uv run dbs-vector mcp

# Granite engine commands
uv run dbs-vector ingest "docs/" --type md-granite
uv run dbs-vector ingest "slow_log.duckdb" --type sql-granite
uv run dbs-vector ingest "http://<host>/api/v1" --type sql-api-granite
```

## Architecture

This is a Clean Architecture, configuration-driven RAG search engine for Apple Silicon (MLX). The dependency flow is: **CLI/MCP → Services → Core Protocols → Infrastructure**.

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

**`mcp/`** — MCP presentation layer.
- `server.py`: FastMCP instance + `start_stdio_server()` entry point.
- `dynamic_tools.py`: `register_search_tools(mcp)` — pre-flight atomic, idempotent, collision-safe.
- `discovery.py`: `register_discovery_tool(mcp)` + `list_engines` tool.
- `state.py`: `_services` dict + `initialize_services()` (transport-agnostic).
- `families/`: `SearchFamily` Protocol + `FamilyRegistry` + built-in `DocumentFamily` and `SqlFamily`.

The MCP server (`dbs-vector mcp`, stdio transport) registers one
`search_<engine>` tool per engine in `config.yaml`, plus a `list_engines`
discovery tool. Granite engines (`md-granite`, `sql-granite`,
`sql-api-granite`) are reachable as MCP tools `search_md_granite`,
`search_sql_granite`, `search_sql_api_granite`. Adding an A/B variant
(e.g., `md-granite-experimental`) requires only a `config.yaml` edit —
no source code changes. See `docs/README_MCP.md` and
`docs/README_PROFILES.md` for the workflow.

FastAPI has been removed. There is no HTTP REST surface; the streamable-
HTTP MCP transport is also not shipped. `dbs-vector serve` is gone.
`initialize_services()` (in `dbs_vector.mcp.state`) loads every
configured engine eagerly at server startup.

**`config.py`** — `Settings` (pydantic-settings) + `EngineConfig` per engine. Loaded from `config.yaml` at startup. Env prefix: `DBS_`. The path can be overridden with `--config-file` or `DBS_CONFIG_FILE` env var. dtype casting behaviour (formerly `attention_mask_dtype` on `EngineConfig`) is now a per-model contract stored in `ModelRegistry` (`core/model_registry.py`) as part of `ModelContract`.

### Configuration-Driven Registry Pattern

Adding a new engine type requires:
1. Implement `IChunker` and `IStoreMapper` concrete classes.
2. Register them in `ComponentRegistry._chunkers` / `ComponentRegistry._mappers`.
3. Add the engine block to `config.yaml` with `mapper_type`, `chunker_type`, `model:` (a `ModelRegistry` key), and `tuning_profile:` (a `profiles:` block key). The fields `model_name`, `vector_dimension`, `max_token_length`, and `attention_mask_dtype` are no longer per-engine fields — they live in `ModelRegistry`.

No changes to services, CLI, or MCP layer are needed.

### Tuning Profiles & Model Registry

Three layers for engine config:

1. **`ModelRegistry` (code, hardcoded)** — `core/model_registry.py` carries
   `vector_dimension`, `model_max_token_length`, `attention_mask_dtype`,
   `compute_dtype_bytes` per model. Adding a model is a `register()` call.
   Built-ins: `gemma-bf16`, `granite-r2`.

2. **`profiles:` block in `config.yaml`** — five numeric knobs per profile:
   `max_token_length` (truncation safety net), `chunk_max_chars` (`.txt`
   fallback only), `batch_size`, and for document engines:
   `chunk_target_tokens` / `chunk_max_tokens` (both **required > 0** for
   `chunker_type: "document"`; unused/inert for SQL profiles). Coherence
   enforced at load: `chunk_target_tokens ≤ chunk_max_tokens ≤ max_token_length`.
   Validated against the engine's model + Metal memory budget at load time.

3. **`engines:` block in `config.yaml`** — references `model:` (registry key)
   and `tuning_profile:` (profile name). Holds pipeline shape (mapper,
   chunker, table, workflow) and prefixes (which vary per engine for the
   same underlying model). Optional `exclusion_filters` list (default `[]`)
   names per-engine content filters; built-ins are `excalidraw` and
   `compressed_json`. Register custom filters via `FilterRegistry.register`
   in `src/dbs_vector/infrastructure/chunking/filters.py`.

Memory budget auto-detected from `mlx.core.metal.device_info()`; override
via `system.memory_budget_gb`.

Module-level `settings = Settings(_env_file=None)` performs zero file I/O at
import (no YAML, no `.env`); CLI callback / API lifespan call
`load_settings(config_file, validate=True)` explicitly and copy fields onto
the singleton via `_populate_singleton_from()`. This makes
`dbs-vector --help` / `--version` survive a malformed or absent `config.yaml`.

Adding a new engine: see spec
`docs/superpowers/specs/2026-05-06-tuning-profiles-design.md`.

### Key Design Details

- **Deduplication**: Content hashes (SHA-256 truncated to 16 chars) are computed at the file level and stored per chunk. Ingestion skips chunks whose hash already exists in the store.
- **Schema evolution**: If `LanceDBStore` detects a schema mismatch on startup, it raises a descriptive `ValueError` that the CLI surfaces with a `--rebuild --force` hint.
- **Asymmetric embeddings**: `MLXEmbedder` prepends different prefixes for passages (`passage_prefix`) vs queries (`query_prefix`), supporting instruction-tuned models like `embeddinggemma`.
- **Thread safety**: `MLXEmbedder` uses a per-model `threading.Lock`.
- **IVF_PQ indexing**: Only created when `total_rows > 256`; partitions scale as `sqrt(total_rows)` capped at 256.

### Test Structure

```
tests/
  unit/         # Mock-based, no I/O — fast
  integration/  # Uses tmpdir LanceDB + real chunkers/mappers
```

Mypy ignores `lancedb`, `pyarrow`, and `mlx_embeddings` (no stubs). Ruff enforces pycodestyle, pyflakes, bugbear, pyupgrade, and isort at line length 100.
