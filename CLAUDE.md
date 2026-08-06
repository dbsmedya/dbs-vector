# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run full validation suite (format, lint, typecheck, test)
uv run poe check

# REQUIRED before opening any PR — re-resolve every dependency to the newest
# version pyproject allows, then run the full gate against that resolution.
uv run poe refresh-deps

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

## Dependency Policy

Every dependency range in `pyproject.toml` is capped at the next major. Two
releases have shipped broken because an uncapped `>=` admitted a breaking
upstream version — transformers 5.13.0, then mcp 2.0.0 — and CI saw neither,
because CI installs `uv sync --frozen` (the pinned lockfile) while users get a
fresh resolution from these constraints.

Two guards close that gap, and both must stay:

- **The caps** stop a breaking major from resolving at all.
- **`uv run poe refresh-deps`** — mirrored by the `fresh-resolution` CI job,
  which also runs daily — re-resolves every dependency to the newest allowed
  version and runs the full gate against it. This is what catches a bad *minor*
  inside an allowed range, which is what transformers 5.13.0 was.

Run `refresh-deps` before opening a PR. If it fails, fix or pin the offender in
the same PR — never merge against a resolution users cannot install.

## Release Policy

Versions are `1.MINOR.PATCH`. **The middle number is the reindex signal.**

- **Minor (`1.7.0`)** — required for any change to what is *stored*. Ships with
  release notes naming the affected engines and stating `--rebuild --force`.
- **Patch (`1.7.1`)** — everything else. **A patch release must never change
  stored data.** Consumers rely on that to adopt fix releases immediately.
- **Major (`2.0.0`)** — breaking CLI, config-file, or MCP contract changes.

**Cut a minor if the change touches any of these.** The list is the test; when
unsure, it is a minor:

- chunker logic that can move a chunk boundary. Note the failure mode is silent:
  chunk hashes key on file content plus position, never chunk content, so when
  boundaries change but file bytes do not, both the full-ingest and watch/upsert
  paths skip every file and write nothing while logging success.
- embedding model, `passage_prefix` / `query_prefix`, or any `ModelRegistry` field
- LanceDB schema — this one is at least loud: `LanceDBStore` raises on mismatch and
  the CLI surfaces a `--rebuild --force` hint
- profile knobs feeding chunking (`chunk_target_tokens`, `chunk_max_tokens`,
  `chunk_max_chars`)
- stored-value normalization (table names, content hashing)

**Safe in a patch:** dependency pins and caps, docs, logging, error messages, and
CLI/MCP surface fixes that do not alter stored rows. `v1.2.1` (transformers cap
relaxed) and `v1.6.1` (dependency caps, tantivy and sqlglot dropped) are the
reference examples.

A reindex release also invalidates `similarity_floor` calibration. Say so in the
notes — that evidence is deployment-local, so only its owner can re-measure it.

## Architecture

This is a Clean Architecture, configuration-driven RAG search engine for Apple Silicon (MLX). The dependency flow is: **CLI/MCP → Services → Core Protocols → Infrastructure**.

### Layers

**`core/`** — Pure domain layer with no external dependencies.
- `models.py`: Pydantic domain models (`Document`, `Chunk`, `SqlChunk`, `SearchResult`, `SqlSearchResult`).
- `ports.py`: Protocol interfaces (`IEmbedder`, `IChunker`, `IVectorStore`, `IStoreMapper`) that decouple infrastructure from services.
- `registry.py`: `ComponentRegistry` maps string names from `config.yaml` to concrete mapper/chunker classes.

**`infrastructure/`** — Concrete implementations of the core protocols.
- `embeddings/mlx_engine.py`: `MLXEmbedder` — runs models on Apple GPU via MLX, casts tensors to NumPy via Unified Memory. Includes a process-level `_MODEL_CACHE` dict to avoid reloading models.
- `storage/lancedb_engine.py`: `LanceDBStore` — Arrow-native storage; uses `IVF_PQ` vector index + LanceDB native FTS. Schema mismatch on startup means `--rebuild --force` is needed.
- `storage/mappers.py`: `DocumentMapper` and `SqlMapper` convert domain chunks ↔ PyArrow `RecordBatch` for zero-copy ingestion and back to domain models on retrieval.
- `chunking/markdown_blocks.py`: `MarkdownBlockParser` — owns `markdown-it-py` configuration and token dispatch; descends into admonition (`!!!`/`???`) and blockquote containers (including GitHub-style `[!ALERT]` blockquotes), producing `_ScopedBlock`s that carry container scope and inherited frame labels. `document.py` never sees a `markdown_it.Token`.
- `chunking/document.py`: `DocumentChunker` — heading-aware, token-sized, four-phase pipeline: parse+descend (via `MarkdownBlockParser`) → pack (scope groups, per-group token budgets, code fences kept atomic) → forward fold (a wholly undersized section folds into the next one; both boundaries render with fold-only `(dbs-vector context: ...)` markers) → compose. Falls back to naive paragraph splitting for `.txt`.
- `chunking/sql.py`: `SqlChunker` — parses JSON slow query log format.
- `watch/watchdog_backend.py`: `WatchdogBackend` — the only module importing
  `watchdog`. Implements the `IWatchBackend` port.

**`services/`** — Orchestration, depend only on protocols.
- `ingestion.py`: `IngestionService` — reads files, chunks, deduplicates via SHA-256 content hashes, batches, embeds, and streams to `IVectorStore`.
- `search.py`: `SearchService` — embeds query and delegates hybrid search; also formats results for CLI output.
- `path_filter.py`: `PathFilter` — the single owner of ingestion path scoping
  (roots, extensions, `ignore_patterns`, gitignore). Shared by CLI discovery,
  watcher events and reconciliation; `None` for non-document engines.
- `watcher.py`: `WatcherService` — debounce map keyed by `(engine, path)`,
  one worker thread that serializes every LanceDB write, 60s FTS refresh timer.

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
   names per-engine content filters; built-ins are `excalidraw`,
   `compressed_json`, and `gitignore` (marker; discovery enforcement is in
   `PathFilter`). Register custom filters via `FilterRegistry.register`
   in `src/dbs_vector/infrastructure/chunking/filters.py`.

Memory budget auto-detected from `mlx.core.metal.device_info()`; override
via `system.memory_budget_gb`.

Module-level `settings = Settings(_env_file=None)` performs zero file I/O at
import (no YAML, no `.env`); CLI callback / API lifespan call
`load_settings(config_file, validate=True)` explicitly and copy fields onto
the singleton via `_populate_singleton_from()`. This makes
`dbs-vector --help` / `--version` survive a malformed or absent `config.yaml`.

Adding a new engine: see `docs/README_PROFILES.md`.

### Key Design Details

- **Deduplication**: Content hashes (SHA-256 truncated to 16 chars) are computed at the file level and stored per chunk. Ingestion skips chunks whose hash already exists in the store.
- **Schema evolution**: If `LanceDBStore` detects a schema mismatch on startup, it raises a descriptive `ValueError` that the CLI surfaces with a `--rebuild --force` hint.
- **Table names**: SQL table names are stored and displayed in original, case-sensitive,
  schema-qualified form (`TryOTODyn.MagentoOrders`) — one copy, no FTS over table names.
  `table_filter` is an equality filter that is case- and schema-insensitive whole-name:
  both the input and each stored name are run through `_normalize_table_name` (lowercase
  + strip schema) at query time and compared exactly. Search resolves matching ids via
  an in-memory `(id, tables)` scan and an `id IN (...)` prefilter; above
  `_TABLE_FILTER_PREFILTER_CAP` it switches to an exact vector-only full-scan ranking
  fallback. `count_matching` uses the same exact scan; browse/top_impacting normalize
  in Polars. `_clean_table_name` (quote/backtick strip, case preserved) populates stored
  data.
- **Asymmetric embeddings**: `MLXEmbedder` prepends different prefixes for passages (`passage_prefix`) vs queries (`query_prefix`), supporting instruction-tuned models like `embeddinggemma`.
- **Thread safety**: `MLXEmbedder` uses a per-model `threading.Lock`.
- **IVF_PQ indexing**: Only created when `total_rows > 256`; partitions scale as `sqrt(total_rows)` capped at 256.
- **Directory watch**: engines with `watch.enabled` re-ingest file changes
  while `dbs-vector mcp` runs. Config lives on the engine (`paths`,
  `ignore_patterns`, `exclusion_filters: [gitignore]`) plus a `watch:` block
  for mechanics only. The index is a rebuildable cache: **after any config
  change to a watched engine, run one `ingest --rebuild --force`**. See
  `docs/README_WATCH.md`.
- **Search scoring**: every result carries `similarity` (exact cosine between query and chunk vectors, computed in NumPy at search time — metric-independent, covers FTS-only rows), `retrieved_by` (channel membership: `both`/`vector`/`fts`), and `rrf_score` (fused RRF value, JSON/debug only). Ranking stays hybrid RRF(K=60); `_build_hybrid` pins `.metric("cosine")`. Admission policy lives in `SearchService` (engine `similarity_floor` / per-call `min_similarity` / `disable_similarity_floor`), which returns a `SearchResponse` envelope (`results`, `floor`, `inspected`, `best_rejected`). Construct services via `build_search_service()` — never hand-wire `SearchService`.
- **Floor calibration**: `similarity_floor` is deployment-local evidence about one corpus and is never committed here — query sets, choice records, and reports live outside the repo (this deployment keeps them in the gitignored `.ayder/`). Every engine ships unset; the `md`/`md-granite` measurements found no safe floor, and unset is a valid outcome. The shipped workflow is `docs/README_CALIBRATE_CORPUS.md`. A different corpus or any model/prefix/chunker/profile, `nprobes`, admission-policy, or pool-geometry change requires recalibration.

### Documentation Layout

**`docs/` is for user-facing guides only.** Everything published there is
written for someone who has never seen this repository's history: how to
configure an engine, how to run a workflow, what a knob means. It must stay
readable without any internal context.

**All internal project documentation lives in `.ayder/superpowers_<YYYYMMDD>/`**
— one gitignored, date-slugged directory per working session:

```
.ayder/superpowers_<YYYYMMDD>/
  specs/          # design specs
  plans/          # implementation plans
  calibration/    # deployment-local measurement evidence
```

Never write a spec, plan, brainstorm, review, session note, or measurement
report into `docs/`. Never commit one. This covers superpowers artifacts and
anything else that documents *how the work happened* rather than *how to use
the result*.

Two consequences worth stating explicitly:

- **Deployment-local evidence never ships.** Calibration query sets, choice
  records, and reports measure one corpus and are meaningless — actively
  misleading — to anyone else. The generic *harness* and *guide* ship; the
  numbers stay local. Same test for anything else that is true only of this
  machine's corpus or config.
- **Shipped docs and code must not cite internal paths.** A reference to a
  spec or plan is a link a user cannot follow, and it rots the moment the
  session directory is archived. Point at the relevant `docs/README_*.md`
  instead — including in error messages raised from `src/`.

### Test Structure

```
tests/
  unit/         # Mock-based, no I/O — fast
  integration/  # Uses tmpdir LanceDB + real chunkers/mappers
```

Mypy ignores `lancedb`, `pyarrow`, and `mlx_embeddings` (no stubs). Ruff enforces pycodestyle, pyflakes, bugbear, pyupgrade, and isort at line length 100.
