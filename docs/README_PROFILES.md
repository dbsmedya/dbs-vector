# Model Profiles and `config.yaml`

This guide explains the profile-based configuration used by `dbs-vector`: what to set in `config.yaml`, which values are model contracts, and which values are safe to tune per machine.

## Configuration Layers

`config.yaml` has three relevant layers:

| Layer | Where | Purpose |
|---|---|---|
| System settings | `system:` | Global runtime settings such as LanceDB path, search probes, and optional memory budget override. |
| Tuning profiles | `profiles:` | Hardware-sensitive knobs: `max_token_length`, `chunk_max_chars`, and `batch_size`. |
| Engines | `engines:` | Data pipeline wiring: model key, chunker, mapper, table, workflow, prefixes, and profile selection. |

Model contract details do **not** live in `config.yaml`. The Hugging Face model path, vector dimension, model context cap, and attention-mask dtype are hardcoded in `src/dbs_vector/core/model_registry.py`.

## System Block

Use `system:` for global settings:

```yaml
system:
  db_path: "./lancedb_dbs_vector"
  nprobes: 20
  # Optional. If unset, dbs-vector tries to auto-detect the Metal max buffer.
  # memory_budget_gb: 22.0
```

Fields:

| Field | Default | What to set |
|---|---:|---|
| `db_path` | `./lancedb_dbs_vector` | LanceDB directory. Change this to keep separate indexes. |
| `nprobes` | `20` | LanceDB vector search probe count. Higher can improve recall with more latency. |
| `log_level` | `INFO` | Optional Loguru level override. |
| `log_serialize` | `false` | Optional JSON log serialization flag. |
| `memory_budget_gb` | unset | Optional Metal memory budget override. Set this if auto-detection fails or if you want a lower safety budget. |

Do not set `system.batch_size`; batching moved to profiles.

## Profiles Block

A profile controls how large each embedding batch can get:

```yaml
profiles:
  gemma-md:           {max_token_length: 2048,  chunk_max_chars: 1000, batch_size: 64}
  gemma-sql-atomic:   {max_token_length: 2048,  chunk_max_chars: 0,    batch_size: 64}
  granite-md-large:   {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  granite-sql-atomic: {max_token_length: 8192,  chunk_max_chars: 0,    batch_size: 32}
```

Profile fields:

| Field | Meaning | Practical guidance |
|---|---|---|
| `max_token_length` | Token limit passed to the model tokenizer. | Must be less than or equal to the model contract cap. Gemma cap is 2048. Granite R2 cap is 32768. |
| `chunk_max_chars` | Character target for document chunk accumulation. | Use a positive value for document engines. Use `0` for atomic SQL/API records. |
| `batch_size` | Number of chunks embedded per MLX forward pass. | Lower this first when memory validation fails or you hit runtime memory pressure. |

The validator checks every engine/profile pairing at config load. It rejects profiles that exceed the model context cap or the Metal memory budget and prints safer suggested values.

Validation walks all configured engines, not just the engine selected for the current command. If one local machine cannot load a Granite profile, either lower that profile's batch/context values, set a smaller explicit `system.memory_budget_gb` and follow the suggestion, or remove/comment the engine block on that machine.

## Engines Block

Each engine selects a model contract and a tuning profile:

```yaml
engines:
  md-granite:
    description: "Markdown & Prose Engine (Granite R2 - long context)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"
```

Required engine fields:

| Field | What to set |
|---|---|
| `description` | Human-readable label. |
| `model` | Model registry key, currently `gemma-bf16` or `granite-r2`. |
| `mapper_type` | Storage mapper, usually `document` or `sql`. |
| `chunker_type` | Input chunker, usually `document`, `duckdb`, or `api`. |
| `table_name` | LanceDB table for this engine. Use a different table when changing embedding model or workflow. |
| `workflow` | Stored workflow label for records. |
| `tuning_profile` | Key under `profiles:`. |

Optional engine fields:

| Field | When to set |
|---|---|
| `passage_prefix` / `query_prefix` | Required for asymmetric models like Gemma. Leave empty for Granite R2 in this project. |
| `duckdb_query` | Override the default DuckDB SQL extraction query. |
| `api_base_url`, `api_key`, `api_page_size`, `api_since_days`, `api_timeout_sec`, `api_min_execution_ms`, `api_database` | Remote SQL API chunker settings. |

## Built-In Model Keys

| Model key | Model | Vector dim | Max model context | Typical engines |
|---|---|---:|---:|---|
| `gemma-bf16` | `mlx-community/embeddinggemma-300m-bf16` | 768 | 2048 | `md`, `sql`, `sql-api` |
| `granite-r2` | `ibm-granite/granite-embedding-311m-multilingual-r2` | 768 | 32768 | `md-granite`, `sql-granite`, `sql-api-granite` |

Use Gemma for the default lightweight engines. Use Granite when you need longer context, stronger multilingual behavior, or the Granite-specific SQL/Turkish behavior this project targets.

## Choosing Profile Values

Start from one of the built-in profiles and change only one variable at a time.

For Markdown/document engines:

```yaml
profiles:
  my-granite-docs: {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}

engines:
  md-granite:
    description: "Markdown & Prose Engine (Granite R2 - long context)"
    model: "granite-r2"
    chunker_type: "document"
    mapper_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "my-granite-docs"
```

For SQL/DuckDB engines:

```yaml
profiles:
  my-granite-sql: {max_token_length: 8192, chunk_max_chars: 0, batch_size: 32}

engines:
  sql-granite:
    description: "SQL Slow Query Log Engine (Granite Clustering)"
    model: "granite-r2"
    chunker_type: "duckdb"
    mapper_type: "sql"
    table_name: "query_vault_granite"
    workflow: "sql_clustering_granite"
    tuning_profile: "my-granite-sql"
```

Rules of thumb:

- If validation says the profile would exceed memory, reduce `batch_size` first.
- If `batch_size: 1` still does not fit, reduce `max_token_length`.
- If truncation warnings appear during ingestion, either raise `max_token_length` or lower `chunk_max_chars`.
- For SQL engines, keep `chunk_max_chars: 0`; each SQL record should remain atomic.
- For document engines, keep `chunk_max_chars` well below the token window. The default Granite document profile uses 6000 chars for a 16384-token window to leave tokenizer slack.

## Common Examples

Smaller Granite document profile for tighter machines:

```yaml
profiles:
  granite-md-medium: {max_token_length: 8192, chunk_max_chars: 3000, batch_size: 16}
```

Longer Granite document profile with lower batch size:

```yaml
profiles:
  granite-md-long: {max_token_length: 32768, chunk_max_chars: 10000, batch_size: 1}
```

Low-memory Gemma profile:

```yaml
profiles:
  gemma-md-small: {max_token_length: 2048, chunk_max_chars: 800, batch_size: 16}
```

Then point an engine at the profile:

```yaml
engines:
  md:
    description: "Markdown & Prose Document Engine (Gemma Search)"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault"
    workflow: "md_search"
    passage_prefix: "title: none | text: "
    query_prefix: "task: search result | query: "
    tuning_profile: "gemma-md-small"
```

## Validation and Error Handling

`dbs-vector` validates the full config when a runtime command loads it:

```bash
uv run dbs-vector ingest docs/ --type md-granite
uv run dbs-vector search "slow query" --type sql-granite
```

The validator catches:

- unknown `model` keys;
- unknown `tuning_profile` keys;
- `max_token_length` above the model contract cap;
- profiles that exceed the Metal memory budget;
- old schema fields such as `system.batch_size`, `model_name`, `vector_dimension`, `attention_mask_dtype`, and per-engine `chunk_max_chars`.

If Metal memory cannot be detected, set an explicit budget:

```yaml
system:
  memory_budget_gb: 16.0
```

## Migration From Old Configs

Old config fields moved as follows:

| Old field | New location |
|---|---|
| `system.batch_size` | `profiles.<name>.batch_size` |
| `engines.<name>.model_name` | `engines.<name>.model` registry key |
| `engines.<name>.vector_dimension` | `ModelRegistry` contract |
| `engines.<name>.max_token_length` | `profiles.<name>.max_token_length` |
| `engines.<name>.attention_mask_dtype` | `ModelRegistry` contract |
| `engines.<name>.chunk_max_chars` | `profiles.<name>.chunk_max_chars` |

When an old config is loaded, `dbs-vector` raises one migration-hint error instead of a long Pydantic error list.

## Related Docs

- [README_EMBEDDINGS.md](README_EMBEDDINGS.md) covers supported embedding models and model-contract behavior.
- [README_DOCS.md](README_DOCS.md) covers Markdown chunking behavior.
- [README_SQL.md](README_SQL.md), [README_duckdb.md](README_duckdb.md), and [README_REMOTE_SQL_API.md](README_REMOTE_SQL_API.md) cover SQL ingestion sources.

## A/B testing tuning profiles

Because each engine name maps to a distinct MCP tool and a distinct
LanceDB table, you can run two profile variants of the same model side
by side and compare results without code changes.

### Step 1: Define a new profile

```yaml
profiles:
  granite-md-large:        {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  granite-md-experimental: {max_token_length: 8192,  chunk_max_chars: 3000, batch_size: 16}
```

### Step 2: Define a new engine that references the variant profile

```yaml
engines:
  md-granite:
    description: "Granite long-context, 6KB chunks (baseline)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"

  md-granite-experimental:
    description: "Granite, smaller chunks for higher recall (A/B candidate)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite_exp"   # MUST differ from baseline
    workflow: "md_search_granite"
    tuning_profile: "granite-md-experimental"
```

### Step 3: Ingest into both engines

```bash
uv run dbs-vector ingest "./docs/" --type md-granite
uv run dbs-vector ingest "./docs/" --type md-granite-experimental
```

### Step 4: Compare via the MCP tools

Start the server (`uv run dbs-vector mcp`) and run the same query through
both `search_md_granite` and `search_md_granite_experimental`. Use
`list_engines` to dump every engine's profile knobs into your
evaluation report so the comparison is reproducible.

### Memory note

Every engine in `engines:` is loaded eagerly at startup. Each
long-context Granite variant consumes roughly 1–2 GB of GPU memory.
Drop the experimental variant from `engines:` once you've decided
which configuration to keep.
