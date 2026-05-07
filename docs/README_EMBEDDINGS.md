# Embedding Models

This document covers the embedding-model layer in `dbs-vector`: which models are supported, how to configure them, and the runtime safeguards (truncation alarm, attention-mask dtype) that keep them honest.

## Tuning Profiles

Each engine references a `tuning_profile:` key in `config.yaml`. A profile is three numeric knobs that control runtime memory and chunking behaviour for that engine:

| Field | What it controls |
|---|---|
| `max_token_length` | Hard truncation limit sent to the tokenizer (must be ≤ the model's `model_max_token_length` in `ModelRegistry`). |
| `chunk_max_chars` | Target character budget per chunk for prose accumulation (`0` = atomic / no splitting, used for SQL). |
| `batch_size` | Number of chunks per MLX forward pass. Tuned to fit the model's embedding matrix in Metal memory budget. |

Profiles are validated at load time against the engine's model contract and the available Metal memory budget (auto-detected from `mlx.core.metal.device_info()`; override with `system.memory_budget_gb`).

Example from `config.yaml`:

```yaml
profiles:
  gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}

engines:
  md:
    description: "Markdown engine"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault"
    workflow: "md_search"
    passage_prefix: "title: none | text: "
    query_prefix: "task: search result | query: "
    tuning_profile: "gemma-md"
```

## Supported Models

| Model | HF repo | Dim | Context | Languages | Engines using it |
|---|---|---|---|---|---|
| **Embeddinggemma 300m** | `mlx-community/embeddinggemma-300m-bf16` | 768 | 2 048 | English-leaning | `md`, `sql`, `sql-api` |
| **Granite 311m R2** | `ibm-granite/granite-embedding-311m-multilingual-r2` | 768 | 32 768 | 200+ (52 enhanced, incl. Turkish) | `md-granite`, `sql-granite`, `sql-api-granite` |

Both load through the same `MLXEmbedder`. Granite is loaded from the original IBM HF repo — `mlx-embeddings` consumes it directly because Granite R2 is ModernBERT-based and ModernBERT is in the supported architecture list.

## Symmetric vs Asymmetric Encoders

- **Embeddinggemma is asymmetric**. It expects a task-instruction prefix on every input. The `md` engine uses `passage_prefix: "title: none | text: "` for documents and `query_prefix: "task: search result | query: "` for queries; `sql`/`sql-api` use `task: clustering | query: ` for both. These prefixes are stripped before storage in LanceDB (only the original text is indexed in the FTS column).
- **Granite is treated as symmetric**. The model card examples don't specify task instructions, and the spike that introduced the model into this project showed sane semantic similarity (cos-sim 0.66 across paraphrases vs 0.42 vs unrelated) without any prefixes. The Granite engine blocks therefore set both `passage_prefix` and `query_prefix` to empty strings. **This is a project implementation choice based on the model card examples and verified spike behaviour, not a contractual term from the card itself.** If a future Granite revision documents prefixes, update the engine config.

## `attention_mask_dtype` (ModelRegistry, not per-engine config)

Some MLX models require the tokenizer's `attention_mask` to be cast to a specific dtype before the forward pass to avoid type-promotion errors. This is now a per-model contract stored in `ModelRegistry` (`core/model_registry.py`) as part of `ModelContract` — it is **not** a field in `EngineConfig` or `config.yaml`.

| Value | Behaviour |
|---|---|
| `None` (default in `ModelContract`) | No cast. Used for ModernBERT-based encoders like Granite. |
| `"float16"` | Mask cast to `mx.float16`. Set in the `gemma-bf16` contract. |
| `"bfloat16"` | Mask cast to `mx.bfloat16`. |
| `"float32"` | Mask cast to `mx.float32`. |

If a model's contract is misconfigured, the next ingest/search will raise a `RuntimeError` whose message names the offending model — both the eager forward path and the lazy `np.array(...)` materialization path are wrapped, so the diagnostic is reliable.

## Runtime Truncation Alarm

`MLXEmbedder` performs an extra (cheap) tokenizer pass per batch with `truncation=False` to compute true input lengths. If any input exceeds `max_token_length`, a single `WARNING` line is logged:

```
Truncating 3/64 inputs above max_token_length=8192 for model 'ibm-granite/...' (longest observed: 11432 tokens, includes task prefix). Consider raising max_token_length or lowering chunk_max_chars.
```

Use this signal to tune `max_token_length` and `chunk_max_chars` in the relevant `profiles:` block in `config.yaml` — the alarm is the project's evidence-based way to decide when to spend more memory on longer context.

## MCP Exposure

`dbs-vector` no longer ships a FastAPI HTTP surface. `dbs-vector mcp` is the only presentation server and uses stdio transport.

Every configured engine is exposed as an MCP tool named `search_<engine_name>`, with dashes replaced by underscores:

| Engine | MCP tool |
|---|---|
| `md` | `search_md` |
| `sql` | `search_sql` |
| `sql-api` | `search_sql_api` |
| `md-granite` | `search_md_granite` |
| `sql-granite` | `search_sql_granite` |
| `sql-api-granite` | `search_sql_api_granite` |

`list_engines` returns the loaded engine metadata, including model key, model repo, profile, table name, and tool name. MCP startup eagerly initializes every configured engine, so multiple Granite/Gemma variants increase startup time and memory use by design.

## Adding a New Model

1. Identify the architecture (BERT, XLM-RoBERTa, ModernBERT, …) and confirm `mlx-embeddings` supports it.
2. Call `ModelRegistry.register()` in `core/model_registry.py` with a short key (e.g. `"my-model"`), the HF repo path, `vector_dimension`, `model_max_token_length`, and `attention_mask_dtype` (only if the model needs a non-default cast).
3. Add one or more `profiles:` entries in `config.yaml` with `max_token_length`, `chunk_max_chars`, and `batch_size` tuned to your Metal memory budget.
4. Add an engine block in `config.yaml` referencing `model: "my-model"` and `tuning_profile: "<your-profile>"`. If the model is symmetric, leave `passage_prefix` and `query_prefix` empty; if asymmetric, copy the prefixes verbatim from the model card.
5. Run a smoke ingest+search via the CLI. Watch for the truncation-alarm warnings.
6. (Optional, recommended) Add a slow-marked integration test under `tests/integration/`.

## Migration from the pre-tuning-profiles schema

If you have an older `config.yaml`, the following fields have moved:

| Legacy field (per-engine) | Where it lives now |
|---|---|
| `model_name: "mlx-community/..."` | `model: "<registry-key>"` in the engine block; HF path is in `ModelRegistry` |
| `vector_dimension` | `ModelContract.vector_dimension` in `core/model_registry.py` |
| `max_token_length` | `max_token_length` in the `profiles:` block |
| `attention_mask_dtype` | `ModelContract.attention_mask_dtype` in `core/model_registry.py` |
| `chunk_max_chars` | `chunk_max_chars` in the `profiles:` block |
| `system.batch_size` (global) | `batch_size` in each `profiles:` block |
