# Embedding Models

This document covers the embedding-model layer in `dbs-vector`: which models are supported, how to configure them, and the runtime safeguards (truncation alarm, attention-mask dtype) that keep them honest.

## Supported Models

| Model | HF repo | Dim | Context | Languages | Engines using it |
|---|---|---|---|---|---|
| **Embeddinggemma 300m** | `mlx-community/embeddinggemma-300m-bf16` | 768 | 2 048 | English-leaning | `md`, `sql`, `sql-api` |
| **Granite 311m R2** | `ibm-granite/granite-embedding-311m-multilingual-r2` | 768 | 32 768 | 200+ (52 enhanced, incl. Turkish) | `md-granite`, `sql-granite`, `sql-api-granite` |

Both load through the same `MLXEmbedder`. Granite is loaded from the original IBM HF repo — `mlx-embeddings` consumes it directly because Granite R2 is ModernBERT-based and ModernBERT is in the supported architecture list.

## Symmetric vs Asymmetric Encoders

- **Embeddinggemma is asymmetric**. It expects a task-instruction prefix on every input. The `md` engine uses `passage_prefix: "title: none | text: "` for documents and `query_prefix: "task: search result | query: "` for queries; `sql`/`sql-api` use `task: clustering | query: ` for both. These prefixes are stripped before storage in LanceDB (only the original text is indexed in the FTS column).
- **Granite is treated as symmetric**. The model card examples don't specify task instructions, and the spike that introduced the model into this project showed sane semantic similarity (cos-sim 0.66 across paraphrases vs 0.42 vs unrelated) without any prefixes. The Granite engine blocks therefore set both `passage_prefix` and `query_prefix` to empty strings. **This is a project implementation choice based on the model card examples and verified spike behaviour, not a contractual term from the card itself.** If a future Granite revision documents prefixes, update the engine config.

## `attention_mask_dtype` (per-engine config field)

Some MLX models require the tokenizer's `attention_mask` to be cast to a specific dtype before the forward pass to avoid type-promotion errors. Other models accept the default integer mask. The `attention_mask_dtype` field in `EngineConfig` (and your `config.yaml`) controls this:

| Value | Behaviour |
|---|---|
| `None` (omitted, the default) | No cast. Use this for ModernBERT-based encoders like Granite. |
| `"float16"` | Mask cast to `mx.float16`. Required for `embeddinggemma-300m-bf16`. |
| `"bfloat16"` | Mask cast to `mx.bfloat16`. |
| `"float32"` | Mask cast to `mx.float32`. |

If you misconfigure this, the next ingest/search will raise a `RuntimeError` whose message names the offending model and recommends setting `attention_mask_dtype` — both the eager forward path and the lazy `np.array(...)` materialization path are wrapped, so the diagnostic is reliable.

## Runtime Truncation Alarm

`MLXEmbedder` performs an extra (cheap) tokenizer pass per batch with `truncation=False` to compute true input lengths. If any input exceeds `max_token_length`, a single `WARNING` line is logged:

```
Truncating 3/64 inputs above max_token_length=8192 for model 'ibm-granite/...' (longest observed: 11432 tokens, includes task prefix). Consider raising max_token_length or lowering chunk_max_chars.
```

Use this signal to tune `max_token_length` and `chunk_max_chars` in your engine config — the alarm is the project's evidence-based way to decide when to spend more memory on longer context.

## API / MCP Constraint (current)

The FastAPI routes (`/search/md`, `/search/sql`) and the MCP tools (`search_documents`, `search_sql_logs`) are hardcoded to the Gemma engines. The Granite engines are **CLI-only** at the route level — `dbs-vector ingest|search --type md-granite|sql-granite|sql-api-granite` works, but they are not reachable over HTTP/MCP without route changes (a tracked Phase 2 follow-up).

**Caveat:** even though Granite routes don't exist, `dbs-vector serve` and `dbs-vector mcp` still call `initialize_services()` which iterates every configured engine. After this change, both binaries will load all six models on startup (~1 GB extra download for Granite, plus the per-process memory cost). Operators who don't want this in the current window can comment out the three Granite engine blocks in `config.yaml` until Phase 2 introduces selective loading.

## Adding a New Model

1. Identify the architecture (BERT, XLM-RoBERTa, ModernBERT, …) and confirm `mlx-embeddings` supports it.
2. Add a new engine block in `config.yaml`. Set `model_name` to the HF repo, pick `vector_dimension` and `max_token_length` from the model card, and set `attention_mask_dtype` only if the model needs a non-default cast.
3. If the model is symmetric, leave `passage_prefix` and `query_prefix` empty. If asymmetric, copy the prefixes verbatim from the model card.
4. Run a smoke ingest+search via the CLI. Watch for the truncation-alarm warnings and the `RuntimeError` recommending `attention_mask_dtype`.
5. (Optional, recommended) Add a slow-marked integration test under `tests/integration/`.
