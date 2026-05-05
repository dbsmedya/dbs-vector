# Design: Granite 311m R2 as a New Embedding Engine

**Date:** 2026-05-06
**Branch:** `granite-311m-r2`
**Status:** Approved (brainstorming complete; ready for implementation plan)

---

## 1. Goal

Add IBM `granite-embedding-311m-multilingual-r2` as a supported embedding model in this project, exposed as three new engines — `md-granite`, `sql-granite`, `sql-api-granite` — alongside the existing `md`, `sql`, and `sql-api` engines. Embeddinggemma stays unchanged; the Granite engines coexist with the Gemma ones. Users can A/B compare both models on the same source documents and SQL workloads.

### Why

- Embeddinggemma (current default for all three engines) is capped at **2 048-token context**. Long markdown sections, schema definitions, and Turkish prose get split aggressively, hurting retrieval quality.
- Granite R2 supports **32 768-token context**, has explicit multilingual support for **52 enhanced languages including Turkish**, and is trained on code/SQL semantics — both motivate Granite for `sql`/`sql-api` clustering as well as `md` search.
- The project's clean architecture (`IEmbedder` protocol, configuration-driven registry) is designed to absorb a new model with near-zero churn — adding three engines is the same code path as adding one.

### Non-goals

- Replacing embeddinggemma. Both models stay supported.
- Adding a second embedding *runtime* (e.g. GGUF / `llama-cpp-python`). The pre-implementation spike confirmed Granite loads cleanly via the existing MLX path; GGUF stays as a documented fallback only if MLX support regresses.
- **Exposing Granite engines via FastAPI or MCP.** The routes in `api/main.py` (`/search/md`, `/search/sql`) and the MCP tools in `api/mcp_server.py` are hardcoded to `_services.get("md")` / `_services.get("sql")`. New engines load at startup (DI is dynamic) but are not reachable over HTTP/MCP without route changes. **This PR is CLI-only for the new engines**; making the API engine-agnostic (e.g. one generalized `/search/{engine}` route + an `engine` param on MCP tools) is its own design effort and tracked as a follow-up.

---

## 2. Pre-implementation Spike (verified 2026-05-06)

A 2-minute REPL spike de-risked the design before writing this spec:

```python
from mlx_embeddings.utils import load
model, tokenizer = load("ibm-granite/granite-embedding-311m-multilingual-r2")
# Loaded in ~62 s (one-time download). Subsequent loads use the HF cache.
```

Confirmed:

| Property | Result |
|---|---|
| Loads via `mlx_embeddings.utils.load` | ✅ No MLX conversion step needed |
| Output interface | `outputs.text_embeds`, shape `(N, 768)` — same as embeddinggemma |
| L2-normalised? | ✅ All output norms = 1.0 |
| Multilingual signal | `cos("Merhaba dünya", "Hello world") = 0.66` vs unrelated SQL at `0.42` |
| Mask-cast required? | ❌ Spike ran without the Gemma-specific `attention_mask → float16` cast. Granite accepts the default integer mask. |

**Conclusion:** zero new code path is required for the embedder. The implementation surface collapses to a config addition plus two surgical edits in `MLXEmbedder` (truncation alarm + config-driven mask-dtype handling).

---

## 3. Architecture summary

### Files touched

| File | Change |
|---|---|
| `config.yaml` | Add three new engine blocks: `md-granite`, `sql-granite`, `sql-api-granite`. Add `attention_mask_dtype: "float16"` to existing `md`, `sql`, `sql-api` blocks. |
| `src/dbs_vector/config.py` | Add `attention_mask_dtype: str \| None = None` field to `EngineConfig`. |
| `src/dbs_vector/services/bootstrap.py` | Pass `attention_mask_dtype` through to `MLXEmbedder`. |
| `src/dbs_vector/infrastructure/embeddings/mlx_engine.py` | Two edits: truncation alarm; config-driven attention-mask cast with `try/except` mapping promotion errors to a config-recommendation message. |
| `tests/unit/test_mlx_engine.py` | Cover the new behaviour (mocked, no model download). |
| `tests/integration/` | Add slow-marked end-to-end tests for the Granite engines (md and sql variants). |
| `docs/README_EMBEDDINGS.md` | **New** — embeddings-focused doc covering supported models, prefixes, `attention_mask_dtype`, truncation alarm. |
| `docs/README_DOCS.md` | Add a paragraph noting that Granite is used without task prefixes (treated as a symmetric encoder, see §4 rationale); link to new embeddings doc. |
| `docs/README_ARCHITECTURE.md` | One-line update in §3.A noting the three new Granite engines alongside the existing `md`/`sql`/`sql-api`. |
| `docs/README.md` | Add link to `README_EMBEDDINGS.md`. |
| `README.md` (root) | Update engine list if present. |
| `CLAUDE.md` | Add `attention_mask_dtype` to `EngineConfig` description; mention the Granite engines in example commands; note that API/MCP currently expose only Gemma engines. |

### Out of scope (deliberately unchanged)

- `core/ports.py`, `core/registry.py` — `IEmbedder` already abstracts the model; no protocol changes.
- `infrastructure/storage/mappers.py`, `infrastructure/storage/lancedb_engine.py` — Granite is dim 768 (same as Gemma); the schema is unchanged.
- `services/ingestion.py`, `services/search.py` — engines are runtime-resolved; no service-level changes.
- `api/main.py`, `api/state.py`, `api/mcp_server.py` — **explicitly out of scope** (see §1 non-goals). The startup path in `state.initialize_services()` will load the new engines because it iterates `settings.engines`, but the hardcoded `/search/md` / `/search/sql` routes and `search_documents` / `search_sql_logs` MCP tools won't expose them. The Granite engines are reachable only via the CLI (`uv run dbs-vector ingest|search --type md-granite|sql-granite|sql-api-granite`).
- `infrastructure/chunking/document.py` — `DocumentChunker` is reused with a larger `chunk_max_chars` value via config.

---

## 4. Config blocks

### `md-granite`

```yaml
md-granite:
  description: "Markdown & Prose Engine (Granite Multilingual R2 - long context)"
  model_name: "ibm-granite/granite-embedding-311m-multilingual-r2"
  vector_dimension: 768
  max_token_length: 8192          # cap below 32K hard limit; truncation alarm flags inputs exceeding this
  table_name: "knowledge_vault_granite"
  mapper_type: "document"
  chunker_type: "document"
  chunk_max_chars: 24000           # ~8K tokens at ~3 chars/tok (Turkish-conservative)
  passage_prefix: ""               # treated as a symmetric encoder — see Decision rationale
  query_prefix: ""
  workflow: "md_search_granite"
  # attention_mask_dtype omitted — defaults to None (Granite accepts the default integer mask, verified by the spike).
```

### `sql-granite` (DuckDB-based slow query log)

```yaml
sql-granite:
  description: "SQL Slow Query Log Engine (Granite Clustering)"
  model_name: "ibm-granite/granite-embedding-311m-multilingual-r2"
  vector_dimension: 768
  max_token_length: 8192
  table_name: "query_vault_granite"
  mapper_type: "sql"
  chunker_type: "duckdb"
  chunk_max_chars: 0
  passage_prefix: ""
  query_prefix: ""
  workflow: "sql_clustering_granite"
  # attention_mask_dtype omitted — defaults to None.
```

### `sql-api-granite` (remote API-based slow query log)

```yaml
sql-api-granite:
  description: "Remote slow query log via HTTP API (Granite)"
  model_name: "ibm-granite/granite-embedding-311m-multilingual-r2"
  vector_dimension: 768
  max_token_length: 8192
  table_name: "query_vault_granite_api"
  mapper_type: "sql"
  chunker_type: "api"
  chunk_max_chars: 0
  passage_prefix: ""
  query_prefix: ""
  workflow: "sql_clustering_granite"
  # ApiChunker fields mirror sql-api (api_base_url, api_key, …) — copy from existing block.
  # attention_mask_dtype omitted — defaults to None.
```

### Existing engines — single-field additions

```yaml
md:
  ...
  attention_mask_dtype: "float16"  # embeddinggemma-bf16 requires this cast
sql:
  ...
  attention_mask_dtype: "float16"
sql-api:
  ...
  attention_mask_dtype: "float16"
```

### Decision rationale

- **`max_token_length: 8192`** for all three Granite engines — Granite's hard limit is 32 768 but starting at 8 192 keeps initial latency and memory bounded. The runtime truncation alarm produces evidence for whether to raise it.
- **`chunk_max_chars: 24000`** for `md-granite` — exploits Granite's long context to keep semantic chunks (whole sections) coherent. Sized for Turkish at ~3 chars/token to stay within the 8 192-token budget. If the alarm fires often, lower this; if it never fires, raise both this and `max_token_length`. SQL engines keep `chunk_max_chars: 0` because slow-query records are always short.
- **Empty `passage_prefix` / `query_prefix`** — the Granite R2 model card examples do not specify task-prefix instructions, and the spike's semantic signal (cos-sim 0.66 paraphrase / 0.42 unrelated) confirms the encoder is usable without prefixes. We treat it as a symmetric encoder. *This is an implementation choice based on the model-card examples and verified behaviour, not a contractual term from the card itself*; if a future Granite revision documents prefixes, update these fields.
- **Distinct `table_name` per Granite engine** — clean A/B; each Granite table can be `--rebuild`'d in isolation. The schema is dim-compatible (768) with the Gemma tables, but mixing model spaces in one ANN index is meaningless.
- **`attention_mask_dtype` field is omitted in all three Granite blocks** — the new field defaults to `None`, which means *no cast*, which is what the spike validated. Listing it explicitly as `null` in YAML is unnecessary; the comment in each block records the intent.

---

## 5. `EngineConfig` field

In `src/dbs_vector/config.py`:

```python
# Some MLX models (e.g. embeddinggemma-bf16) require the attention_mask cast
# to a specific dtype to avoid type-promotion errors. Leave unset for models
# that accept the default integer mask (e.g. ModernBERT / Granite).
attention_mask_dtype: str | None = None  # accepted: None, "float16", "bfloat16", "float32"
```

`bootstrap.build_dependencies` propagates the field to `MLXEmbedder`:

```python
embedder = MLXEmbedder(
    model_name=config.model_name,
    max_token_length=config.max_token_length,
    dimension=config.vector_dimension,
    passage_prefix=config.passage_prefix,
    query_prefix=config.query_prefix,
    attention_mask_dtype=config.attention_mask_dtype,
)
```

---

## 6. `MLXEmbedder` changes

The two edits, both inside `_execute_mlx`:

### 6.1 Truncation alarm

Inserted *before* the existing truncation+padding tokenizer call. The alarm measures **pre-truncation tokenized input lengths after prefix application** — i.e. the lengths of `passage_prefix + text` (for `embed_batch`) or `query_prefix + text` (for `embed_query`) as `_execute_mlx` already receives them, since prefixing happens upstream in `embed_batch` / `embed_query`. This is the right metric for "did we exceed the model's budget?" — not the raw source-chunk character length. The cost is one extra (fast) tokenizer pass per batch — negligible vs. the model forward.

```python
no_trunc = self.tokenizer._tokenizer(
    texts,
    padding=False,
    truncation=False,
    add_special_tokens=True,
)
lengths = [len(ids) for ids in no_trunc["input_ids"]]
max_len = max(lengths) if lengths else 0
if max_len > self._max_token_length:
    over_count = sum(1 for n in lengths if n > self._max_token_length)
    logger.warning(
        "Truncating {}/{} inputs above max_token_length={} for model '{}' "
        "(longest observed: {} tokens, includes task prefix). "
        "Consider raising max_token_length or lowering chunk_max_chars.",
        over_count,
        len(texts),
        self._max_token_length,
        self._model_name,
        max_len,
    )
```

### 6.2 Config-driven attention-mask dtype + promotion-error mapping

Replaces the current unconditional `attention_mask → float16` cast. **Membership check uses `"attention_mask" in inputs`**, not `hasattr` — `BatchEncoding` and plain dict mocks both support `in`, while `hasattr` would silently skip the cast under a dict-based test fixture.

```python
if self._attention_mask_dtype and "attention_mask" in inputs:
    import mlx.core as mx
    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    if self._attention_mask_dtype not in dtype_map:
        raise ValueError(
            f"Unsupported attention_mask_dtype '{self._attention_mask_dtype}'. "
            f"Allowed: {list(dtype_map)}"
        )
    inputs["attention_mask"] = inputs["attention_mask"].astype(
        dtype_map[self._attention_mask_dtype]
    )
```

The `try/except` must wrap **both the forward and the materialization**. MLX evaluates lazily: a type-promotion error can fire either inside `self.model(...)` (eager kernel dispatch on some paths) *or* during `np.array(embeds_mlx)` (which forces graph evaluation per the existing comment in this file). Wrapping only the forward would let the lazy-evaluated case bypass the friendly diagnostic and surface as a raw MLX exception.

```python
try:
    outputs = self.model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
    embeds_mlx = (
        outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs["text_embeds"]
    )
    # np.array(...) forces MLX lazy evaluation. A type-promotion error can
    # surface here rather than at the model() call.
    vectors_np: NDArray[np.float32] = np.array(embeds_mlx).astype(np.float32)
except Exception as e:
    if "promote" in str(e).lower():
        raise RuntimeError(
            f"MLX type-promotion error while running model '{self._model_name}'. "
            f"This usually means the model requires the attention_mask cast to a specific dtype. "
            f"Set `attention_mask_dtype` in this engine's block in config.yaml — "
            f"try \"float16\" (common for bf16 models like embeddinggemma) or \"bfloat16\". "
            f"Original error: {e}"
        ) from e
    raise
```

### 6.3 Behaviour matrix

| Scenario | Outcome |
|---|---|
| Granite engine, `attention_mask_dtype` unset (None) | No cast. Model runs cleanly. |
| Gemma engine, `attention_mask_dtype: "float16"` | Mask cast to fp16. Model runs cleanly. |
| Gemma engine, `attention_mask_dtype` accidentally unset | `try` fails with "Cannot promote types…" → `RuntimeError` raised that names the offending model and recommends the config field to set. |
| Any unrelated MLX error | Re-raised verbatim, no obfuscation. |
| Unknown dtype string in config | `ValueError` listing allowed values, raised before the model forward. |

---

## 7. Migration & rollback

### Forward path

1. Pull the change. `config.yaml` now declares `md`, `sql`, `sql-api` (each with `attention_mask_dtype: "float16"`) plus the three new Granite engines.
2. Existing tables (`knowledge_vault`, `query_vault`) are untouched — schema, dim, and workflow tag are unchanged. The Gemma engines see only an explicit `attention_mask_dtype` in config that produces identical behaviour to today's hardcoded cast.
3. First Granite ingest creates fresh tables: `knowledge_vault_granite`, `query_vault_granite`, `query_vault_granite_api`. Model download is one-time (~60 s); the same model file is shared across all three Granite engines via the in-process `_MODEL_CACHE`.
4. Search via CLI: `uv run dbs-vector search … --type md-granite|sql-granite|sql-api-granite`. The Gemma engines (`md`, `sql`, `sql-api`) continue to work in parallel — direct A/B.
5. **API/MCP traffic is unaffected** — the existing routes still point at the Gemma engines. New engines are CLI-only this PR.

### Rollback (single revert)

1. `git revert <merge-commit>` — config and embedder edits gone.
2. Optionally `rm -rf lancedb_dbs_vector/knowledge_vault_granite.lance lancedb_dbs_vector/query_vault_granite.lance lancedb_dbs_vector/query_vault_granite_api.lance` to drop the Granite tables.
3. The existing Gemma engines were never touched at the data level; they continue working.

---

## 8. Testing

### Unit tests (`tests/unit/test_mlx_engine.py`)

All mock `mlx_embeddings.utils.load`; no model download.

| Test | Verifies |
|---|---|
| `test_attention_mask_dtype_none_skips_cast` | `attention_mask_dtype=None` → mask is **not** cast (Granite default path). |
| `test_attention_mask_dtype_float16_applies_cast` | `attention_mask_dtype="float16"` → mask cast to `mx.float16` (Gemma path). |
| `test_attention_mask_dtype_invalid_raises` | Unknown dtype string raises `ValueError` listing allowed values. |
| `test_truncation_warning_emitted` | Batch with one input > `max_token_length` logs `WARNING` containing model name and observed length (uses `caplog`). |
| `test_truncation_warning_silent_when_under_budget` | All inputs ≤ `max_token_length` → no warning. |
| `test_promote_error_remapped_to_config_hint` | Mock model raising `ValueError("Cannot promote types: bf16, int32")` → `RuntimeError` whose message contains `"attention_mask_dtype"`. |
| `test_unrelated_model_error_passes_through` | Mock raising `ValueError("something else")` → re-raised verbatim, no wrapping. |

### Integration test (`tests/integration/`)

| Test | Verifies |
|---|---|
| `test_md_granite_engine_end_to_end` | Loads `md-granite` config from a fixture YAML, ingests two short markdown files into a `tmp_path` LanceDB, searches, asserts a result is returned with non-zero score. **Slow-marked**: `@pytest.mark.skipif(os.getenv("RUN_SLOW_TESTS") != "1")` — opt-in because it downloads the real model. |
| `test_sql_granite_engine_end_to_end` | Same shape as above for `sql-granite` — loads a tiny DuckDB-style fixture, ingests, searches. Slow-marked. |

### Integration-test wiring (resolves the “fixture YAML vs. global singleton” ambiguity)

`bootstrap.build_dependencies` reads `from dbs_vector.config import settings` — a module-level singleton that's loaded once at import time. To inject a fixture config, the test must replace this singleton, not just set `DBS_CONFIG_FILE`. Pattern:

```python
def test_md_granite_engine_end_to_end(tmp_path, monkeypatch):
    fixture = tmp_path / "config.yaml"
    fixture.write_text(textwrap.dedent(f"""
        system:
          db_path: {tmp_path / "lancedb"}
          batch_size: 8
        engines:
          md-granite:
            description: "test"
            model_name: "ibm-granite/granite-embedding-311m-multilingual-r2"
            vector_dimension: 768
            max_token_length: 512
            table_name: "knowledge_vault_granite"
            mapper_type: "document"
            chunker_type: "document"
            chunk_max_chars: 2000
            passage_prefix: ""
            query_prefix: ""
            workflow: "md_search_granite"
    """))

    from dbs_vector import config as config_module
    from dbs_vector.services import bootstrap as bootstrap_module

    fixture_settings = config_module.load_settings(str(fixture))
    monkeypatch.setattr(config_module, "settings", fixture_settings)
    monkeypatch.setattr(bootstrap_module, "settings", fixture_settings)

    deps = bootstrap_module.build_dependencies("md-granite")
    # … then exercise IngestionService + SearchService against deps …
```

`monkeypatch` reverts both module references at test teardown; no global state leaks between tests. The same pattern is reused for `test_sql_granite_engine_end_to_end`.

### Manual validation (post-merge)

```bash
# md-granite — ingest and A/B vs. Gemma
uv run dbs-vector ingest "docs/" --type md-granite
uv run dbs-vector search "Türkçe doküman araması" --type md-granite
uv run dbs-vector search "Türkçe doküman araması" --type md

# sql-granite — ingest from DuckDB fixture and A/B vs. Gemma
uv run dbs-vector ingest "queries.json" --type sql-granite
uv run dbs-vector search "find users by email" --type sql-granite
uv run dbs-vector search "find users by email" --type sql

# sql-api-granite — exercise the remote-API path
uv run dbs-vector ingest "" --type sql-api-granite        # ApiChunker pulls from configured endpoint
uv run dbs-vector search "slow join on orders" --type sql-api-granite

# Regression smoke checks for Gemma engines after the attention_mask_dtype migration
uv run dbs-vector search "anything" --type md
uv run dbs-vector search "anything" --type sql
```

---

## 9. Documentation

| File | Action |
|---|---|
| `docs/README_EMBEDDINGS.md` | **New.** Covers: supported models (Gemma, Granite); MLX backend constraints; symmetric vs asymmetric models; the `attention_mask_dtype` config field with model-by-model recommendations; the truncation alarm and how to interpret it; how to add a new model. |
| `docs/README_DOCS.md` | Add a paragraph in the "Task Prefixes" section: Granite is symmetric (empty prefixes), with a pointer to `README_EMBEDDINGS.md`. |
| `docs/README_ARCHITECTURE.md` | One-line addition in §3.A — mention `md-granite` alongside `md`/`sql`. |
| `docs/README.md` | Add link to `README_EMBEDDINGS.md`. |
| `README.md` (root) | Update engine list/table if present. |
| `CLAUDE.md` | Add `attention_mask_dtype` to `EngineConfig` description; add `md-granite` to example commands. |

---

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `mlx_embeddings` returns wrong pooling for ModernBERT → degraded recall | Low (spike showed sane semantic signal) | Manual A/B against Gemma. If recall is poor, fall back to GGUF + `llama-cpp-python` backend (separate design effort). |
| Missing `attention_mask_dtype: "float16"` on Gemma engines after the config change | Medium during the migration commit only | Single commit updates all three Gemma engines + adds Granite. `test_promote_error_remapped_to_config_hint` ensures the error is actionable. Manual smoke check (`search --type md`) catches any regression. |
| Long-context tempting aggressive `max_token_length` bumps → OOM or slow ingest | Medium long-term | Truncation alarm gives quantitative evidence. Default capped at 8 192. Ingest batch size unchanged at 64. |
| `chunk_max_chars=24000` exceeds the 8 192-token budget for Turkish/code-heavy text | Medium | Alarm fires immediately and visibly; operator either lowers `chunk_max_chars` or raises `max_token_length`. Mitigated, not prevented. |
| HuggingFace cache pulls ~1 GB on first Granite run | Certain | One-time, ~60 s on a normal connection. Documented in `README_EMBEDDINGS.md`. |
| Upstream `mlx_embeddings` change breaks Granite path | Low | Pinned via `uv.lock`. Slow-marked integration test catches breakage on dependency upgrade. |
| Future engine config typo points Granite at the Gemma table | Low | The two schemas are identical (dim 768, same fields), so a typo would **not** trigger a schema-mismatch error — instead, the two model spaces would silently mix in one ANN index, degrading recall without a visible failure. Mitigations: (a) `workflow` column is distinct per engine (`md_search` vs `md_search_granite`), so chunks are tagged at the row level and a follow-up audit query can detect mixing; (b) naming convention (`<engine>_<table>` style) makes the mapping obvious; (c) PR review on `config.yaml` changes. |

### Documented fallback path (Plan B)

If MLX-native loading regresses or pooling proves wrong: a published GGUF variant (`mykor/granite-embedding-311m-multilingual-r2-GGUF`) declares `modernbert` architecture, ships a working `llama_cpp.Llama(embedding=True)` example, and the BF16 build cosine-matches fp32 at 0.9999. Adopting it would require a new `LlamaCppEmbedder` implementation of `IEmbedder` and a `llama-cpp-python` dependency — a follow-up design, not part of this work.

### Known follow-up: API/MCP engine generalization

The hardcoded `/search/md` and `/search/sql` FastAPI routes (and `search_documents` / `search_sql_logs` MCP tools) prevent any non-`md`/`sql` engine — including the three new Granite engines — from being reachable over HTTP/MCP. The right fix is *not* to add three more hardcoded routes/tools (that doubles the surface and entrenches the wrong abstraction) but to generalize:

- One `/search/{engine}` POST route that resolves `_services.get(engine)` dynamically. Request schema picks `SqlSearchRequest` shape vs `SearchRequest` shape based on the engine's `mapper_type`.
- One `search` MCP tool that takes an `engine` parameter (with discriminated-union response shape).

This is a separate brainstorm/spec because it changes the public API contract.

---

## 11. Acceptance criteria

- [ ] `uv run poe check` passes (format, lint, typecheck, all tests).
- [ ] `uv run dbs-vector ingest "docs/" --type md-granite` ingests successfully and creates the `knowledge_vault_granite` table.
- [ ] `uv run dbs-vector ingest <fixture> --type sql-granite` ingests successfully and creates the `query_vault_granite` table.
- [ ] `uv run dbs-vector ingest "" --type sql-api-granite` ingests successfully from the configured remote API and creates `query_vault_granite_api`.
- [ ] `uv run dbs-vector search "<query>" --type md-granite|sql-granite|sql-api-granite` returns ranked results.
- [ ] `uv run dbs-vector search "<query>" --type md|sql|sql-api` (existing Gemma engines) returns identical behaviour to the pre-change baseline — no schema rebuild, no quality regression.
- [ ] FastAPI `/search/md` and `/search/sql` continue to serve only the Gemma engines (Granite engines are not exposed over HTTP/MCP this PR — verify both routes return Gemma-tagged results, and that no `/search/md-granite` route exists).
- [ ] Truncation alarm fires when an input exceeds `max_token_length`; produces a single warning line per batch.
- [ ] Promotion-error path raises a `RuntimeError` recommending `attention_mask_dtype` — verified for both the eager-forward path and the lazy `np.array(...)` materialization path.
- [ ] All new and updated docs are linked from `docs/README.md`.
