# Tuning Profiles & Model Contract Registry — Design

**Status:** Draft (2026-05-06)
**Author:** sinan + Claude (brainstorming session)
**Implements:** Hardware-aware per-engine tuning of `max_token_length`, `chunk_max_chars`, `batch_size`; closes the OOM crash hit when `md-granite` was bumped from 8K → 16K context.

---

## 1. Goal & Non-Goals

### Goal

Make ingestion safely tunable per engine without forcing the user to reason about Metal buffer math. Specifically:

1. **Per-engine `batch_size`** — currently global in `system:` block, forcing every engine to share the smallest workable value.
2. **Hardware-aware fail-fast** — refuse to start ingestion when the configured profile would exceed the Metal buffer budget; print a recommendation instead of crashing mid-batch.
3. **Strict model contracts** — properties the model *requires to function correctly* (`vector_dimension`, `model_max_token_length`, `attention_mask_dtype`) are hardcoded in code and cannot be misconfigured in YAML.
4. **Tunable profiles** — the three numeric knobs that vary with hardware (`max_token_length`, `chunk_max_chars`, `batch_size`) live in a named preset table in `config.yaml`. Engines reference profiles by name.

### Non-Goals

- **Auto-tuning at runtime.** The validator computes a recommendation; it does not silently rewrite config or retry on OOM.
- **CLI surface (`dbs-vector tune validate / recommend / list`).** Deferred to **Phase 3**. Validation runs at `load_settings()` only; the recommender lives as an internal Python helper used by tests.
- **Jira chunker (`md-jira-granite`).** Deferred to **Phase 2** (separate spec). Profiles are designed so adding a jira profile later is a config-only change.
- **Migration tooling.** Single-shot YAML rewrite is documented in this spec; no automated migrator.
- **Profile auto-selection from hardware.** User picks profile name explicitly. The validator confirms it fits.

### Calibration data point

The user hit `[metal::malloc] Attempting to allocate 41818614016 bytes which is greater than the maximum allowed buffer size of 22613000192 bytes` when running `md-granite` with `max_token_length=16384, batch_size=64`. This is the canonical example the validator must catch *before* ingestion starts.

---

## 2. Architecture — Three Layers

```
┌──────────────────────────────────────────────────────────────┐
│  ModelRegistry (code, open/closed)                           │
│  Hardcoded ModelContract entries. Adding a model = code      │
│  change. Holds: model_name, vector_dimension,                │
│         model_max_token_length, attention_mask_dtype         │
└──────────────────────────────────────────────────────────────┘
                              ▲
                              │ resolved by name
┌──────────────────────────────────────────────────────────────┐
│  EngineConfig (config.yaml, user-tunable)                    │
│  description, model (key), mapper_type, chunker_type,        │
│  table_name, workflow, tuning_profile (key),                 │
│  passage_prefix, query_prefix, chunker-specific kwargs       │
└──────────────────────────────────────────────────────────────┘
                              ▲
                              │ resolved by name
┌──────────────────────────────────────────────────────────────┐
│  TuningProfile (config.yaml, user-tunable)                   │
│  Three numbers: max_token_length, chunk_max_chars,           │
│                 batch_size. Validated against engine + HW.   │
└──────────────────────────────────────────────────────────────┘
```

**The split is principled:**
- **ModelRegistry** = "what the model *requires* to function correctly." Misconfigure these and you corrupt embeddings or crash MLX. Hardcoded.
- **EngineConfig** = "what *this deployment* uses." Pipeline shape, table names, workflow labels, prefixes. Tunable per deployment.
- **TuningProfile** = "how big to chunk and batch." Hardware-bound. Tunable per deployment.

**Why prefixes are on the engine, not the contract.** Same underlying gemma model serves both `md` (search prefix) and `sql` (clustering prefix). Putting prefixes on the contract would force two contracts for one model. Putting them on the engine keeps the contract count low and makes prefix experimentation a config change, not a code change.

---

## 3. ModelRegistry

### `core/model_registry.py` — new module

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelContract:
    """Immutable model contract. All fields are properties of the model itself,
    not of any particular deployment that uses the model."""

    model_name: str                  # the MLX repo path passed to mx_load
    vector_dimension: int            # output embedding size
    model_max_token_length: int      # hard cap from the model card
    attention_mask_dtype: str | None # None | "float16" | "bfloat16" | "float32"
    compute_dtype_bytes: int = 2     # model's internal compute dtype (bf16=2)


class ModelRegistry:
    """Open/closed registry of model contracts. Adding a model = register() call."""

    _models: dict[str, ModelContract] = {}

    @classmethod
    def register(cls, key: str, contract: ModelContract) -> None:
        if key in cls._models:
            raise ValueError(f"Model contract '{key}' already registered")
        cls._models[key] = contract

    @classmethod
    def get(cls, key: str) -> ModelContract:
        if key not in cls._models:
            known = sorted(cls._models)
            raise KeyError(f"Unknown model contract '{key}'. Known: {known}")
        return cls._models[key]

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._models)
```

### Built-in registrations (at module import)

```python
ModelRegistry.register("gemma-bf16", ModelContract(
    model_name="mlx-community/embeddinggemma-300m-bf16",
    vector_dimension=768,
    model_max_token_length=2048,
    attention_mask_dtype="float16",
    compute_dtype_bytes=2,
))

ModelRegistry.register("granite-r2", ModelContract(
    model_name="ibm-granite/granite-embedding-311m-multilingual-r2",
    vector_dimension=768,
    model_max_token_length=32768,
    attention_mask_dtype=None,
    compute_dtype_bytes=2,
))
```

Two contracts cover all six existing engines. Adding a new model later means one PR adding a `register()` call.

---

## 4. EngineConfig (revised)

### `config.py` — `EngineConfig`

```python
class EngineConfig(BaseModel):
    """Per-deployment engine config. References model contract + tuning profile."""

    description: str
    model: str                  # key into ModelRegistry  (NEW)
    mapper_type: str
    chunker_type: str
    table_name: str
    workflow: str
    tuning_profile: str         # key into Settings.profiles  (NEW)

    # Model-deployment-specific (kept on engine because they vary per workflow
    # using the same underlying model):
    passage_prefix: str = ""
    query_prefix: str = ""

    # Chunker-specific (unchanged):
    duckdb_query: str | None = None
    api_base_url: str = ""
    api_key: str = ""
    api_page_size: int = 200
    api_since_days: int = 15
    api_timeout_sec: int = 30
    api_min_execution_ms: float = 0.0
    api_database: str = ""

    def chunker_kwargs(
        self, query_override: str | None = None, url_override: str | None = None
    ) -> dict[str, object]:
        """Resolve chunker initialization kwargs. Reads chunk_max_chars from the
        resolved tuning profile via settings.profiles (caller injects)."""
        # signature change documented in §6
```

### Removed fields

The following move to `ModelRegistry`:
- `model_name`
- `vector_dimension`
- `max_token_length` (replaced by `model_max_token_length` on the contract; runtime value comes from profile)
- `attention_mask_dtype`

The following moves to `TuningProfile`:
- `chunk_max_chars`

### Why `passage_prefix`/`query_prefix` stay on the engine

Same gemma-bf16 model is shared between `md` (search prefix) and `sql` (clustering prefix). Two engines, one contract. Prefixes encode the embedding *task*, which is a deployment choice, not a model property.

---

## 5. TuningProfile + Settings

### `config.py` — `TuningProfile`

```python
class TuningProfile(BaseModel):
    """Three numeric knobs validated against the engine's model contract and
    available memory at load time."""

    max_token_length: int       # ≤ contract.model_max_token_length
    chunk_max_chars: int        # 0 = atomic chunks (SQL); >0 = merge until limit
    batch_size: int             # passed to IngestionService._batched()
```

### `config.py` — `Settings`

```python
class Settings(BaseSettings):
    db_path: str = "./lancedb_dbs_vector"
    nprobes: int = 20
    log_level: str = "INFO"
    log_serialize: bool = False

    memory_budget_gb: float | None = None  # None = auto-detect from MLX (NEW)

    profiles: dict[str, TuningProfile] = {}  # NEW
    engines: dict[str, EngineConfig] = {}

    # REMOVED: batch_size  (now per-profile)

    model_config = SettingsConfigDict(env_prefix="DBS_", env_file=".env")
```

### Built-in profiles (in default `config.yaml`)

```yaml
profiles:
  gemma-md:           {max_token_length: 2048,  chunk_max_chars: 1000, batch_size: 64}
  gemma-sql-atomic:   {max_token_length: 2048,  chunk_max_chars: 0,    batch_size: 64}
  granite-md-large:   {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  granite-sql-atomic: {max_token_length: 8192,  chunk_max_chars: 0,    batch_size: 32}
```

These four profiles cover today's six engines. The user adds new profiles freely; the validator catches mistakes.

---

## 6. End-to-end resolution flow

`build_dependencies("md-granite")` after this change:

```python
def build_dependencies(engine_name, query_override=None, url_override=None):
    if engine_name not in settings.engines:
        raise ValueError(f"Unknown engine: '{engine_name}'.")

    engine = settings.engines[engine_name]
    contract = ModelRegistry.get(engine.model)            # KeyError if unknown
    profile = settings.profiles[engine.tuning_profile]    # KeyError if unknown

    # validation already happened at load_settings(); see §7
    embedder = MLXEmbedder(
        model_name=contract.model_name,
        max_token_length=profile.max_token_length,
        dimension=contract.vector_dimension,
        passage_prefix=engine.passage_prefix,
        query_prefix=engine.query_prefix,
        attention_mask_dtype=contract.attention_mask_dtype,
    )

    MapperClass = ComponentRegistry.get_mapper(engine.mapper_type)
    ChunkerClass = ComponentRegistry.get_chunker(engine.chunker_type)
    mapper = MapperClass(vector_dimension=contract.vector_dimension)
    chunker = ChunkerClass(**engine.chunker_kwargs(
        chunk_max_chars=profile.chunk_max_chars,
        query_override=query_override,
        url_override=url_override,
    ))

    store = LanceDBStore(
        db_path=settings.db_path,
        table_name=engine.table_name,
        vector_dimension=contract.vector_dimension,
        mapper=mapper,
        nprobes=settings.nprobes,
    )
    return EngineDeps(embedder, store, chunker, engine.workflow)
```

`IngestionService` reads `batch_size` from `profile.batch_size` (passed in via a constructor arg or via the resolved deps), no longer from `settings.batch_size`.

`EngineConfig.chunker_kwargs()` gains a `chunk_max_chars: int` parameter — caller injects the value from the resolved profile. The `if self.chunk_max_chars > 0:` branch becomes `if chunk_max_chars > 0:`.

---

## 7. Validation chain (fail-fast at `load_settings()`)

After parsing YAML and constructing all profiles + engines, run a `_validate_config(settings)` pass. Any failure raises `ValueError` with a clear remediation message before `load_settings()` returns.

For each `(engine_name, engine)` in `settings.engines.items()`:

1. **Model contract exists.**
   `ModelRegistry.get(engine.model)` — else: `"Engine 'md-granite' references unknown model contract 'granite-r3'. Known: ['gemma-bf16', 'granite-r2']."`

2. **Tuning profile exists.**
   `settings.profiles[engine.tuning_profile]` — else: `"Engine 'md-granite' references unknown tuning profile 'granite-md-extreme'. Known: [...]"`

3. **Profile fits model.**
   `profile.max_token_length ≤ contract.model_max_token_length` — else: `"Profile 'granite-md-large' requires 16384 tokens but engine 'md' uses model 'gemma-bf16' (cap 2048). Lower profile.max_token_length or pick a different model."`

4. **Profile fits memory budget.**
   `estimate_peak_buffer_bytes(profile, contract) ≤ memory_budget_bytes × 0.9` — else: `"Profile 'granite-md-extreme' would allocate ~41 GB; memory budget is 22 GB. Suggested values from recommend_profile(): max_token_length=16384, batch_size=N. Edit config.yaml."`
   The `× 0.9` leaves 10% headroom for OS/other allocations. The "suggested values" are produced by calling `recommend_profile(contract, memory_budget_gb, target_chunker=engine.chunker_type)` (§9) so the message is always self-consistent with the formula.

5. **Chunk-vs-token sanity (warn only).**
   `profile.chunk_max_chars > 0 and profile.chunk_max_chars > profile.max_token_length × 4` → `logger.warning(...)`.
   *(Rough char-per-token ratio of 4. Caller may legitimately want oversized chunks if they expect heavy truncation; this is a heads-up, not a fail.)*

Validation runs once at startup. CLI ingestion / `serve` / `mcp` all go through the same `load_settings()` so all three benefit.

---

## 8. Memory budget detection

### Layering

The detector touches MLX (an external/heavy import); the math is pure. They go in different modules to keep `core/` infrastructure-free per the project's layering rule (CLAUDE.md: "core/ has no external dependencies").

- **`infrastructure/hardware.py`** — `detect_memory_budget_gb()` (lazy `mlx.core` import).
- **`core/profile_math.py`** — pure helpers used by the validator and recommender (§9).

### `infrastructure/hardware.py` — new module

```python
from loguru import logger


def detect_memory_budget_gb() -> float | None:
    """Try to read Metal's max buffer length. Return None if unavailable."""
    try:
        import mlx.core as mx
        info = mx.metal.device_info()
        max_bytes = info.get("max_buffer_length")
        if max_bytes:
            return max_bytes / (1024 ** 3)
    except Exception as e:  # noqa: BLE001 — any failure means "fall back to config"
        logger.debug("Metal device_info unavailable: {}", e)
    return None


def resolve_memory_budget_gb(configured: float | None) -> float:
    """Resolve final memory budget. Configured wins; else auto-detect; else raise."""
    if configured is not None:
        return configured
    detected = detect_memory_budget_gb()
    if detected is not None:
        logger.info("Auto-detected memory budget: {:.1f} GB", detected)
        return detected
    raise ValueError(
        "Could not auto-detect Metal memory budget. "
        "Set system.memory_budget_gb in config.yaml (e.g., 16.0)."
    )
```

**No new dependencies.** `mlx` is already pinned. If `mlx.metal.device_info()` doesn't exist on the installed MLX version, the `try/except` falls through and the user gets a clear "set it explicitly" error.

---

## 9. Memory equation

### `core/profile_math.py` — new module (pure, no external deps)

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbs_vector.config import TuningProfile
    from dbs_vector.core.model_registry import ModelContract


# Calibrated empirically against the user's OOM:
# batch=64, seq=16384, bf16 (2 bytes) → real allocation 41 GB
# Raw: 64 × 16384² × 2 = 34.4 GB. Real / raw = 1.19× per-element overhead.
# We use 3.0× for safety because the 41 GB is just the largest single buffer;
# total memory pressure is higher (weights, KV cache, activations).
_PEAK_BUFFER_OVERHEAD = 3.0

# Approximate char-per-token ratio for English+code; used by the recommender.
_CHARS_PER_TOKEN = 2.5


def estimate_peak_buffer_bytes(profile: "TuningProfile", contract: "ModelContract") -> int:
    """Approximate peak Metal memory pressure during a forward pass.

    Dominated by attention: O(batch × seq² × dtype_bytes), with a 3× safety
    factor for temporaries, weights, and KV cache. Hidden_dim drops out because
    the attention matrix is the long pole, not the activations.

    The dtype is the model's compute dtype (contract.compute_dtype_bytes), not
    the attention_mask cast — the mask is cheap; the attention buffer is what
    blows up."""
    return int(
        _PEAK_BUFFER_OVERHEAD
        * profile.batch_size
        * profile.max_token_length ** 2
        * contract.compute_dtype_bytes
    )
```

This formula is intentionally simple. Calibration knobs:
- `_PEAK_BUFFER_OVERHEAD = 3.0` — covers attention temporaries + weights + KV cache.
- `contract.compute_dtype_bytes = 2` — bf16 for both gemma and granite.

If users hit false-positive rejections, lower `_PEAK_BUFFER_OVERHEAD`; if they hit OOM with a green validator, raise it. One scalar to maintain.

### Recommender (internal helper, not CLI)

```python
def recommend_profile(
    contract: "ModelContract",
    memory_budget_gb: float,
    target_chunker: str = "document",
) -> dict[str, int]:
    """Suggest profile values that fit memory_budget_gb for this contract.

    Strategy: start from the model's max seq len; pick batch_size that fits
    memory; if no batch ≥ 1 fits, halve seq_len and retry. The chunk_max_chars
    heuristic depends on chunker_type:
      - 'duckdb' / 'api'  → 0 (atomic SQL records)
      - else (document)   → seq_len × _CHARS_PER_TOKEN × 0.5 (50% of seq, leaves
                            room for tokenization slack and prefix tokens)
    """
    budget = int(memory_budget_gb * 1024 ** 3 * 0.9)
    seq = contract.model_max_token_length
    while seq >= 512:
        max_batch = budget // int(
            _PEAK_BUFFER_OVERHEAD * seq ** 2 * contract.compute_dtype_bytes
        )
        if max_batch >= 1:
            chunk = (
                0
                if target_chunker in ("duckdb", "api")
                else int(seq * _CHARS_PER_TOKEN * 0.5)
            )
            return {
                "max_token_length": seq,
                "chunk_max_chars": chunk,
                "batch_size": int(max_batch),
            }
        seq //= 2
    raise ValueError(
        f"No profile fits {memory_budget_gb} GB for model with cap "
        f"{contract.model_max_token_length}. Reduce model or increase budget."
    )
```

---

## 10. config.yaml after migration

```yaml
system:
  db_path: "./lancedb_dbs_vector"
  nprobes: 20
  # memory_budget_gb auto-detected from Metal; uncomment to override:
  # memory_budget_gb: 22.0

profiles:
  gemma-md:           {max_token_length: 2048,  chunk_max_chars: 1000, batch_size: 64}
  gemma-sql-atomic:   {max_token_length: 2048,  chunk_max_chars: 0,    batch_size: 64}
  granite-md-large:   {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  granite-sql-atomic: {max_token_length: 8192,  chunk_max_chars: 0,    batch_size: 32}

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
    tuning_profile: "gemma-md"

  sql:
    description: "SQL Slow Query Log Engine (Gemma Clustering)"
    model: "gemma-bf16"
    mapper_type: "sql"
    chunker_type: "duckdb"
    table_name: "query_vault"
    workflow: "sql_clustering"
    passage_prefix: "task: clustering | query: "
    query_prefix: "task: clustering | query: "
    tuning_profile: "gemma-sql-atomic"

  sql-api:
    description: "Remote slow query log via HTTP API"
    model: "gemma-bf16"
    mapper_type: "sql"
    chunker_type: "api"
    table_name: "query_vault"
    workflow: "sql_clustering"
    passage_prefix: "task: clustering | query: "
    query_prefix: "task: clustering | query: "
    tuning_profile: "gemma-sql-atomic"
    api_base_url: "http://localhost:8080/api/v1"
    api_key: "..."
    api_page_size: 200
    api_since_days: 60
    api_timeout_sec: 30
    api_min_execution_ms: 0
    api_database: ""

  md-granite:
    description: "Markdown & Prose Engine (Granite R2 - long context)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"

  sql-granite:
    description: "SQL Slow Query Log Engine (Granite Clustering)"
    model: "granite-r2"
    mapper_type: "sql"
    chunker_type: "duckdb"
    table_name: "query_vault_granite"
    workflow: "sql_clustering_granite"
    tuning_profile: "granite-sql-atomic"

  sql-api-granite:
    description: "Remote slow query log via HTTP API (Granite)"
    model: "granite-r2"
    mapper_type: "sql"
    chunker_type: "api"
    table_name: "query_vault_granite_api"
    workflow: "sql_clustering_granite"
    tuning_profile: "granite-sql-atomic"
    api_base_url: "http://localhost:8080/api/v1"
    api_key: "..."
    api_page_size: 200
    api_since_days: 60
    api_timeout_sec: 30
    api_min_execution_ms: 0
    api_database: ""
```

Each engine drops from ~12 lines to ~8 lines for non-API engines. Model contract details are gone from config entirely.

---

## 11. Migration

This is a **breaking config schema change**. Single PR, no automated migrator (YAGNI; the migration is mechanical).

### Code changes

1. **New module** `core/model_registry.py` with `ModelContract`, `ModelRegistry`, two built-in `register()` calls.
2. **New module** `core/profile_math.py` with `estimate_peak_buffer_bytes`, `recommend_profile`, `_PEAK_BUFFER_OVERHEAD`, `_CHARS_PER_TOKEN`. Pure, no external deps.
3. **New module** `infrastructure/hardware.py` with `detect_memory_budget_gb`, `resolve_memory_budget_gb`. Lazy `mlx.core` import.
4. **`config.py`**:
   - Add `TuningProfile` model.
   - Add `Settings.profiles`, `Settings.memory_budget_gb`.
   - Remove `Settings.batch_size`.
   - Shrink `EngineConfig`: remove `model_name`, `vector_dimension`, `max_token_length`, `attention_mask_dtype`, `chunk_max_chars`. Add `model`, `tuning_profile`. Update `chunker_kwargs()` signature to accept `chunk_max_chars`.
   - Add `_validate_config(settings)` and call it at the end of `load_settings()`.
5. **`services/bootstrap.py`** `build_dependencies`:
   - Resolve contract + profile + memory budget.
   - Pass `chunk_max_chars=profile.chunk_max_chars` into `engine.chunker_kwargs(...)`.
   - Inject `profile.batch_size` into `EngineDeps` (new field) so `IngestionService` can read it.
6. **`services/ingestion.py`**:
   - `IngestionService.__init__` accepts `batch_size: int` (constructor arg), drop the `from dbs_vector.config import settings` line at line 11 (now passed in).
   - Replace `settings.batch_size` at line 106 with `self.batch_size`.
7. **`config.yaml`** rewritten per §10.
8. **CLI** (`cli.py`) — no API change visible to users; internal wiring updated where it constructs `IngestionService` to pass `batch_size`.
9. **Tests** updated:
   - `tests/unit/test_bootstrap.py` — `mock_settings` fixture now includes `profiles` dict and `engines` referencing them; `MockEngine.model` instead of `model_name`, etc.
   - New `tests/unit/test_model_registry.py` — register/get/duplicate/unknown.
   - New `tests/unit/test_profile_math.py` — `estimate_peak_buffer_bytes`, `recommend_profile` happy path + cap reduction loop.
   - New `tests/unit/test_hardware.py` — `detect_memory_budget_gb` with mocked `mx.metal.device_info`; missing → None; `resolve_memory_budget_gb` precedence (configured > detected > raise).
   - New `tests/unit/test_config_validation.py` — every fail-fast path in `_validate_config`.
   - New `tests/unit/test_tuning_profile.py` — TuningProfile parses, defaults, edge cases.
   - Existing `tests/integration/test_granite_engines.py` updated for new YAML shape.
10. **Docs** updated:
    - `CLAUDE.md` — config schema change, batch_size moved to profile, memory_budget_gb auto-detected.
    - `docs/README_EMBEDDINGS.md` — new "Tuning profiles" section.
    - `docs/superpowers/specs/2026-05-06-tuning-profiles-design.md` — this file.

### What does *not* change

- LanceDB table schemas (no embeddings re-indexed).
- `MLXEmbedder` constructor signature (same kwargs, different upstream source).
- CLI argument surface for `ingest` / `search` / `serve` / `mcp`.
- `ComponentRegistry` for mappers/chunkers (untouched).

### What the user must do after upgrading

Rewrite `config.yaml` per §10. The validator will print clear errors against the old schema (missing `model`, missing `tuning_profile`) so the migration path is debuggable.

---

## 12. Testing strategy

| Module | Test file | Coverage |
|---|---|---|
| `core/model_registry.py` | `tests/unit/test_model_registry.py` | register / get / duplicate raises / unknown raises / built-ins present |
| `core/profile_math.py` | `tests/unit/test_profile_math.py` | `estimate_peak_buffer_bytes` with known calibration point; `recommend_profile` happy path + cap reduction loop + chunker-type heuristic |
| `infrastructure/hardware.py` | `tests/unit/test_hardware.py` | `detect_memory_budget_gb` with mocked `mx.metal.device_info`; missing → None; `resolve_memory_budget_gb` precedence (configured > detected > raise) |
| `config.py` | `tests/unit/test_config_validation.py` | unknown model → ValueError; unknown profile → ValueError; profile.max_token > model cap → ValueError; profile would OOM → ValueError; chunk_max_chars warning logged |
| `config.py` | `tests/unit/test_tuning_profile.py` | YAML round-trip; required fields; field types |
| `services/bootstrap.py` | `tests/unit/test_bootstrap.py` (updated) | resolves contract + profile; passes correct values to MLXEmbedder; passes batch_size to EngineDeps |
| `services/ingestion.py` | `tests/unit/test_ingestion.py` (updated) | accepts batch_size kwarg; uses it in `_batched` |
| Integration | `tests/integration/test_granite_engines.py` (updated) | end-to-end with new YAML shape; both md-granite and sql-granite |

Every fail-fast path gets a unit test. The validator must produce error messages that name the offending engine + the offending value + the remediation.

### Negative tests are mandatory

For each validation rule, write a test that constructs the failing config, calls `load_settings()`, and asserts the expected error message substring. Without these, the validation logic is just a comment.

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| **`_PEAK_BUFFER_OVERHEAD` mis-calibrated.** Real MLX overhead has constants we don't perfectly model. | Calibrated from the user's empirical OOM (41 GB at batch=64 seq=16384, raw 34.4 GB → 1.19× per-element). Set to 3.0× to also cover weights + KV cache + activations. Single tunable scalar; bump it if false-negatives appear. |
| **`mx.metal.device_info()` unavailable on user's MLX version.** | Fall through to "set memory_budget_gb explicitly" error with clear remediation. Tested. |
| **Config migration friction.** | The new schema is materially simpler. Validator produces engine-specific error messages. README_EMBEDDINGS.md gets a migration section. |
| **Profile-name typos.** | Validator catches at startup. Error names the engine + the bad key + lists valid keys. |
| **Runtime memory pressure exceeds budget mid-batch** (other processes, OS pressure). | Out of scope for fail-fast validator. Document `system.memory_budget_gb` override for users in shared environments. |
| **Adding a new model still requires code changes.** | Intentional. The contract is exactly the set of fields where misconfiguration silently corrupts data; we *want* code review on additions. Document the `register()` pattern in README_EMBEDDINGS.md. |

---

## 14. Acceptance criteria

This PR is done when:

1. `uv run poe check` passes (213+ tests, no regressions).
2. The user's calibration crash (`md-granite`, max_token=16384, batch=64) is caught by `_validate_config` at `load_settings()` time, producing a remediation message naming the failing profile and suggesting concrete safer values (computed by `recommend_profile`).
3. `dbs-vector ingest "docs/" --type md-granite` works with `tuning_profile: granite-md-large` (16384 / 6000 / 8).
4. Removing `batch_size` from `system:` does not break ingestion; per-engine `batch_size` from profile is what the ingester sees.
5. Six existing engines continue to ingest and search correctly with the new YAML shape.
6. `_validate_config` rejects: unknown model key, unknown profile key, profile.max_token > model cap, profile that would OOM. Each with a test.
7. Memory budget auto-detection succeeds on the user's Apple Silicon machine; `system.memory_budget_gb: 22.0` override path is exercised by a test.
8. CLAUDE.md and README_EMBEDDINGS.md describe the new config shape and the model-registration pattern.

---

## 15. Phase 2 / Phase 3 (out of scope)

### Phase 2 — Jira chunker

Independent feature. Requires:
- New `JiraChunker` class implementing `IChunker`.
- Register in `ComponentRegistry._chunkers["jira"]`.
- Optional new fields on `ModelContract` if jira's content shape benefits from a separate model contract (likely reuse `granite-r2`).
- New profile, e.g. `granite-jira` or reuse `granite-md-large`.
- New engine block `md-jira-granite` referencing them.
- Cross-referencing jira issues ↔ SQL queries is itself a third feature (search-time join or stored linkage).

This spec is designed so Phase 2 is a config + new chunker class change, with no rewiring of the core registry/profile system.

### Phase 3 — `dbs-vector tune` CLI

```
dbs-vector tune validate
dbs-vector tune recommend --engine <name>
dbs-vector tune list
```

The validation logic and `recommend_profile` helper from this spec are reused as-is; Phase 3 is purely a CLI layer on top.
