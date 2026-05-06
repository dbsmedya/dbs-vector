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
        self,
        chunk_max_chars: int,
        query_override: str | None = None,
        url_override: str | None = None,
    ) -> dict[str, object]:
        """Resolve chunker init kwargs. `chunk_max_chars` is injected by the
        caller from the resolved tuning profile (no longer a field on Engine)."""
        if self.chunker_type == "duckdb":
            return {"query": query_override or self.duckdb_query}
        if self.chunker_type == "api":
            kwargs: dict[str, object] = {
                "base_url": url_override or self.api_base_url,
                "api_key": self.api_key,
                "page_size": self.api_page_size,
                "since_days": self.api_since_days,
                "timeout_sec": self.api_timeout_sec,
                "min_execution_ms": self.api_min_execution_ms,
            }
            if self.api_database:
                kwargs["database"] = self.api_database
            if query_override:
                kwargs["custom_query"] = query_override
            return kwargs
        if chunk_max_chars > 0:
            return {"max_chars": chunk_max_chars}
        return {}
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
from pydantic import BaseModel, ConfigDict, Field

class TuningProfile(BaseModel):
    """Three numeric knobs validated against the engine's model contract and
    available memory at load time."""

    model_config = ConfigDict(extra="forbid")  # reject unknown fields → typo guard

    max_token_length: int = Field(gt=0)        # ≤ contract.model_max_token_length
    chunk_max_chars: int = Field(ge=0)         # 0 = atomic; >0 = merge until limit
    batch_size: int = Field(gt=0)              # passed to IngestionService
```

Pydantic enforces the numeric bounds at parse time; the engine-vs-model and memory checks run later in `_validate_config`. Negative or zero values fail with a clean Pydantic error before the validator gets involved.

### `config.py` — `Settings`

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DBS_",
        env_file=".env",
        extra="ignore",
        # NOTE: extra="ignore" because BaseSettings reads environment variables
        # at construction time, and dev/CI environments often have stale or
        # unrelated DBS_* env vars. extra="forbid" would crash on those. YAML
        # strictness (typo detection in the system: block) is enforced
        # separately by _apply_system_config — see §7.
    )

    db_path: str = "./lancedb_dbs_vector"
    nprobes: int = Field(default=20, gt=0)
    log_level: str = "INFO"
    log_serialize: bool = False

    memory_budget_gb: float | None = Field(default=None, gt=0)   # NEW
    # None = auto-detect from MLX. Must be > 0 if set.

    profiles: dict[str, TuningProfile] = {}    # NEW
    engines: dict[str, EngineConfig] = {}

    # REMOVED: batch_size  (now per-profile)
```

`EngineConfig` also gets `model_config = ConfigDict(extra="forbid")` so legacy fields like `model_name`, `vector_dimension`, `attention_mask_dtype`, `max_token_length`, `chunk_max_chars`, `batch_size` will surface as a Pydantic error when migrating from old YAML — see Fix 7 / §11 migration handling.

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
    return EngineDeps(
        embedder=embedder,
        store=store,
        chunker=chunker,
        workflow=engine.workflow,
        batch_size=profile.batch_size,        # NEW field — injected for IngestionService
    )
```

### Final `EngineDeps` shape

```python
class EngineDeps(NamedTuple):
    """Resolved per-engine runtime dependencies."""
    embedder: Any
    store: Any
    chunker: Any
    workflow: str
    batch_size: int   # NEW: from the resolved TuningProfile
```

`IngestionService.__init__` accepts `batch_size: int` and stores it as `self.batch_size`. `cli.py` passes `deps.batch_size` when constructing the service. The module-level `from dbs_vector.config import settings` import in `services/ingestion.py` is removed (no longer reads `settings.batch_size`).

---

## 7. Validation chain (fail-fast on explicit config load)

### When validation runs (and when config is read at all)

**The module-level singleton must perform zero file IO at import time.** Previously `settings = load_settings()` would `yaml.safe_load(config_file)` at import; that fails on malformed YAML *before* any validator runs, breaking `dbs-vector --help`, `dbs-vector --version`, IDE module loaders, and `pytest` collection whenever a project's `config.yaml` is broken or experimental. Pydantic-settings additionally reads `.env` if `env_file=".env"` is set on `model_config`, so a naive `settings = Settings()` would still touch the disk.

The fix has two parts:

1. **Module singleton skips both YAML and `.env` read.** `settings = Settings(_env_file=None)` — `_env_file=None` is pydantic-settings' explicit override that disables the configured env-file for this instance. Defaults + non-file env vars only. No YAML read. No `.env` read. No MLX import. No validation.

2. **Every runtime caller explicitly loads.** Before any code consumes `settings.engines`, the caller must call `load_settings(config_file, validate=True)` (which constructs its own `Settings()` *without* `_env_file=None`, so `.env` IS read at runtime) and copy the loaded fields onto the singleton via `_populate_singleton_from(new_settings)`.

Caller table:

| Caller | Exact call | Validates? |
|---|---|---|
| Module-import singleton (`config.py` bottom) | `settings = Settings()` | **No I/O** |
| CLI callback (`cli.py main()`) | `new = load_settings(config_file, validate=True)`; copy onto `settings` | **Yes** |
| API lifespan (`api/main.py`) | `new = load_settings(os.environ.get("DBS_CONFIG_FILE", "config.yaml"), validate=True)`; copy onto `settings`; then `initialize_services()` | **Yes** |
| MCP standalone (`api/mcp_server.py` if invoked outside FastAPI) | Same pattern as API lifespan | **Yes** |
| Tests: malformed-config negative test | `load_settings(path, validate=False)` (file IO occurs but validation skipped) | **No** |
| Tests: validation-coverage test | `load_settings(path, validate=True)` | **Yes** |

The function signature: `load_settings(config_file: str, validate: bool = False) -> Settings`. Default is `validate=False` (cheaper for tests that want raw parsing); runtime callers explicitly pass `validate=True`.

```python
# config.py — module bottom
# _env_file=None disables pydantic-settings' .env file read for this instance.
# Zero I/O at import; populated later by CLI callback or API lifespan.
settings = Settings(_env_file=None)
```

```python
# cli.py — main() callback (when subcommand is invoked)
new_settings = load_settings(config_file, validate=True)
settings.db_path = new_settings.db_path
settings.nprobes = new_settings.nprobes
settings.engines = new_settings.engines
settings.profiles = new_settings.profiles
settings.memory_budget_gb = new_settings.memory_budget_gb
settings.log_level = new_settings.log_level
settings.log_serialize = new_settings.log_serialize
```

```python
# api/main.py — lifespan (NEW: load config before initialize_services)
@asynccontextmanager
async def lifespan(app):
    config_file = os.environ.get("DBS_CONFIG_FILE", "config.yaml")
    new_settings = load_settings(config_file, validate=True)
    # copy onto module-level singleton (same field list as cli.py)
    for field in ("db_path", "nprobes", "engines", "profiles",
                  "memory_budget_gb", "log_level", "log_serialize"):
        setattr(settings, field, getattr(new_settings, field))
    initialize_services()
    ...
```

Why this is provably import-safe:

| Failure mode | Behavior |
|---|---|
| `config.yaml` missing | `settings = Settings()` succeeds; CLI/API loader handles missing file at runtime. |
| `config.yaml` malformed YAML | `import dbs_vector.config` succeeds; the YAML error is raised by the *runtime* `load_settings()` call inside CLI callback or API lifespan, where it can be reported cleanly. |
| `config.yaml` old schema | Same as above — caught by `_apply_system_config` / `_raise_migration_hint` at runtime, not at import. |
| `dbs-vector --help` / `--version` | Typer callback short-circuits before `load_settings()` (`cli.py:54-55`); module import did no I/O. |
| Direct `python -c "import dbs_vector.api.main"` (no Typer) | Module imports succeed with empty `settings`. App routes that read `settings.engines` get `KeyError` until lifespan populates them — same lifecycle as before, just made explicit. |

`_validate_config(settings)` additionally short-circuits when `not settings.engines` as defense-in-depth.

### Validation rules

After parsing YAML and constructing all profiles + engines, run `_validate_config(settings)`. Any failure raises `ValueError` with a clear remediation message before `load_settings()` returns.

For each `(engine_name, engine)` in `settings.engines.items()`:

1. **Model contract exists.**
   `ModelRegistry.get(engine.model)` — else: `"Engine 'md-granite' references unknown model contract 'granite-r3'. Known: ['gemma-bf16', 'granite-r2']."`

2. **Tuning profile exists.**
   `settings.profiles[engine.tuning_profile]` — else: `"Engine 'md-granite' references unknown tuning profile 'granite-md-extreme'. Known: [...]"`

3. **Profile fits model.**
   `profile.max_token_length ≤ contract.model_max_token_length` — else: `"Profile 'granite-md-large' requires 16384 tokens but engine 'md' uses model 'gemma-bf16' (cap 2048). Lower profile.max_token_length or pick a different model."`

4. **Profile fits memory budget.**
   `estimate_peak_buffer_bytes(profile, contract) ≤ memory_budget_bytes × 0.9` — else:
   ```
   Profile 'granite-md-extreme' on engine 'md-granite' fails the memory budget:
     conservative estimate: 103.2 GB  (3.0× safety factor over raw attention buffer)
     budget (Metal max buffer × 0.9): 20.4 GB
   For reference: the raw attention buffer alone at this config is ~34 GB; the
   2025-05 calibration crash measured 41 GB before OOM. The 3.0× factor adds
   headroom for KV cache, weights, and activations.
   Suggested values from recommend_profile(): max_token_length=16384, batch_size=N.
   Edit config.yaml.
   ```
   The `× 0.9` leaves 10% headroom for OS/other allocations. **The error message distinguishes the conservative estimate (used to fail) from the raw / observed values** (cited for context) so the user understands why a config that "should fit 22 GB" is being rejected. Suggested values come from `recommend_profile(contract, memory_budget_gb, target_chunker=engine.chunker_type)` (§9).

5. **Chunk-vs-token sanity (warn only).**
   `profile.chunk_max_chars > 0 and profile.chunk_max_chars > profile.max_token_length × 4` → `logger.warning(...)`.
   *(Rough char-per-token ratio of 4. Caller may legitimately want oversized chunks if they expect heavy truncation; this is a heads-up, not a fail.)*

### Validation scope: all configured engines

Validation walks **every** `(engine_name, engine)` in `settings.engines`, not just the engine the CLI is about to use. Reasons:
- The user is editing `config.yaml`; they should see all errors at once (`serve`/`mcp` load all engines).
- Per-engine checks are O(1); engines are few.
- A latent misconfiguration in an unused engine becomes a future production-time crash.

The memory check is **per-engine**, not summed across engines. Concurrent memory pressure (multiple engines loaded simultaneously by `serve`/`mcp`) is out of scope for the fail-fast validator — MLX models load lazily on first use and the 22 GB cap is per-buffer, not total. Document `system.memory_budget_gb` as an override for shared environments.

Validation runs once on explicit config load. CLI `ingest` / `search` / `serve` / `mcp` all hit the validated path through the typer callback; the module-import singleton stays unvalidated by design (see "When validation runs" above).

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

The recommender prioritizes **preserving the user's intended sequence length** over throughput. If the user has a profile that fails validation, they almost certainly want their target context length and need a smaller batch — not a smaller context.

**Strategy (in order):**
1. Start from `target_seq_len` (the user's *intended* `max_token_length`, defaulting to the existing failing profile's value or `contract.model_max_token_length` if no current profile).
2. Find the largest `batch_size ≥ 1` that fits the memory budget at that seq length.
3. If `batch_size = 0` (even batch=1 won't fit), halve `seq_len` and retry — but *only as a last resort*; the message must call this out so the user understands their context budget was reduced.
4. Pick `chunk_max_chars` from the chunker type heuristic.

```python
def recommend_profile(
    contract: "ModelContract",
    memory_budget_gb: float,
    target_chunker: str = "document",
    target_seq_len: int | None = None,
) -> dict[str, int | bool]:
    """Suggest profile values that fit memory_budget_gb for this contract.

    Args:
        target_seq_len: The user's intended max_token_length. The recommender
            will preserve this value if at all possible (favoring smaller batch
            size) and only halve it as a last resort. Defaults to
            contract.model_max_token_length.

    Returns: dict with keys:
        max_token_length, chunk_max_chars, batch_size,
        seq_len_reduced (bool — True if step 3 fired, signaling that the
        recommendation could not preserve the requested context length).
    """
    budget = int(memory_budget_gb * 1024 ** 3 * 0.9)
    seq = target_seq_len or contract.model_max_token_length
    seq = min(seq, contract.model_max_token_length)  # clamp to model cap
    seq_len_reduced = False

    while seq >= 512:
        per_sample = int(_PEAK_BUFFER_OVERHEAD * seq ** 2 * contract.compute_dtype_bytes)
        max_batch = budget // per_sample if per_sample > 0 else 0
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
                "seq_len_reduced": seq_len_reduced,
            }
        seq //= 2
        seq_len_reduced = True

    raise ValueError(
        f"No profile fits {memory_budget_gb} GB for model with cap "
        f"{contract.model_max_token_length}. Reduce model or increase budget."
    )
```

**Validator integration:** when validation rule #4 fires, the validator passes the failing profile's `max_token_length` as `target_seq_len`. So the user with `(seq=16384, batch=64)` gets a recommendation of `(seq=16384, batch=N)` — same context, smaller batch — not `(seq=32768, batch=tiny)` (the original spec text was wrong about this). If `seq_len_reduced=True` in the result, the error message includes a "your context length was reduced from X to Y" line so the user understands the trade-off.

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
8. **`config.py` module bottom**: replace `settings = load_settings()` with `settings = Settings()` (no I/O at import). Update `load_settings()` signature default to `validate=False`. See §7 for full rationale.
9. **`api/main.py` lifespan**: explicitly call `load_settings(os.environ.get("DBS_CONFIG_FILE", "config.yaml"), validate=True)` and copy fields onto the singleton *before* `initialize_services()`. Add an import for `os` and `load_settings`. See §7 example. Also update `/health` endpoint at `api/main.py:90-91` (`config.model_name`) — `model_name` is no longer on `EngineConfig`; resolve via `ModelRegistry.get(config.model).model_name`.
10. **`api/mcp_server.py`** (if it can be invoked standalone outside the FastAPI lifespan): same lifespan-style explicit load. The current `dbs-vector mcp` CLI subcommand routes through Typer first, so the CLI callback already handles it; only direct `python -m dbs_vector.api.mcp_server` style invocations need extra wiring (note in spec, no code change needed if not used).
11. **CLI** (`cli.py`):
   - The Typer `main()` callback (`cli.py:36-74`) currently copies fields from the freshly-loaded `new_settings` onto the global singleton. Update the field copy:
     ```python
     # cli.py main() — before
     settings.db_path = new_settings.db_path
     settings.batch_size = new_settings.batch_size      # REMOVE
     settings.nprobes = new_settings.nprobes
     settings.engines = new_settings.engines
     settings.log_level = new_settings.log_level
     settings.log_serialize = new_settings.log_serialize

     # cli.py main() — after
     settings.db_path = new_settings.db_path
     settings.nprobes = new_settings.nprobes
     settings.engines = new_settings.engines
     settings.log_level = new_settings.log_level
     settings.log_serialize = new_settings.log_serialize
     settings.profiles = new_settings.profiles                  # NEW
     settings.memory_budget_gb = new_settings.memory_budget_gb  # NEW
     ```
   - Where each command builds `IngestionService(...)`, pass `batch_size=deps.batch_size` (from the new `EngineDeps.batch_size` field).
   - No public CLI argument changes. `dbs-vector --help` and `dbs-vector --version` continue to work because the callback short-circuits before `load_settings()` when no subcommand is invoked (`cli.py:54-55` already handles this).
12. **Tests** updated:
    - `tests/unit/test_bootstrap.py` — `mock_settings` fixture now includes `profiles` dict and `engines` referencing them; `MockEngine.model` instead of `model_name`, etc.
    - New `tests/unit/test_model_registry.py` — register/get/duplicate/unknown.
    - New `tests/unit/test_profile_math.py` — `estimate_peak_buffer_bytes`, `recommend_profile` happy path + cap reduction loop.
    - New `tests/unit/test_hardware.py` — `detect_memory_budget_gb` with mocked `mx.metal.device_info`; missing → None; `resolve_memory_budget_gb` precedence (configured > detected > raise).
    - New `tests/unit/test_config_validation.py` — every fail-fast path in `_validate_config`.
    - New `tests/unit/test_tuning_profile.py` — TuningProfile parses, defaults, edge cases.
    - New `tests/unit/test_config_import_safety.py` — importing `dbs_vector.config` performs no file I/O (assert via mock that `Path.open` is not called); `dbs_vector.config.settings` after import equals `Settings()` defaults.
    - Existing `tests/integration/test_granite_engines.py` updated for new YAML shape.
13. **Docs** updated:
    - `CLAUDE.md` — config schema change, batch_size moved to profile, memory_budget_gb auto-detected, model registry pattern.
    - `docs/README_EMBEDDINGS.md` — new "Tuning profiles" section + migration walkthrough referenced from `_raise_migration_hint`.
    - `docs/README_DOCS.md` — currently describes a global `batch_size`; rewrite to point at profiles + per-engine batch.
    - `docs/README_SQL.md`, `docs/README_REMOTE_SQL_API.md`, `docs/README_ARCHITECTURE.md`, `docs/README.md`, root `README.md` — sweep for any references to `system.batch_size` or per-engine `max_token_length`/`chunk_max_chars` and update to the profile-based shape.
    - `docs/superpowers/specs/2026-05-06-tuning-profiles-design.md` — this file.

### What does *not* change

- LanceDB table schemas (no embeddings re-indexed).
- `MLXEmbedder` constructor signature (same kwargs, different upstream source).
- CLI argument surface for `ingest` / `search` / `serve` / `mcp`.
- `ComponentRegistry` for mappers/chunkers (untouched).

### What the user must do after upgrading

Rewrite `config.yaml` per §10.

### Migration error path (old YAML → new schema)

Old-schema YAML can fail in two places:

1. **Engine block** — `EngineConfig(**v)` with Pydantic's `ValidationError` because `extra="forbid"` rejects `model_name`, `vector_dimension`, `attention_mask_dtype`, etc., AND because `model` and `tuning_profile` are now required and missing.
2. **System block** — currently `system:` is loaded with a `for key, value in data["system"].items(): setattr(...)` loop using `hasattr`. Unknown keys (the legacy `batch_size`) are silently ignored. We must add an explicit gate.

`load_settings()` validates *both* blocks before model construction:

```python
def load_settings(config_file=None, validate=False):
    base_settings = Settings()
    if config_file is None:
        config_file = os.getenv("DBS_CONFIG_FILE", "config.yaml")
    yaml_path = Path(config_file)
    if not yaml_path.exists():
        return base_settings
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 1. system block — explicit allow-list, legacy-key detection
    if "system" in data and isinstance(data["system"], dict):
        _apply_system_config(data["system"], base_settings, config_file)

    # 2. profiles block
    if "profiles" in data and isinstance(data["profiles"], dict):
        try:
            base_settings.profiles = {
                k: TuningProfile(**v) for k, v in data["profiles"].items()
            }
        except ValidationError as e:
            _raise_migration_hint(e, config_file, where="profiles")

    # 3. engines block
    if "engines" in data and isinstance(data["engines"], dict):
        try:
            base_settings.engines = {
                k: EngineConfig(**v) for k, v in data["engines"].items()
            }
        except ValidationError as e:
            _raise_migration_hint(e, config_file, where="engines")

    if validate:
        _validate_config(base_settings, config_file)
    return base_settings


_LEGACY_SYSTEM_KEYS = {"batch_size"}   # moved to TuningProfile
_KNOWN_SYSTEM_KEYS = {                  # current Settings system-block fields
    "db_path", "nprobes", "log_level", "log_serialize", "memory_budget_gb",
}


def _apply_system_config(system: dict, settings: Settings, config_file: str) -> None:
    """Apply system: keys onto the Settings instance with strict validation."""
    legacy = sorted(set(system) & _LEGACY_SYSTEM_KEYS)
    unknown = sorted(set(system) - _KNOWN_SYSTEM_KEYS - _LEGACY_SYSTEM_KEYS)
    if legacy:
        raise ValueError(
            f"Config schema mismatch in {config_file} (system: block).\n"
            f"  Legacy keys found: {legacy}\n"
            f"  These moved to TuningProfile. Define a profile under "
            f"`profiles:` and reference it from each engine via "
            f"`tuning_profile:`. See spec §10 / README_EMBEDDINGS.md."
        )
    if unknown:
        raise ValueError(
            f"Unknown keys in {config_file} system: block: {unknown}. "
            f"Allowed: {sorted(_KNOWN_SYSTEM_KEYS)}."
        )
    for key, value in system.items():
        setattr(settings, key, value)


_LEGACY_ENGINE_FIELDS = {
    "model_name", "vector_dimension", "max_token_length",
    "attention_mask_dtype", "chunk_max_chars", "batch_size",
}
_REQUIRED_ENGINE_FIELDS = {"model", "tuning_profile"}


def _raise_migration_hint(err: ValidationError, config_file: str, where: str) -> None:
    """Detect old-schema fields and rewrite as a single migration message."""
    seen_legacy = {
        e["loc"][-1] for e in err.errors()
        if e["loc"][-1] in _LEGACY_ENGINE_FIELDS
    }
    missing_required = {
        e["loc"][-1] for e in err.errors()
        if e["type"] == "missing" and e["loc"][-1] in _REQUIRED_ENGINE_FIELDS
    }
    if seen_legacy or missing_required:
        raise ValueError(
            f"Config schema mismatch in {config_file} ({where}: block).\n"
            f"  Legacy per-engine fields found: {sorted(seen_legacy) or 'none'}\n"
            f"  Missing new required fields: {sorted(missing_required) or 'none'}\n"
            f"See spec §10 / README_EMBEDDINGS.md for the new schema."
        ) from err
    raise err  # genuine new-schema error — propagate
```

Migration coverage:

| Legacy thing | Where it's detected | Outcome |
|---|---|---|
| `system.batch_size: 64` | `_apply_system_config` legacy-key check | Raise migration hint, name `batch_size`, point at `profiles:` |
| Unknown `system:` key (typo) | `_apply_system_config` unknown-key check | Raise with allow-list |
| `engines.md.model_name: ...` | `_raise_migration_hint(where="engines")` via Pydantic | Raise migration hint, name legacy fields |
| Missing `engines.md.tuning_profile` | Same | Raise migration hint, name missing required |
| Genuine bug in new-shape YAML | Falls through, raw `ValidationError` | Propagate unchanged |

---

## 12. Testing strategy

| Module | Test file | Coverage |
|---|---|---|
| `core/model_registry.py` | `tests/unit/test_model_registry.py` | register / get / duplicate raises / unknown raises / built-ins present |
| `core/profile_math.py` | `tests/unit/test_profile_math.py` | `estimate_peak_buffer_bytes` against known calibration point (raw 34.4 GB → conservative ~103 GB at 3.0×); `recommend_profile` preserves `target_seq_len`; `seq_len_reduced` flag set when halving fires; chunker-type heuristic for chunk_max_chars |
| `infrastructure/hardware.py` | `tests/unit/test_hardware.py` | `detect_memory_budget_gb` with mocked `mx.metal.device_info`; missing-MLX path → None; `resolve_memory_budget_gb` precedence (configured > detected > raise) |
| `config.py` | `tests/unit/test_config_validation.py` | unknown model → ValueError; unknown profile → ValueError; profile.max_token > model cap → ValueError; profile would OOM → ValueError including conservative/raw/observed numbers; chunk_max_chars warning logged; module-import singleton (no args) does NOT validate |
| `config.py` | `tests/unit/test_tuning_profile.py` | YAML round-trip; required fields; numeric bounds (`batch_size <= 0`, `max_token_length <= 0`, `chunk_max_chars < 0` rejected by Pydantic); `extra="forbid"` rejects unknown fields |
| `config.py` | `tests/unit/test_migration_hint.py` (NEW) | `_apply_system_config` raises migration hint for `system.batch_size`; raises with allow-list for unknown system keys; `_raise_migration_hint` fires for legacy per-engine fields and for missing `model`/`tuning_profile`; does NOT fire for genuinely-broken-but-new-shape YAML (raw `ValidationError` propagates) |
| `services/bootstrap.py` | `tests/unit/test_bootstrap.py` (updated) | resolves contract + profile; passes correct values to MLXEmbedder; populates `EngineDeps.batch_size` from profile |
| `services/ingestion.py` | `tests/unit/test_ingestion.py` (updated) | accepts `batch_size` kwarg in `__init__`; uses `self.batch_size` in `_batched`; module no longer imports `settings` |
| `cli.py` | `tests/unit/test_cli_callback.py` (NEW) | `dbs-vector --help` and `dbs-vector --version` succeed without loading config (tmpdir without `config.yaml`); subcommand path copies `profiles` and `memory_budget_gb` onto the singleton; does not copy `batch_size` |
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
2. The user's calibration crash (`md-granite`, max_token=16384, batch=64) is caught by `_validate_config` at `load_settings()` time, producing a remediation message that:
   - Names the failing engine + profile.
   - Distinguishes conservative estimate (~103 GB) from raw attention buffer (~34 GB) and the observed OOM data point (~41 GB).
   - Suggests concrete safer values from `recommend_profile()` that **preserve `max_token_length=16384`** and lower `batch_size` (not the other way around).
3. `dbs-vector ingest "docs/" --type md-granite` works with `tuning_profile: granite-md-large` (16384 / 6000 / 8).
4. Removing `batch_size` from `system:` does not break ingestion; per-engine `batch_size` from profile is what the ingester sees (`EngineDeps.batch_size` → `IngestionService.batch_size`).
5. Six existing engines continue to ingest and search correctly with the new YAML shape.
6. `_validate_config` rejects: unknown model key, unknown profile key, profile.max_token > model cap, profile that would OOM. Each with a test. TuningProfile fields with `batch_size <= 0`, `max_token_length <= 0`, `chunk_max_chars < 0` are rejected by Pydantic before the validator runs (separate test).
7. Memory budget auto-detection succeeds on the user's Apple Silicon machine; `system.memory_budget_gb: 22.0` override path is exercised by a test; missing-MLX path falls through to the explicit "set memory_budget_gb" error (mocked test).
8. CLAUDE.md and README_EMBEDDINGS.md describe the new config shape and the model-registration pattern. All docs in the §11 update list are swept for stale `system.batch_size` references.
9. **`dbs-vector --help` and `dbs-vector --version` work without performing any config file I/O at module import time**, asserted in three scenarios:
   - tmpdir with **no `config.yaml`** present;
   - tmpdir with a **malformed `config.yaml`** (broken YAML syntax — `yaml.safe_load` would raise `YAMLError` if called);
   - tmpdir with an **old-schema `config.yaml`** (legacy `system.batch_size`, legacy per-engine fields).
   In all three, `dbs-vector --help` and `--version` exit 0 with their normal output. **The malformed-config case is the proof point**: a module that read YAML at import would crash here. The fix is `settings = Settings()` at module bottom — zero I/O — with all real loading deferred to the CLI callback or API lifespan. A separate unit test (`test_config_import_safety.py`) asserts `Path.open` is never called during `import dbs_vector.config`.
10. **Old-schema YAML produces a single migration-hint error** that names the legacy fields detected and points at `docs/superpowers/specs/2026-05-06-tuning-profiles-design.md` §10 + `docs/README_EMBEDDINGS.md`. Asserted by:
    - a test for `_apply_system_config` that the message names `system.batch_size`;
    - a test for `_raise_migration_hint` that the message names legacy per-engine fields (e.g., `model_name`, `attention_mask_dtype`);
    - a test that the message names missing required fields (`model`, `tuning_profile`);
    - a negative test that genuinely-broken new-schema YAML produces the raw `ValidationError`, not the migration hint.

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
