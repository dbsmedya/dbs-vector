# Tuning Profiles & Model Contract Registry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-layer config system (hardcoded `ModelRegistry`, YAML `EngineConfig`, YAML `TuningProfile`) that closes the OOM hit when `md-granite` is bumped from 8K → 16K context, by validating per-engine memory budgets at runtime config-load time.

**Architecture:** Hardcoded model contracts (open/closed registry, code change to add) carry `vector_dimension`, `model_max_token_length`, `attention_mask_dtype`, `compute_dtype_bytes`. Engines reference contracts and tuning profiles by name. Profiles hold the three numeric knobs (`max_token_length`, `chunk_max_chars`, `batch_size`) that vary with hardware. Config validation happens only on explicit runtime loads (CLI callback, API lifespan); module import performs zero file I/O so `--help` / `--version` survive a malformed `config.yaml`.

**Tech Stack:** Python 3.12, Pydantic v2 (`BaseModel`, `BaseSettings`, `ConfigDict(extra="forbid")`, `Field(gt=0)`), pytest, MLX (`mlx.core.metal.device_info` for memory budget auto-detect), Typer, FastAPI.

**Spec:** `docs/superpowers/specs/2026-05-06-tuning-profiles-design.md`

---

## File Structure

### New source files
- `src/dbs_vector/core/model_registry.py` — `ModelContract` dataclass + `ModelRegistry` open/closed class + 2 built-in registrations.
- `src/dbs_vector/core/profile_math.py` — `estimate_peak_buffer_bytes`, `recommend_profile`, `_PEAK_BUFFER_OVERHEAD`, `_CHARS_PER_TOKEN`. Pure, no external deps.
- `src/dbs_vector/infrastructure/hardware.py` — `detect_memory_budget_gb`, `resolve_memory_budget_gb`. Lazy `mlx.core` import.

### Modified source files
- `src/dbs_vector/config.py` — add `TuningProfile`, `Settings.profiles`, `Settings.memory_budget_gb`; remove `Settings.batch_size`; shrink `EngineConfig` (remove model-contract fields, add `model` + `tuning_profile`); update `chunker_kwargs()` signature; replace module-bottom `settings = load_settings()` with `settings = Settings(_env_file=None)` (zero I/O at import — disables `.env` read); flip `load_settings(config_file: str | None = None, validate: bool = False)`; add `_apply_system_config`, `_raise_migration_hint`, `_validate_config`.
- `src/dbs_vector/services/bootstrap.py` — resolve through `ModelRegistry` + `Settings.profiles`; populate new `EngineDeps.batch_size`.
- `src/dbs_vector/services/ingestion.py` — `IngestionService.__init__` accepts `batch_size: int`; drop the module-level `from dbs_vector.config import settings` import; replace `settings.batch_size` with `self.batch_size`.
- `src/dbs_vector/cli.py` — Typer callback singleton-copy adds `profiles` + `memory_budget_gb`, drops `batch_size`; CLI commands pass `deps.batch_size` to `IngestionService`.
- `src/dbs_vector/api/main.py` — `lifespan` explicitly calls `load_settings(config_file, validate=True)` and copies fields onto the singleton before `initialize_services()`; `/health` resolves `model_name` via `ModelRegistry.get(config.model).model_name`.

### Modified config + docs
- `config.yaml` — full rewrite per spec §10.
- `CLAUDE.md`, `docs/README_EMBEDDINGS.md`, `docs/README_DOCS.md`, `docs/README_SQL.md`, `docs/README_REMOTE_SQL_API.md`, `docs/README_ARCHITECTURE.md`, `docs/README.md`, root `README.md` — sweep for `system.batch_size` and per-engine `max_token_length`/`chunk_max_chars` references.

### New test files
- `tests/unit/test_model_registry.py`
- `tests/unit/test_profile_math.py`
- `tests/unit/test_hardware.py`
- `tests/unit/test_tuning_profile.py`
- `tests/unit/test_migration_hint.py`
- `tests/unit/test_config_validation.py`
- `tests/unit/test_config_import_safety.py`
- `tests/unit/test_cli_callback.py`

### Modified test files
- `tests/unit/test_bootstrap.py` — `mock_settings` shape change.
- `tests/unit/test_ingestion.py` — `IngestionService` constructor arg.
- `tests/integration/test_granite_engines.py` — YAML fixture in new schema.

---

## Task 1: ModelRegistry

**Files:**
- Create: `src/dbs_vector/core/model_registry.py`
- Test: `tests/unit/test_model_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_model_registry.py
import pytest

from dbs_vector.core.model_registry import ModelContract, ModelRegistry


def test_get_returns_gemma_contract():
    contract = ModelRegistry.get("gemma-bf16")
    assert contract.model_name == "mlx-community/embeddinggemma-300m-bf16"
    assert contract.vector_dimension == 768
    assert contract.model_max_token_length == 2048
    assert contract.attention_mask_dtype == "float16"
    assert contract.compute_dtype_bytes == 2


def test_get_returns_granite_contract():
    contract = ModelRegistry.get("granite-r2")
    assert contract.model_name == "ibm-granite/granite-embedding-311m-multilingual-r2"
    assert contract.vector_dimension == 768
    assert contract.model_max_token_length == 32768
    assert contract.attention_mask_dtype is None
    assert contract.compute_dtype_bytes == 2


def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown model contract 'nonexistent'"):
        ModelRegistry.get("nonexistent")


def test_get_unknown_lists_known_keys():
    with pytest.raises(KeyError, match=r"Known: \['gemma-bf16', 'granite-r2'\]"):
        ModelRegistry.get("nonexistent")


def test_register_duplicate_raises_valueerror():
    duplicate = ModelContract(
        model_name="x",
        vector_dimension=1,
        model_max_token_length=1,
        attention_mask_dtype=None,
        compute_dtype_bytes=2,
    )
    with pytest.raises(ValueError, match="already registered"):
        ModelRegistry.register("gemma-bf16", duplicate)


def test_keys_returns_sorted():
    keys = ModelRegistry.keys()
    assert "gemma-bf16" in keys
    assert "granite-r2" in keys
    assert keys == sorted(keys)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_model_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.core.model_registry'`

- [ ] **Step 3: Implement ModelContract + ModelRegistry**

```python
# src/dbs_vector/core/model_registry.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContract:
    """Immutable model contract. All fields are properties of the model itself,
    not of any particular deployment that uses the model."""

    model_name: str
    vector_dimension: int
    model_max_token_length: int
    attention_mask_dtype: str | None
    compute_dtype_bytes: int = 2


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


# Built-in registrations
ModelRegistry.register(
    "gemma-bf16",
    ModelContract(
        model_name="mlx-community/embeddinggemma-300m-bf16",
        vector_dimension=768,
        model_max_token_length=2048,
        attention_mask_dtype="float16",
        compute_dtype_bytes=2,
    ),
)

ModelRegistry.register(
    "granite-r2",
    ModelContract(
        model_name="ibm-granite/granite-embedding-311m-multilingual-r2",
        vector_dimension=768,
        model_max_token_length=32768,
        attention_mask_dtype=None,
        compute_dtype_bytes=2,
    ),
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_model_registry.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/core/model_registry.py tests/unit/test_model_registry.py
git commit -m "feat(core): add ModelRegistry with gemma-bf16 and granite-r2 contracts"
```

---

## Task 2: profile_math (estimator + recommender)

**Files:**
- Create: `src/dbs_vector/core/profile_math.py`
- Test: `tests/unit/test_profile_math.py`

- [ ] **Step 1: Write failing tests for the estimator**

```python
# tests/unit/test_profile_math.py
import pytest

from dbs_vector.core.model_registry import ModelContract
from dbs_vector.core.profile_math import (
    _PEAK_BUFFER_OVERHEAD,
    estimate_peak_buffer_bytes,
    recommend_profile,
)


class FakeProfile:
    def __init__(self, max_token_length: int, batch_size: int, chunk_max_chars: int = 0) -> None:
        self.max_token_length = max_token_length
        self.batch_size = batch_size
        self.chunk_max_chars = chunk_max_chars


GRANITE = ModelContract(
    model_name="ibm-granite/granite-embedding-311m-multilingual-r2",
    vector_dimension=768,
    model_max_token_length=32768,
    attention_mask_dtype=None,
    compute_dtype_bytes=2,
)


def test_estimate_calibration_point():
    """Calibration: batch=64, seq=16384, bf16 → raw 34.4 GB; with 3.0× factor ~103 GB."""
    profile = FakeProfile(max_token_length=16384, batch_size=64)
    bytes_estimated = estimate_peak_buffer_bytes(profile, GRANITE)
    raw_attention = 64 * (16384 ** 2) * 2
    expected = int(_PEAK_BUFFER_OVERHEAD * raw_attention)
    assert bytes_estimated == expected
    # Sanity bounds: the estimate is ~103 GB (decimal). Lower bound is in
    # decimal GB to match the docstring; upper bound is in GiB for headroom.
    assert bytes_estimated > 100 * 1000 ** 3
    assert bytes_estimated < 110 * 1024 ** 3


def test_estimate_small_profile_fits_easily():
    profile = FakeProfile(max_token_length=2048, batch_size=64)
    bytes_estimated = estimate_peak_buffer_bytes(profile, GRANITE)
    assert bytes_estimated < 2 * 1024 ** 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_profile_math.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.core.profile_math'`

- [ ] **Step 3: Implement estimator**

```python
# src/dbs_vector/core/profile_math.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbs_vector.config import TuningProfile
    from dbs_vector.core.model_registry import ModelContract


# Calibrated empirically against the user's 2025-05 OOM:
#   batch=64, seq=16384, bf16 (2 bytes) → real allocation 41 GB
#   raw: 64 × 16384² × 2 = 34.4 GB. real / raw = 1.19× per-element overhead.
# We use 3.0× because the 41 GB is just the largest single buffer; total
# memory pressure also includes weights, KV cache, and activations.
_PEAK_BUFFER_OVERHEAD = 3.0

# Approximate char-per-token ratio for English+code; used by the recommender.
_CHARS_PER_TOKEN = 2.5


def estimate_peak_buffer_bytes(
    profile: "TuningProfile", contract: "ModelContract"
) -> int:
    """Approximate peak Metal memory pressure during a forward pass.

    Dominated by attention: O(batch × seq² × dtype_bytes), with a 3× safety
    factor for temporaries, weights, and KV cache. Hidden_dim drops out
    because the attention matrix is the long pole, not the activations.

    The dtype is the model's compute dtype (contract.compute_dtype_bytes),
    not the attention_mask cast — the mask is cheap; the attention buffer
    is what blows up.
    """
    return int(
        _PEAK_BUFFER_OVERHEAD
        * profile.batch_size
        * profile.max_token_length ** 2
        * contract.compute_dtype_bytes
    )
```

- [ ] **Step 4: Run estimator tests**

Run: `uv run pytest tests/unit/test_profile_math.py -v`
Expected: 2 passed

- [ ] **Step 5: Add failing tests for the recommender**

Append to `tests/unit/test_profile_math.py`:

```python
def test_recommend_preserves_target_seq_len():
    """The 16K/64 OOM case should be recommended as 16K/<smaller batch>."""
    result = recommend_profile(GRANITE, memory_budget_gb=22.0, target_seq_len=16384)
    assert result["max_token_length"] == 16384
    assert result["batch_size"] >= 1
    assert result["seq_len_reduced"] is False


def test_recommend_clamps_to_model_cap():
    result = recommend_profile(GRANITE, memory_budget_gb=22.0, target_seq_len=99999)
    assert result["max_token_length"] <= GRANITE.model_max_token_length


def test_recommend_atomic_chunks_for_sql():
    result = recommend_profile(
        GRANITE, memory_budget_gb=22.0, target_chunker="duckdb", target_seq_len=8192
    )
    assert result["chunk_max_chars"] == 0


def test_recommend_doc_chunks_use_char_heuristic():
    result = recommend_profile(
        GRANITE, memory_budget_gb=22.0, target_chunker="document", target_seq_len=8192
    )
    expected_chunk = int(8192 * 2.5 * 0.5)
    assert result["chunk_max_chars"] == expected_chunk


def test_recommend_seq_reduced_flag_when_halving():
    """A tiny budget that can't fit even batch=1 at target seq forces halving."""
    result = recommend_profile(GRANITE, memory_budget_gb=0.5, target_seq_len=32768)
    assert result["seq_len_reduced"] is True
    assert result["max_token_length"] < 32768


def test_recommend_raises_if_no_fit():
    """A pathologically tiny budget can't fit even seq=512."""
    with pytest.raises(ValueError, match="No profile fits"):
        recommend_profile(GRANITE, memory_budget_gb=0.0001, target_seq_len=512)


def test_recommend_defaults_target_to_model_cap():
    result = recommend_profile(GRANITE, memory_budget_gb=22.0)
    assert result["max_token_length"] >= 1
```

- [ ] **Step 6: Run recommender tests to verify failure**

Run: `uv run pytest tests/unit/test_profile_math.py -v`
Expected: FAIL on the new tests with `ImportError: cannot import name 'recommend_profile'`

- [ ] **Step 7: Implement recommender**

Append to `src/dbs_vector/core/profile_math.py`:

```python
def recommend_profile(
    contract: "ModelContract",
    memory_budget_gb: float,
    target_chunker: str = "document",
    target_seq_len: int | None = None,
) -> dict[str, int | bool]:
    """Suggest profile values that fit memory_budget_gb for this contract.

    Strategy (preserves user's intended context length over throughput):
      1. Start from target_seq_len (defaults to contract.model_max_token_length).
      2. Pick the largest batch_size ≥ 1 that fits at that seq length.
      3. If no batch fits, halve seq_len and retry; set seq_len_reduced=True.
      4. Pick chunk_max_chars from chunker-type heuristic.

    Returns:
        dict with keys: max_token_length, chunk_max_chars, batch_size,
        seq_len_reduced (bool — True if step 3 fired).
    """
    budget = int(memory_budget_gb * 1024 ** 3 * 0.9)
    seq = target_seq_len if target_seq_len is not None else contract.model_max_token_length
    seq = min(seq, contract.model_max_token_length)
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

- [ ] **Step 8: Run all profile_math tests**

Run: `uv run pytest tests/unit/test_profile_math.py -v`
Expected: 9 passed

- [ ] **Step 9: Commit**

```bash
git add src/dbs_vector/core/profile_math.py tests/unit/test_profile_math.py
git commit -m "feat(core): add profile_math estimator + recommender"
```

---

## Task 3: Hardware memory detection

**Files:**
- Create: `src/dbs_vector/infrastructure/hardware.py`
- Test: `tests/unit/test_hardware.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_hardware.py
from unittest.mock import patch

import pytest

from dbs_vector.infrastructure.hardware import (
    detect_memory_budget_gb,
    resolve_memory_budget_gb,
)


def test_detect_returns_gb_from_mlx():
    fake_info = {"max_buffer_length": 22 * 1024 ** 3}
    with patch("mlx.core.metal.device_info", return_value=fake_info):
        assert detect_memory_budget_gb() == pytest.approx(22.0, rel=1e-6)


def test_detect_returns_none_when_mlx_unavailable():
    with patch("mlx.core.metal.device_info", side_effect=ImportError("no metal")):
        assert detect_memory_budget_gb() is None


def test_detect_returns_none_when_key_missing():
    with patch("mlx.core.metal.device_info", return_value={}):
        assert detect_memory_budget_gb() is None


def test_resolve_configured_wins():
    with patch(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=22.0
    ):
        assert resolve_memory_budget_gb(8.0) == 8.0


def test_resolve_falls_back_to_detected():
    with patch(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=22.0
    ):
        assert resolve_memory_budget_gb(None) == 22.0


def test_resolve_raises_when_neither_available():
    with patch(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=None
    ):
        with pytest.raises(ValueError, match="Could not auto-detect"):
            resolve_memory_budget_gb(None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_hardware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.infrastructure.hardware'`

- [ ] **Step 3: Implement hardware module**

```python
# src/dbs_vector/infrastructure/hardware.py
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_hardware.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/infrastructure/hardware.py tests/unit/test_hardware.py
git commit -m "feat(infra): add Metal memory budget detection"
```

---

## Task 4: TuningProfile model (additive only)

**Files:**
- Modify: `src/dbs_vector/config.py`
- Test: `tests/unit/test_tuning_profile.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tuning_profile.py
import pytest
from pydantic import ValidationError

from dbs_vector.config import TuningProfile


def test_parses_valid_profile():
    profile = TuningProfile(max_token_length=8192, chunk_max_chars=4000, batch_size=8)
    assert profile.max_token_length == 8192
    assert profile.chunk_max_chars == 4000
    assert profile.batch_size == 8


def test_atomic_chunks_allowed():
    profile = TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=64)
    assert profile.chunk_max_chars == 0


def test_zero_max_token_length_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=0, chunk_max_chars=100, batch_size=8)


def test_negative_max_token_length_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=-1, chunk_max_chars=100, batch_size=8)


def test_zero_batch_size_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=2048, chunk_max_chars=100, batch_size=0)


def test_negative_chunk_max_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=2048, chunk_max_chars=-1, batch_size=8)


def test_extra_fields_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        TuningProfile(
            max_token_length=2048,
            chunk_max_chars=100,
            batch_size=8,
            unknown_field="x",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tuning_profile.py -v`
Expected: FAIL with `ImportError: cannot import name 'TuningProfile' from 'dbs_vector.config'`

- [ ] **Step 3: Add TuningProfile to config.py**

In `src/dbs_vector/config.py`, change the imports at the top:

```python
import os
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
```

Then add the new class right after the imports (before `class EngineConfig`):

```python
class TuningProfile(BaseModel):
    """Three numeric knobs validated against the engine's model contract and
    available memory at load time."""

    model_config = ConfigDict(extra="forbid")

    max_token_length: int = Field(gt=0)
    chunk_max_chars: int = Field(ge=0)
    batch_size: int = Field(gt=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tuning_profile.py -v`
Expected: 7 passed

- [ ] **Step 5: Run full unit suite to ensure nothing else broke**

Run: `uv run pytest tests/unit -x -q`
Expected: all pass (we only added a new class).

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/config.py tests/unit/test_tuning_profile.py
git commit -m "feat(config): add TuningProfile model with numeric bounds"
```

---

## Task 5: Settings additions, module-bottom, load_settings signature

**Files:**
- Modify: `src/dbs_vector/config.py`
- Test: `tests/unit/test_config_import_safety.py` (new)

- [ ] **Step 1: Write failing import-safety test**

```python
# tests/unit/test_config_import_safety.py
import importlib
import sys
from unittest.mock import patch


def test_importing_config_does_not_open_files(tmp_path, monkeypatch):
    """Module import must not perform any file I/O — neither config.yaml nor .env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DBS_LOG_LEVEL=DEBUG\n")  # tempt pydantic-settings
    (tmp_path / "config.yaml").write_text("system: : :\nbroken: :")  # tempt yaml.safe_load
    sys.modules.pop("dbs_vector.config", None)
    with patch("pathlib.Path.open") as mock_open, patch("builtins.open") as builtin_open:
        importlib.import_module("dbs_vector.config")
    assert mock_open.call_count == 0, f"Path.open called: {mock_open.call_args_list}"
    assert builtin_open.call_count == 0, f"builtins.open called: {builtin_open.call_args_list}"


def test_module_singleton_is_default_settings():
    """settings = Settings(_env_file=None) at module bottom — empty defaults, no I/O."""
    sys.modules.pop("dbs_vector.config", None)
    config = importlib.import_module("dbs_vector.config")
    assert config.settings.engines == {}
    assert config.settings.profiles == {}
    assert config.settings.memory_budget_gb is None


def test_load_settings_default_does_not_validate(tmp_path):
    """load_settings(path) without validate=True must not call _validate_config."""
    sys.modules.pop("dbs_vector.config", None)
    from dbs_vector.config import load_settings

    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")
    s = load_settings(str(yaml_path))
    assert s.engines == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_import_safety.py -v`
Expected: FAIL — `settings.profiles` and `settings.memory_budget_gb` don't exist yet, and module import still does file I/O.

- [ ] **Step 3: Update Settings class**

In `src/dbs_vector/config.py`, replace the `Settings` class with:

```python
class Settings(BaseSettings):
    """Global configuration for the dbs-vector application."""

    model_config = SettingsConfigDict(
        env_prefix="DBS_",
        env_file=".env",
        extra="ignore",  # env vars often include unrelated keys; ignore them
    )

    # General system
    db_path: str = "./lancedb_dbs_vector"
    nprobes: int = Field(default=20, gt=0)
    log_level: str = "INFO"
    log_serialize: bool = False

    # NEW: memory budget (None → auto-detect via MLX in resolve_memory_budget_gb)
    memory_budget_gb: float | None = Field(default=None, gt=0)

    # NEW: profile dict
    profiles: dict[str, TuningProfile] = {}

    # Engines dictionary (shape changes in Task 7)
    engines: dict[str, EngineConfig] = {}

    # REMOVED: batch_size (now per-profile)
```

Note: `extra="ignore"` on `Settings` because pydantic-settings reads environment variables that often include unrelated keys (CI env, system env). Strict checking happens for the YAML payload via `_apply_system_config` (Task 6).

- [ ] **Step 4: Replace module-bottom singleton + flip load_settings default**

At the bottom of `src/dbs_vector/config.py`, replace:

```python
# Global singleton instance
settings = load_settings()
```

with:

```python
# Module-level singleton: ZERO file I/O at import. We pass _env_file=None
# explicitly to disable pydantic-settings' .env file reading for this
# instance — otherwise BaseSettings will stat() the .env path and (if it
# exists) read it at import time, violating the import-safety contract.
# Runtime callers (cli.py callback, api/main.py lifespan) call
# load_settings(config_file, validate=True), which constructs a fresh
# Settings() (without _env_file=None, so .env IS loaded then) and copies
# fields onto this singleton.
settings = Settings(_env_file=None)
```

And update the `load_settings` signature:

```python
def load_settings(config_file: str | None = None, validate: bool = False) -> Settings:
    """Load and (optionally) validate settings from a YAML file.

    Default validate=False: useful for tests and any caller that wants raw
    parsing. Runtime callers (CLI callback, API lifespan) pass validate=True.
    """
    base_settings = Settings()

    if config_file is None:
        config_file = os.getenv("DBS_CONFIG_FILE", "config.yaml")

    yaml_path = Path(config_file)
    if not yaml_path.exists():
        logger.warning("Configuration file '{}' not found, using defaults", yaml_path)
        return base_settings

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # System block (allow-list / legacy detection added in Task 6)
    if "system" in data and isinstance(data["system"], dict):
        for key, value in data["system"].items():
            if hasattr(base_settings, key):
                setattr(base_settings, key, value)

    # Profiles block
    if "profiles" in data and isinstance(data["profiles"], dict):
        base_settings.profiles = {
            k: TuningProfile(**v) for k, v in data["profiles"].items()
        }

    # Engines block
    if "engines" in data and isinstance(data["engines"], dict):
        base_settings.engines = {k: EngineConfig(**v) for k, v in data["engines"].items()}

    # _validate_config wired up in Task 8
    return base_settings
```

(The `validate` parameter is wired but unused for now; `_validate_config` is added in Task 8.)

- [ ] **Step 5: Run import-safety tests**

Run: `uv run pytest tests/unit/test_config_import_safety.py -v`
Expected: 3 passed

- [ ] **Step 6: Run full unit suite to confirm nothing else broke**

Run: `uv run pytest tests/unit -x -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/config.py tests/unit/test_config_import_safety.py
git commit -m "refactor(config): make module import zero-IO; add Settings.profiles + memory_budget_gb"
```

---

## Task 6: System-block validator + migration hint for `system.batch_size`

**Files:**
- Modify: `src/dbs_vector/config.py`
- Test: `tests/unit/test_migration_hint.py` (new)

- [ ] **Step 1: Write failing tests for `_apply_system_config`**

```python
# tests/unit/test_migration_hint.py
import textwrap

import pytest

from dbs_vector.config import load_settings


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


def test_legacy_system_batch_size_raises_migration_hint(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        system:
          db_path: "./lance"
          batch_size: 8
        """,
    )
    with pytest.raises(ValueError, match="Legacy keys found.*batch_size"):
        load_settings(yaml_path)


def test_unknown_system_key_raises_with_allowlist(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        system:
          db_path: "./lance"
          unknown_key: true
        """,
    )
    with pytest.raises(ValueError, match="Unknown keys.*unknown_key"):
        load_settings(yaml_path)


def test_known_system_keys_pass_through(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        system:
          db_path: "/tmp/lance"
          nprobes: 30
          memory_budget_gb: 16.0
        """,
    )
    s = load_settings(yaml_path)
    assert s.db_path == "/tmp/lance"
    assert s.nprobes == 30
    assert s.memory_budget_gb == 16.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_migration_hint.py -v`
Expected: FAIL — current loader silently ignores `batch_size` and unknown keys.

- [ ] **Step 3: Add `_apply_system_config` and rewire load_settings**

In `src/dbs_vector/config.py`, add module-level constants and helper function (place above `load_settings`):

```python
_LEGACY_SYSTEM_KEYS = {"batch_size"}  # moved to TuningProfile in profiles: block
_KNOWN_SYSTEM_KEYS = {
    "db_path",
    "nprobes",
    "log_level",
    "log_serialize",
    "memory_budget_gb",
}


def _apply_system_config(
    system: dict[str, object], settings: "Settings", config_file: str
) -> None:
    """Apply system: keys onto the Settings instance with strict validation.

    - Legacy keys (e.g., batch_size) raise a migration hint.
    - Unknown keys raise with the allow-list to catch typos.
    - Known keys pass through to setattr().
    """
    legacy = sorted(set(system) & _LEGACY_SYSTEM_KEYS)
    unknown = sorted(set(system) - _KNOWN_SYSTEM_KEYS - _LEGACY_SYSTEM_KEYS)
    if legacy:
        raise ValueError(
            f"Config schema mismatch in {config_file} (system: block).\n"
            f"  Legacy keys found: {legacy}\n"
            f"  These moved to TuningProfile. Define profiles under "
            f"`profiles:` and reference them from each engine via "
            f"`tuning_profile:`. See "
            f"docs/superpowers/specs/2026-05-06-tuning-profiles-design.md "
            f"§10 / docs/README_EMBEDDINGS.md."
        )
    if unknown:
        raise ValueError(
            f"Unknown keys in {config_file} system: block: {unknown}. "
            f"Allowed: {sorted(_KNOWN_SYSTEM_KEYS)}."
        )
    for key, value in system.items():
        setattr(settings, key, value)
```

In the same file, replace the system-block parsing inside `load_settings`:

```python
    # System block — strict validation (legacy / unknown key rejection)
    if "system" in data and isinstance(data["system"], dict):
        _apply_system_config(data["system"], base_settings, config_file)
```

- [ ] **Step 4: Run migration tests**

Run: `uv run pytest tests/unit/test_migration_hint.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full suite to confirm no other tests broke**

Run: `uv run pytest tests -x -q`
Expected: tests using `system.batch_size` in config.yaml *will fail* — that's the next task. Document any failures and proceed; we'll fix the engine-block + config.yaml together in Task 7.

If any unit test other than ones using the real `config.yaml` fails, fix in this commit.

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/config.py tests/unit/test_migration_hint.py
git commit -m "feat(config): add _apply_system_config to detect legacy system.batch_size"
```

---

## Task 7: EngineConfig shrink + `_raise_migration_hint` + config.yaml rewrite

**Files:**
- Modify: `src/dbs_vector/config.py`
- Modify: `config.yaml`
- Modify: `tests/unit/test_bootstrap.py` (mock_settings fixture)
- Test: `tests/unit/test_migration_hint.py` (extend)

- [ ] **Step 1: Extend failing tests for engine-block migration hint**

Append to `tests/unit/test_migration_hint.py`:

```python
def test_legacy_engine_field_raises_migration_hint(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "x"
            model_name: "mlx-community/embeddinggemma-300m-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """,
    )
    with pytest.raises(ValueError, match="Legacy per-engine fields found.*model_name"):
        load_settings(yaml_path)


def test_missing_required_engine_fields_raises_migration_hint(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "x"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
        """,
    )
    with pytest.raises(ValueError, match="Missing new required fields.*model"):
        load_settings(yaml_path)


def test_genuine_validation_error_propagates_unchanged(tmp_path):
    """A genuine validation bug in new schema should NOT be wrapped as migration."""
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: 12345  # wrong type — should be string
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """,
    )
    with pytest.raises(ValueError, match="mapper_type"):
        load_settings(yaml_path)
    # Ensure the migration hint phrase is NOT present
    try:
        load_settings(yaml_path)
    except ValueError as e:
        assert "Legacy per-engine fields" not in str(e)
        assert "Missing new required fields" not in str(e)


def test_valid_new_schema_loads(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "Gemma markdown"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "md_search"
            passage_prefix: "title: none | text: "
            query_prefix: "task: search result | query: "
            tuning_profile: "gemma-md"
        """,
    )
    s = load_settings(yaml_path)
    assert s.engines["md"].model == "gemma-bf16"
    assert s.engines["md"].tuning_profile == "gemma-md"
    assert s.engines["md"].passage_prefix == "title: none | text: "
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_migration_hint.py::test_valid_new_schema_loads -v`
Expected: FAIL — `EngineConfig` still requires legacy fields.

- [ ] **Step 3: Replace `EngineConfig` with shrunk shape**

In `src/dbs_vector/config.py`, replace the `EngineConfig` class entirely:

```python
class EngineConfig(BaseModel):
    """Per-deployment engine config. References model contract + tuning profile."""

    model_config = ConfigDict(extra="forbid")

    description: str
    model: str  # key into ModelRegistry
    mapper_type: str
    chunker_type: str
    table_name: str
    workflow: str
    tuning_profile: str  # key into Settings.profiles

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

- [ ] **Step 4: Add `_raise_migration_hint` and wire into `load_settings`**

Add helper near the other `_*` helpers in `config.py`:

```python
_LEGACY_ENGINE_FIELDS = {
    "model_name",
    "vector_dimension",
    "max_token_length",
    "attention_mask_dtype",
    "chunk_max_chars",
    "batch_size",
}
_REQUIRED_ENGINE_FIELDS = {"model", "tuning_profile"}


def _raise_migration_hint(err, config_file: str, where: str) -> None:
    """Detect old-schema fields in a Pydantic ValidationError and rewrap as a
    single migration message. If the error is unrelated to migration, propagate."""
    seen_legacy = {
        e["loc"][-1]
        for e in err.errors()
        if e["loc"][-1] in _LEGACY_ENGINE_FIELDS
    }
    missing_required = {
        e["loc"][-1]
        for e in err.errors()
        if e["type"] == "missing" and e["loc"][-1] in _REQUIRED_ENGINE_FIELDS
    }
    if seen_legacy or missing_required:
        raise ValueError(
            f"Config schema mismatch in {config_file} ({where}: block).\n"
            f"  Legacy per-engine fields found: {sorted(seen_legacy) or 'none'}\n"
            f"  Missing new required fields: {sorted(missing_required) or 'none'}\n"
            f"See docs/superpowers/specs/2026-05-06-tuning-profiles-design.md §10 "
            f"or docs/README_EMBEDDINGS.md for the new schema."
        ) from err
    raise err
```

In `load_settings`, wrap the engines + profiles block parsing:

```python
    # Profiles block
    if "profiles" in data and isinstance(data["profiles"], dict):
        try:
            base_settings.profiles = {
                k: TuningProfile(**v) for k, v in data["profiles"].items()
            }
        except ValidationError as e:
            _raise_migration_hint(e, config_file, where="profiles")

    # Engines block
    if "engines" in data and isinstance(data["engines"], dict):
        try:
            base_settings.engines = {
                k: EngineConfig(**v) for k, v in data["engines"].items()
            }
        except ValidationError as e:
            _raise_migration_hint(e, config_file, where="engines")
```

Add the import at the top of `config.py`:

```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError
```

- [ ] **Step 5: Rewrite `config.yaml` to new schema**

Replace `config.yaml` entirely:

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
    api_key: "0Byuf9P9e5UxUNIsngNjr6b9u8sldoHd1ek_ImBbxiI"
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
    api_key: "0Byuf9P9e5UxUNIsngNjr6b9u8sldoHd1ek_ImBbxiI"
    api_page_size: 200
    api_since_days: 60
    api_timeout_sec: 30
    api_min_execution_ms: 0
    api_database: ""
```

- [ ] **Step 6: Update `tests/unit/test_bootstrap.py` mock fixture**

Replace the `mock_settings` fixture so it matches the new shape (we'll update bootstrap consumption in Task 9, but the fixture must satisfy the new EngineConfig shape now):

```python
# tests/unit/test_bootstrap.py
from unittest.mock import MagicMock, patch

import pytest

from dbs_vector.services.bootstrap import EngineDeps, build_dependencies


@pytest.fixture
def mock_settings():
    """Minimal settings fixture: one engine + one profile registered."""
    engine_config = MagicMock()
    engine_config.model = "gemma-bf16"
    engine_config.mapper_type = "document"
    engine_config.chunker_type = "document"
    engine_config.table_name = "t"
    engine_config.workflow = "default"
    engine_config.tuning_profile = "test-profile"
    engine_config.passage_prefix = ""
    engine_config.query_prefix = ""
    engine_config.chunker_kwargs.return_value = {"max_chars": 500}

    profile = MagicMock()
    profile.max_token_length = 2048
    profile.chunk_max_chars = 500
    profile.batch_size = 64

    with patch("dbs_vector.services.bootstrap.settings") as s:
        s.engines = {"md": engine_config}
        s.profiles = {"test-profile": profile}
        s.db_path = "./test.db"
        s.nprobes = 10
        s.memory_budget_gb = 22.0
        yield s


def test_build_dependencies_unknown_engine_raises(mock_settings):
    with pytest.raises(ValueError, match="Unknown engine"):
        build_dependencies("no-such-engine")
```

(Other tests in this file will be updated in Task 9 when bootstrap actually uses these fields.)

- [ ] **Step 7: Run migration + bootstrap tests**

Run: `uv run pytest tests/unit/test_migration_hint.py tests/unit/test_bootstrap.py::test_build_dependencies_unknown_engine_raises -v`
Expected: tests from Step 1 + the surviving bootstrap test pass.

- [ ] **Step 8: Commit**

```bash
git add src/dbs_vector/config.py config.yaml tests/unit/test_migration_hint.py tests/unit/test_bootstrap.py
git commit -m "feat(config): EngineConfig shrink, migration hints, config.yaml rewrite"
```

---

## Task 8: `_validate_config` (model+profile+memory checks)

**Files:**
- Modify: `src/dbs_vector/config.py`
- Test: `tests/unit/test_config_validation.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_config_validation.py
import textwrap

import pytest

from dbs_vector.config import load_settings


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


GENERAL_PROFILES = """
profiles:
  gemma-md:           {max_token_length: 2048,  chunk_max_chars: 1000, batch_size: 64}
  gemma-too-big:      {max_token_length: 99999, chunk_max_chars: 1000, batch_size: 64}
  granite-oom:        {max_token_length: 16384, chunk_max_chars: 1000, batch_size: 64}
  granite-md-large:   {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
"""


def test_unknown_model_raises(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + """
        engines:
          md:
            description: "x"
            model: "nonexistent-model"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """,
    )
    with pytest.raises(ValueError, match="unknown model contract 'nonexistent-model'"):
        load_settings(yaml_path, validate=True)


def test_unknown_profile_raises(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + """
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "nonexistent-profile"
        """,
    )
    with pytest.raises(ValueError, match="unknown tuning profile 'nonexistent-profile'"):
        load_settings(yaml_path, validate=True)


def test_profile_exceeds_model_cap_raises(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + """
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-too-big"
        """,
    )
    with pytest.raises(ValueError, match="requires 99999 tokens.*cap 2048"):
        load_settings(yaml_path, validate=True)


def test_profile_oom_raises_with_recommendation(tmp_path):
    """The user's calibration crash: 16K seq × batch 64 on granite, 22 GB cap."""
    yaml_path = _write_yaml(
        tmp_path,
        f"""
        system:
          memory_budget_gb: 22.0
        {GENERAL_PROFILES.strip()}
        engines:
          md-granite:
            description: "x"
            model: "granite-r2"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "granite-oom"
        """,
    )
    with pytest.raises(ValueError) as exc_info:
        load_settings(yaml_path, validate=True)
    msg = str(exc_info.value)
    assert "granite-oom" in msg
    assert "md-granite" in msg
    assert "conservative estimate" in msg
    assert "raw attention buffer" in msg
    assert "41 GB" in msg  # observed OOM data point from calibration note
    assert "16384" in msg  # recommendation preserves seq len


def test_unknown_model_fires_before_memory_check(tmp_path, monkeypatch):
    """Validation ordering: unknown model must fail BEFORE memory budget resolution.

    Without lazy resolution, an MLX-unavailable environment would mask the real
    config error with "Could not auto-detect Metal memory budget."
    """
    # Force memory detection to fail; we should never reach it.
    monkeypatch.setattr(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb",
        lambda: None,
    )
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + """
        engines:
          md:
            description: "x"
            model: "nonexistent-model"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """,
    )
    with pytest.raises(ValueError, match="unknown model contract"):
        load_settings(yaml_path, validate=True)
    # Specifically NOT a memory-budget error
    with pytest.raises(ValueError) as exc_info:
        load_settings(yaml_path, validate=True)
    assert "Could not auto-detect" not in str(exc_info.value)


def test_validate_false_skips_checks(tmp_path):
    """Default validate=False does not run the chain; broken profile is loaded."""
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + """
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-too-big"
        """,
    )
    s = load_settings(yaml_path)  # default validate=False
    assert "md" in s.engines


def test_validate_empty_engines_is_noop(tmp_path):
    yaml_path = _write_yaml(tmp_path, "system:\n  db_path: \"/tmp\"\n")
    s = load_settings(yaml_path, validate=True)
    assert s.engines == {}


def test_valid_config_passes_validation(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        f"""
        system:
          memory_budget_gb: 22.0
        {GENERAL_PROFILES.strip()}
        engines:
          md-granite:
            description: "x"
            model: "granite-r2"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "granite-md-large"
        """,
    )
    s = load_settings(yaml_path, validate=True)
    assert "md-granite" in s.engines
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_validation.py -v`
Expected: FAIL — `_validate_config` doesn't exist yet, so `validate=True` is ignored.

- [ ] **Step 3: Implement `_validate_config` and wire it in**

Add to `src/dbs_vector/config.py` (near the other helpers):

```python
_CALIBRATION_NOTE = (
    "Calibration reference: 2025-05 OOM measured 41 GB at "
    "batch=64, seq=16384, bf16; the 3.0× safety factor preserves headroom."
)


def _validate_config(settings: "Settings", config_file: str) -> None:
    """Run the validation chain over every configured engine.

    Rules (each fails-fast with a remediation message), executed per engine:
      1. Engine.model exists in ModelRegistry.
      2. Engine.tuning_profile exists in settings.profiles.
      3. profile.max_token_length ≤ contract.model_max_token_length.
      4. estimate_peak_buffer_bytes ≤ memory_budget × 0.9.
      5. (warn) chunk_max_chars routinely exceeds max_token_length × 4.

    Memory budget is resolved lazily (only when rule 4 actually runs) so a
    config with an unknown model/profile fails on rule 1/2 BEFORE we attempt
    Metal auto-detection — otherwise an MLX-unavailable environment would
    mask real config errors with "Could not auto-detect Metal memory budget."
    """
    from dbs_vector.core.model_registry import ModelRegistry
    from dbs_vector.core.profile_math import (
        estimate_peak_buffer_bytes,
        recommend_profile,
    )
    from dbs_vector.infrastructure.hardware import resolve_memory_budget_gb

    if not settings.engines:
        return

    # Lazy: only resolved on first memory check (rule 4).
    budget_gb: float | None = None

    for engine_name, engine in settings.engines.items():
        # Rule 1: model contract exists
        try:
            contract = ModelRegistry.get(engine.model)
        except KeyError as e:
            raise ValueError(
                f"Engine '{engine_name}' references {e}".replace("KeyError: ", "")
            ) from e

        # Rule 2: profile exists
        if engine.tuning_profile not in settings.profiles:
            known = sorted(settings.profiles)
            raise ValueError(
                f"Engine '{engine_name}' references unknown tuning profile "
                f"'{engine.tuning_profile}'. Known: {known}"
            )
        profile = settings.profiles[engine.tuning_profile]

        # Rule 3: profile fits model cap
        if profile.max_token_length > contract.model_max_token_length:
            raise ValueError(
                f"Profile '{engine.tuning_profile}' requires "
                f"{profile.max_token_length} tokens but engine '{engine_name}' "
                f"uses model '{engine.model}' (cap {contract.model_max_token_length}). "
                f"Lower profile.max_token_length or pick a different model."
            )

        # Rule 4: profile fits memory budget — resolve budget lazily here.
        if budget_gb is None:
            budget_gb = resolve_memory_budget_gb(settings.memory_budget_gb)
        budget_bytes = int(budget_gb * 1024 ** 3)
        peak = estimate_peak_buffer_bytes(profile, contract)
        cap = int(budget_bytes * 0.9)
        if peak > cap:
            raw_attention = (
                profile.batch_size
                * profile.max_token_length ** 2
                * contract.compute_dtype_bytes
            )
            suggested = recommend_profile(
                contract,
                budget_gb,
                target_chunker=engine.chunker_type,
                target_seq_len=profile.max_token_length,
            )
            raise ValueError(
                f"Profile '{engine.tuning_profile}' on engine '{engine_name}' "
                f"fails the memory budget:\n"
                f"  conservative estimate: {peak / 1024**3:.1f} GB "
                f"(3.0× safety factor over raw attention buffer)\n"
                f"  raw attention buffer: {raw_attention / 1024**3:.1f} GB\n"
                f"  budget (Metal max buffer × 0.9): {cap / 1024**3:.1f} GB\n"
                f"  {_CALIBRATION_NOTE}\n"
                f"Suggested values: max_token_length={suggested['max_token_length']}, "
                f"batch_size={suggested['batch_size']}"
                f"{' (context length reduced)' if suggested['seq_len_reduced'] else ''}. "
                f"Edit {config_file}."
            )

        # Rule 5: chunk-vs-token sanity (warn only)
        if profile.chunk_max_chars > 0 and profile.chunk_max_chars > profile.max_token_length * 4:
            logger.warning(
                "Engine '{}' profile '{}': chunk_max_chars={} likely exceeds "
                "max_token_length={} (× 4 char/token heuristic). Chunks may truncate.",
                engine_name,
                engine.tuning_profile,
                profile.chunk_max_chars,
                profile.max_token_length,
            )
```

In `load_settings`, before `return base_settings`:

```python
    if validate:
        _validate_config(base_settings, config_file)
    return base_settings
```

- [ ] **Step 4: Run validation tests**

Run: `uv run pytest tests/unit/test_config_validation.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run full unit suite**

Run: `uv run pytest tests/unit -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/config.py tests/unit/test_config_validation.py
git commit -m "feat(config): add _validate_config with model+profile+memory checks"
```

---

## Task 9: bootstrap + EngineDeps.batch_size

**Files:**
- Modify: `src/dbs_vector/services/bootstrap.py`
- Modify: `tests/unit/test_bootstrap.py`

- [ ] **Step 1: Write failing tests for new bootstrap behavior**

Replace `tests/unit/test_bootstrap.py` with:

```python
# tests/unit/test_bootstrap.py
from unittest.mock import MagicMock, patch

import pytest

from dbs_vector.services.bootstrap import EngineDeps, build_dependencies


@pytest.fixture
def mock_settings():
    """Minimal settings fixture: one engine + one profile registered."""
    engine_config = MagicMock()
    engine_config.model = "gemma-bf16"
    engine_config.mapper_type = "document"
    engine_config.chunker_type = "document"
    engine_config.table_name = "t"
    engine_config.workflow = "default"
    engine_config.tuning_profile = "test-profile"
    engine_config.passage_prefix = "P:"
    engine_config.query_prefix = "Q:"
    engine_config.chunker_kwargs.return_value = {"max_chars": 500}

    profile = MagicMock()
    profile.max_token_length = 2048
    profile.chunk_max_chars = 500
    profile.batch_size = 64

    with patch("dbs_vector.services.bootstrap.settings") as s:
        s.engines = {"md": engine_config}
        s.profiles = {"test-profile": profile}
        s.db_path = "./test.db"
        s.nprobes = 10
        s.memory_budget_gb = 22.0
        yield s, engine_config, profile


def test_unknown_engine_raises(mock_settings):
    with pytest.raises(ValueError, match="Unknown engine"):
        build_dependencies("no-such-engine")


def test_returns_engine_deps_with_batch_size(mock_settings):
    _, _, profile = mock_settings
    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder"),
        patch("dbs_vector.services.bootstrap.LanceDBStore"),
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockRegistry.get_mapper.return_value = MagicMock()
        MockRegistry.get_chunker.return_value = MagicMock()
        deps = build_dependencies("md")
    assert isinstance(deps, EngineDeps)
    assert deps.batch_size == 64
    assert deps.workflow == "default"


def test_resolves_via_model_registry(mock_settings):
    """MLXEmbedder is constructed with model_name etc. from ModelRegistry, not engine."""
    _, engine_config, profile = mock_settings
    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder") as MockEmbedder,
        patch("dbs_vector.services.bootstrap.LanceDBStore"),
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockRegistry.get_mapper.return_value = MagicMock()
        MockRegistry.get_chunker.return_value = MagicMock()
        build_dependencies("md")
    _, kwargs = MockEmbedder.call_args
    assert kwargs["model_name"] == "mlx-community/embeddinggemma-300m-bf16"
    assert kwargs["max_token_length"] == 2048
    assert kwargs["dimension"] == 768
    assert kwargs["passage_prefix"] == "P:"
    assert kwargs["query_prefix"] == "Q:"
    assert kwargs["attention_mask_dtype"] == "float16"


def test_passes_chunk_max_chars_to_chunker_kwargs(mock_settings):
    _, engine_config, _ = mock_settings
    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder"),
        patch("dbs_vector.services.bootstrap.LanceDBStore"),
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockRegistry.get_mapper.return_value = MagicMock()
        MockRegistry.get_chunker.return_value = MagicMock()
        build_dependencies("md")
    args, kwargs = engine_config.chunker_kwargs.call_args
    # chunk_max_chars is the only positional or first keyword arg
    assert kwargs.get("chunk_max_chars") == 500 or (args and args[0] == 500)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_bootstrap.py -v`
Expected: FAIL — bootstrap still reads legacy fields off the engine config.

- [ ] **Step 3: Rewrite `services/bootstrap.py`**

Replace contents of `src/dbs_vector/services/bootstrap.py`:

```python
"""Dependency-injection factory for engines."""

import os
from typing import Any, NamedTuple

from dbs_vector.config import settings
from dbs_vector.core.model_registry import ModelRegistry
from dbs_vector.core.registry import ComponentRegistry
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore


class EngineDeps(NamedTuple):
    """Resolved per-engine runtime dependencies."""

    embedder: Any
    store: Any
    chunker: Any
    workflow: str
    batch_size: int


def build_dependencies(
    engine_name: str,
    query_override: str | None = None,
    url_override: str | None = None,
) -> EngineDeps:
    """Resolve the chunker / mapper / embedder / store stack for an engine."""
    if engine_name not in settings.engines:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Check {os.environ.get('DBS_CONFIG_FILE', 'config.yaml')}."
        )

    engine = settings.engines[engine_name]
    contract = ModelRegistry.get(engine.model)

    if engine.tuning_profile not in settings.profiles:
        raise ValueError(
            f"Engine '{engine_name}' references unknown tuning profile "
            f"'{engine.tuning_profile}'. Known: {sorted(settings.profiles)}"
        )
    profile = settings.profiles[engine.tuning_profile]

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
    chunker = ChunkerClass(
        **engine.chunker_kwargs(
            chunk_max_chars=profile.chunk_max_chars,
            query_override=query_override,
            url_override=url_override,
        )
    )

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
        batch_size=profile.batch_size,
    )
```

- [ ] **Step 4: Run bootstrap tests**

Run: `uv run pytest tests/unit/test_bootstrap.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/services/bootstrap.py tests/unit/test_bootstrap.py
git commit -m "feat(bootstrap): resolve via ModelRegistry + TuningProfile, expose batch_size"
```

---

## Task 10: IngestionService takes batch_size as constructor arg

**Files:**
- Modify: `src/dbs_vector/services/ingestion.py`
- Modify: `tests/unit/test_ingestion.py` (if it exists; otherwise create minimal coverage)

- [ ] **Step 1: Check current state of test_ingestion.py**

Run: `ls tests/unit/test_ingestion.py 2>/dev/null && uv run pytest tests/unit/test_ingestion.py --collect-only -q || echo "MISSING"`

If the file is missing, create `tests/unit/test_ingestion.py` with the failing test below. Otherwise, append.

- [ ] **Step 2: Write failing test**

```python
# tests/unit/test_ingestion.py (or append)
from unittest.mock import MagicMock

from dbs_vector.services.ingestion import IngestionService


def test_ingestion_service_accepts_batch_size_kwarg():
    chunker = MagicMock()
    embedder = MagicMock()
    store = MagicMock()
    svc = IngestionService(
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
        workflow="w",
        batch_size=8,
    )
    assert svc.batch_size == 8


def test_ingestion_service_uses_self_batch_size_in_batched():
    """_batched yields batches sized by self.batch_size."""
    svc = IngestionService(
        chunker=MagicMock(),
        embedder=MagicMock(),
        vector_store=MagicMock(),
        workflow="w",
        batch_size=3,
    )
    batches = list(svc._batched(iter(range(10)), svc.batch_size))
    assert [len(b) for b in batches] == [3, 3, 3, 1]
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_ingestion.py -v`
Expected: FAIL — `IngestionService.__init__` doesn't accept `batch_size`.

- [ ] **Step 4: Update IngestionService**

In `src/dbs_vector/services/ingestion.py`:

Remove the `from dbs_vector.config import settings` import at the top (line 11 in current code).

Update `__init__`:

```python
class IngestionService:
    """Orchestrates the chunking, embedding, and storage of documents."""

    def __init__(
        self,
        chunker: IChunker,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        workflow: str = "default",
        batch_size: int = 64,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.workflow = workflow
        self.batch_size = batch_size
```

Replace `settings.batch_size` (line ~106) with `self.batch_size`:

```python
        for batch in self._batched(_chunk_generator(), self.batch_size):
```

- [ ] **Step 5: Run ingestion tests**

Run: `uv run pytest tests/unit/test_ingestion.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run full unit suite**

Run: `uv run pytest tests/unit -x -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/services/ingestion.py tests/unit/test_ingestion.py
git commit -m "refactor(ingestion): accept batch_size via constructor, drop settings import"
```

---

## Task 11: CLI callback singleton-mutation

**Files:**
- Modify: `src/dbs_vector/cli.py`
- Test: `tests/unit/test_cli_callback.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_cli_callback.py
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(args: list[str], cwd: Path, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "dbs_vector.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_help_works_with_no_config(tmp_path):
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0
    assert "dbs-vector" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_help_works_with_malformed_config(tmp_path):
    (tmp_path / "config.yaml").write_text("system: : :\nthis: is: not: yaml")
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0


def test_help_works_with_old_schema_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
system:
  db_path: "./lance"
  batch_size: 64
engines:
  md:
    description: "Old"
    model_name: "mlx-community/embeddinggemma-300m-bf16"
    vector_dimension: 768
    max_token_length: 2048
    chunk_max_chars: 1000
    table_name: "t"
    mapper_type: "document"
    chunker_type: "document"
    workflow: "w"
"""
    )
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0


def test_version_works_with_malformed_config(tmp_path):
    (tmp_path / "config.yaml").write_text("not: valid: yaml: at: all: : :")
    result = _run_cli(["--version"], cwd=tmp_path)
    assert result.returncode == 0
    assert "dbs-vector" in result.stdout.lower()


# Direct unit tests of the singleton-mutation helper (does not require subprocess).

from unittest.mock import MagicMock


def _make_fake_new_settings():
    fake = MagicMock()
    fake.db_path = "/tmp/lance"
    fake.nprobes = 30
    fake.engines = {"md": object()}
    fake.profiles = {"gemma-md": object()}
    fake.memory_budget_gb = 22.0
    fake.log_level = "DEBUG"
    fake.log_serialize = True
    return fake


def test_populate_singleton_copies_profiles_and_memory_budget():
    """Helper extracted from main() callback must copy profiles + memory_budget_gb."""
    from dbs_vector.cli import _populate_singleton_from
    from dbs_vector.config import settings

    new = _make_fake_new_settings()
    _populate_singleton_from(new)

    assert settings.db_path == "/tmp/lance"
    assert settings.nprobes == 30
    assert settings.engines == new.engines
    assert settings.profiles == new.profiles
    assert settings.memory_budget_gb == 22.0
    assert settings.log_level == "DEBUG"
    assert settings.log_serialize is True


def test_populate_singleton_does_not_set_legacy_batch_size():
    """The new schema has no Settings.batch_size; the helper must not set it."""
    from dbs_vector.cli import _populate_singleton_from
    from dbs_vector.config import Settings, settings

    new = _make_fake_new_settings()
    _populate_singleton_from(new)
    # Settings has no batch_size field after the schema change.
    assert "batch_size" not in Settings.model_fields
    # And the singleton instance does not have one either.
    assert not hasattr(settings, "batch_size")
```

- [ ] **Step 2: Verify which fields current CLI copies**

Read `src/dbs_vector/cli.py` lines 36-74 to confirm the current copy block. Expect:

```python
    settings.db_path = new_settings.db_path
    settings.batch_size = new_settings.batch_size
    settings.nprobes = new_settings.nprobes
    settings.engines = new_settings.engines
    settings.log_level = new_settings.log_level
    settings.log_serialize = new_settings.log_serialize
```

- [ ] **Step 3: Run tests to verify import-safety failures**

Run: `uv run pytest tests/unit/test_cli_callback.py -v`
Expected: tests pass for `--help`/`--version` if the import-safety changes from Tasks 5-7 are wired (no file IO at import). May still fail if CLI subcommand path tries to read config — those will fail in a later step.

- [ ] **Step 4: Extract a singleton-mutation helper, then call it from the callback**

In `src/dbs_vector/cli.py`, add a module-level helper (just above the `main()` callback):

```python
def _populate_singleton_from(new_settings: "Settings") -> None:
    """Copy fields from a freshly-loaded Settings onto the module-level singleton.

    Extracted as a top-level function so it can be unit-tested without driving
    the entire Typer callback. The field list is the source of truth for what
    runtime callers (CLI, API lifespan) propagate from disk to the singleton.
    """
    from dbs_vector.config import settings

    settings.db_path = new_settings.db_path
    settings.nprobes = new_settings.nprobes
    settings.engines = new_settings.engines
    settings.profiles = new_settings.profiles
    settings.memory_budget_gb = new_settings.memory_budget_gb
    settings.log_level = new_settings.log_level
    settings.log_serialize = new_settings.log_serialize
```

Add the type-only import at the top of `cli.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbs_vector.config import Settings
```

Then modify the Typer `main()` callback to use it:

```python
@app.callback()
def main(
    ctx: typer.Context,
    config_file: Annotated[
        str, typer.Option("--config-file", "-c", help="Path to config.yaml file.")
    ] = "config.yaml",
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show the version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """dbs-vector: Configurable Arrow-Native Search Engine."""
    # Skip config loading when just showing help or version (no subcommand invoked)
    if ctx.invoked_subcommand is None:
        return

    import os

    from dbs_vector.config import load_settings, settings

    # Export to environment so uvicorn subprocesses (in API mode) inherit it
    os.environ["DBS_CONFIG_FILE"] = config_file

    # Load AND validate the config; copy fields onto the singleton.
    new_settings = load_settings(config_file, validate=True)
    _populate_singleton_from(new_settings)

    # Configure logger based on settings
    configure_logger(level=settings.log_level, serialize=settings.log_serialize)
```

- [ ] **Step 5: Update `IngestionService` construction sites**

Search for `IngestionService(` calls in `cli.py`. They currently look like:

```python
service = IngestionService(
    chunker=deps.chunker,
    embedder=deps.embedder,
    vector_store=deps.store,
    workflow=deps.workflow,
)
```

Add `batch_size=deps.batch_size`:

```python
service = IngestionService(
    chunker=deps.chunker,
    embedder=deps.embedder,
    vector_store=deps.store,
    workflow=deps.workflow,
    batch_size=deps.batch_size,
)
```

Verify: `grep -n "IngestionService(" src/dbs_vector/cli.py` — update every occurrence.

- [ ] **Step 6: Run CLI callback tests**

Run: `uv run pytest tests/unit/test_cli_callback.py -v`
Expected: 6 passed (4 subprocess help/version tests + 2 direct singleton-mutation tests).

- [ ] **Step 7: Run full unit suite**

Run: `uv run pytest tests/unit -x -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/dbs_vector/cli.py tests/unit/test_cli_callback.py
git commit -m "feat(cli): callback copies profiles+memory_budget_gb; passes batch_size"
```

---

## Task 12: API lifespan + /health endpoint

**Files:**
- Modify: `src/dbs_vector/api/main.py`
- Test: `tests/unit/test_api_lifespan.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_api_lifespan.py
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_lifespan_loads_config_before_initialize_services(monkeypatch, tmp_path):
    """Lifespan must call load_settings(validate=True) before initialize_services."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("system:\n  db_path: \"./lance\"\n")
    monkeypatch.setenv("DBS_CONFIG_FILE", str(yaml_path))

    call_order: list[str] = []

    def fake_load_settings(config_file, validate=False):
        call_order.append(f"load_settings(validate={validate})")
        s = MagicMock()
        s.db_path = "./lance"
        s.engines = {}
        s.profiles = {}
        s.memory_budget_gb = None
        s.nprobes = 20
        s.log_level = "INFO"
        s.log_serialize = False
        return s

    def fake_initialize_services():
        call_order.append("initialize_services")

    with (
        patch("dbs_vector.api.main.load_settings", side_effect=fake_load_settings),
        patch(
            "dbs_vector.api.main.initialize_services",
            side_effect=fake_initialize_services,
        ),
        patch("dbs_vector.api.main.mcp") as fake_mcp,
    ):
        fake_session_manager = MagicMock()

        class FakeAsyncContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                return None

        fake_session_manager.run.return_value = FakeAsyncContext()
        fake_mcp.session_manager = fake_session_manager

        from dbs_vector.api.main import lifespan

        async with lifespan(MagicMock()):
            pass

    assert call_order == ["load_settings(validate=True)", "initialize_services"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_api_lifespan.py -v`
Expected: FAIL — `load_settings` not yet imported in `api/main.py`.

- [ ] **Step 3: Update `api/main.py`**

At the top of `src/dbs_vector/api/main.py`, add imports:

```python
import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from dbs_vector.api.mcp_server import mcp
from dbs_vector.api.state import _services, initialize_services
from dbs_vector.config import load_settings, settings
from dbs_vector.core.model_registry import ModelRegistry
from dbs_vector.core.models import SearchResult, SqlSearchResult
```

Add an import for the shared singleton-mutation helper:

```python
from dbs_vector.cli import _populate_singleton_from
```

(This is the same helper extracted in Task 11. Reusing it keeps the field
list in one place; both the CLI callback and the API lifespan stay in sync
when fields change later.)

Replace the `lifespan` function:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown events for the API."""
    logger.info("Initializing MLX Embedders and LanceDB connections")

    # Explicit config load: module-level singleton is empty Settings(_env_file=None)
    # until this point. Must run before initialize_services consumes settings.engines.
    config_file = os.environ.get("DBS_CONFIG_FILE", "config.yaml")
    new_settings = load_settings(config_file, validate=True)
    _populate_singleton_from(new_settings)

    try:
        initialize_services()
        logger.success("API is ready to accept concurrent requests")
    except Exception as e:
        logger.error("Failed to initialize search services: {}", e)
        raise

    async with mcp.session_manager.run():
        yield

    logger.info("Cleaning up resources")
    _services.clear()
```

Update the `/health` endpoint to resolve `model_name` via `ModelRegistry`:

```python
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    if not _services:
        raise HTTPException(status_code=503, detail="Search service initializing or failed")

    status_dict = {"status": "healthy"}
    for engine_name, engine in settings.engines.items():
        contract = ModelRegistry.get(engine.model)
        status_dict[f"{engine_name}_model"] = contract.model_name

    return status_dict
```

- [ ] **Step 4: Confirm pytest-asyncio is available (likely already present)**

Check if it's already a dev dependency:

```bash
grep pytest-asyncio pyproject.toml || echo "MISSING"
```

If `MISSING` prints, add it to the dev-dependency section in `pyproject.toml` and run `uv sync`. The Step 7 commit should then include `pyproject.toml` and `uv.lock`. **Otherwise (already present), do not stage `uv.lock`** — staging an unrelated lock-file diff is a common accidental commit.

Ensure the test file has `@pytest.mark.asyncio` on the async test (already shown in Step 1). No `conftest.py` change required.

- [ ] **Step 5: Run lifespan test**

Run: `uv run pytest tests/unit/test_api_lifespan.py -v`
Expected: 1 passed.

- [ ] **Step 6: Run full unit suite**

Run: `uv run pytest tests/unit -x -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
# Only include pyproject.toml + uv.lock if Step 4 actually added pytest-asyncio.
git add src/dbs_vector/api/main.py tests/unit/test_api_lifespan.py
# If pytest-asyncio was newly added in Step 4, also:
#   git add pyproject.toml uv.lock
git commit -m "feat(api): lifespan loads config before initialize_services; /health via ModelRegistry"
```

---

## Task 13: Update integration test (test_granite_engines.py)

**Files:**
- Modify: `tests/integration/test_granite_engines.py`

- [ ] **Step 1: Inspect current YAML fixture**

Run: `grep -A 30 "config.yaml\|yaml.dump\|engines:" tests/integration/test_granite_engines.py | head -100`

The test writes a YAML fixture to a temp dir. The YAML uses the old engine schema (`model_name`, `max_token_length`, etc.). It must be updated.

- [ ] **Step 2: Update the YAML fixture in the test**

Replace the YAML content used by the test with the new schema. The new fixture should look like:

```python
GRANITE_TEST_CONFIG = """
system:
  db_path: "{db_path}"
  nprobes: 10
  memory_budget_gb: 22.0

profiles:
  granite-md-large:   {{max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}}
  granite-sql-atomic: {{max_token_length: 8192,  chunk_max_chars: 0,    batch_size: 32}}

engines:
  md-granite:
    description: "Granite markdown test"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite_test"
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"

  sql-granite:
    description: "Granite SQL test"
    model: "granite-r2"
    mapper_type: "sql"
    chunker_type: "duckdb"
    table_name: "query_vault_granite_test"
    workflow: "sql_clustering_granite"
    tuning_profile: "granite-sql-atomic"
"""
```

Update the `_patch_settings_singleton` helper (or equivalent) to also patch `profiles` and `memory_budget_gb` onto the singleton, alongside `engines`. The exact code depends on the current implementation; verify against the file.

- [ ] **Step 3: Run integration tests**

These tests are slow-marked and gated. Run them explicitly:

Run: `DBS_VECTOR_RUN_SLOW_GRANITE=1 uv run pytest tests/integration/test_granite_engines.py -v`
Expected: tests pass (download Granite model on first run, may take several minutes).

If you don't want to run the slow tests during development, ensure they at least *collect* without errors:

Run: `uv run pytest tests/integration/test_granite_engines.py --collect-only -q`
Expected: tests collected, no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_granite_engines.py
git commit -m "test(integration): update test_granite_engines.py YAML fixture to new schema"
```

---

## Task 14: Documentation sweep

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/README_EMBEDDINGS.md`
- Modify: `docs/README_DOCS.md`
- Modify: `docs/README_SQL.md`
- Modify: `docs/README_REMOTE_SQL_API.md`
- Modify: `docs/README_ARCHITECTURE.md`
- Modify: `docs/README.md`
- Modify: `README.md`

- [ ] **Step 1: Sweep for stale references**

Run: `grep -rn "system\.\?batch_size\|attention_mask_dtype\|max_token_length\|chunk_max_chars\|model_name" CLAUDE.md README.md docs/ 2>/dev/null | grep -v "docs/superpowers"`

Expected: hits in multiple docs files. Each hit needs a content review:
- References to `system.batch_size` → describe the new per-profile `batch_size` field.
- References to per-engine `max_token_length` / `chunk_max_chars` → describe profile-based config and link to spec §10.
- References to per-engine `attention_mask_dtype` → describe model-registry-based contracts and link to spec §3-4.
- References to per-engine `model_name` → describe `model` field referencing `ModelRegistry`.

- [ ] **Step 2: Update `CLAUDE.md`**

Add a section on the new config schema near the existing "Configuration-Driven Registry Pattern" section:

```markdown
### Tuning Profiles & Model Registry

Three layers for engine config:

1. **`ModelRegistry` (code, hardcoded)** — `core/model_registry.py` carries
   `vector_dimension`, `model_max_token_length`, `attention_mask_dtype`,
   `compute_dtype_bytes` per model. Adding a model is a `register()` call.
   Built-ins: `gemma-bf16`, `granite-r2`.

2. **`profiles:` block in `config.yaml`** — three numeric knobs per profile:
   `max_token_length`, `chunk_max_chars`, `batch_size`. Validated against
   the engine's model + Metal memory budget at load time.

3. **`engines:` block in `config.yaml`** — references `model:` (registry key)
   and `tuning_profile:` (profile name). Holds pipeline shape (mapper,
   chunker, table, workflow) and prefixes (which vary per engine for the
   same underlying model).

Memory budget auto-detected from `mlx.core.metal.device_info()`; override
via `system.memory_budget_gb`.

Module-level `settings = Settings(_env_file=None)` performs zero file I/O at
import (no YAML, no `.env`); CLI callback / API lifespan call
`load_settings(config_file, validate=True)` explicitly and copy fields onto
the singleton via `_populate_singleton_from()`. This makes
`dbs-vector --help` / `--version` survive a malformed or absent `config.yaml`.

Adding a new engine: see spec
`docs/superpowers/specs/2026-05-06-tuning-profiles-design.md`.
```

In the existing "Architecture" section, update the `EngineConfig` description to reflect the slimmer shape (drop references to `model_name`, `max_token_length`, etc.; keep `mapper_type`, `chunker_type`, `table_name`, `workflow`, prefixes).

In the existing "Commands" section, the `dbs-vector ingest`/`search` commands work as before — no CLI argument changes — so leave that section unchanged.

- [ ] **Step 3: Update `docs/README_EMBEDDINGS.md`**

Add a new section "Tuning Profiles" near the top, and a "Migrating from the pre-tuning-profiles schema" subsection at the bottom. Sample content:

```markdown
## Tuning Profiles

The three numeric knobs that scale with hardware (`max_token_length`,
`chunk_max_chars`, `batch_size`) live in a `profiles:` block in
`config.yaml`. Each engine references one profile by name.

```yaml
profiles:
  granite-md-large: {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}

engines:
  md-granite:
    model: "granite-r2"
    tuning_profile: "granite-md-large"
    # ... pipeline shape ...
```

Adding a profile: edit `config.yaml`. Adding a model: register in
`src/dbs_vector/core/model_registry.py`.

The validator at config-load time refuses profiles that exceed the model's
context cap or the Metal memory budget, with a recommendation that
preserves your intended `max_token_length`.

## Migration from the pre-tuning-profiles schema

If your `config.yaml` has any of these legacy fields:
- `system.batch_size` → moved to per-profile `batch_size`
- `engines.<name>.model_name` → replaced by `engines.<name>.model` (registry key)
- `engines.<name>.vector_dimension` / `max_token_length` /
  `attention_mask_dtype` → moved to `ModelRegistry` (hardcoded)
- `engines.<name>.chunk_max_chars` → moved to per-profile `chunk_max_chars`

You'll get a single migration-hint error pointing at this section.
See spec `docs/superpowers/specs/2026-05-06-tuning-profiles-design.md` §10
for a complete example of the new schema.
```

- [ ] **Step 4: Update remaining docs**

For each of `docs/README_DOCS.md`, `docs/README_SQL.md`, `docs/README_REMOTE_SQL_API.md`, `docs/README_ARCHITECTURE.md`, `docs/README.md`, root `README.md`:
- Find references to `system.batch_size`. Replace with per-profile `batch_size` and link to `docs/README_EMBEDDINGS.md` "Tuning Profiles" section.
- Find references to per-engine `max_token_length` / `chunk_max_chars`. Same treatment.
- Find references to per-engine `model_name` / `attention_mask_dtype`. Update to mention `model:` (registry key).

If a doc only briefly mentions any of these in passing (e.g., "you can tune `chunk_max_chars`"), update the example to reference the profile field instead.

- [ ] **Step 5: Verify no stale references remain**

Run: `grep -rn "system\.\?batch_size\|attention_mask_dtype:\|model_name:" CLAUDE.md README.md docs/ 2>/dev/null | grep -v "docs/superpowers/specs" | grep -v "docs/superpowers/plans"`
Expected: no hits in non-spec / non-plan files.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md docs/
git commit -m "docs: sweep references to legacy config schema; describe tuning profiles"
```

---

## Task 15: Final acceptance gate

**Files:**
- None (verification only).

- [ ] **Step 1: Run the full check suite**

Run: `uv run poe check`
Expected: format clean, lint clean, mypy clean, all tests pass (~213+).

- [ ] **Step 2: Verify §14 acceptance criteria #2 — calibration crash is caught**

Create a temporary YAML reproducing the user's OOM (manual sanity check; do not commit):

```bash
cat > /tmp/oom_test.yaml <<'EOF'
system:
  memory_budget_gb: 22.0

profiles:
  granite-oom: {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 64}

engines:
  md-granite:
    description: "test"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "t"
    workflow: "w"
    tuning_profile: "granite-oom"
EOF
```

Run: `DBS_CONFIG_FILE=/tmp/oom_test.yaml uv run dbs-vector ingest /tmp 2>&1 | head -30`

Expected output contains:
- `granite-oom` (profile name)
- `md-granite` (engine name)
- `conservative estimate` (memory wording)
- `16384` (suggested max_token_length, preserved from input)
- A suggested smaller `batch_size`

Clean up: `rm /tmp/oom_test.yaml`

- [ ] **Step 3: Verify §14 acceptance criterion #4 — batch_size from profile**

The `IngestionService` constructor now takes `batch_size`. The test `test_ingestion_service_uses_self_batch_size_in_batched` from Task 10 covers this. Verify it passed in Step 1.

- [ ] **Step 4: Verify §14 acceptance criteria #5 — six engines parse from config**

Run:

```bash
uv run python -c "
from dbs_vector.config import load_settings
s = load_settings('config.yaml', validate=True)
print('Engines:', sorted(s.engines.keys()))
print('Profiles:', sorted(s.profiles.keys()))
"
```

Expected output:
- `Engines: ['md', 'md-granite', 'sql', 'sql-api', 'sql-api-granite', 'sql-granite']`
- `Profiles: ['gemma-md', 'gemma-sql-atomic', 'granite-md-large', 'granite-sql-atomic']`

- [ ] **Step 5: Verify §14 acceptance criterion #9 — --help/--version with broken config**

```bash
mkdir -p /tmp/dbs_help_test
cd /tmp/dbs_help_test
echo "this is: not: yaml: at all : :" > config.yaml
DBS_CONFIG_FILE=$(pwd)/config.yaml uv run dbs-vector --help
DBS_CONFIG_FILE=$(pwd)/config.yaml uv run dbs-vector --version
cd -
rm -rf /tmp/dbs_help_test
```

Expected: both commands exit 0 with normal output, neither raises a YAML parse error.

- [ ] **Step 6: Verify §14 acceptance criterion #10 — old-schema migration hint**

```bash
mkdir -p /tmp/dbs_migration_test
cd /tmp/dbs_migration_test
cat > config.yaml <<'EOF'
system:
  batch_size: 8
EOF
DBS_CONFIG_FILE=$(pwd)/config.yaml uv run dbs-vector ingest /tmp 2>&1 | grep -E "Legacy keys found|Config schema mismatch"
cd -
rm -rf /tmp/dbs_migration_test
```

Expected: output contains `Legacy keys found` and `batch_size`.

- [ ] **Step 7: Verify import safety end-to-end**

```bash
uv run python -c "
import sys
from unittest.mock import patch
with patch('pathlib.Path.open') as p, patch('builtins.open') as b:
    import dbs_vector.config
    print('Path.open calls:', p.call_count)
    print('builtins.open calls:', b.call_count)
"
```

Expected output:
```
Path.open calls: 0
builtins.open calls: 0
```

- [ ] **Step 8: Final commit (if any docs/test fixes were needed)**

If Steps 1-7 surfaced any small fixes, commit them:

```bash
git status
# review and commit if needed
```

If everything passed cleanly, no commit is needed for this task.

- [ ] **Step 9: Summary log**

Print and capture:

```bash
git log --oneline main..HEAD
```

This is the full set of commits introduced by this PR.
