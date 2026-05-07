# Dynamic Engine Exposure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `search_documents` / `search_sql_logs` MCP tools with a config-driven, per-engine MCP surface that auto-registers from `settings.engines`. Drop FastAPI and the streamable-HTTP transport entirely; `dbs-vector mcp` (stdio) becomes the sole presentation surface.

**Architecture:** A two-registry split decouples presentation from infrastructure. `core.families.FamilyKeyRegistry` (lightweight, no presentation imports) holds valid family keys for `config.py` validation. `mcp.families.FamilyRegistry` holds full `SearchFamily` instances exposing `run_search` / `format_results` / `make_handler`. At startup, `register_search_tools(mcp)` iterates `settings.engines`, runs a pre-flight validation pass, then registers one MCP tool per engine via the family's `make_handler(engine_name)` closure. `register_discovery_tool(mcp)` adds a `list_engines` tool sharing the same `_dbs_vector_registrations` idempotency dict.

**Tech Stack:** Python 3.12+, Pydantic v2, FastMCP (`mcp.server.fastmcp.FastMCP`), Typer, pytest, ruff, mypy. Apple MLX for embeddings, LanceDB for storage.

**Spec:** `docs/superpowers/specs/2026-05-07-dynamic-engine-exposure-design.md` (commits `8ad930d` → `677a18a`).

---

## File Structure

**New files:**
- `src/dbs_vector/core/families.py` — `FamilyKeyRegistry` class + built-in key registrations (no presentation imports)
- `src/dbs_vector/mcp/__init__.py` — empty package marker
- `src/dbs_vector/mcp/families/__init__.py` — built-in `FamilyRegistry` registrations
- `src/dbs_vector/mcp/families/base.py` — `SearchFamily` Protocol
- `src/dbs_vector/mcp/families/registry.py` — `FamilyRegistry` class
- `src/dbs_vector/mcp/families/document.py` — `DocumentFamily` (run_search, format_results, make_handler)
- `src/dbs_vector/mcp/families/sql.py` — `SqlFamily` (with min_time)
- `src/dbs_vector/mcp/dynamic_tools.py` — `register_search_tools(mcp)`
- `src/dbs_vector/mcp/discovery.py` — `_list_engines` + `register_discovery_tool(mcp)`
- `tests/unit/test_family_key_registry.py`
- `tests/unit/test_family_registry.py`
- `tests/unit/test_document_family.py`
- `tests/unit/test_sql_family.py`
- `tests/unit/test_dynamic_tools.py`
- `tests/unit/test_list_engines_tool.py`

**Moved files (git mv):**
- `src/dbs_vector/api/state.py` → `src/dbs_vector/mcp/state.py`
- `src/dbs_vector/api/mcp_server.py` → `src/dbs_vector/mcp/server.py`

**Modified files:**
- `src/dbs_vector/config.py` — `EngineConfig.family` field + `resolved_family` property + validation rules 6 & 7
- `src/dbs_vector/cli.py` — remove `serve` subcommand; update `mcp()` to call `start_stdio_server()` while preserving `--config-file` override
- `tests/unit/test_cli_callback.py` — extend with three-form override coverage
- `tests/unit/test_config_validation.py` — extend with engine-name and family-key tests
- `tests/unit/test_config_import_safety.py` — extend with `dbs_vector.mcp` not in sys.modules check
- `tests/integration/test_cli.py` — drop the serve subprocess assertions
- `pyproject.toml` — remove `fastapi` and `uvicorn`
- `uv.lock` — regenerate
- `docs/README_MCP.md` — full rewrite (drop streamable-HTTP section; new tool naming)
- `docs/README_PROFILES.md` — add A/B testing section
- `CLAUDE.md` — drop FastAPI references; update commands and architecture

**Deleted files:**
- `src/dbs_vector/api/main.py`
- `src/dbs_vector/api/__init__.py`
- `tests/integration/test_api.py`
- `tests/unit/test_api_lifespan.py`
- `tests/unit/api/test_mcp_server.py` (legacy tool tests; replaced by family tests)
- `docs/README_API.md`

---

## Task ordering rationale

Tasks 1-9 add new code without breaking the existing FastAPI / legacy MCP surface. Task 7 (the `state.py` move) updates importers in `api/main.py`, `api/mcp_server.py`, and `cli.py` so they continue to work via the new path. Task 10 then moves `mcp_server.py` and removes legacy tools. Task 11 updates the CLI. Task 12 deletes the now-orphaned files. Task 13 cleans up dependencies. Tasks 14-15 finish docs and verify everything.

---

## Task 1: FamilyKeyRegistry (core layer)

**Files:**
- Create: `src/dbs_vector/core/families.py`
- Create: `tests/unit/test_family_key_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_family_key_registry.py`:

```python
"""Tests for FamilyKeyRegistry — the lightweight, presentation-agnostic registry."""

import pytest

from dbs_vector.core.families import FamilyKeyRegistry


@pytest.fixture(autouse=True)
def _restore_keys():
    """Snapshot/restore the registry around each test so reset_for_testing
    in one test cannot leak into another."""
    snapshot = set(FamilyKeyRegistry.keys())
    yield
    FamilyKeyRegistry._reset_for_testing()
    for key in snapshot:
        FamilyKeyRegistry.register(key)


def test_builtin_keys_are_registered():
    assert FamilyKeyRegistry.is_valid("document")
    assert FamilyKeyRegistry.is_valid("sql")


def test_register_adds_new_key():
    FamilyKeyRegistry.register("jira")
    assert FamilyKeyRegistry.is_valid("jira")


def test_is_valid_returns_false_for_unknown_key():
    assert FamilyKeyRegistry.is_valid("nonexistent") is False


def test_keys_returns_sorted_list():
    keys = FamilyKeyRegistry.keys()
    assert keys == sorted(keys)
    assert "document" in keys
    assert "sql" in keys


def test_reset_for_testing_clears_keys():
    FamilyKeyRegistry._reset_for_testing()
    assert FamilyKeyRegistry.keys() == []
    assert FamilyKeyRegistry.is_valid("document") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_family_key_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.core.families'`

- [ ] **Step 3: Implement FamilyKeyRegistry**

Create `src/dbs_vector/core/families.py`:

```python
"""Lightweight family-key registry. No presentation-layer imports.

This module is imported by `dbs_vector.config` to validate that an engine's
`resolved_family` is a known family key, without dragging FastMCP or any
presentation-layer modules into the config import path. The full
SearchFamily registry (with run_search / format_results / make_handler
implementations) lives at `dbs_vector.mcp.families.registry` and is loaded
only by runtime callers.
"""


class FamilyKeyRegistry:
    """Open/closed registry of valid search family keys."""

    _keys: set[str] = set()

    @classmethod
    def register(cls, key: str) -> None:
        cls._keys.add(key)

    @classmethod
    def is_valid(cls, key: str) -> bool:
        return key in cls._keys

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._keys)

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Clear all keys. Tests must restore prior state via fixture cleanup."""
        cls._keys.clear()


# Built-in keys. Adding a new family also requires registering its key here
# (in addition to registering the SearchFamily instance in mcp/families/).
FamilyKeyRegistry.register("document")
FamilyKeyRegistry.register("sql")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_family_key_registry.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/core/families.py tests/unit/test_family_key_registry.py
git commit -m "feat(core): add FamilyKeyRegistry for presentation-agnostic family validation"
```

---

## Task 2: EngineConfig.family field + validation rules

**Files:**
- Modify: `src/dbs_vector/config.py`
- Modify: `tests/unit/test_config_validation.py`

- [ ] **Step 1: Write failing tests for the new field and validation rules**

Append to `tests/unit/test_config_validation.py`:

```python
def test_resolved_family_falls_back_to_mapper_type():
    """When EngineConfig.family is None, resolved_family equals mapper_type."""
    from dbs_vector.config import EngineConfig

    engine = EngineConfig(
        description="t",
        model="gemma-bf16",
        mapper_type="document",
        chunker_type="document",
        table_name="t",
        workflow="t",
        tuning_profile="p",
    )
    assert engine.family is None
    assert engine.resolved_family == "document"


def test_resolved_family_uses_explicit_family_when_set():
    """When EngineConfig.family is set, resolved_family uses it."""
    from dbs_vector.config import EngineConfig

    engine = EngineConfig(
        description="t",
        model="gemma-bf16",
        mapper_type="custom-mapper",
        chunker_type="document",
        table_name="t",
        workflow="t",
        tuning_profile="p",
        family="document",
    )
    assert engine.resolved_family == "document"


def test_validate_rejects_illegal_engine_name():
    """Engine names must match ^[a-z0-9][a-z0-9_-]*$ (Rule 6)."""
    import tempfile
    from pathlib import Path

    from dbs_vector.config import load_settings

    config_yaml = """
profiles:
  p: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}

engines:
  Bad-Name:
    description: "x"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "t"
    workflow: "w"
    tuning_profile: "p"
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(config_yaml)
        with pytest.raises(ValueError, match="Engine name 'Bad-Name'"):
            load_settings(str(path), validate=True)


def test_validate_rejects_unknown_family():
    """An engine whose resolved_family is not in FamilyKeyRegistry raises (Rule 7)."""
    import tempfile
    from pathlib import Path

    from dbs_vector.config import load_settings

    config_yaml = """
profiles:
  p: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}

engines:
  oddball:
    description: "x"
    model: "gemma-bf16"
    mapper_type: "nope-not-a-family"
    chunker_type: "document"
    table_name: "t"
    workflow: "w"
    tuning_profile: "p"
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(config_yaml)
        with pytest.raises(ValueError, match="unknown family"):
            load_settings(str(path), validate=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_validation.py -v -k "resolved_family or illegal_engine_name or unknown_family"`
Expected: 4 failures with attribute or schema errors.

- [ ] **Step 3: Add `family` field and `resolved_family` property to EngineConfig**

In `src/dbs_vector/config.py`, modify the `EngineConfig` class. Find the existing class definition (it has `mapper_type: str` near the top). Add `family: str | None = None` immediately after `mapper_type`, and add a `resolved_family` property at the end of the class:

```python
class EngineConfig(BaseModel):
    """Per-deployment engine config. References model contract + tuning profile."""

    model_config = ConfigDict(extra="forbid")

    description: str
    model: str  # key into ModelRegistry
    mapper_type: str
    family: str | None = None  # optional override; defaults to mapper_type
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

    @property
    def resolved_family(self) -> str:
        """Family key for presentation-layer dispatch.

        Defaults to mapper_type for backwards compatibility; overridable via
        the `family:` config field when an engine's mapper differs from its
        intended presentation surface.
        """
        return self.family or self.mapper_type

    def chunker_kwargs(
        self,
        chunk_max_chars: int,
        query_override: str | None = None,
        url_override: str | None = None,
    ) -> dict[str, object]:
        """Resolve chunker init kwargs. `chunk_max_chars` is injected by the
        caller from the resolved tuning profile (no longer a field on Engine)."""
        # ... existing body unchanged ...
```

- [ ] **Step 4: Add validation rules 6 and 7 to `_validate_config`**

In the same file, find `_validate_config(settings, config_file)`. Add the engine-name regex import at the top of the function and add Rule 6 + Rule 7 inside the `for engine_name, engine in settings.engines.items():` loop, BEFORE the existing Rule 1:

```python
import re
from dbs_vector.core.families import FamilyKeyRegistry

_ENGINE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
```

(Place the imports at the top of `_validate_config` body — they're cheap and keeps the dependency local. The `_ENGINE_NAME_PATTERN` constant goes at module top alongside `_LEGACY_ENGINE_FIELDS`.)

Inside the loop, immediately after `for engine_name, engine in settings.engines.items():`, add:

```python
        # Rule 6: legal engine name (presentation layer requires this for
        # MCP tool naming and predictable URLs).
        if not _ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match "
                f"{_ENGINE_NAME_PATTERN.pattern}. Allowed: lowercase letters, "
                f"digits, dash, underscore (must start with letter or digit). "
                f"Edit {config_file}."
            )

        # Rule 7: family resolves to a known FamilyKeyRegistry entry.
        if not FamilyKeyRegistry.is_valid(engine.resolved_family):
            raise ValueError(
                f"Engine '{engine_name}' references unknown family "
                f"'{engine.resolved_family}'. Known families: "
                f"{FamilyKeyRegistry.keys()}. Edit {config_file}."
            )
```

The `_ENGINE_NAME_PATTERN` constant should be defined at the top of `config.py` (module level) so both `_validate_config` and any future caller can reuse it. Add it next to `_LEGACY_ENGINE_FIELDS`:

```python
_ENGINE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
```

And add `import re` at the top of the file if not already imported.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config_validation.py -v`
Expected: all green including the 4 new ones.

Run: `uv run pytest tests/unit -v`
Expected: all green (no regressions in other tests using the existing 6 engines, all of which have lowercase mapper_type values that pass Rule 7).

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/config.py tests/unit/test_config_validation.py
git commit -m "feat(config): add EngineConfig.family + validation rules 6/7

Rule 6 enforces engine-name pattern ^[a-z0-9][a-z0-9_-]*\$ for
predictable MCP tool naming. Rule 7 verifies resolved_family is a
known key in FamilyKeyRegistry. resolved_family property defaults to
mapper_type, preserving backward compat with all six existing engines."
```

---

## Task 3: SearchFamily Protocol + FamilyRegistry

**Files:**
- Create: `src/dbs_vector/mcp/__init__.py`
- Create: `src/dbs_vector/mcp/families/__init__.py` (deferred — populated in Task 6)
- Create: `src/dbs_vector/mcp/families/base.py`
- Create: `src/dbs_vector/mcp/families/registry.py`
- Create: `tests/unit/test_family_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_family_registry.py`:

```python
"""Tests for FamilyRegistry — the full presentation-layer registry."""

from typing import Any

import pytest

from dbs_vector.core.families import FamilyKeyRegistry
from dbs_vector.mcp.families.registry import FamilyRegistry


class _StubFamily:
    """Test double matching the SearchFamily Protocol — only `name` is used
    in registry tests; the Protocol's other methods aren't called here."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def format_results(self, results: list[Any], query: str) -> str:
        return ""

    def make_handler(self, engine_name: str) -> Any:
        async def handler() -> str:
            return ""
        return handler


@pytest.fixture(autouse=True)
def _restore_registries():
    key_snapshot = set(FamilyKeyRegistry.keys())
    fam_snapshot = dict(FamilyRegistry._families)
    yield
    FamilyRegistry._reset_for_testing()
    FamilyKeyRegistry._reset_for_testing()
    for k in key_snapshot:
        FamilyKeyRegistry.register(k)
    for fam in fam_snapshot.values():
        FamilyRegistry.register(fam)


def test_register_with_known_key_succeeds():
    FamilyKeyRegistry.register("document")
    fam = _StubFamily("document")
    FamilyRegistry.register(fam)
    assert FamilyRegistry.get("document") is fam


def test_register_with_unknown_key_raises_runtime_error():
    FamilyKeyRegistry._reset_for_testing()  # no keys registered
    fam = _StubFamily("ghost")
    with pytest.raises(RuntimeError, match="no matching key in FamilyKeyRegistry"):
        FamilyRegistry.register(fam)


def test_duplicate_registration_raises():
    FamilyKeyRegistry.register("document")
    FamilyRegistry.register(_StubFamily("document"))
    with pytest.raises(ValueError, match="already registered"):
        FamilyRegistry.register(_StubFamily("document"))


def test_get_unknown_raises_with_known_list():
    FamilyKeyRegistry.register("document")
    FamilyRegistry.register(_StubFamily("document"))
    with pytest.raises(KeyError, match=r"Unknown.*Known: \['document'\]"):
        FamilyRegistry.get("nonexistent")


def test_keys_returns_sorted():
    FamilyKeyRegistry.register("document")
    FamilyKeyRegistry.register("sql")
    FamilyRegistry.register(_StubFamily("sql"))
    FamilyRegistry.register(_StubFamily("document"))
    assert FamilyRegistry.keys() == ["document", "sql"]


def test_reset_for_testing_clears_families():
    FamilyKeyRegistry.register("document")
    FamilyRegistry.register(_StubFamily("document"))
    FamilyRegistry._reset_for_testing()
    assert FamilyRegistry.keys() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_family_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.mcp'`

- [ ] **Step 3: Create empty package markers and the Protocol**

```bash
mkdir -p src/dbs_vector/mcp/families
```

Create `src/dbs_vector/mcp/__init__.py`:

```python
"""MCP presentation layer for dbs-vector."""
```

Create `src/dbs_vector/mcp/families/base.py`:

```python
"""SearchFamily Protocol: contract that each search family implements."""

from typing import Any, Protocol

from dbs_vector.services.search import SearchService


class SearchFamily(Protocol):
    """Self-contained MCP-layer plugin for a class of search engines.

    Each family owns:
      - A search dispatcher (translate kwargs → service call → list of results).
      - A result formatter (translate results → human-readable string).
      - A handler factory (build a per-engine async function with a concrete
        signature that FastMCP will introspect for its tool schema).

    The handler signature returned by make_handler() IS the family's public
    argument schema. There is no separate args_model — duplication risks
    drift, and FastMCP's introspection works on the handler directly.
    """

    name: str  # e.g., "document", "sql"; must match a key in FamilyKeyRegistry

    def run_search(
        self,
        service: SearchService,
        query: str,
        limit: int,
        source_filter: str | None,
        **family_kwargs: Any,
    ) -> list[Any]:
        """Run the search and return the raw result list."""
        ...

    def format_results(self, results: list[Any], query: str) -> str:
        """Render results for an MCP tool's stdout."""
        ...

    def make_handler(self, engine_name: str) -> Any:
        """Build a per-engine async handler whose explicit signature FastMCP
        will introspect for the tool schema."""
        ...
```

- [ ] **Step 4: Create the FamilyRegistry**

Create `src/dbs_vector/mcp/families/registry.py`:

```python
"""FamilyRegistry: full SearchFamily implementations, keyed by family name.

Cross-checks against FamilyKeyRegistry on registration to ensure every
presentation-layer family has a corresponding core key. config.py relies on
the core key registry to validate engine.resolved_family without importing
this module.
"""

from dbs_vector.core.families import FamilyKeyRegistry
from dbs_vector.mcp.families.base import SearchFamily


class FamilyRegistry:
    """Open/closed registry of full SearchFamily implementations."""

    _families: dict[str, SearchFamily] = {}

    @classmethod
    def register(cls, family: SearchFamily) -> None:
        if not FamilyKeyRegistry.is_valid(family.name):
            raise RuntimeError(
                f"SearchFamily '{family.name}' has no matching key in "
                f"FamilyKeyRegistry. Add `FamilyKeyRegistry.register({family.name!r})` "
                f"in core/families.py before registering the implementation."
            )
        if family.name in cls._families:
            raise ValueError(f"Search family '{family.name}' already registered")
        cls._families[family.name] = family

    @classmethod
    def get(cls, key: str) -> SearchFamily:
        if key not in cls._families:
            known = sorted(cls._families)
            raise KeyError(f"Unknown search family '{key}'. Known: {known}")
        return cls._families[key]

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._families)

    @classmethod
    def _reset_for_testing(cls) -> None:
        cls._families.clear()
```

Create `src/dbs_vector/mcp/families/__init__.py` (placeholder — populated in Task 6):

```python
"""Built-in family registrations live here (populated in Task 6)."""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_family_registry.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/mcp/__init__.py src/dbs_vector/mcp/families/__init__.py src/dbs_vector/mcp/families/base.py src/dbs_vector/mcp/families/registry.py tests/unit/test_family_registry.py
git commit -m "feat(mcp): add SearchFamily Protocol and FamilyRegistry

FamilyRegistry cross-checks against core.families.FamilyKeyRegistry on
registration so presentation keys cannot drift from the core allow-list."
```

---

## Task 4: DocumentFamily

**Files:**
- Create: `src/dbs_vector/mcp/families/document.py`
- Create: `tests/unit/test_document_family.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_document_family.py`:

```python
"""Tests for DocumentFamily run_search / format_results / make_handler."""

import inspect
from unittest.mock import MagicMock

import pytest

from dbs_vector.core.models import Chunk, SearchResult
from dbs_vector.mcp.families.document import DocumentFamily


def test_run_search_calls_service_with_kwargs():
    fam = DocumentFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="hello", limit=3, source_filter="docs/")

    service.execute_query.assert_called_once_with("hello", "docs/", 3, extra_filters={})


def test_format_results_includes_source_and_text():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="hello world", source="doc.md", content_hash="abc"),
            score=None,
            distance=0.1234,
            is_fts_match=False,
        ),
    ]
    out = fam.format_results(results, query="q")
    assert "Found 1 results for 'q'" in out
    assert "Source: doc.md" in out
    assert "hello world" in out
    assert "0.1234" in out


def test_format_results_uses_score_when_distance_none():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="t", source="s.md", content_hash="h"),
            score=0.0325,
            distance=None,
            is_fts_match=False,
        ),
    ]
    out = fam.format_results(results, query="q")
    assert "0.0325" in out
    assert "FTS" not in out


def test_format_results_marks_fts_match_with_no_score_or_distance():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="t", source="s.md", content_hash="h"),
            score=None,
            distance=None,
            is_fts_match=True,
        ),
    ]
    out = fam.format_results(results, query="q")
    assert "FTS" in out


def test_format_results_empty_returns_no_results_message():
    fam = DocumentFamily()
    out = fam.format_results([], query="zzz")
    assert out == "No results found for query: 'zzz'"


def test_make_handler_signature_has_expected_parameters():
    """FastMCP introspects this signature to build the tool schema."""
    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    sig = inspect.signature(handler)
    params = sig.parameters
    assert list(params) == ["query", "limit", "source_filter"]
    assert params["query"].annotation is str
    assert params["limit"].default == 5
    assert params["source_filter"].default is None


@pytest.mark.asyncio
async def test_make_handler_returns_error_when_service_missing(monkeypatch):
    """Handler reports a clear error if _services has no entry for the engine."""
    import dbs_vector.mcp.state as state_mod

    monkeypatch.setattr(state_mod, "_services", {})
    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    out = await handler(query="x")
    assert "search service 'md-test' is not initialized" in out


@pytest.mark.asyncio
async def test_make_handler_runs_search_and_formats(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [
        SearchResult(
            chunk=Chunk(id="x_0", text="content", source="f.md", content_hash="h"),
            score=None,
            distance=0.5,
            is_fts_match=False,
        ),
    ]
    monkeypatch.setattr(state_mod, "_services", {"md-test": service})

    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    out = await handler(query="q", limit=1)

    service.execute_query.assert_called_once_with("q", None, 1, extra_filters={})
    assert "Found 1 results for 'q'" in out
    assert "Source: f.md" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_document_family.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.mcp.families.document'`. Also fails on `dbs_vector.mcp.state` — that module gets created in Task 7. For now, just confirm the import error and proceed.

- [ ] **Step 3: Implement DocumentFamily**

Create `src/dbs_vector/mcp/families/document.py`:

```python
"""DocumentFamily — search engines whose results are document chunks."""

import asyncio
from typing import Any

from dbs_vector.services.search import SearchService


class DocumentFamily:
    """SearchFamily implementation for document-style engines (markdown,
    prose, etc.)."""

    name: str = "document"

    def run_search(
        self,
        service: SearchService,
        query: str,
        limit: int,
        source_filter: str | None,
        **family_kwargs: Any,
    ) -> list[Any]:
        return service.execute_query(query, source_filter, limit, extra_filters={})

    def format_results(self, results: list[Any], query: str) -> str:
        if not results:
            return f"No results found for query: '{query}'"

        output = [f"Found {len(results)} results for '{query}':\n"]
        for res in results:
            if res.distance is not None:
                dist_str = f"{res.distance:.4f}"
            elif res.score is not None:
                dist_str = f"{res.score:.4f}"
            else:
                dist_str = "N/A (FTS)"
            chunk = res.chunk
            output.append(
                f"--- Result (Score: {dist_str}) ---\n"
                f"Source: {chunk.source}\n"
                f"Content:\n{chunk.text}\n"
            )
        return "\n".join(output)

    def make_handler(self, engine_name: str) -> Any:
        family = self  # closure capture

        async def handler(
            query: str,
            limit: int = 5,
            source_filter: str | None = None,
        ) -> str:
            from dbs_vector.mcp.state import _services  # lazy import

            service = _services.get(engine_name)
            if service is None:
                return f"Error: search service '{engine_name}' is not initialized."
            try:
                results = await asyncio.to_thread(
                    family.run_search,
                    service,
                    query,
                    limit,
                    source_filter,
                )
                return family.format_results(results, query)
            except Exception as e:
                return f"Search execution failed: {e}"

        return handler
```

- [ ] **Step 4: Run tests to verify they pass**

The tests import `dbs_vector.mcp.state`, which doesn't exist yet. To unblock these tests now, create a placeholder `src/dbs_vector/mcp/state.py`:

```python
"""Placeholder; real implementation moved here in Task 7."""

_services: dict = {}
```

Run: `uv run pytest tests/unit/test_document_family.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/families/document.py src/dbs_vector/mcp/state.py tests/unit/test_document_family.py
git commit -m "feat(mcp): add DocumentFamily — run_search/format_results/make_handler

Includes a placeholder mcp/state.py so the family handler can import
_services. The placeholder is replaced in Task 7 by the moved api/state.py."
```

---

## Task 5: SqlFamily

**Files:**
- Create: `src/dbs_vector/mcp/families/sql.py`
- Create: `tests/unit/test_sql_family.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_sql_family.py`:

```python
"""Tests for SqlFamily run_search / format_results / make_handler — includes
the family-specific min_time filter."""

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from dbs_vector.core.models import SqlChunk, SqlSearchResult
from dbs_vector.mcp.families.sql import SqlFamily


def _make_sql_result() -> SqlSearchResult:
    return SqlSearchResult(
        chunk=SqlChunk(
            id="sql_0",
            text="SELECT * FROM t",
            raw_query="SELECT * FROM t WHERE id=1",
            source="prod_db",
            execution_time_ms=500.5,
            calls=2,
            content_hash="h",
            latest_ts=datetime.now(),
        ),
        score=None,
        distance=0.5678,
        is_fts_match=False,
    )


def test_run_search_passes_min_time_filter():
    fam = SqlFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="q", limit=2, source_filter=None, min_time=100.0)

    service.execute_query.assert_called_once_with(
        "q", None, 2, extra_filters={"min_time": 100.0}
    )


def test_run_search_omits_min_time_when_unset():
    fam = SqlFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="q", limit=2, source_filter=None)

    service.execute_query.assert_called_once_with("q", None, 2, extra_filters={})


def test_format_results_includes_execution_time_calls_and_raw_query():
    fam = SqlFamily()
    out = fam.format_results([_make_sql_result()], query="q")
    assert "Found 1 results for 'q'" in out
    assert "Source Database: prod_db" in out
    assert "Execution Time: 500.5ms (Calls: 2)" in out
    assert "SELECT * FROM t WHERE id=1" in out
    assert "0.5678" in out


def test_format_results_empty_returns_no_results_message():
    fam = SqlFamily()
    out = fam.format_results([], query="zzz")
    assert out == "No results found for query: 'zzz'"


def test_make_handler_signature_includes_min_time():
    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    sig = inspect.signature(handler)
    params = sig.parameters
    assert list(params) == ["query", "limit", "source_filter", "min_time"]
    assert params["min_time"].default is None


@pytest.mark.asyncio
async def test_make_handler_runs_search_and_formats(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [_make_sql_result()]
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    out = await handler(query="q", limit=1, min_time=200.0)

    service.execute_query.assert_called_once_with(
        "q", None, 1, extra_filters={"min_time": 200.0}
    )
    assert "Source Database: prod_db" in out


@pytest.mark.asyncio
async def test_make_handler_handles_exception(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.side_effect = Exception("DB down")
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    out = await handler(query="q")

    assert "Search execution failed: DB down" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sql_family.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.mcp.families.sql'`.

- [ ] **Step 3: Implement SqlFamily**

Create `src/dbs_vector/mcp/families/sql.py`:

```python
"""SqlFamily — search engines whose results are SQL query log entries."""

import asyncio
from typing import Any

from dbs_vector.services.search import SearchService


class SqlFamily:
    """SearchFamily implementation for SQL-log-style engines.

    Adds a family-specific `min_time` filter (minimum execution time in
    milliseconds) on top of the standard query/limit/source_filter args.
    """

    name: str = "sql"

    def run_search(
        self,
        service: SearchService,
        query: str,
        limit: int,
        source_filter: str | None,
        **family_kwargs: Any,
    ) -> list[Any]:
        extra_filters: dict[str, Any] = {}
        min_time = family_kwargs.get("min_time")
        if min_time is not None:
            extra_filters["min_time"] = min_time
        return service.execute_query(query, source_filter, limit, extra_filters=extra_filters)

    def format_results(self, results: list[Any], query: str) -> str:
        if not results:
            return f"No results found for query: '{query}'"

        output = [f"Found {len(results)} results for '{query}':\n"]
        for res in results:
            if res.distance is not None:
                dist_str = f"{res.distance:.4f}"
            elif res.score is not None:
                dist_str = f"{res.score:.4f}"
            else:
                dist_str = "N/A (FTS)"
            chunk = res.chunk
            output.append(
                f"--- Result (Score: {dist_str}) ---\n"
                f"Source Database: {chunk.source}\n"
                f"Execution Time: {chunk.execution_time_ms}ms (Calls: {chunk.calls})\n"
                f"SQL Query:\n{chunk.raw_query}\n"
            )
        return "\n".join(output)

    def make_handler(self, engine_name: str) -> Any:
        family = self  # closure capture

        async def handler(
            query: str,
            limit: int = 5,
            source_filter: str | None = None,
            min_time: float | None = None,
        ) -> str:
            from dbs_vector.mcp.state import _services  # lazy import

            service = _services.get(engine_name)
            if service is None:
                return f"Error: search service '{engine_name}' is not initialized."
            try:
                results = await asyncio.to_thread(
                    family.run_search,
                    service,
                    query,
                    limit,
                    source_filter,
                    min_time=min_time,
                )
                return family.format_results(results, query)
            except Exception as e:
                return f"Search execution failed: {e}"

        return handler
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sql_family.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/families/sql.py tests/unit/test_sql_family.py
git commit -m "feat(mcp): add SqlFamily with min_time filter passthrough"
```

---

## Task 6: Wire built-in family registrations

**Files:**
- Modify: `src/dbs_vector/mcp/families/__init__.py`

- [ ] **Step 1: Add a smoke test for the registrations**

Append to `tests/unit/test_family_registry.py`:

```python
def test_builtin_registrations_present():
    """Importing dbs_vector.mcp.families registers DocumentFamily and SqlFamily."""
    # Reset both registries to a clean state
    FamilyRegistry._reset_for_testing()
    FamilyKeyRegistry._reset_for_testing()
    # Re-register core keys (the dbs_vector.core.families module-level
    # registrations only run once at import time)
    FamilyKeyRegistry.register("document")
    FamilyKeyRegistry.register("sql")

    # Re-import the package's __init__ to re-trigger registrations
    import importlib

    import dbs_vector.mcp.families

    importlib.reload(dbs_vector.mcp.families)

    assert "document" in FamilyRegistry.keys()
    assert "sql" in FamilyRegistry.keys()
    assert FamilyRegistry.get("document").name == "document"
    assert FamilyRegistry.get("sql").name == "sql"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_family_registry.py::test_builtin_registrations_present -v`
Expected: FAIL — the families package `__init__.py` does not register anything yet.

- [ ] **Step 3: Populate `mcp/families/__init__.py`**

Replace `src/dbs_vector/mcp/families/__init__.py` with:

```python
"""Built-in SearchFamily registrations.

Importing this package registers DocumentFamily and SqlFamily with the
FamilyRegistry. The registrations cross-check against
dbs_vector.core.families.FamilyKeyRegistry on the way in.

To add a new family:
  1. Register the key in `dbs_vector/core/families.py` at module top.
  2. Implement the SearchFamily here in a new module.
  3. Add `FamilyRegistry.register(NewFamily())` below.
"""

from dbs_vector.mcp.families.document import DocumentFamily
from dbs_vector.mcp.families.registry import FamilyRegistry
from dbs_vector.mcp.families.sql import SqlFamily


def _register_builtins() -> None:
    """Idempotent registration: skip if already registered (module reload)."""
    for fam in (DocumentFamily(), SqlFamily()):
        if fam.name not in FamilyRegistry.keys():
            FamilyRegistry.register(fam)


_register_builtins()


__all__ = ["FamilyRegistry"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_family_registry.py -v`
Expected: PASS, 7 tests (the 6 from Task 3 + the new builtin smoke test).

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/families/__init__.py tests/unit/test_family_registry.py
git commit -m "feat(mcp): register DocumentFamily and SqlFamily as built-ins

Importing dbs_vector.mcp.families now wires both families into
FamilyRegistry. Cross-check against FamilyKeyRegistry remains in place."
```

---

## Task 7: Move api/state.py → mcp/state.py

**Files:**
- Move: `src/dbs_vector/api/state.py` → `src/dbs_vector/mcp/state.py`
- Modify: `src/dbs_vector/api/main.py` (import path)
- Modify: `src/dbs_vector/api/mcp_server.py` (import path)
- Modify: `src/dbs_vector/cli.py` (import path in `mcp` subcommand)

- [ ] **Step 1: Run baseline tests**

Run: `uv run poe check`
Expected: all green (current state).

- [ ] **Step 2: Delete the placeholder created in Task 4 and do the real move**

```bash
rm src/dbs_vector/mcp/state.py
git mv src/dbs_vector/api/state.py src/dbs_vector/mcp/state.py
```

- [ ] **Step 3: Update importers**

In `src/dbs_vector/api/main.py`, change the import (line 12):

```python
from dbs_vector.mcp.state import _services, initialize_services
```

In `src/dbs_vector/api/mcp_server.py`, change the import (line 5):

```python
from dbs_vector.mcp.state import _services
```

In `src/dbs_vector/cli.py`, change the import inside `mcp()` (line 209):

```python
    from dbs_vector.mcp.state import initialize_services
```

- [ ] **Step 4: Run tests to verify nothing regressed**

Run: `uv run poe check`
Expected: all green. The Document/Sql family tests that monkey-patched `dbs_vector.mcp.state` already work with the moved module; the FastAPI app, legacy MCP tools, and CLI all import from the new location.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/state.py src/dbs_vector/api/main.py src/dbs_vector/api/mcp_server.py src/dbs_vector/cli.py
git commit -m "refactor: move api/state.py → mcp/state.py

Update three importers (api/main.py, api/mcp_server.py, cli.py).
Behavior unchanged; this is preparation for removing the api/ package."
```

---

## Task 8: register_search_tools (dynamic_tools.py)

**Files:**
- Create: `src/dbs_vector/mcp/dynamic_tools.py`
- Create: `tests/unit/test_dynamic_tools.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_dynamic_tools.py`:

```python
"""Tests for register_search_tools — pre-flight validation, idempotency,
collision detection, and naming convention."""

from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

import dbs_vector.config as config_mod
import dbs_vector.mcp.dynamic_tools as dyn
from dbs_vector.config import EngineConfig, Settings, TuningProfile


def _make_settings(engines: dict[str, EngineConfig]) -> Settings:
    s = Settings()
    s.engines = engines
    s.profiles = {"p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)}
    return s


def _make_engine(mapper: str = "document", desc: str = "test") -> EngineConfig:
    return EngineConfig(
        description=desc,
        model="gemma-bf16",
        mapper_type=mapper,
        chunker_type=mapper,
        table_name="t",
        workflow="w",
        tuning_profile="p",
    )


@pytest.fixture
def fresh_mcp() -> FastMCP:
    return FastMCP("test-dbs-vector")


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    """Each test gets a fresh Settings stub patched into both the
    config singleton and the dynamic_tools module."""
    s = Settings()
    monkeypatch.setattr(config_mod, "settings", s)
    monkeypatch.setattr(dyn, "settings", s)
    yield s


def test_normalize_tool_name_replaces_dashes_with_underscores():
    assert dyn._normalize_tool_name("md-granite") == "search_md_granite"
    assert dyn._normalize_tool_name("md") == "search_md"
    assert dyn._normalize_tool_name("sql-api-granite") == "search_sql_api_granite"


def test_register_search_tools_registers_one_tool_per_engine(fresh_mcp, _clean_settings):
    _clean_settings.engines = {
        "md": _make_engine("document", "Markdown engine"),
        "sql": _make_engine("sql", "SQL engine"),
    }
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)

    tool_names = {t.name for t in fresh_mcp._tool_manager.list_tools()}
    assert "search_md" in tool_names
    assert "search_sql" in tool_names


def test_register_search_tools_legacy_names_absent(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"md": _make_engine("document"), "sql": _make_engine("sql")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)

    tool_names = {t.name for t in fresh_mcp._tool_manager.list_tools()}
    assert "search_documents" not in tool_names
    assert "search_sql_logs" not in tool_names


def test_register_search_tools_uses_engine_description(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"md": _make_engine("document", "Markdown & Prose")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)

    tool = next(t for t in fresh_mcp._tool_manager.list_tools() if t.name == "search_md")
    assert tool.description == "Markdown & Prose"


def test_invalid_engine_name_raises(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"Bad-Name": _make_engine("document")}
    with pytest.raises(ValueError, match="Engine name 'Bad-Name'"):
        dyn.register_search_tools(fresh_mcp)


def test_collision_detection_raises(fresh_mcp, _clean_settings):
    _clean_settings.engines = {
        "md-granite": _make_engine("document"),
        "md_granite": _make_engine("document"),
    }
    with pytest.raises(ValueError, match="MCP tool name collision"):
        dyn.register_search_tools(fresh_mcp)


def test_unknown_family_raises(fresh_mcp, _clean_settings):
    bad_engine = _make_engine("document")
    bad_engine.family = "ghost"  # bypasses config-time validation
    _clean_settings.engines = {"x": bad_engine}
    with pytest.raises(KeyError, match="Unknown search family 'ghost'"):
        dyn.register_search_tools(fresh_mcp)


def test_idempotent_registration_with_identical_settings(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"md": _make_engine("document")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)
    count_after_first = len(fresh_mcp._tool_manager.list_tools())
    dyn.register_search_tools(fresh_mcp)
    count_after_second = len(fresh_mcp._tool_manager.list_tools())

    assert count_after_first == count_after_second


def test_stale_registration_with_different_family_raises(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"x": _make_engine("document")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }
    dyn.register_search_tools(fresh_mcp)

    # Now mutate to a different family for the same engine name
    _clean_settings.engines["x"] = _make_engine("sql")
    with pytest.raises(RuntimeError, match="Stale tool registration"):
        dyn.register_search_tools(fresh_mcp)


def test_pre_flight_atomicity_no_partial_registration(fresh_mcp, _clean_settings):
    """If the LAST engine has a bad family, NONE of the earlier engines' tools
    should be registered."""
    bad_engine = _make_engine("document")
    bad_engine.family = "ghost"
    _clean_settings.engines = {
        "md": _make_engine("document"),
        "sql": _make_engine("sql"),
        "broken": bad_engine,
    }
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    with pytest.raises(KeyError):
        dyn.register_search_tools(fresh_mcp)

    tools = fresh_mcp._tool_manager.list_tools()
    assert tools == []
    assert getattr(fresh_mcp, "_dbs_vector_registrations", {}) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_dynamic_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.mcp.dynamic_tools'`.

- [ ] **Step 3: Implement register_search_tools**

Create `src/dbs_vector/mcp/dynamic_tools.py`:

```python
"""Dynamic MCP tool registration: one tool per engine in settings.engines.

Reads the populated dbs_vector.config.settings singleton. Tests monkey-patch
this module's `settings` import for isolation.
"""

import re

from mcp.server.fastmcp import FastMCP

from dbs_vector.config import settings
from dbs_vector.mcp.families.registry import FamilyRegistry

_ENGINE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _normalize_tool_name(engine_name: str) -> str:
    """Convert engine name to a valid MCP tool name. Dashes → underscores."""
    return f"search_{engine_name.replace('-', '_')}"


def register_search_tools(mcp: FastMCP) -> None:
    """Iterate settings.engines and register one MCP tool per engine.

    Reads from the module-level `settings` singleton (already populated by
    the CLI callback via _populate_singleton_from). Tests monkey-patch
    `dbs_vector.mcp.dynamic_tools.settings` for isolation.

    Idempotency rules:
      - Skip if the same (engine_name, family_key) is already registered.
      - Raise if the same tool name is registered with a DIFFERENT
        (engine_name, family_key) — settings are expected to be immutable
        for the lifetime of a FastMCP instance.

    Pre-flight failures (raise before any tool is registered):
      - Engine name not matching _ENGINE_NAME_PATTERN.
      - Two distinct engines normalize to the same MCP tool name.
      - Engine references a family not in FamilyRegistry.

    Pre-flight resolves and validates all engines BEFORE any tool is added,
    so a config with N engines where the last has an unknown family will
    not leave the first N-1 tools half-registered.
    """
    if not hasattr(mcp, "_dbs_vector_registrations"):
        mcp._dbs_vector_registrations = {}  # tool_name → (engine_name, family_key)
    registrations: dict[str, tuple[str, str]] = mcp._dbs_vector_registrations

    # Pre-flight: name pattern + collision + family resolution.
    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str]] = []
    for engine_name, engine in settings.engines.items():
        if not _ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {_ENGINE_NAME_PATTERN.pattern}. "
                f"Allowed: lowercase, digits, dash, underscore (must start with letter or digit)."
            )
        tool_name = _normalize_tool_name(engine_name)
        if tool_name in seen:
            raise ValueError(
                f"MCP tool name collision: engines '{seen[tool_name]}' and "
                f"'{engine_name}' both normalize to '{tool_name}'. "
                f"Rename one of them in config.yaml."
            )
        seen[tool_name] = engine_name

        family_key = engine.resolved_family
        FamilyRegistry.get(family_key)  # raises KeyError if unknown
        resolved.append((engine_name, tool_name, family_key))

    # Registration phase — every engine has been validated.
    for engine_name, tool_name, family_key in resolved:
        family = FamilyRegistry.get(family_key)
        prior = registrations.get(tool_name)
        if prior is not None:
            if prior == (engine_name, family_key):
                continue  # idempotent — same registration
            raise RuntimeError(
                f"Stale tool registration for '{tool_name}': previously registered "
                f"as engine={prior[0]} family={prior[1]}, now requested as "
                f"engine={engine_name} family={family_key}. Reset the FastMCP "
                f"instance instead of re-registering with different settings."
            )

        engine = settings.engines[engine_name]
        handler = family.make_handler(engine_name)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=engine.description,
        )
        registrations[tool_name] = (engine_name, family_key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_dynamic_tools.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/dynamic_tools.py tests/unit/test_dynamic_tools.py
git commit -m "feat(mcp): register_search_tools — pre-flight atomic, idempotent, collision-safe

Pre-flight validates engine names, family resolution, and tool-name
collisions BEFORE any mcp.add_tool call. Idempotency uses a per-mcp
_dbs_vector_registrations dict; identical re-registration is a no-op
while mismatched re-registration raises."
```

---

## Task 9: list_engines discovery tool

**Files:**
- Create: `src/dbs_vector/mcp/discovery.py`
- Create: `tests/unit/test_list_engines_tool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_list_engines_tool.py`:

```python
"""Tests for list_engines MCP tool and register_discovery_tool."""

import json

import pytest
from mcp.server.fastmcp import FastMCP

import dbs_vector.config as config_mod
import dbs_vector.mcp.discovery as discovery_mod
import dbs_vector.mcp.state as state_mod
from dbs_vector.config import EngineConfig, Settings, TuningProfile


def _make_engine(mapper: str = "document", desc: str = "test") -> EngineConfig:
    return EngineConfig(
        description=desc,
        model="gemma-bf16",
        mapper_type=mapper,
        chunker_type=mapper,
        table_name=f"{mapper}_table",
        workflow="w",
        tuning_profile="p",
    )


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    s = Settings()
    s.engines = {"md": _make_engine("document"), "sql": _make_engine("sql")}
    s.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=1000, batch_size=64)
    }
    monkeypatch.setattr(config_mod, "settings", s)
    monkeypatch.setattr(discovery_mod, "settings", s)
    monkeypatch.setattr(state_mod, "_services", {})
    yield s


@pytest.fixture
def fresh_mcp() -> FastMCP:
    return FastMCP("test-dbs-vector")


@pytest.mark.asyncio
async def test_list_engines_returns_metadata_per_engine():
    out_str = await discovery_mod._list_engines()
    out = json.loads(out_str)
    names = {e["name"] for e in out}
    assert names == {"md", "sql"}


@pytest.mark.asyncio
async def test_list_engines_includes_profile_knobs():
    out = json.loads(await discovery_mod._list_engines())
    md_entry = next(e for e in out if e["name"] == "md")
    assert md_entry["profile"]["max_token_length"] == 2048
    assert md_entry["profile"]["chunk_max_chars"] == 1000
    assert md_entry["profile"]["batch_size"] == 64
    assert md_entry["profile"]["name"] == "p"


@pytest.mark.asyncio
async def test_list_engines_marks_unloaded_engines(_clean_state):
    out = json.loads(await discovery_mod._list_engines())
    for entry in out:
        assert entry["loaded"] is False


@pytest.mark.asyncio
async def test_list_engines_marks_loaded_engines(_clean_state, monkeypatch):
    monkeypatch.setattr(state_mod, "_services", {"md": object()})
    out = json.loads(await discovery_mod._list_engines())
    md = next(e for e in out if e["name"] == "md")
    sql = next(e for e in out if e["name"] == "sql")
    assert md["loaded"] is True
    assert sql["loaded"] is False


@pytest.mark.asyncio
async def test_list_engines_works_with_partial_services_map(_clean_state, monkeypatch):
    """list_engines tolerates _services missing entries — does not crash."""
    monkeypatch.setattr(state_mod, "_services", {"md": object()})
    out = json.loads(await discovery_mod._list_engines())
    assert len(out) == 2  # both still listed; sql.loaded == False


@pytest.mark.asyncio
async def test_list_engines_includes_mcp_tool_name():
    out = json.loads(await discovery_mod._list_engines())
    md = next(e for e in out if e["name"] == "md")
    assert md["mcp_tool"] == "search_md"


def test_register_discovery_tool_registers_list_engines(fresh_mcp):
    discovery_mod.register_discovery_tool(fresh_mcp)
    tool_names = {t.name for t in fresh_mcp._tool_manager.list_tools()}
    assert "list_engines" in tool_names


def test_register_discovery_tool_idempotent(fresh_mcp):
    discovery_mod.register_discovery_tool(fresh_mcp)
    count_first = len(fresh_mcp._tool_manager.list_tools())
    discovery_mod.register_discovery_tool(fresh_mcp)
    count_second = len(fresh_mcp._tool_manager.list_tools())
    assert count_first == count_second


def test_register_discovery_tool_raises_on_name_clash(fresh_mcp):
    """If something else registered list_engines under a different sentinel,
    the discovery registrar refuses to silently overwrite."""
    fresh_mcp._dbs_vector_registrations = {"list_engines": ("foo", "bar")}
    with pytest.raises(RuntimeError, match="non-discovery sentinel"):
        discovery_mod.register_discovery_tool(fresh_mcp)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_list_engines_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbs_vector.mcp.discovery'`.

- [ ] **Step 3: Implement the discovery module**

Create `src/dbs_vector/mcp/discovery.py`:

```python
"""list_engines MCP tool and its registration helper.

list_engines reads from settings + ModelRegistry directly; it does not
depend on the runtime _services dict beyond reporting which entries are
loaded. The MCP server itself is all-or-nothing at startup, so when
list_engines is reachable, every engine that initialized successfully
will report loaded: true. The flag exists for tests with a partial
_services map and for any future partial-loader work (out of scope).
"""

import json

from mcp.server.fastmcp import FastMCP

from dbs_vector.config import settings

# Sentinel tracked in the same _dbs_vector_registrations dict that
# register_search_tools uses, so idempotency state is shared.
_DISCOVERY_SENTINEL = ("__discovery__", "__discovery__")


async def _list_engines() -> str:
    """List configured search engines and their tuning profiles.

    Returns a JSON-encoded list of engine metadata: name, family, model,
    description, table name, profile knobs (max_token_length,
    chunk_max_chars, batch_size), MCP tool name, and whether a runtime
    service object is currently registered for that engine. Useful for A/B
    testing harnesses and for clients that want to enumerate available
    variants programmatically.
    """
    from dbs_vector.core.model_registry import ModelRegistry
    from dbs_vector.mcp.state import _services

    out = []
    for name, engine in settings.engines.items():
        contract = ModelRegistry.get(engine.model)
        profile = settings.profiles[engine.tuning_profile]
        out.append({
            "name": name,
            "family": engine.resolved_family,
            "model": engine.model,
            "model_name": contract.model_name,
            "description": engine.description,
            "table_name": engine.table_name,
            "profile": {
                "name": engine.tuning_profile,
                "max_token_length": profile.max_token_length,
                "chunk_max_chars": profile.chunk_max_chars,
                "batch_size": profile.batch_size,
            },
            "mcp_tool": f"search_{name.replace('-', '_')}",
            "loaded": name in _services,
        })
    return json.dumps(out, indent=2)


def register_discovery_tool(mcp: FastMCP) -> None:
    """Register the list_engines MCP tool.

    Skip-if-identical when our discovery sentinel is already in
    mcp._dbs_vector_registrations. Raise on a non-discovery occupation
    of the `list_engines` slot.
    """
    if not hasattr(mcp, "_dbs_vector_registrations"):
        mcp._dbs_vector_registrations = {}
    registrations: dict = mcp._dbs_vector_registrations

    prior = registrations.get("list_engines")
    if prior == _DISCOVERY_SENTINEL:
        return
    if prior is not None:
        raise RuntimeError(
            f"Tool 'list_engines' already registered with non-discovery "
            f"sentinel {prior!r}. Reset the FastMCP instance instead of "
            f"re-registering."
        )

    mcp.add_tool(
        _list_engines,
        name="list_engines",
        description="List configured search engines and their tuning profiles.",
    )
    registrations["list_engines"] = _DISCOVERY_SENTINEL
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_list_engines_tool.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/mcp/discovery.py tests/unit/test_list_engines_tool.py
git commit -m "feat(mcp): add list_engines discovery tool with idempotent registration

Shares _dbs_vector_registrations with register_search_tools via a
discovery sentinel. Tolerates partial _services maps for tests and
future partial-loader work."
```

---

## Task 10: Move api/mcp_server.py → mcp/server.py and add start_stdio_server()

**Files:**
- Move: `src/dbs_vector/api/mcp_server.py` → `src/dbs_vector/mcp/server.py`
- Modify: `src/dbs_vector/api/main.py` (drop import of legacy `mcp` symbol if any)
- Modify: `src/dbs_vector/cli.py` (update import in `mcp` subcommand)
- Delete: `tests/unit/api/test_mcp_server.py` (legacy tool tests; replaced by family tests)

- [ ] **Step 1: Move the file**

```bash
git mv src/dbs_vector/api/mcp_server.py src/dbs_vector/mcp/server.py
```

- [ ] **Step 2: Rewrite server.py — drop legacy tools, add start_stdio_server**

Replace the contents of `src/dbs_vector/mcp/server.py` with:

```python
"""FastMCP server instance + stdio entry point.

Tool registration is dynamic: per-engine search tools come from
register_search_tools(); the list_engines tool comes from
register_discovery_tool(). Both run inside start_stdio_server() before the
mcp.run() loop begins.
"""

from mcp.server.fastmcp import FastMCP

from dbs_vector.mcp.discovery import register_discovery_tool
from dbs_vector.mcp.dynamic_tools import register_search_tools
from dbs_vector.mcp.state import initialize_services

mcp = FastMCP(
    "dbs-vector",
    stateless_http=True,
)


def start_stdio_server() -> None:
    """Initialize services, register all tools, and run stdio MCP.

    Takes no arguments — dbs_vector.config.settings is already populated by
    the CLI callback's _populate_singleton_from(...) call before this runs.
    initialize_services(), register_search_tools(mcp), and
    register_discovery_tool(mcp) all read from the singleton too, so
    settings ownership is consistent across the lifecycle.
    """
    initialize_services()
    register_search_tools(mcp)
    register_discovery_tool(mcp)
    mcp.run()
```

- [ ] **Step 3: Update FastAPI `main.py` to import from the new path**

`src/dbs_vector/api/main.py` currently has `from dbs_vector.api.mcp_server import mcp`. Change to:

```python
from dbs_vector.mcp.server import mcp
```

(This file gets deleted in Task 12; the change is just to keep tests green between Tasks 10 and 12.)

- [ ] **Step 4: Update cli.py mcp() subcommand import**

In `src/dbs_vector/cli.py`, find this block inside `mcp()` (currently around line 208):

```python
    from dbs_vector.api.mcp_server import mcp as mcp_server
    from dbs_vector.api.state import initialize_services
    from dbs_vector.config import _populate_singleton_from, load_settings
```

Replace with:

```python
    from dbs_vector.config import _populate_singleton_from, load_settings
    from dbs_vector.mcp.server import start_stdio_server
```

Then change the body of `mcp()` from the existing form (which calls `initialize_services()` and `mcp_server.run()` directly) to call `start_stdio_server()`. The full updated body:

```python
    """Starts the FastMCP standard input/output (stdio) server for integrations."""
    import os

    from dbs_vector.config import _populate_singleton_from, load_settings
    from dbs_vector.mcp.server import start_stdio_server

    # If the subcommand was given a config file (e.g., `dbs-vector mcp -c X`),
    # re-load and re-populate the singleton; otherwise rely on what the global
    # callback already loaded. Also re-export DBS_CONFIG_FILE so spawned
    # subprocesses see the same path.
    if config_file is not None:
        os.environ["DBS_CONFIG_FILE"] = config_file
        new_settings = load_settings(config_file, validate=True)
        _populate_singleton_from(new_settings)

    logger.info("Initializing MLX Embedders and LanceDB connections")
    try:
        start_stdio_server()
    except Exception as e:
        logger.error("Failed to initialize search services: {}", e)
        raise
```

- [ ] **Step 5: Delete the legacy MCP server tests**

```bash
git rm tests/unit/api/test_mcp_server.py
# Remove the empty directory if it exists:
rmdir tests/unit/api 2>/dev/null || true
```

- [ ] **Step 6: Run tests to verify nothing else regressed**

Run: `uv run poe check`

Some FastAPI/CLI tests may still be green (they don't depend on the legacy MCP tool tests we just removed). Tests in `tests/integration/test_cli.py` that assert `serve` invocation still pass — Task 11 deletes them. Tests for `mcp -c Y` callback already pass — they don't depend on legacy tools.

Expected: green or with only `tests/integration/test_cli.py`'s serve subprocess assertions still passing (they are removed in Task 11).

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/mcp/server.py src/dbs_vector/api/main.py src/dbs_vector/cli.py
git rm tests/unit/api/test_mcp_server.py
git commit -m "refactor(mcp): move mcp_server.py to mcp/server.py; add start_stdio_server

- Drop hardcoded search_documents and search_sql_logs tools from the
  module. Per-engine tools come from register_search_tools(mcp); the
  list_engines tool comes from register_discovery_tool(mcp). Both are
  invoked inside start_stdio_server() before mcp.run().
- cli.py mcp() subcommand calls start_stdio_server() while preserving
  the existing -c / --config-file override path.
- Delete tests/unit/api/test_mcp_server.py (legacy tools removed)."
```

---

## Task 11: Remove serve subcommand; extend CLI callback tests

**Files:**
- Modify: `src/dbs_vector/cli.py` (delete `serve` function)
- Modify: `tests/integration/test_cli.py` (drop the four serve-subprocess assertions)
- Modify: `tests/unit/test_cli_callback.py` (extend with override-path coverage)

- [ ] **Step 1: Add new tests for the override paths**

Append to `tests/unit/test_cli_callback.py`:

```python
def test_mcp_uses_global_config_when_no_subcommand_override(tmp_path, monkeypatch):
    """`dbs-vector mcp` (no -c) uses the global callback's loaded config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'system:\n  db_path: "./global_db"\n'
        'profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n'
        'engines: {}\n'
    )
    monkeypatch.setenv("DBS_CONFIG_FILE", str(config_path))

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod

        called = {"yes": False}

        def fake_start():
            called["yes"] = True

        ctx.setattr(server_mod, "start_stdio_server", fake_start)
        result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0
    assert called["yes"] is True

    from dbs_vector.config import settings
    assert settings.db_path == "./global_db"


def test_mcp_uses_global_callback_config(tmp_path, monkeypatch):
    """`dbs-vector -c X mcp` uses global config X."""
    config_path = tmp_path / "global.yaml"
    config_path.write_text(
        'system:\n  db_path: "./X_db"\n'
        'profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n'
        'engines: {}\n'
    )

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod
        ctx.setattr(server_mod, "start_stdio_server", lambda: None)
        result = runner.invoke(app, ["-c", str(config_path), "mcp"])
    assert result.exit_code == 0

    from dbs_vector.config import settings
    assert settings.db_path == "./X_db"


def test_mcp_subcommand_config_overrides_global(tmp_path, monkeypatch):
    """`dbs-vector mcp -c Y` reloads from Y AFTER the global callback ran."""
    global_path = tmp_path / "global.yaml"
    global_path.write_text(
        'system:\n  db_path: "./GLOBAL_db"\n'
        'profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n'
        'engines: {}\n'
    )
    sub_path = tmp_path / "sub.yaml"
    sub_path.write_text(
        'system:\n  db_path: "./SUB_db"\n'
        'profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n'
        'engines: {}\n'
    )

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod
        ctx.setattr(server_mod, "start_stdio_server", lambda: None)
        result = runner.invoke(app, ["-c", str(global_path), "mcp", "-c", str(sub_path)])
    assert result.exit_code == 0

    from dbs_vector.config import settings
    assert settings.db_path == "./SUB_db"


def test_mcp_subcommand_config_sets_env_var(tmp_path, monkeypatch):
    """`dbs-vector mcp -c Y` re-exports DBS_CONFIG_FILE so spawned subprocesses
    inherit Y, not the global value."""
    import os

    global_path = tmp_path / "global.yaml"
    global_path.write_text(
        'system:\n  db_path: "./G"\n'
        'profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n'
        'engines: {}\n'
    )
    sub_path = tmp_path / "sub.yaml"
    sub_path.write_text(
        'system:\n  db_path: "./S"\n'
        'profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n'
        'engines: {}\n'
    )

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod
        ctx.setattr(server_mod, "start_stdio_server", lambda: None)
        runner.invoke(app, ["-c", str(global_path), "mcp", "-c", str(sub_path)])

    assert os.environ.get("DBS_CONFIG_FILE") == str(sub_path)


def test_serve_subcommand_no_longer_exists():
    """The serve CLI command was deleted in the MCP-only revision."""
    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Delete the `serve` function and its uvicorn import from cli.py**

In `src/dbs_vector/cli.py`, find and DELETE the entire `serve` function:

```python
@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", "-h", help="Host to bind the API server to.")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to bind the API server to.")
    ] = 8000,
    reload: Annotated[
        bool, typer.Option("--reload", help="Enable auto-reload for development.")
    ] = False,
) -> None:
    """Starts the asynchronous FastAPI search server."""
    import uvicorn

    logger.info("Starting dbs-vector API server at http://{}:{}", host, port)
    uvicorn.run("dbs_vector.api.main:app", host=host, port=port, reload=reload)
```

Also DELETE the now-misleading inline comment near line 61 (`# Export to environment so uvicorn subprocesses (in API mode) inherit it`). Replace it with:

```python
    # Export to environment so any spawned subprocesses (e.g., MCP stdio
    # transport invoked via uv run) inherit the same config.
```

- [ ] **Step 3: Delete the `class TestServeCommand` block from `tests/integration/test_cli.py`**

The file contains a class `TestServeCommand` with four methods that assert `uvicorn.run("dbs_vector.api.main:app", ...)`. Delete the entire class. The current span is approximately lines 385–443 (locate the class by name in case other edits shifted line numbers).

Concretely, delete this block:

```python
class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_default_options(self, mock_settings):
        """Test serve with default options."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            result = runner.invoke(app, ["serve"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="127.0.0.1",
                port=8000,
                reload=False,
            )

    def test_serve_custom_host_port(self, mock_settings):
        """Test serve with custom host and port."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9000"])

            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="0.0.0.0",
                port=9000,
                reload=False,
            )

    def test_serve_with_reload(self, mock_settings):
        """Test serve with reload option."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            runner.invoke(app, ["serve", "--reload"])

            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="127.0.0.1",
                port=8000,
                reload=True,
            )

    def test_serve_short_options(self, mock_settings):
        """Test short options for serve command."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            runner.invoke(app, ["serve", "-h", "0.0.0.0", "-p", "8080"])

            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="0.0.0.0",
                port=8080,
                reload=False,
            )
```

Verify the deletion:

```bash
rg -n "TestServeCommand|main:app|uvicorn" tests/integration/test_cli.py
```

Expected: no matches. Replacement coverage for "serve no longer exists" lives in `tests/unit/test_cli_callback.py::test_serve_subcommand_no_longer_exists` (added in Step 1).

- [ ] **Step 4: Run tests to verify the new ones pass and nothing else broke**

Run: `uv run pytest tests/unit/test_cli_callback.py -v`
Expected: PASS — including the 5 new tests.

Run: `uv run pytest tests/integration/test_cli.py -v`
Expected: PASS — the four serve tests are gone; remaining ingest/search/mcp tests still pass.

Run: `uv run poe check`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/cli.py tests/unit/test_cli_callback.py tests/integration/test_cli.py
git commit -m "refactor(cli): remove serve subcommand; codify mcp -c override paths

The serve subcommand previously launched uvicorn-of-FastAPI; under the
MCP-stdio-only design it has no purpose. Five new tests in
test_cli_callback.py cover:
  - dbs-vector mcp uses the global callback's loaded config
  - dbs-vector -c X mcp uses global config X
  - dbs-vector mcp -c Y reloads + repopulates from Y
  - dbs-vector mcp -c Y re-exports DBS_CONFIG_FILE for subprocesses
  - dbs-vector serve --help fails with no-such-command"
```

---

## Task 12: Delete the api/ package, the FastAPI app, and obsolete tests

**Files:**
- Delete: `src/dbs_vector/api/main.py`
- Delete: `src/dbs_vector/api/__init__.py`
- Delete: `src/dbs_vector/api/` (directory)
- Delete: `tests/integration/test_api.py`
- Delete: `tests/unit/test_api_lifespan.py`

- [ ] **Step 1: Delete the FastAPI app and the api package**

```bash
git rm src/dbs_vector/api/main.py
git rm src/dbs_vector/api/__init__.py
rmdir src/dbs_vector/api
```

- [ ] **Step 2: Delete the FastAPI/lifespan tests**

```bash
git rm tests/integration/test_api.py
git rm tests/unit/test_api_lifespan.py
```

- [ ] **Step 3: Run tests to verify nothing in the surviving test suite broke**

Run: `uv run poe check`

If any test or source still references `dbs_vector.api.*`, the import will fail. Re-check:

```bash
rg -n "dbs_vector\.api" src tests
```

Expected: no matches. If any appear, fix them (they'll be in tests that should have been migrated earlier; update the import path or delete the test).

- [ ] **Step 4: Verify acceptance #9 passes**

```bash
uv run python -c "import dbs_vector.api"
```

Expected: `ModuleNotFoundError: No module named 'dbs_vector.api'`

```bash
uv run python -c "import dbs_vector.api.main"
```

Expected: `ModuleNotFoundError: No module named 'dbs_vector.api'`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete dbs_vector.api package — FastAPI removed

- Delete src/dbs_vector/api/main.py (FastAPI app)
- Delete src/dbs_vector/api/__init__.py (empty after moves)
- Delete tests/integration/test_api.py (replaced by family/dynamic_tools tests)
- Delete tests/unit/test_api_lifespan.py (FastAPI lifespan no longer exists)

The api/ directory is gone. All MCP code lives under dbs_vector.mcp.*."
```

---

## Task 13: Remove fastapi and uvicorn dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (regenerated)

- [ ] **Step 1: Verify there are no remaining imports**

```bash
rg -n "^(import|from) (fastapi|uvicorn)" src tests
```

Expected: no matches. If any appear, they MUST be cleaned up before removing the dependencies.

- [ ] **Step 2: Verify no transitive consumer requires them**

```bash
uv tree | grep -E "(fastapi|uvicorn)" || true
```

If anything other than `fastapi` or `uvicorn` themselves shows up as a parent of these packages, document it inline in `pyproject.toml`. Otherwise both are safe to remove.

- [ ] **Step 3: Remove the entries from pyproject.toml**

In `pyproject.toml`, find the dependencies section and delete the two lines:

```toml
"fastapi>=0.136.0",
...
"uvicorn>=0.45.0",
```

Save the file.

- [ ] **Step 4: Regenerate the lockfile and re-sync**

```bash
uv lock
uv sync
```

- [ ] **Step 5: Verify acceptance #11**

```bash
uv run python -c "
import sys
src_imports = open('pyproject.toml').read()
assert 'fastapi' not in src_imports, 'fastapi still in pyproject.toml'
assert 'uvicorn' not in src_imports, 'uvicorn still in pyproject.toml'
print('OK')
"
```

Expected: prints `OK`.

```bash
rg -n "import fastapi|from fastapi|import uvicorn|from uvicorn" src tests
```

Expected: no matches.

- [ ] **Step 6: Run the full check**

Run: `uv run poe check`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): drop fastapi and uvicorn

Both packages were used solely by the removed FastAPI app and the
deleted serve subcommand. No project source imports either; transitive
consumers (verified via uv tree) do not require them."
```

---

## Task 14: Documentation updates

**Files:**
- Delete: `docs/README_API.md`
- Modify: `docs/README_MCP.md`
- Modify: `docs/README_PROFILES.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Delete the API doc**

```bash
git rm docs/README_API.md
```

- [ ] **Step 2: Rewrite README_MCP.md**

The full content has these structural changes from the existing version:

1. Update the "Transport Methods" section: remove the table row for "Streamable HTTP". Replace the section with:

```markdown
## Transport

`dbs-vector` ships an MCP **stdio** transport only. The AI assistant
spawns `dbs-vector mcp` as a subprocess and communicates over its
standard input/output. No network ports are opened. Each client process
loads its own copy of the MLX models (~1.2 GB GPU memory each).

Streamable-HTTP MCP transport is not currently shipped — see the design
spec at `docs/superpowers/specs/2026-05-07-dynamic-engine-exposure-design.md`
for rationale and re-introduction notes.
```

2. Delete the entire "Method 2: Streamable HTTP" section. Renumber "Method 1" to remove the "Method 1" prefix (it's the only method now).

3. Delete the "Option B — Streamable HTTP" subsection from the Claude Desktop integration. Keep only the stdio configuration.

4. Delete the Cursor section (Cursor uses HTTP transport). Replace with a note:

```markdown
## Integrating with Cursor

Cursor's MCP integration currently expects an HTTP endpoint. Since
`dbs-vector` ships only stdio, Cursor cannot connect directly. If your
team needs Cursor support, you can wrap stdio with an external bridge
(e.g., `mcp-proxy`) — but this is not officially supported.
```

5. Replace the entire "Tools Provided" section with:

```markdown
## Tools Provided

`dbs-vector` registers one MCP tool per engine in `config.yaml`, plus
one `list_engines` discovery tool. Tool names follow the pattern
`search_<engine_name>` with dashes (`-`) replaced by underscores.

For the default `config.yaml` shipped with the project:

| Tool name | Engine | Family | Description |
|-----------|--------|--------|-------------|
| `search_md` | `md` | document | Markdown & Prose Document Engine (Gemma) |
| `search_sql` | `sql` | sql | SQL Slow Query Log Engine (Gemma) |
| `search_md_granite` | `md-granite` | document | Markdown & Prose (Granite, long context) |
| `search_sql_granite` | `sql-granite` | sql | SQL Slow Query Log (Granite) |
| `search_sql_api_granite` | `sql-api-granite` | sql | Remote slow query log API (Granite) |
| `list_engines` | — | — | Lists configured engines and tuning profiles |

### Search tools (per family)

**Document family** (`search_md`, `search_md_granite`, etc.) takes:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Semantic search query |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to a file path or pattern |

**SQL family** (`search_sql`, `search_sql_granite`, `search_sql_api_granite`) takes:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `query` | string | yes | Natural language or partial SQL |
| `limit` | int | no | Max results (default 5, max 100) |
| `source_filter` | string | no | Restrict to a database name |
| `min_time` | float | no | Minimum execution time in ms |

### `list_engines`

Returns a JSON-encoded array describing every configured engine: name,
family, model, description, table name, profile knobs
(`max_token_length`, `chunk_max_chars`, `batch_size`), MCP tool name,
and a `loaded` flag indicating whether the runtime service object is
currently registered. Useful for A/B-testing harnesses and for clients
that want to enumerate engines programmatically.

## Migration from legacy tool names

`dbs-vector` previously exposed two hardcoded tools: `search_documents`
and `search_sql_logs`. Both are **removed** in this revision. Update
your MCP client config or LLM prompts:

- `search_documents` → `search_md`
- `search_sql_logs` → `search_sql`

The new naming convention covers every engine in `config.yaml`,
including the Granite variants which were previously unreachable.

## A/B testing tuning profiles

Adding an experimental engine variant requires only a config edit:

\```yaml
profiles:
  granite-md-experimental: {max_token_length: 8192, chunk_max_chars: 3000, batch_size: 16}

engines:
  md-granite-experimental:
    description: "Granite, smaller chunks (A/B candidate vs md-granite)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite_exp"   # MUST differ from baseline
    workflow: "md_search_granite"
    tuning_profile: "granite-md-experimental"
\```

After ingesting into the new engine and restarting `dbs-vector mcp`, a
new MCP tool `search_md_granite_experimental` becomes available. Use
`list_engines` to confirm both variants are loaded and to compare their
profile knobs in your evaluation report.
```

6. In the "Troubleshooting" section, delete the rows that reference HTTP 404 and `mcp-proxy`. Keep only stdio-relevant rows.

7. Delete the "Test the Streamable HTTP endpoint" subsection.

8. In "Architecture Notes", replace the existing FastAPI / mounted-MCP description with:

```markdown
## Architecture Notes

- The MCP server is a `FastMCP` instance (`stateless_http=True`) created
  once in `src/dbs_vector/mcp/server.py`.
- Tool registration is dynamic: `register_search_tools(mcp)` iterates
  `settings.engines` and registers one `search_<engine>` tool per engine
  via the family's `make_handler(engine_name)` factory.
  `register_discovery_tool(mcp)` adds the `list_engines` tool.
- Both registration helpers run inside `start_stdio_server()` before
  `mcp.run()`. They share an idempotency dict (`_dbs_vector_registrations`)
  attached to the FastMCP instance.
- All engines defined in `config.yaml` are loaded once at startup
  (transport-agnostic — `initialize_services()` is in
  `dbs_vector.mcp.state`). Each `dbs-vector mcp` process loads its own
  engine instances.
```

- [ ] **Step 3: Add A/B testing section to README_PROFILES.md**

Append this section to `docs/README_PROFILES.md`:

```markdown
## A/B testing tuning profiles

Because each engine name maps to a distinct MCP tool and a distinct
LanceDB table, you can run two profile variants of the same model side
by side and compare results without code changes.

### Step 1: Define a new profile

\```yaml
profiles:
  granite-md-large:        {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  granite-md-experimental: {max_token_length: 8192,  chunk_max_chars: 3000, batch_size: 16}
\```

### Step 2: Define a new engine that references the variant profile

\```yaml
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
\```

### Step 3: Ingest into both engines

\```bash
uv run dbs-vector ingest "./docs/" --type md-granite
uv run dbs-vector ingest "./docs/" --type md-granite-experimental
\```

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
```

- [ ] **Step 4: Update CLAUDE.md**

Open `CLAUDE.md`. In the **Commands** section, delete the line:

```bash
uv run dbs-vector serve
```

In the **Architecture** section, find the paragraph that begins:

> The FastAPI routes (`/search/md`, `/search/sql`) and MCP tools (`search_documents`, `search_sql_logs`) are currently hardcoded to the Gemma engines. The Granite engines (`md-granite`, `sql-granite`, `sql-api-granite`) are CLI-only this PR; generalizing the routes is a tracked Phase 2 follow-up.

Replace it with:

```markdown
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
```

In the **`api/`** layer description, replace the entire bullet block with:

```markdown
**`mcp/`** — MCP presentation layer.
- `server.py`: FastMCP instance + `start_stdio_server()` entry point.
- `dynamic_tools.py`: `register_search_tools(mcp)` — pre-flight atomic, idempotent, collision-safe.
- `discovery.py`: `register_discovery_tool(mcp)` + `list_engines` tool.
- `state.py`: `_services` dict + `initialize_services()` (transport-agnostic).
- `families/`: `SearchFamily` Protocol + `FamilyRegistry` + built-in `DocumentFamily` and `SqlFamily`.
```

- [ ] **Step 5: Run the full check**

Run: `uv run poe check`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add docs/README_MCP.md docs/README_PROFILES.md CLAUDE.md
git rm docs/README_API.md
git commit -m "docs: rewrite for MCP-stdio-only dynamic engine exposure

- README_MCP.md: drop Streamable HTTP method; new per-engine tool
  naming convention; A/B testing workflow; legacy-tool migration note.
- README_PROFILES.md: add 'A/B testing tuning profiles' section.
- CLAUDE.md: drop serve example; replace api/ description with mcp/;
  rewrite the Phase-2-routes paragraph.
- Delete docs/README_API.md (FastAPI removed)."
```

---

## Task 15: Final acceptance check

**Files:**
- (no source changes)

- [ ] **Step 1: Run the full validation suite**

```bash
uv run poe check
```

Expected: ruff clean, mypy clean, all tests pass.

- [ ] **Step 2: Verify acceptance #1 — unit-test command**

```bash
uv run pytest \
  tests/unit/test_dynamic_tools.py \
  tests/unit/test_document_family.py \
  tests/unit/test_sql_family.py \
  tests/unit/test_family_key_registry.py \
  tests/unit/test_family_registry.py \
  tests/unit/test_list_engines_tool.py \
  tests/unit/test_config_validation.py \
  tests/unit/test_config_import_safety.py -v
```

Expected: all green.

- [ ] **Step 3: Verify acceptance #2 — import safety**

```bash
uv run python -c "
import dbs_vector.config
import sys
assert 'dbs_vector.mcp' not in sys.modules, sorted(sys.modules)
print('OK')
"
```

Expected: prints `OK`. If `dbs_vector.mcp` shows up, find what triggered the import (likely a config.py edit pulled in `mcp/families`) and refactor to lazy-import.

- [ ] **Step 4: Verify acceptances #3 + #4 — MCP tool listing and list_engines**

Start the server in one terminal:

```bash
uv run dbs-vector mcp
```

In another terminal, send a `tools/list` JSON-RPC request via stdin (or use an MCP client). Expected presence: `search_md`, `search_sql`, `search_md_granite`, `search_sql_granite`, `search_sql_api_granite`, `list_engines`. Expected absence: `search_documents`, `search_sql_logs`.

Then call `list_engines` and assert the JSON output contains all six configured engines with `loaded: true` and the right profile knobs.

(In CI, this is implicitly covered by the test_dynamic_tools.py and test_list_engines_tool.py unit tests, which are sufficient acceptance.)

- [ ] **Step 5: Verify acceptance #5 — config-only A/B variant**

Edit `config.yaml`, append the experimental engine and profile from the README_PROFILES.md A/B section. Restart `dbs-vector mcp` and confirm via the unit-test pattern that `search_md_granite_experimental` is in the registered tools list.

(In CI: covered by test_dynamic_tools.py — adding any engine name to settings produces the corresponding tool.)

- [ ] **Step 6: Verify acceptances #7 + #8 — collision and pre-flight atomicity**

Already covered by `test_collision_detection_raises` and `test_pre_flight_atomicity_no_partial_registration` in test_dynamic_tools.py.

- [ ] **Step 7: Verify acceptance #9 — api/ package gone**

```bash
uv run python -c "import dbs_vector.api"
```

Expected: `ModuleNotFoundError: No module named 'dbs_vector.api'`.

```bash
uv run python -c "import dbs_vector.api.main"
```

Expected: `ModuleNotFoundError: No module named 'dbs_vector.api'`.

- [ ] **Step 8: Verify acceptance #10 — serve subcommand gone**

```bash
uv run dbs-vector serve --help
```

Expected: Typer error "No such command 'serve'." (non-zero exit).

- [ ] **Step 9: Verify acceptance #11 — direct dependencies cleaned**

```bash
grep -E '^\s*"(fastapi|uvicorn)' pyproject.toml || echo "OK: not in pyproject.toml"
rg -n "^(import|from) (fastapi|uvicorn)" src tests || echo "OK: no project imports"
```

Expected: both print `OK`.

- [ ] **Step 10: Verify acceptance #12 — CLI override paths**

```bash
uv run pytest tests/unit/test_cli_callback.py -v -k "mcp_uses_global or mcp_subcommand_config or mcp_uses_global_config_when_no_subcommand or serve_subcommand_no_longer_exists"
```

Expected: 5 tests pass.

- [ ] **Step 11: Final commit if any cleanup was needed**

If the acceptance checks identified any small fix-ups (typos, missed imports), apply them and commit:

```bash
git add -A
git commit -m "chore: post-acceptance cleanup"
```

If everything passes without further changes, no commit is needed for this task.

---

## Spec coverage check

| Spec section / decision | Implementing task |
|---|---|
| §3.1 / Decision 11: two-registry split | Tasks 1, 3 |
| §3, §4.1: SearchFamily Protocol | Task 3 |
| §4.2: FamilyKeyRegistry | Task 1 |
| §4.3: FamilyRegistry + cross-check | Task 3 |
| §4.4: DocumentFamily make_handler | Task 4 |
| §4.4: SqlFamily make_handler | Task 5 |
| §4.5 / Decision 8, 9: register_search_tools (idempotency, pre-flight atomic, collision) | Task 8 |
| §4.6 / Decision 10: list_engines + register_discovery_tool (shared registrations dict) | Task 9 |
| §4.7: start_stdio_server() (settings-free) | Task 10 |
| §5.1 / Decision 16: EngineConfig.family + resolved_family | Task 2 |
| §5.2: validation rules 6 (engine name) and 7 (family resolves) | Task 2 |
| §6.1 / Decisions 4, 5, 6: per-engine tools, no aliases, legacy absent | Tasks 8, 10 |
| §6.5 / Decisions 2, 3: stdio only; serve removed | Tasks 11, 12 |
| §7.1: new files | Tasks 1, 3, 4, 5, 6, 8, 9 |
| §7.2: api/ → mcp/ moves | Tasks 7, 10 |
| §7.3: cli.py + config.py + pyproject.toml + uv.lock | Tasks 2, 11, 13 |
| §7.4: api/main.py + api/__init__.py deletion | Task 12 |
| §7.5: deleted test files | Tasks 10, 12 |
| §7.6 / Decision 19: dependency audit | Task 13 |
| §8 / Decision 18: api package deletion | Task 12 |
| §8.3 / Decision 21: doc updates | Task 14 |
| §9: testing strategy (per-family, dynamic, discovery, config validation, import safety, CLI callback) | Tasks 1, 3, 4, 5, 8, 9, 11 |
| §11 Decision 14: `_reset_for_testing` on both registries | Tasks 1, 3 |
| §11 Decision 15: E2E Granite tests gated | (existing test file, no change in scope) |
| §11 Decision 22: `dbs-vector mcp -c Y` override preserved | Tasks 10, 11 |
| §12: acceptance criteria 1-12 | Task 15 |
