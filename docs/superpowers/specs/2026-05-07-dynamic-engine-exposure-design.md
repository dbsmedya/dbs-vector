# Dynamic Engine Exposure (API + MCP) Design

**Date:** 2026-05-07
**Status:** Draft for review
**Goal:** Replace hardcoded `/search/md` and `/search/sql` HTTP routes and the `search_documents` / `search_sql_logs` MCP tools with a config-driven, per-engine surface that auto-registers from `settings.engines`. Granite engines (`md-granite`, `sql-granite`, `sql-api-granite`) become reachable via API and MCP without code changes; future families (e.g., `jira-chunker`) plug in as self-contained modules without modifying central API/MCP wiring.

---

## 1. Motivation

Today the API and MCP layers expose only two engines, hardcoded by name:

- `api/main.py` — `_services.get("md")` at line 108, `_services.get("sql")` at line 128
- `api/mcp_server.py` — `@mcp.tool()` decorators bound to `_services.get("md")` and `_services.get("sql")`

`initialize_services()` already loads every engine in `settings.engines` (including the Granite variants), but those services are unreachable. The CLAUDE.md docstring explicitly flags this as a Phase-2 follow-up.

Beyond exposing the existing Granite engines, the design must accommodate:

1. **A/B testing of tuning profiles.** Adding a new engine like `md-granite-experimental` (different `tuning_profile`, separate `table_name`) should produce a working API route + MCP tool without code changes.
2. **Future families** (e.g., `jira-chunker`) without modifying central API/MCP code. New family ⇒ one new module + one registration line.
3. **Discoverability** so A/B harnesses, eval scripts, and MCP clients can introspect what engines are loaded and what their profile knobs are.

---

## 2. Architectural Approach: Family-Registry Plugin Pattern

Each search "family" (document, sql, future jira) is a self-contained module that owns its request model, response model, dispatch logic, and MCP result formatting. A `FamilyRegistry` maps a string key (`"document"`, `"sql"`, ...) to a `SearchFamily` instance.

At server startup, both FastAPI and FastMCP iterate `settings.engines`, look up the family for each engine via `engine.resolved_family`, and dynamically register one HTTP route + one MCP tool per engine.

**OCP guarantees:**
- Existing family modules (`document.py`, `sql.py`) are never modified when a new family is added.
- Central registration modules (`dynamic_routes.py`, `dynamic_mcp.py`) iterate engines but contain no family-specific code.
- Adding a new family = one new module + one registration line in `families/__init__.py` (or wherever `FamilyRegistry.register()` is called).

This mirrors the existing `core/registry.py` ComponentRegistry pattern (chunkers, mappers) extended to the presentation layer.

---

## 3. Components

### 3.1 `SearchFamily` Protocol

**File:** `src/dbs_vector/api/families/base.py`

```python
from typing import Any, Protocol
from pydantic import BaseModel

from dbs_vector.services.search import SearchService


class SearchFamily(Protocol):
    """Self-contained presentation-layer plugin for a class of search engines.

    Each family owns:
      - The request schema (family-specific filters live here, e.g. `min_time`).
      - The response schema (concrete, not a Union).
      - HTTP dispatch (translate request → service call → response).
      - MCP result formatting (translate results → human-readable string).
    """

    name: str  # e.g., "document", "sql"; key in FamilyRegistry
    request_model: type[BaseModel]
    response_model: type[BaseModel]

    def dispatch_http(
        self,
        service: SearchService,
        request: BaseModel,
    ) -> BaseModel:
        """Run the search and build the typed response."""
        ...

    def format_mcp_result(self, results: list[Any], query: str) -> str:
        """Render results to a human-readable string for an MCP tool's stdout."""
        ...
```

**Why a Protocol (not ABC):** Structural typing matches the rest of the codebase (`IEmbedder`, `IChunker`, `IVectorStore` are all Protocols). No forced inheritance hierarchy.

### 3.2 `FamilyRegistry`

**File:** `src/dbs_vector/api/families/registry.py`

```python
from dbs_vector.api.families.base import SearchFamily


class FamilyRegistry:
    """Open/closed registry of search families. Adding a family = register() call."""

    _families: dict[str, SearchFamily] = {}

    @classmethod
    def register(cls, family: SearchFamily) -> None:
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
```

Built-in registrations live in `families/__init__.py`:

```python
from dbs_vector.api.families.registry import FamilyRegistry
from dbs_vector.api.families.document import DocumentFamily
from dbs_vector.api.families.sql import SqlFamily

FamilyRegistry.register(DocumentFamily())
FamilyRegistry.register(SqlFamily())
```

### 3.3 `DocumentFamily`

**File:** `src/dbs_vector/api/families/document.py`

Owns:
- `SearchRequest` (query, limit, source_filter)
- `SearchResponse` (query, results: list[SearchResult])
- `dispatch_http(service, request) → SearchResponse`
- `format_mcp_result(results, query) → str` — formatted as "Source: ... Content: ..."

`SearchRequest` and `SearchResponse` move from `api/main.py` into this module unchanged.

### 3.4 `SqlFamily`

**File:** `src/dbs_vector/api/families/sql.py`

Owns:
- `SqlSearchRequest` (query, limit, source_filter, **min_time**)
- `SqlSearchResponse` (query, results: list[SqlSearchResult])
- `dispatch_http(service, request) → SqlSearchResponse` — passes `min_time` via `extra_filters` if set
- `format_mcp_result(results, query) → str` — formatted as "Source Database: ... Execution Time: ...ms ... SQL Query: ..."

### 3.5 Dynamic HTTP Route Registration

**File:** `src/dbs_vector/api/dynamic_routes.py`

```python
def register_search_routes(app: FastAPI, settings: Settings) -> None:
    """Iterate settings.engines and register POST /search/{engine_name} per engine.

    Idempotent: if an engine is already registered on this app, skip silently.
    Clears app.openapi_schema after registration to invalidate any cached schema.
    """
    if not hasattr(app.state, "dynamic_engines"):
        app.state.dynamic_engines = set()

    for engine_name, engine in settings.engines.items():
        if engine_name in app.state.dynamic_engines:
            continue

        family = FamilyRegistry.get(engine.resolved_family)
        path = f"/search/{engine_name}"

        # Capture engine_name and family in a closure factory to avoid late-binding.
        endpoint = _make_endpoint(engine_name, family)

        app.add_api_route(
            path=path,
            endpoint=endpoint,
            methods=["POST"],
            response_model=family.response_model,
            tags=[engine.resolved_family],
            summary=engine.description,
        )
        app.state.dynamic_engines.add(engine_name)

    app.openapi_schema = None  # force OpenAPI regen if anything changed
```

`_make_endpoint(engine_name, family)` returns an async callable that:
1. Looks up `_services.get(engine_name)`, returns 503 if missing.
2. Validates request as `family.request_model`.
3. Offloads the synchronous `family.dispatch_http(service, request)` to `asyncio.to_thread`.
4. Returns the response, or wraps exceptions in HTTP 500.

### 3.6 Dynamic MCP Tool Registration

**File:** `src/dbs_vector/api/dynamic_mcp.py`

```python
import re

# Engine name allow-list: lowercase + digits + dash + underscore.
_ENGINE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _normalize_tool_name(engine_name: str) -> str:
    """Convert engine name to a valid MCP tool name. Dashes → underscores."""
    return f"search_{engine_name.replace('-', '_')}"


def register_search_tools(mcp: FastMCP, settings: Settings) -> None:
    """Iterate settings.engines and register one MCP tool per engine.

    Idempotent: tracks registered names on the FastMCP instance.
    Fails fast on engine name collisions (e.g., `md-granite` vs `md_granite`
    both normalize to `search_md_granite`).
    """
    if not hasattr(mcp, "_dynamic_tools"):
        mcp._dynamic_tools = set()

    # Pre-flight: collision detection across normalized names
    seen: dict[str, str] = {}
    for engine_name in settings.engines:
        if not _ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {_ENGINE_NAME_PATTERN.pattern}. "
                f"Allowed: lowercase, digits, dash, underscore."
            )
        tool_name = _normalize_tool_name(engine_name)
        if tool_name in seen:
            raise ValueError(
                f"MCP tool name collision: engines '{seen[tool_name]}' and "
                f"'{engine_name}' both normalize to '{tool_name}'. "
                f"Rename one of them in config.yaml."
            )
        seen[tool_name] = engine_name

    for engine_name, engine in settings.engines.items():
        tool_name = _normalize_tool_name(engine_name)
        if tool_name in mcp._dynamic_tools:
            continue

        family = FamilyRegistry.get(engine.resolved_family)
        handler = _make_mcp_handler(engine_name, family)

        mcp.add_tool(
            handler,
            name=tool_name,
            description=engine.description,
        )
        mcp._dynamic_tools.add(tool_name)
```

`_make_mcp_handler(engine_name, family)` returns an async callable whose argument schema mirrors `family.request_model` fields and whose body:
1. Looks up `_services.get(engine_name)`, returns an error string if missing.
2. Calls `family.dispatch_http(service, request)`.
3. Calls `family.format_mcp_result(results, query)` and returns the string.

### 3.7 `GET /engines` Discovery Endpoint

**File:** `src/dbs_vector/api/main.py` (added inline alongside `/health`)

Returns structured metadata for every configured engine, drawing from `settings.engines`, `settings.profiles`, and `ModelRegistry`. Does not require `_services` to be populated — works even if some engines failed to load.

```python
@app.get("/engines")
async def list_engines() -> list[dict]:
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
            "route": f"/search/{name}",
            "mcp_tool": f"search_{name.replace('-', '_')}",
            "loaded": name in _services,
        })
    return out
```

---

## 4. Configuration Changes

### 4.1 New optional field on `EngineConfig`

```python
class EngineConfig(BaseModel):
    ...
    mapper_type: str
    family: str | None = None  # NEW — optional override; defaults to mapper_type

    @property
    def resolved_family(self) -> str:
        return self.family or self.mapper_type
```

**Why optional with mapper_type fallback:** Today `mapper_type` and family align 1:1, so existing config.yaml works unchanged. The optional `family` field decouples the *concept* (presentation surface) from the *implementation* (Arrow ↔ domain mapping), so a future engine can specify e.g. `family: jira` while keeping `mapper_type: jira` — same value today, but the codebase is open for future divergence (e.g., a custom mapper that exposes the document family API).

### 4.2 No changes required to existing `config.yaml`

All six existing engines remain valid. Granite engines auto-expose with no config edits.

### 4.3 Adding an A/B variant (config-only)

```yaml
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
```

Restart server → `POST /search/md-granite-experimental` and MCP tool `search_md_granite_experimental` auto-register. `GET /engines` lists both variants with their profile values, enabling automated comparison reports.

### 4.4 Validation chain additions

`_validate_config` (in `config.py`) gains two rules executed per engine:

- **Rule 6 (legal engine name):** `engine_name` matches `^[a-z0-9][a-z0-9_-]*$`. Rejects empty, uppercase, leading-dash, or special-char names with a remediation message.
- **Rule 7 (family resolves):** `engine.resolved_family in FamilyRegistry.keys()`. Rejects misspelled families (e.g., `mapper_type: docs` should be `document`).

The collision-detection check is performed in `register_search_tools()` rather than `_validate_config` because it is presentation-layer-specific (the route layer doesn't suffer the same collision).

---

## 5. API Contract

### 5.1 HTTP Routes

| Route | Method | Status | Notes |
|---|---|---|---|
| `POST /search/{engine_name}` | POST | NEW (dynamic) | One per engine in `settings.engines`. Path is the engine name verbatim. |
| `GET /engines` | GET | NEW | Lists configured engines with profile metadata. |
| `GET /health` | GET | unchanged | Continues to list `{engine}_model` for each engine. |
| `POST /search/md` | POST | unchanged behavior | Now generated dynamically for the `md` engine. Same request/response shape as before. |
| `POST /search/sql` | POST | unchanged behavior | Now generated dynamically for the `sql` engine. Same request/response shape as before. |

**Error responses:**
- `404` on `POST /search/{unknown_engine}` — automatic via FastAPI route-not-found.
- `503` on a known engine when `_services.get(engine_name)` is None — engine configured but failed to load.
- `500` on dispatch exception — wrapped with the original error in `detail`.

### 5.2 Request/Response Models

`SearchRequest`, `SearchResponse`, `SqlSearchRequest`, `SqlSearchResponse` move from `api/main.py` into their respective family modules without schema changes. Existing JSON contracts are preserved.

---

## 6. MCP Contract

### 6.1 Tools

| Tool name | Status | Engine |
|---|---|---|
| `search_md` | NEW (replaces `search_documents`) | `md` |
| `search_sql` | NEW (replaces `search_sql_logs`) | `sql` |
| `search_md_granite` | NEW | `md-granite` |
| `search_sql_granite` | NEW | `sql-granite` |
| `search_sql_api_granite` | NEW | `sql-api-granite` |
| `search_documents` | **REMOVED** | — |
| `search_sql_logs` | **REMOVED** | — |

### 6.2 Tool naming convention

`search_<engine_name>` with dashes (`-`) replaced by underscores. The legal engine-name pattern `^[a-z0-9][a-z0-9_-]*$` guarantees the result is a valid Python identifier. Collision detection (see §3.6) prevents `md-granite` and `md_granite` from silently shadowing each other.

### 6.3 Tool description

Sourced from `engine.description` in config.yaml. Lets the user customize what an LLM sees about each variant — critical for A/B testing, where the description is how Claude picks between `search_md_granite` ("baseline") and `search_md_granite_experimental` ("smaller chunks for higher recall").

### 6.4 Tool argument schema

Mirrors the family's `request_model`:
- Document family: `query`, `limit`, `source_filter`
- SQL family: `query`, `limit`, `source_filter`, `min_time`

---

## 7. File Structure

### 7.1 New files

```
src/dbs_vector/api/families/
  __init__.py           # FamilyRegistry singleton + built-in registrations
  base.py               # SearchFamily Protocol
  registry.py           # FamilyRegistry class
  document.py           # DocumentFamily + SearchRequest + SearchResponse
  sql.py                # SqlFamily + SqlSearchRequest + SqlSearchResponse

src/dbs_vector/api/dynamic_routes.py   # register_search_routes(app, settings)
src/dbs_vector/api/dynamic_mcp.py      # register_search_tools(mcp, settings)
```

### 7.2 Modified files

```
src/dbs_vector/api/main.py
  - Drop hardcoded SearchRequest, SqlSearchRequest, SearchResponse, SqlSearchResponse
    (they move to family modules; re-import from there only if main.py needs them)
  - Drop hardcoded /search/md, /search/sql route definitions
  - Add register_search_routes(app, settings) call inside the lifespan, after
    initialize_services() succeeds
  - Add GET /engines endpoint inline

src/dbs_vector/api/mcp_server.py
  - Drop @mcp.tool() decorators for search_documents and search_sql_logs
  - Keep the FastMCP instance creation
  - Optional: add a module-level run helper that calls register_search_tools(mcp, settings)
    before starting

src/dbs_vector/cli.py
  - In mcp() command: after _populate_singleton_from(...), call
    register_search_tools(mcp_server, settings) before mcp_server.run()
  - In serve() command: no changes needed — FastAPI lifespan handles registration

src/dbs_vector/config.py
  - EngineConfig: add `family: str | None = None`
  - EngineConfig: add `resolved_family` property
  - _validate_config: add Rule 6 (legal engine name) and Rule 7 (family resolves)
```

### 7.3 Deleted code (in modified files)

- `mcp_server.py`: `search_documents()` and `search_sql_logs()` function bodies and decorators
- `main.py`: hardcoded `/search/md`, `/search/sql` routes; the four request/response model classes (moved to families)

---

## 8. Migration / Breaking Changes

### 8.1 Surface changes

| Surface | Before | After | Client action |
|---|---|---|---|
| HTTP `POST /search/md` | hardcoded | dynamic for `md` engine | none |
| HTTP `POST /search/sql` | hardcoded | dynamic for `sql` engine | none |
| HTTP `POST /search/md-granite` | 404 | works | none (new capability) |
| HTTP `POST /search/sql-granite` | 404 | works | none (new capability) |
| HTTP `POST /search/sql-api-granite` | 404 | works | none (new capability) |
| HTTP `GET /engines` | 404 | new endpoint | none (new capability) |
| MCP `search_documents` | exists | **REMOVED** | call `search_md` instead |
| MCP `search_sql_logs` | exists | **REMOVED** | call `search_sql` instead |
| MCP `search_md_granite`, ... | absent | exists | none (new capability) |

### 8.2 No compatibility shim

Per design decision: rename is a breaking change for MCP clients that hardcoded the legacy tool names. The codebase is in active pre-release development; tolerance for breaking changes is high *now*, and rises with every release that keeps the aliases. Clean cut is cheaper than a deprecation cycle.

### 8.3 Documentation updates (in scope)

- **`docs/README_MCP.md`** — full rewrite of "Tools Provided" section: list per-engine tools, naming convention (`search_<engine_name>`), A/B testing workflow, breaking-change migration note, refreshed integration examples for Claude Desktop / Claude Code / Cursor
- **`docs/README_API.md`** — replace fixed `/search/md` and `/search/sql` sections with the dynamic per-engine pattern; document new `GET /engines` endpoint with sample response
- **`docs/README_PROFILES.md`** — add "A/B testing tuning profiles" section showing how to define a variant engine + profile and compare via `GET /engines`
- **`CLAUDE.md`** — replace the paragraph that begins "The FastAPI routes (`/search/md`, `/search/sql`) and MCP tools (`search_documents`, `search_sql_logs`) are currently hardcoded to the Gemma engines…" with the new dynamic-registration description

---

## 9. Testing Strategy

### 9.1 New unit tests

- **`tests/unit/test_family_registry.py`**
  - `test_register_and_get`
  - `test_duplicate_registration_raises`
  - `test_get_unknown_raises_with_known_list`
  - `test_keys_returns_sorted`

- **`tests/unit/test_document_family.py`**
  - `test_request_response_model_shapes`
  - `test_dispatch_http_calls_service_with_request_fields`
  - `test_format_mcp_result_includes_source_and_text`
  - `test_format_mcp_result_handles_empty_results`

- **`tests/unit/test_sql_family.py`**
  - All of the document tests, plus:
  - `test_dispatch_http_passes_min_time_filter`
  - `test_dispatch_http_omits_min_time_when_unset`
  - `test_format_mcp_result_includes_execution_time_and_calls`

- **`tests/unit/test_dynamic_routes.py`**
  - `test_routes_registered_per_engine` (assert path + method + response_model)
  - `test_idempotent_registration` (call twice, assert no duplicates)
  - `test_openapi_schema_invalidated_after_registration`
  - `test_unknown_engine_returns_404`
  - `test_engine_with_uninitialized_service_returns_503`

- **`tests/unit/test_dynamic_mcp.py`**
  - `test_tools_registered_per_engine`
  - `test_tool_name_normalization` (dash → underscore)
  - `test_legacy_tool_names_absent` (asserts `search_documents`, `search_sql_logs` NOT in registered tools)
  - `test_collision_detection_raises` (engine names `md-granite` and `md_granite` both present → ValueError)
  - `test_invalid_engine_name_raises` (uppercase, leading dash, special chars)
  - `test_idempotent_registration`

- **`tests/unit/test_engines_endpoint.py`**
  - `test_engines_endpoint_returns_metadata_per_engine`
  - `test_engines_endpoint_includes_profile_knobs`
  - `test_engines_endpoint_marks_unloaded_engines`

- **`tests/unit/test_config_validation.py` (extend)**
  - `test_engine_name_pattern_validation`
  - `test_unknown_family_rejected`
  - `test_resolved_family_falls_back_to_mapper_type`

### 9.2 Modified integration tests

- **`tests/integration/test_api.py`**
  - Replace hardcoded `/search/md` and `/search/sql` test fixtures with a parametrized fixture iterating engines
  - Add tests for `/search/md-granite` (using a mocked granite service)
  - Add `GET /engines` smoke test

- **`tests/integration/test_granite_engines.py`**
  - End-to-end: ingest small fixture into a granite engine, query via `/search/<granite_engine>`, assert results

### 9.3 Test invariants

- Iteration order over `settings.engines` matches `config.yaml` load order (Python dict insertion order is preserved). Tests that compare against expected lists may rely on this.
- `_services` dict and dynamic registration sets stay in sync after lifespan startup
- `register_search_routes` and `register_search_tools` are pure with respect to `settings.engines` snapshot — calling them twice with the same settings produces identical state

---

## 10. Out of Scope (Tracked Follow-ups)

### 10.1 Lazy engine loading
Today `initialize_services()` loads every configured engine eagerly, consuming GPU memory for all variants regardless of use. The A/B testing workflow makes this worse (multiple variants pinned simultaneously). Lazy loading — building the embedder + store on first request to that engine — is a worthwhile follow-up but introduces thread-safety, cold-start UX, and cache-eviction questions that deserve their own design.

### 10.2 Compare endpoint
A `POST /search/{engine}/_compare` endpoint that takes a list of sibling engines and returns paired results in one round-trip would be convenient for eval scripts. Skipped because clients can fan out to N per-engine routes themselves, and a compare endpoint forces a Union response type that re-introduces the OCP concerns this design avoids.

### 10.3 Dynamic family registration from config
Today families are registered in code (in `families/__init__.py`). A future enhancement could let users register custom families via Python entry points or a `families:` block in config.yaml. Out of scope for this PR.

---

## 11. Decision Log

| # | Decision | Rationale |
|---|---|---|
| 1 | Family-Registry plugin pattern over polymorphic Union response | OCP — Union sites are modification points |
| 2 | Path-based per-engine routes `/search/{engine_name}` | OpenAPI shows concrete (non-Union) response models per engine; URL documents available engines |
| 3 | `GET /engines` discovery endpoint | A/B harnesses and MCP clients need structured metadata, not just OpenAPI route listing |
| 4 | Defer lazy engine loading | Independent design; eager loading is current behavior; no regression |
| 5 | No compare endpoint | Client-side fan-out preserves OCP and per-route typing |
| 6 | MCP per-engine tools `search_<engine_name>` | One tool per engine matches the per-route HTTP surface |
| 7 | Drop legacy MCP tool names without shim | Pre-release codebase; cost of cleanup rises with every release |
| 8 | Tests assert legacy names ABSENT | Negative assertion prevents quiet drift |
| 9 | Engine name allow-list `^[a-z0-9][a-z0-9_-]*$` + collision detection | Predictable URL/tool names; fail-fast on shadowing |
| 10 | Idempotent registration (skip-if-exists) | Tests, uvicorn `--reload`, and shared FastMCP instances all benefit |
| 11 | `app.openapi_schema = None` after registration | Standard FastAPI cache-invalidation gotcha |
| 12 | Optional `family: str \| None` with `mapper_type` fallback | Decouples presentation concern (family) from infrastructure concern (mapper); existing config unchanged |
| 13 | Family owns dispatch + MCP formatting + tool signature | Central modules stay closed; new families ship as one self-contained file |
| 14 | Granite model name `ibm-granite/granite-embedding-311m-multilingual-r2` | Verified against `core/model_registry.py` |
| 15 | Update `README_MCP.md`, `README_API.md`, `README_PROFILES.md`, `CLAUDE.md` in scope | Surface change is user-visible and breaking; docs must ship with the code |

---

## 12. Acceptance Criteria

A reviewer can verify the implementation by:

1. Restarting the server with the existing `config.yaml`. `POST /search/md`, `POST /search/sql`, `POST /search/md-granite`, `POST /search/sql-granite`, `POST /search/sql-api-granite` all return 200 for valid queries (engines that have ingested data).
2. `curl http://127.0.0.1:8000/engines` returns metadata for all six configured engines including profile knobs.
3. `tools/list` over MCP returns `search_md`, `search_sql`, `search_md_granite`, `search_sql_granite`, `search_sql_api_granite`, and **does not** return `search_documents` or `search_sql_logs`.
4. Adding an experimental engine to `config.yaml` and restarting produces a new route + new MCP tool with no code changes.
5. `uv run poe check` passes (ruff, mypy, pytest).
6. Negative test passes: introducing two engine names that normalize to the same MCP tool name raises a `ValueError` at startup.
