# Dynamic Engine Exposure (MCP-only, stdio-only) Design

**Date:** 2026-05-07
**Status:** Draft for review (revised — FastAPI removed; streamable-HTTP transport not shipped)
**Goal:** Replace the hardcoded `search_documents` / `search_sql_logs` MCP tools with a config-driven, per-engine MCP surface that auto-registers from `settings.engines`. Granite engines (`md-granite`, `sql-granite`, `sql-api-granite`) become reachable without code changes; future families (e.g., `jira-chunker`) plug in as self-contained modules without modifying central wiring. **FastAPI is removed entirely** and **streamable-HTTP transport is not shipped** — `dbs-vector mcp` (stdio) is the sole presentation surface.

---

## 1. Scope Decision: MCP stdio only

This revision drops FastAPI completely **and** retires the `serve` CLI subcommand. Rationale:

- The codebase's primary integration target is MCP-compatible AI assistants (Claude Desktop, Claude Code, Cursor). HTTP REST clients are not in active use.
- Maintaining two presentation layers (FastAPI HTTP + FastMCP) doubles the surface area for dynamic-registration correctness, request/response model duplication, and CORS / health / docs concerns.
- Streamable-HTTP MCP transport could be re-introduced via `mcp.streamable_http_app()` + uvicorn, but no current MCP-aware coding tool requires it; every supported integration target works with stdio. Shipping streamable-HTTP would force us to design FastMCP session-manager lifecycle handling and an app-factory for `uvicorn --reload`, neither of which is paying its way at this time.

**Removed entirely:**
- FastAPI `app` instance and the `dbs_vector.api.main` module
- `POST /search/md` and `POST /search/sql` HTTP routes
- `GET /health` HTTP route
- CORS middleware
- The `app.mount("/mcp", mcp.streamable_http_app())` mount point
- `/docs` and `/openapi.json` (FastAPI auto-generated)
- `SearchRequest`, `SqlSearchRequest`, `SearchResponse`, `SqlSearchResponse` Pydantic classes (replaced by family-owned handler signatures)
- The `dbs-vector serve` CLI subcommand
- `fastapi` and `uvicorn` Python dependencies (subject to dependency-graph verification — see §7.6)
- The legacy `src/dbs_vector/api/` package (its remaining contents move to `src/dbs_vector/mcp/`)

**Retained:**
- FastMCP server instance (moved to `src/dbs_vector/mcp/server.py`)
- `_services` dict and `initialize_services()` — transport-agnostic, moved to `src/dbs_vector/mcp/state.py`
- The `mcp` CLI subcommand (stdio transport) — sole entry point

---

## 2. Motivation

`api/mcp_server.py` currently hardcodes two tools:

- `search_documents` — bound to `_services.get("md")`
- `search_sql_logs` — bound to `_services.get("sql")`

`initialize_services()` already loads every engine in `settings.engines` (including Granite variants), but those services are unreachable. CLAUDE.md explicitly flags this as a Phase-2 follow-up.

Beyond exposing existing Granite engines, the design must accommodate:

1. **A/B testing of tuning profiles.** Adding `md-granite-experimental` (different `tuning_profile`, separate `table_name`) should produce a working MCP tool without code changes.
2. **Future families** (e.g., `jira-chunker`) without modifying central code. New family ⇒ one new module + one registration line.
3. **Discoverability** so eval scripts and MCP clients can introspect what engines are loaded and what their profile knobs are. Provided via a `list_engines` MCP tool.

---

## 3. Architectural Approach: Family-Registry Plugin Pattern

Each search "family" (document, sql, future jira) is a self-contained module that owns its argument schema, dispatch logic, and result formatting. A `FamilyRegistry` (presentation layer) maps a string key to a `SearchFamily` instance. A separate `FamilyKeyRegistry` (core layer) holds just the valid key set, used by `config.py` for validation without dragging FastMCP into config import paths.

At server startup (stdio transport, the only one shipped), `register_search_tools(mcp)` reads from the populated `dbs_vector.config.settings` singleton, iterates `settings.engines`, looks up each engine's family via `engine.resolved_family`, and registers one MCP tool per engine using a per-family handler factory. The same lifecycle hook also calls `register_discovery_tool(mcp)` to register the `list_engines` tool. Both functions take only the FastMCP instance as an argument — config flows through the already-populated singleton, matching the pattern used elsewhere in the codebase (`initialize_services()`, the CLI callback's `_populate_singleton_from(new_settings)`). Tests monkey-patch `dbs_vector.config.settings` (or the module-level `settings` import in the function under test) for isolation.

**OCP guarantees:**
- Existing family modules are never modified when a new family is added.
- Central registration (`dynamic_tools.py`) iterates engines but contains no family-specific code.
- Adding a new family = one new module + two registration lines (one in each registry).

This mirrors the existing `core/registry.py` ComponentRegistry pattern (chunkers, mappers).

### 3.1 Two-registry split (resolves Finding 5)

Two registries with distinct responsibilities and import boundaries:

**`core/families.py` — `FamilyKeyRegistry`** (lightweight, presentation-agnostic)
- Only stores valid family-name strings.
- No FastMCP, no Pydantic request models, no presentation imports.
- Imported by `config.py` for `_validate_config` Rule 7 (family resolves to a known key).
- Built-in keys registered at module top: `document`, `sql`.

**`mcp/families/registry.py` — `FamilyRegistry`** (full presentation-layer registry)
- Stores `SearchFamily` instances (with argument schemas, dispatch, formatting).
- Cross-checks against `FamilyKeyRegistry` at registration time — every concrete family's `name` MUST appear in `FamilyKeyRegistry.keys()`. If a family is registered in the presentation layer without a corresponding core key, raise.

**Import-safety acceptance:** A test asserts `python -c "import dbs_vector.config"` does NOT pull `mcp` modules into `sys.modules`.

---

## 4. Components

### 4.1 `SearchFamily` Protocol

**File:** `src/dbs_vector/mcp/families/base.py`

```python
from typing import Any, Protocol

from dbs_vector.services.search import SearchService


class SearchFamily(Protocol):
    """Self-contained MCP-layer plugin for a class of search engines.

    Each family owns:
      - A search dispatcher (translate kwargs → service call → list of results).
      - A result formatter (translate results → human-readable string).
      - A handler factory (build a per-engine async function with a concrete
        signature that FastMCP will introspect for its tool schema).

    The handler signature returned by make_handler() IS the family's
    public argument schema. There is no separate args_model — duplication
    risks drift, and FastMCP's introspection works on the handler directly.
    """

    name: str  # e.g., "document", "sql"; key in FamilyKeyRegistry

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
        """Build a per-engine async handler with an explicit, FastMCP-friendly
        signature. This is the resolution to the schema-generation question
        (Finding 2). See §4.4 for the pattern."""
        ...
```

**Why the Protocol exposes `make_handler` instead of a generic dispatch method:** FastMCP infers tool argument schemas from a function's `inspect.Signature`. A generic `def handler(*args, **kwargs)` would produce an empty MCP schema. Each family's `make_handler` returns a closure with concrete typed parameters — see §4.4 for the canonical pattern.

### 4.2 `FamilyKeyRegistry` (core)

**File:** `src/dbs_vector/core/families.py`

```python
class FamilyKeyRegistry:
    """Lightweight registry of valid family keys. No presentation-layer imports.

    Imported by config.py to validate engine.resolved_family without pulling
    FastMCP into the config import path. The full SearchFamily registry lives
    in mcp/families/registry.py and is loaded only at runtime.
    """

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


# Built-in keys. Adding a new family also requires registering its key here.
FamilyKeyRegistry.register("document")
FamilyKeyRegistry.register("sql")
```

### 4.3 `FamilyRegistry` (presentation)

**File:** `src/dbs_vector/mcp/families/registry.py`

```python
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

**Built-in registrations** live in `src/dbs_vector/mcp/families/__init__.py`:

```python
from dbs_vector.mcp.families.registry import FamilyRegistry
from dbs_vector.mcp.families.document import DocumentFamily
from dbs_vector.mcp.families.sql import SqlFamily

FamilyRegistry.register(DocumentFamily())
FamilyRegistry.register(SqlFamily())
```

**Import order:** `mcp.families.registry` imports `FamilyKeyRegistry` from `dbs_vector.core.families`, which executes its top-level `FamilyKeyRegistry.register("document")` / `register("sql")` calls during the import. By the time `mcp.families.__init__` calls `FamilyRegistry.register(DocumentFamily())`, the cross-check against `FamilyKeyRegistry` already finds the matching key.

### 4.4 Per-family handler factory pattern (resolves Finding 2)

This is the canonical answer to "how does FastMCP get a real signature?"

**`DocumentFamily.make_handler` (file: `src/dbs_vector/mcp/families/document.py`):**

```python
class DocumentFamily:
    name = "document"

    def run_search(self, service, query, limit, source_filter, **_):
        return service.execute_query(query, source_filter, limit, extra_filters={})

    def format_results(self, results, query):
        # current search_documents formatting logic
        ...

    def make_handler(self, engine_name: str):
        """Return an async function whose explicit signature FastMCP will
        introspect to build the tool schema. The engine_name is captured in
        the closure; the runtime keyword arguments are introspectable."""
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

**`SqlFamily.make_handler`** is identical in structure but its handler signature includes `min_time: float | None = None`:

```python
async def handler(
    query: str,
    limit: int = 5,
    source_filter: str | None = None,
    min_time: float | None = None,
) -> str:
    ...
```

**Why this works:** `mcp.add_tool(handler, name=..., description=...)` calls `inspect.signature(handler)` internally, which sees the concrete annotations on the closure's parameters. Each engine gets a tool whose schema reflects its family's parameters with no signature synthesis or `__signature__` mutation required.

### 4.5 Dynamic MCP tool registration

**File:** `src/dbs_vector/mcp/dynamic_tools.py`

```python
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

    Idempotency rules (resolves Finding 6):
      - If a tool with this normalized name is already registered AND its
        (engine_name, family_key) match the prior registration, skip silently.
      - If the same normalized name is registered with a DIFFERENT engine_name
        or family_key, raise a clear error. Settings are expected to be
        immutable for the lifetime of a FastMCP instance; mismatch indicates
        a stale registration that must be reset, not silently overwritten.

    Pre-flight failures (raise before any registration):
      - Engine name not matching _ENGINE_NAME_PATTERN.
      - Two distinct engines normalize to the same MCP tool name (collision).
      - Engine references a family not in FamilyRegistry.

    Pre-flight resolves and validates all engines BEFORE any tool is added,
    so a config with N engines where the last has an unknown family will not
    leave the first N-1 tools half-registered (resolves Finding 7).
    """
    if not hasattr(mcp, "_dbs_vector_registrations"):
        mcp._dbs_vector_registrations = {}  # tool_name → (engine_name, family_key)
    registrations: dict[str, tuple[str, str]] = mcp._dbs_vector_registrations

    # Pre-flight: name pattern + collision + family resolution. No mutation
    # of `mcp` until every engine passes.
    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str]] = []  # (engine_name, tool_name, family_key)
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
        # Raises KeyError with the known list if family is unknown:
        FamilyRegistry.get(family_key)
        resolved.append((engine_name, tool_name, family_key))

    # Registration phase — all engines have been validated.
    for engine_name, tool_name, family_key in resolved:
        family = FamilyRegistry.get(family_key)

        prior = registrations.get(tool_name)
        if prior is not None:
            if prior == (engine_name, family_key):
                continue  # truly idempotent — same registration
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

### 4.6 `list_engines` MCP tool (replaces `GET /engines`)

**File:** `src/dbs_vector/mcp/discovery.py`

```python
from mcp.server.fastmcp import FastMCP

# Sentinel used to track list_engines in the same registrations dict that
# register_search_tools uses, so both helpers share idempotency state.
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
    import json

    from dbs_vector.config import settings
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
    """Register the list_engines MCP tool with the same idempotency rules
    as register_search_tools (resolves Finding 6).

    Skip-if-identical: if `list_engines` is already registered with the
    discovery sentinel, do nothing. If `list_engines` is registered under a
    different sentinel (i.e., something else has stolen the name), raise.
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

**Partial-load behavior (resolves Finding 5):** `list_engines` reads from `settings.engines`, `settings.profiles`, and `ModelRegistry` directly — none of which depend on `_services`. The `loaded` flag on each entry simply reports whether `name in _services` is true at call time. **The MCP server itself remains all-or-nothing at startup**: today's `initialize_services()` raises if any engine fails to load and the server does not start, so users only reach `list_engines` once every engine has loaded successfully. The `loaded` flag is therefore meaningful in two scenarios:
1. Tests that pre-populate `_services` with a partial map.
2. Future partial-loader work that catches per-engine failures (out of scope — see §10.5).

`register_discovery_tool(mcp)` is invoked by the same lifecycle hook that calls `register_search_tools(mcp, settings)` — see §4.7.

### 4.7 Server lifecycle (stdio)

**File:** `src/dbs_vector/mcp/server.py` (moved from `src/dbs_vector/api/mcp_server.py`)

```python
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

    Takes no arguments — `dbs_vector.config.settings` is already populated
    by the CLI callback's `_populate_singleton_from(...)` call before this
    runs. `initialize_services()`, `register_search_tools(mcp)`, and
    `register_discovery_tool(mcp)` all read from the singleton too, so
    settings ownership is consistent across the lifecycle.
    """
    initialize_services()
    register_search_tools(mcp)
    register_discovery_tool(mcp)
    mcp.run()
```

The `streamable_http_path="/"` argument that previously customized the mount path is no longer needed — streamable-HTTP transport is not shipped (see §1).

---

## 5. Configuration Changes

### 5.1 New optional field on `EngineConfig`

```python
class EngineConfig(BaseModel):
    ...
    mapper_type: str
    family: str | None = None  # NEW — optional override; defaults to mapper_type

    @property
    def resolved_family(self) -> str:
        return self.family or self.mapper_type
```

**Why optional with `mapper_type` fallback:** Today `mapper_type` and family align 1:1, so existing config.yaml works unchanged. The optional `family` field decouples the *concept* (presentation surface) from the *implementation* (Arrow ↔ domain mapping), preserving an extension point for future engines whose mapper is custom but whose presentation surface is shared.

### 5.2 Validation chain additions

`_validate_config` (in `config.py`) gains two rules executed per engine:

- **Rule 6 (legal engine name):** `engine_name` matches `^[a-z0-9][a-z0-9_-]*$`. Rejects empty, uppercase, leading-dash, or special-char names.
- **Rule 7 (family resolves):** `engine.resolved_family in FamilyKeyRegistry.keys()`. Imports only `dbs_vector.core.families` — no presentation-layer imports.

Collision detection across normalized MCP tool names is performed in `register_search_tools()` (presentation-layer-specific), not in `_validate_config`.

### 5.3 No changes required to existing `config.yaml`

All six existing engines remain valid. Granite engines auto-expose with no config edits.

### 5.4 Adding an A/B variant (config-only)

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

Restart server → MCP tool `search_md_granite_experimental` auto-registers and shows up in `list_engines`.

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
| `list_engines` | NEW | — (discovery, no engine bound) |
| `search_documents` | **REMOVED** | — |
| `search_sql_logs` | **REMOVED** | — |

### 6.2 Tool naming convention

`search_<engine_name>` with dashes (`-`) replaced by underscores. The legal engine-name pattern `^[a-z0-9][a-z0-9_-]*$` guarantees the result is a valid Python identifier. Collision detection (§4.5) prevents `md-granite` and `md_granite` from silently shadowing each other.

### 6.3 Tool description

Sourced from `engine.description` in config.yaml. This is how Claude (and other MCP-aware LLMs) distinguish between A/B variants like `search_md_granite` ("baseline") vs `search_md_granite_experimental` ("smaller chunks for higher recall").

### 6.4 Tool argument schema

Sourced from each family's `make_handler(engine_name)` return value via FastMCP's built-in `inspect.signature` introspection. The handler's parameter list IS the schema:

- Document family handlers: `query: str, limit: int = 5, source_filter: str | None = None`
- SQL family handlers: `query: str, limit: int = 5, source_filter: str | None = None, min_time: float | None = None`

### 6.5 Transports

**stdio only.** FastMCP also supports a streamable-HTTP transport, but it is not shipped by this codebase. The only entry point is:

- **stdio** — `dbs-vector mcp` subcommand. Process I/O. Each invocation a fresh process. All Claude Desktop / Claude Code / Cursor integration patterns use stdio (see `docs/README_MCP.md`).

If a future use case demands streamable-HTTP (or any other) transport, it can be re-introduced as a follow-up without touching the dynamic-registration design — see §10.6.

---

## 7. File Structure

### 7.1 New files

```
src/dbs_vector/core/families.py                  # FamilyKeyRegistry (lightweight, presentation-agnostic)

src/dbs_vector/mcp/__init__.py                   # empty
src/dbs_vector/mcp/families/
  __init__.py                                    # FamilyRegistry singleton + built-in registrations
  base.py                                        # SearchFamily Protocol
  registry.py                                    # FamilyRegistry class
  document.py                                    # DocumentFamily
  sql.py                                         # SqlFamily

src/dbs_vector/mcp/dynamic_tools.py              # register_search_tools(mcp, settings)
src/dbs_vector/mcp/discovery.py                  # register_discovery_tool(mcp) — list_engines tool
```

### 7.2 Moved files (git mv)

```
src/dbs_vector/api/mcp_server.py → src/dbs_vector/mcp/server.py
  - Drop @mcp.tool() decorators for search_documents and search_sql_logs
  - Keep the FastMCP instance creation (rename module-level `mcp` if needed
    to avoid collision with the package name)
  - Add a public helper start_stdio_server() that:
      initialize_services()
      register_search_tools(mcp)
      register_discovery_tool(mcp)
      mcp.run()
    Takes no settings argument — see §3 for the singleton-ownership rationale.

src/dbs_vector/api/state.py → src/dbs_vector/mcp/state.py
  - No changes to initialize_services() — it remains transport-agnostic
  - Keep _services dict
  - All call sites updated to import from dbs_vector.mcp.state
```

### 7.3 Modified files

```
src/dbs_vector/cli.py
  - mcp() command: import start_stdio_server from dbs_vector.mcp.server
    and call it directly (no args). The current "initialize_services() +
    mcp.run()" body is replaced by the single helper call. The CLI callback
    has already populated the singleton via _populate_singleton_from before
    this command runs.
  - serve() command: REMOVED entirely (resolves Findings 1, 2 and the
    "no coding tool requires HTTP MCP" decision).

src/dbs_vector/config.py
  - EngineConfig: add `family: str | None = None`
  - EngineConfig: add `resolved_family` property
  - _validate_config: add Rule 6 (legal engine name) and Rule 7 (family in
    FamilyKeyRegistry — imports only dbs_vector.core.families)

pyproject.toml
  - Remove `fastapi` from dependencies
  - Remove `uvicorn` from dependencies (subject to verification — see §7.6)

uv.lock
  - Regenerate via `uv lock`
```

### 7.4 Deleted files

```
src/dbs_vector/api/main.py     # FastAPI app
src/dbs_vector/api/__init__.py # legacy package — empty after moves; delete
```

After §7.2 moves and §7.4 deletions, the `src/dbs_vector/api/` directory should not exist. If any leftover modules remain in the directory, they too move to a relevant package (`mcp/` or `services/`); the spec assumes a clean removal.

### 7.5 Migration of test files

```
tests/integration/test_api.py    → DELETE (covered FastAPI; replaced by tests/integration/test_mcp_server.py)
tests/unit/test_api_lifespan.py  → DELETE (FastAPI lifespan no longer exists)
```

Any test that imports `dbs_vector.api.state` or `dbs_vector.api.mcp_server` is updated to import from `dbs_vector.mcp.state` / `dbs_vector.mcp.server`.

### 7.6 Dependency audit (resolves Finding 4)

Before removing `fastapi` and `uvicorn` from `pyproject.toml`, the implementer must:

1. Run `uv pip list --tree` (or equivalent) to confirm no other dependency in the project transitively requires either package.
2. Check that no remaining test file imports `fastapi.testclient` or `uvicorn`.
3. Run `uv run poe check` after removal to confirm imports still resolve.

If a transitive dependency (e.g., a test or doc-tool) needs one of them, that dependency is documented in `pyproject.toml` with an inline comment explaining why it stays.

---

## 8. Migration / Breaking Changes

### 8.1 Surface changes

| Surface | Before | After | Client action |
|---|---|---|---|
| FastAPI `app` | exists | **REMOVED** | clients using HTTP REST must migrate to stdio MCP |
| HTTP `POST /search/md` | route | **REMOVED** | call MCP tool `search_md` (via stdio) |
| HTTP `POST /search/sql` | route | **REMOVED** | call MCP tool `search_sql` (via stdio) |
| HTTP `GET /health` | route | **REMOVED** | use MCP `tools/list` for liveness |
| HTTP `GET /docs`, `/openapi.json` | FastAPI auto | **REMOVED** | — |
| MCP mount at `/mcp` | yes | **GONE** — no HTTP transport | switch client config to stdio (see README_MCP.md) |
| `dbs-vector serve` CLI | uvicorn-of-FastAPI | **REMOVED** | use `dbs-vector mcp` (stdio) instead |
| MCP `search_documents` | exists | **REMOVED** | call `search_md` |
| MCP `search_sql_logs` | exists | **REMOVED** | call `search_sql` |
| MCP `search_md_granite`, etc. | absent | NEW | new capability |
| MCP `list_engines` | absent | NEW | new capability |
| `dbs_vector.api` package | exists | **REMOVED** | imports update to `dbs_vector.mcp.*` |

### 8.2 No compatibility shim

Per locked design decision: rename + FastAPI removal is a clean breaking change. The codebase is in active pre-release development; cost of cleanup rises with every release that keeps legacy aliases.

### 8.3 Documentation updates (in scope)

- **`docs/README_MCP.md`** — full rewrite. Remove the "Method 2: Streamable HTTP" section entirely. Remove all integration examples that reference `http://127.0.0.1:8000/mcp`. The "Tools Provided" section is rewritten with the new per-engine naming convention, A/B testing workflow, breaking-change migration note, and refreshed integration examples for Claude Desktop / Claude Code / Cursor — all stdio
- **`docs/README_API.md`** — DELETED. The file announces FastAPI removal and is no longer relevant; pointers to README_MCP.md replace any outbound links from CLAUDE.md / README.md
- **`docs/README_PROFILES.md`** — add "A/B testing tuning profiles" section showing how to define a variant engine + profile and inspect via `list_engines`
- **`CLAUDE.md`** — replace the paragraph that begins "The FastAPI routes (`/search/md`, `/search/sql`) and MCP tools (`search_documents`, `search_sql_logs`) are currently hardcoded to the Gemma engines…" with the new MCP-stdio-only dynamic-registration description; remove the `serve` example from the Commands section; update Architecture section to drop the `api/` package reference and add `mcp/`

---

## 9. Testing Strategy

### 9.1 New unit tests

- **`tests/unit/test_family_key_registry.py`**
  - `test_register_and_is_valid`
  - `test_keys_returns_sorted`
  - `test_reset_for_testing_clears_keys`

- **`tests/unit/test_family_registry.py`**
  - `test_register_with_known_key_succeeds`
  - `test_register_with_unknown_key_raises_runtime_error` (cross-check against FamilyKeyRegistry)
  - `test_duplicate_registration_raises`
  - `test_get_unknown_raises_with_known_list`
  - `test_reset_for_testing_clears_families`

- **`tests/unit/test_document_family.py`**
  - `test_run_search_calls_service_with_kwargs`
  - `test_format_results_includes_source_and_text`
  - `test_format_results_handles_empty_results`
  - `test_make_handler_signature` (uses `inspect.signature` to assert parameter names + defaults match the spec)

- **`tests/unit/test_sql_family.py`**
  - All of the document tests, plus:
  - `test_run_search_passes_min_time_filter`
  - `test_run_search_omits_min_time_when_unset`
  - `test_format_results_includes_execution_time_and_calls`
  - `test_make_handler_signature_includes_min_time`

- **`tests/unit/test_dynamic_tools.py`**
  - `test_tools_registered_per_engine`
  - `test_tool_name_normalization` (dash → underscore)
  - `test_legacy_tool_names_absent` — asserts `search_documents` and `search_sql_logs` are NOT in registered tools (resolves Finding 8.b on negative assertions)
  - `test_collision_detection_raises` — `md-granite` and `md_granite` both present → ValueError
  - `test_invalid_engine_name_raises` — uppercase, leading dash, special chars
  - `test_idempotent_registration_with_identical_settings`
  - `test_stale_registration_with_different_family_raises` (resolves Finding 6)

- **`tests/unit/test_list_engines_tool.py`**
  - `test_list_engines_returns_metadata_per_engine`
  - `test_list_engines_includes_profile_knobs`
  - `test_list_engines_marks_unloaded_engines`
  - `test_list_engines_works_when_services_partial` (engine missing from `_services` → `loaded: false`)

- **`tests/unit/test_config_validation.py` (extend)**
  - `test_engine_name_pattern_validation`
  - `test_unknown_family_rejected`
  - `test_resolved_family_falls_back_to_mapper_type`

- **`tests/unit/test_config_import_safety.py` (extend)**
  - `test_importing_config_does_not_load_mcp_modules` — asserts `dbs_vector.mcp` not in `sys.modules` after `import dbs_vector.config` (resolves Finding 5)

### 9.2 New integration tests (MCP-only)

- **`tests/integration/test_mcp_server.py`**
  - Start FastMCP in-process with mocked services
  - Issue `tools/list` JSON-RPC request, assert expected tool names present and legacy names absent
  - Call `search_md` with a mocked service, assert formatted output
  - Call `search_md_granite` with a mocked service, assert formatted output
  - Call `list_engines`, assert structured metadata

- **`tests/integration/test_granite_engines.py`**
  - **Gated behind `DBS_RUN_E2E_GRANITE=1`** (resolves Finding 8). Tests skipped by default.
  - End-to-end: ingest small fixture into a granite engine, query via MCP `search_md_granite`, assert results

### 9.3 Deleted test files

- `tests/integration/test_api.py` — replaced by `test_mcp_server.py`
- `tests/unit/test_api_lifespan.py` — FastAPI lifespan removed

### 9.4 Test invariants and fixtures

- Shared fixture `clean_family_registries()` calls `FamilyRegistry._reset_for_testing()` and `FamilyKeyRegistry._reset_for_testing()` between tests, then re-runs the built-in registrations
- Each MCP server test creates a fresh `FastMCP` instance — never shared across tests
- Iteration order over `settings.engines` matches `config.yaml` load order (Python dict insertion order is preserved)

---

## 10. Out of Scope (Tracked Follow-ups)

### 10.1 Lazy engine loading
Today `initialize_services()` loads every configured engine eagerly. The A/B testing workflow makes this worse (multiple variants pinned simultaneously). Lazy loading is a worthwhile follow-up but introduces thread-safety, cold-start UX, and cache-eviction questions that deserve their own design.

### 10.2 Multi-engine "compare" tool
A `compare_engines` MCP tool that takes a list of sibling engines and returns paired results would be convenient for eval scripts. Skipped because clients can call N per-engine tools themselves.

### 10.3 Dynamic family registration from config
Today families are registered in code (in `families/__init__.py`). A future enhancement could let users register custom families via Python entry points or a `families:` block in config.yaml. Out of scope for this PR.

### 10.4 Streamable-HTTP authentication
N/A — streamable-HTTP transport is not shipped (see §10.6). If it is re-introduced, bearer-token auth or origin checks become a corresponding follow-up.

### 10.5 Partial engine loading
Today `initialize_services()` is all-or-nothing. A future loader could catch per-engine failures, log them, and start the server with the surviving engines. `list_engines` is already structured to report `loaded: false` for engines whose service object is missing, which is the consumer-facing piece of that design. Out of scope for this PR — the loader changes deserve their own design (failure-mode policy, partial-init UX, A/B impact).

### 10.6 Streamable-HTTP MCP transport
Re-introducing an HTTP MCP server (via uvicorn over `mcp.streamable_http_app()`) is straightforward — call `register_search_tools` and `register_discovery_tool` on the same global `mcp` instance and wrap the ASGI app behind an app-factory for `--reload`. Out of scope for this PR; no current MCP-aware coding tool requires it.

---

## 11. Decision Log

| # | Decision | Rationale |
|---|---|---|
| 1 | Family-Registry plugin pattern | OCP — central registration code never touched per family |
| 2 | **Drop FastAPI entirely; MCP is sole presentation layer** | Lower surface area; primary integration target is MCP-aware AI assistants |
| 3 | **Drop streamable-HTTP transport too; `serve` subcommand removed** | No coding tool requires HTTP MCP today; complexity (lifespan, --reload, auth) deferred until needed |
| 4 | `list_engines` MCP tool replaces `GET /engines` | Same use case (discovery), MCP-native delivery |
| 5 | Per-engine MCP tools `search_<engine_name>`, no aliases | Clean cut; pre-release codebase |
| 6 | Tests assert legacy tool names ABSENT | Negative assertion prevents quiet drift |
| 7 | Engine name allow-list `^[a-z0-9][a-z0-9_-]*$` + collision detection | Predictable tool names; fail-fast on shadowing |
| 8 | Idempotent registration: skip-if-identical, raise on mismatch | Prevents stale-registration silent overwrite |
| 9 | Pre-flight resolves all engines (incl. family lookup) before any tool registration | Resolves Finding 7; never leave half-registered state |
| 10 | Discovery tool shares the `_dbs_vector_registrations` dict with search-tool registration | Resolves Finding 6; uniform idempotency policy |
| 11 | Two-registry split: `core/families.py` (keys) + `mcp/families/registry.py` (impls) | Resolves Finding 5; preserves config.py import safety |
| 12 | Per-family `make_handler` factory with explicit signatures | FastMCP introspects naturally |
| 13 | Family contract: `run_search` + `format_results` + `make_handler` | Methods testable independently |
| 14 | `_reset_for_testing` on both registries | Test isolation |
| 15 | E2E Granite tests gated behind `DBS_RUN_E2E_GRANITE=1` | CI runs fast by default |
| 16 | Optional `family: str \| None` with `mapper_type` fallback | Decouples presentation from infrastructure mapping |
| 17 | Granite model name `ibm-granite/granite-embedding-311m-multilingual-r2` | Verified against `core/model_registry.py` |
| 18 | Move `api/mcp_server.py → mcp/server.py` and `api/state.py → mcp/state.py`; delete `api/` package | Resolves Finding 3; package name reflects what it does |
| 19 | Drop `fastapi` and `uvicorn` from pyproject.toml (subject to dependency audit) | Resolves Finding 4; honest about what's required |
| 20 | `list_engines.loaded` flag is honest about partial maps but startup is still all-or-nothing | Resolves Finding 5 wording |
| 21 | Update `README_MCP.md`, `README_PROFILES.md`, `CLAUDE.md` in scope; delete `README_API.md` | Surface change is breaking; docs ship with code |

---

## 12. Acceptance Criteria

A reviewer can verify the implementation by:

1. **Unit tests pass** — `uv run pytest tests/unit/test_dynamic_tools.py tests/unit/test_document_family.py tests/unit/test_sql_family.py tests/unit/test_family_key_registry.py tests/unit/test_family_registry.py tests/unit/test_list_engines_tool.py tests/unit/test_config_validation.py tests/unit/test_config_import_safety.py -v` all green.
2. **Import safety** — `uv run python -c "import dbs_vector.config; import sys; assert 'dbs_vector.mcp' not in sys.modules"` succeeds.
3. **MCP tool listing** — Starting the server (`dbs-vector mcp`, stdio) and issuing a `tools/list` request returns:
   - Present: `search_md`, `search_sql`, `search_md_granite`, `search_sql_granite`, `search_sql_api_granite`, `list_engines`
   - Absent: `search_documents`, `search_sql_logs`
4. **Discovery tool** — Calling `list_engines` returns JSON metadata for all six configured engines with profile knobs; `loaded: true` for every engine when startup succeeds.
5. **Config-only A/B variant** — Adding `md-granite-experimental` to `config.yaml` and restarting produces a new MCP tool `search_md_granite_experimental` with no source-code changes.
6. **Static checks** — `uv run poe check` passes (ruff, mypy, pytest).
7. **Negative MCP test (collision)** — Two engine names that normalize to the same MCP tool name raise a `ValueError` at startup with both names in the message.
8. **Negative MCP test (pre-flight atomicity)** — A config where the *last* engine has an unknown family raises before any tool is registered; `mcp._dbs_vector_registrations` is empty after the exception.
9. **`api/` package gone** — `uv run python -c "import dbs_vector.api"` raises `ModuleNotFoundError`. `uv run python -c "import dbs_vector.api.main"` raises `ModuleNotFoundError`.
10. **`serve` subcommand gone** — `uv run dbs-vector serve` exits with a Typer "no such command" error.
11. **Direct dependencies cleaned** — `fastapi` and `uvicorn` are absent from `pyproject.toml`'s `[project] dependencies` (and any extras / optional groups). No source file under `src/dbs_vector/` or `tests/` imports `fastapi` or `uvicorn`. (The packages MAY still be importable if a transitive dependency keeps them installed — that is acceptable per §7.6 and not asserted here.)
12. **(Gated, opt-in) End-to-end Granite** — `DBS_RUN_E2E_GRANITE=1 uv run pytest tests/integration/test_granite_engines.py` passes when local Granite indices and model are available.
