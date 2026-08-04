"""list_engines MCP tool and its registration helper.

list_engines reads from settings + ModelRegistry directly; it does not
depend on the runtime _services dict beyond reporting which entries are
loaded. The MCP server itself is all-or-nothing at startup, so when
list_engines is reachable, every engine that initialized successfully
will report loaded: true. The flag exists for tests with a partial
_services map and for any future partial-loader work (out of scope).
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from dbs_vector.config import settings
from dbs_vector.core.naming import normalize_tool_name


def _build_engine_listing(engine_names: set[str] | None) -> str:
    """List configured search engines and their tuning profiles.

    Returns a JSON-encoded list of engine metadata: name, family, model,
    description, table name, profile knobs (max_token_length,
    chunk_max_chars, batch_size), search/read MCP tool names, and whether a
    runtime service object is currently registered for that engine. Useful
    for A/B testing harnesses and clients enumerating variants programmatically.

    `engine_names`: None (default) lists every engine (stdio behavior). A
    subset restricts the listing to a token-scoped HTTP sub-app's engines —
    the same structural scoping as the search/read/browse/triage registrars.
    """
    from dbs_vector.core.model_registry import ModelRegistry
    from dbs_vector.mcp.state import _services

    out = []
    for name, engine in settings.engines.items():
        if engine_names is not None and name not in engine_names:
            continue
        contract = ModelRegistry.get(engine.model)
        profile = settings.profiles[engine.tuning_profile]
        out.append(
            {
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
                "mcp_tool": normalize_tool_name(name),
                "read_tool": (
                    normalize_tool_name(name, verb="read")
                    if engine.resolved_family == "document"
                    else None
                ),
                "loaded": name in _services,
            }
        )
    return json.dumps(out, indent=2)


def register_discovery_tool(mcp: FastMCP, *, engine_names: set[str] | None = None) -> None:
    """Register the list_engines MCP tool.

    `engine_names`: None (default) lists every engine — the stdio behavior.
    A subset scopes the listing to a token group's own engines, matching the
    search/read/browse/triage registrars' structural scoping.

    Skip-if-identical when our discovery sentinel is already in
    mcp._dbs_vector_registrations. Raise on a non-discovery occupation
    of the `list_engines` slot.
    """
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}
    registrations: dict = mcp_any._dbs_vector_registrations

    # Scope-aware sentinel: re-registering the SAME scope is idempotent;
    # a DIFFERENT scope on the same instance raises (mirrors the
    # register_search_tools stale-registration rule).
    sentinel = ("__discovery__", frozenset(engine_names) if engine_names is not None else None)
    prior = registrations.get("list_engines")
    if prior == sentinel:
        return
    if prior is not None:
        raise RuntimeError(
            f"Tool 'list_engines' already registered with sentinel {prior!r}. "
            f"Reset the FastMCP instance instead of re-registering."
        )

    async def _list_engines() -> str:
        return _build_engine_listing(engine_names)

    mcp.add_tool(
        _list_engines,
        name="list_engines",
        description="List configured search engines and their tuning profiles.",
    )
    registrations["list_engines"] = sentinel
