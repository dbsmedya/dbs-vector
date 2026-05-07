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
                "mcp_tool": f"search_{name.replace('-', '_')}",
                "loaded": name in _services,
            }
        )
    return json.dumps(out, indent=2)


def register_discovery_tool(mcp: FastMCP) -> None:
    """Register the list_engines MCP tool.

    Skip-if-identical when our discovery sentinel is already in
    mcp._dbs_vector_registrations. Raise on a non-discovery occupation
    of the `list_engines` slot.
    """
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}
    registrations: dict = mcp_any._dbs_vector_registrations

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
