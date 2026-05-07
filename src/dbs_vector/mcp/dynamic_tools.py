"""Dynamic MCP tool registration: one tool per engine in settings.engines.

Reads the populated dbs_vector.config.settings singleton. Tests monkey-patch
this module's `settings` import for isolation.
"""

import re
from typing import Any

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
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}  # tool_name → (engine_name, family_key)
    registrations: dict[str, tuple[str, str]] = mcp_any._dbs_vector_registrations

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
