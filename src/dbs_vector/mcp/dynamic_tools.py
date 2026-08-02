"""Dynamic MCP tool registration from settings.engines.

Reads the populated dbs_vector.config.settings singleton. Tests monkey-patch
this module's `settings` import for isolation.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from dbs_vector.config import settings
from dbs_vector.core.naming import ENGINE_NAME_PATTERN, normalize_tool_name
from dbs_vector.mcp.families.base import BrowseFamily, ReadFamily, TriageFamily
from dbs_vector.mcp.families.registry import FamilyRegistry


def register_search_tools(mcp: FastMCP, allow_raw_queries: bool = False) -> None:
    """Iterate settings.engines and register one MCP tool per engine.

    `allow_raw_queries` is the server-level egress flag (from
    --allow-raw-queries); it is threaded into each family's make_handler and
    recorded in the registration tuple so a second call with a different flag
    raises instead of silently keeping a stale handler. Mirrors
    register_browse_tools. Defaults to False (fail-closed).

    Reads from the module-level `settings` singleton (already populated by
    the CLI callback via _populate_singleton_from). Tests monkey-patch
    `dbs_vector.mcp.dynamic_tools.settings` for isolation.

    Idempotency rules:
      - Skip if the same (engine_name, family_key, allow_raw_queries) is
        already registered.
      - Raise if the same tool name is registered with DIFFERENT settings —
        settings are expected to be immutable for the lifetime of a FastMCP
        instance.

    Pre-flight failures (raise before any tool is registered):
      - Engine name not matching ENGINE_NAME_PATTERN.
      - Two distinct engines normalize to the same MCP tool name.
      - Engine references a family not in FamilyRegistry.

    Pre-flight resolves and validates all engines BEFORE any tool is added,
    so a config with N engines where the last has an unknown family will
    not leave the first N-1 tools half-registered.
    """
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}  # tool_name → (engine_name, family_key, allow_raw_queries)
    registrations: dict[str, tuple[str, str, bool]] = mcp_any._dbs_vector_registrations

    # Pre-flight: name pattern + collision + family resolution.
    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str]] = []
    for engine_name, engine in settings.engines.items():
        if not ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {ENGINE_NAME_PATTERN.pattern}. "
                f"Allowed: lowercase, digits, dash, underscore (must start with letter or digit)."
            )
        tool_name = normalize_tool_name(engine_name)
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
        current = (engine_name, family_key, allow_raw_queries)
        prior = registrations.get(tool_name)
        if prior is not None:
            if prior == current:
                continue  # idempotent — same registration
            raise RuntimeError(
                f"Stale tool registration for '{tool_name}': previously "
                f"{prior}, now {current}. Reset the FastMCP instance instead "
                f"of re-registering with different settings."
            )

        engine = settings.engines[engine_name]
        handler = family.make_handler(engine_name, allow_raw_queries)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=family.search_description(engine_name, engine),
        )
        registrations[tool_name] = current


def register_read_tools(mcp: FastMCP) -> None:
    """Register one exact adjacent-chunk reader per read-capable engine."""
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}
    registrations: dict[str, tuple] = mcp_any._dbs_vector_registrations

    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str, Any]] = []
    for engine_name, engine in settings.engines.items():
        if not ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {ENGINE_NAME_PATTERN.pattern}."
            )
        family_key = engine.resolved_family
        family = FamilyRegistry.get(family_key)
        if not isinstance(family, ReadFamily):
            continue
        tool_name = normalize_tool_name(engine_name, verb="read")
        if tool_name in seen:
            raise ValueError(
                f"MCP tool name collision: '{seen[tool_name]}' and '{engine_name}' "
                f"both normalize to '{tool_name}'."
            )
        seen[tool_name] = engine_name
        resolved.append((engine_name, tool_name, family_key, engine))

    for engine_name, tool_name, family_key, engine in resolved:
        family = FamilyRegistry.get(family_key)
        if not isinstance(family, ReadFamily):  # defensive against registry mutation
            raise RuntimeError(
                f"Family '{family_key}' for engine '{engine_name}' no longer supports read."
            )
        current = (engine_name, family_key, "read")
        prior = registrations.get(tool_name)
        if prior is not None:
            if prior == current:
                continue
            raise RuntimeError(
                f"Stale read tool registration for '{tool_name}': previously "
                f"{prior}, now {current}. Reset the FastMCP instance instead "
                f"of re-registering with different settings."
            )
        mcp.add_tool(
            family.make_read_handler(engine_name),
            name=tool_name,
            description=family.read_description(engine_name, engine),
        )
        registrations[tool_name] = current


def register_browse_tools(mcp: FastMCP, allow_raw_queries: bool) -> None:
    """Register one browse_<engine> tool per SQL-family engine.

    Mirrors register_search_tools' pre-flight (name pattern, collision, family
    resolution, idempotency) but registers ONLY engines whose
    resolved_family == "sql", uses verb="browse" tool names, and sources the
    description from family.browse_description(engine, allow_raw_queries).
    """
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}
    registrations: dict[str, tuple] = mcp_any._dbs_vector_registrations

    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str, Any]] = []
    for engine_name, engine in settings.engines.items():
        if engine.resolved_family != "sql":
            continue
        if not ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {ENGINE_NAME_PATTERN.pattern}."
            )
        tool_name = normalize_tool_name(engine_name, verb="browse")
        if tool_name in seen:
            raise ValueError(
                f"MCP tool name collision: '{seen[tool_name]}' and '{engine_name}' "
                f"both normalize to '{tool_name}'."
            )
        seen[tool_name] = engine_name
        family_key = engine.resolved_family
        FamilyRegistry.get(family_key)
        resolved.append((engine_name, tool_name, family_key, engine))

    for engine_name, tool_name, family_key, engine in resolved:
        family = FamilyRegistry.get(family_key)
        if not isinstance(family, BrowseFamily):
            raise RuntimeError(
                f"Family '{family_key}' for engine '{engine_name}' does not "
                f"support browse (missing make_browse_handler/browse_description)."
            )
        prior = registrations.get(tool_name)
        current = (engine_name, family_key, allow_raw_queries)
        if prior is not None:
            if prior == current:
                continue
            raise RuntimeError(
                f"Stale browse tool registration for '{tool_name}': previously "
                f"{prior}, now {current}. Reset the FastMCP instance instead of "
                f"re-registering with different settings."
            )
        handler = family.make_browse_handler(engine_name, allow_raw_queries)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=family.browse_description(engine_name, engine, allow_raw_queries),
        )
        registrations[tool_name] = current


def register_triage_tools(mcp: FastMCP, allow_raw_queries: bool) -> None:
    """Register one top_impacting_<engine> tool per SQL-family engine.

    Mirrors register_browse_tools' pre-flight (name pattern, collision, family
    resolution, idempotency) but registers ONLY engines whose
    resolved_family == "sql", uses verb="top_impacting" tool names, and sources
    the description from family.triage_description(engine, allow_raw_queries).
    """
    mcp_any: Any = mcp
    if not hasattr(mcp_any, "_dbs_vector_registrations"):
        mcp_any._dbs_vector_registrations = {}
    registrations: dict[str, tuple] = mcp_any._dbs_vector_registrations

    seen: dict[str, str] = {}
    resolved: list[tuple[str, str, str, Any]] = []
    for engine_name, engine in settings.engines.items():
        if engine.resolved_family != "sql":
            continue
        if not ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match {ENGINE_NAME_PATTERN.pattern}."
            )
        tool_name = normalize_tool_name(engine_name, verb="top_impacting")
        if tool_name in seen:
            raise ValueError(
                f"MCP tool name collision: '{seen[tool_name]}' and '{engine_name}' "
                f"both normalize to '{tool_name}'."
            )
        seen[tool_name] = engine_name
        family_key = engine.resolved_family
        FamilyRegistry.get(family_key)
        resolved.append((engine_name, tool_name, family_key, engine))

    for engine_name, tool_name, family_key, engine in resolved:
        family = FamilyRegistry.get(family_key)
        if not isinstance(family, TriageFamily):
            raise RuntimeError(
                f"Family '{family_key}' for engine '{engine_name}' does not "
                f"support triage (missing make_triage_handler/triage_description)."
            )
        prior = registrations.get(tool_name)
        current = (engine_name, family_key, allow_raw_queries)
        if prior is not None:
            if prior == current:
                continue
            raise RuntimeError(
                f"Stale triage tool registration for '{tool_name}': previously "
                f"{prior}, now {current}. Reset the FastMCP instance instead of "
                f"re-registering with different settings."
            )
        handler = family.make_triage_handler(engine_name, allow_raw_queries)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=family.triage_description(engine_name, engine, allow_raw_queries),
        )
        registrations[tool_name] = current
