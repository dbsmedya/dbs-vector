"""Shared naming rules for engines and their MCP tools.

Single source of truth imported by config validation, dynamic tool
registration, and the discovery tool — so the regex and the tool-name
convention cannot drift across the three call sites.
"""

import re

# Engine names must be MCP-tool-safe and produce predictable URLs:
# lowercase letters, digits, dash, underscore; must start with letter or digit.
ENGINE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def normalize_tool_name(engine_name: str) -> str:
    """Convert an engine name to its MCP tool name (dashes → underscores)."""
    return f"search_{engine_name.replace('-', '_')}"
