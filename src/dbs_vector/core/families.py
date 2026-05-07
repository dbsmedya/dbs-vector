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
