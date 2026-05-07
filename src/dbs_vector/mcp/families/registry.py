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
