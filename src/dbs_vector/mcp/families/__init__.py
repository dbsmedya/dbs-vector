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
