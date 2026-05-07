"""Tests for FamilyRegistry — the full presentation-layer registry."""

from typing import Any

import pytest

from dbs_vector.core.families import FamilyKeyRegistry
from dbs_vector.mcp.families.registry import FamilyRegistry


class _StubFamily:
    """Test double matching the SearchFamily Protocol — only `name` is used
    in registry tests; the Protocol's other methods aren't called here."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def format_results(self, results: list[Any], query: str) -> str:
        return ""

    def make_handler(self, engine_name: str) -> Any:
        async def handler() -> str:
            return ""
        return handler


@pytest.fixture(autouse=True)
def _restore_registries():
    key_snapshot = set(FamilyKeyRegistry.keys())
    fam_snapshot = dict(FamilyRegistry._families)
    yield
    FamilyRegistry._reset_for_testing()
    FamilyKeyRegistry._reset_for_testing()
    for k in key_snapshot:
        FamilyKeyRegistry.register(k)
    for fam in fam_snapshot.values():
        FamilyRegistry.register(fam)


def test_register_with_known_key_succeeds():
    FamilyKeyRegistry.register("document")
    fam = _StubFamily("document")
    FamilyRegistry.register(fam)
    assert FamilyRegistry.get("document") is fam


def test_register_with_unknown_key_raises_runtime_error():
    FamilyKeyRegistry._reset_for_testing()  # no keys registered
    fam = _StubFamily("ghost")
    with pytest.raises(RuntimeError, match="no matching key in FamilyKeyRegistry"):
        FamilyRegistry.register(fam)


def test_duplicate_registration_raises():
    FamilyKeyRegistry.register("document")
    FamilyRegistry.register(_StubFamily("document"))
    with pytest.raises(ValueError, match="already registered"):
        FamilyRegistry.register(_StubFamily("document"))


def test_get_unknown_raises_with_known_list():
    FamilyKeyRegistry.register("document")
    FamilyRegistry.register(_StubFamily("document"))
    with pytest.raises(KeyError, match=r"Unknown.*Known: \['document'\]"):
        FamilyRegistry.get("nonexistent")


def test_keys_returns_sorted():
    FamilyKeyRegistry.register("document")
    FamilyKeyRegistry.register("sql")
    FamilyRegistry.register(_StubFamily("sql"))
    FamilyRegistry.register(_StubFamily("document"))
    assert FamilyRegistry.keys() == ["document", "sql"]


def test_reset_for_testing_clears_families():
    FamilyKeyRegistry.register("document")
    FamilyRegistry.register(_StubFamily("document"))
    FamilyRegistry._reset_for_testing()
    assert FamilyRegistry.keys() == []
