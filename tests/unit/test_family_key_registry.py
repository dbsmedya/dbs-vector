"""Tests for FamilyKeyRegistry — the lightweight, presentation-agnostic registry."""

import pytest

from dbs_vector.core.families import FamilyKeyRegistry


@pytest.fixture(autouse=True)
def _restore_keys():
    """Snapshot/restore the registry around each test so reset_for_testing
    in one test cannot leak into another."""
    snapshot = set(FamilyKeyRegistry.keys())
    yield
    FamilyKeyRegistry._reset_for_testing()
    for key in snapshot:
        FamilyKeyRegistry.register(key)


def test_builtin_keys_are_registered():
    assert FamilyKeyRegistry.is_valid("document")
    assert FamilyKeyRegistry.is_valid("sql")


def test_register_adds_new_key():
    FamilyKeyRegistry.register("jira")
    assert FamilyKeyRegistry.is_valid("jira")


def test_is_valid_returns_false_for_unknown_key():
    assert FamilyKeyRegistry.is_valid("nonexistent") is False


def test_keys_returns_sorted_list():
    keys = FamilyKeyRegistry.keys()
    assert keys == sorted(keys)
    assert "document" in keys
    assert "sql" in keys


def test_reset_for_testing_clears_keys():
    FamilyKeyRegistry._reset_for_testing()
    assert FamilyKeyRegistry.keys() == []
    assert FamilyKeyRegistry.is_valid("document") is False
