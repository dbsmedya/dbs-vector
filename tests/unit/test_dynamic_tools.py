"""Tests for register_search_tools — pre-flight validation, idempotency,
collision detection, and naming convention."""

import pytest
from mcp.server.fastmcp import FastMCP

import dbs_vector.config as config_mod
import dbs_vector.mcp.dynamic_tools as dyn
from dbs_vector.config import EngineConfig, Settings, TuningProfile


def _make_settings(engines: dict[str, EngineConfig]) -> Settings:
    s = Settings()
    s.engines = engines
    s.profiles = {"p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)}
    return s


def _make_engine(mapper: str = "document", desc: str = "test") -> EngineConfig:
    return EngineConfig(
        description=desc,
        model="gemma-bf16",
        mapper_type=mapper,
        chunker_type=mapper,
        table_name="t",
        workflow="w",
        tuning_profile="p",
    )


@pytest.fixture
def fresh_mcp() -> FastMCP:
    return FastMCP("test-dbs-vector")


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    """Each test gets a fresh Settings stub patched into both the
    config singleton and the dynamic_tools module."""
    s = Settings()
    monkeypatch.setattr(config_mod, "settings", s)
    monkeypatch.setattr(dyn, "settings", s)
    yield s


def test_register_search_tools_registers_one_tool_per_engine(fresh_mcp, _clean_settings):
    _clean_settings.engines = {
        "md": _make_engine("document", "Markdown engine"),
        "sql": _make_engine("sql", "SQL engine"),
    }
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)

    tool_names = {t.name for t in fresh_mcp._tool_manager.list_tools()}
    assert "search_md" in tool_names
    assert "search_sql" in tool_names


def test_register_search_tools_legacy_names_absent(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"md": _make_engine("document"), "sql": _make_engine("sql")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)

    tool_names = {t.name for t in fresh_mcp._tool_manager.list_tools()}
    assert "search_documents" not in tool_names
    assert "search_sql_logs" not in tool_names


def test_register_search_tools_uses_engine_description(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"md": _make_engine("document", "Markdown & Prose")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)

    tool = next(t for t in fresh_mcp._tool_manager.list_tools() if t.name == "search_md")
    assert tool.description == "Markdown & Prose"


def test_invalid_engine_name_raises(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"Bad-Name": _make_engine("document")}
    with pytest.raises(ValueError, match="Engine name 'Bad-Name'"):
        dyn.register_search_tools(fresh_mcp)


def test_collision_detection_raises(fresh_mcp, _clean_settings):
    _clean_settings.engines = {
        "md-granite": _make_engine("document"),
        "md_granite": _make_engine("document"),
    }
    with pytest.raises(ValueError, match="MCP tool name collision"):
        dyn.register_search_tools(fresh_mcp)


def test_unknown_family_raises(fresh_mcp, _clean_settings):
    bad_engine = _make_engine("document")
    bad_engine.family = "ghost"  # bypasses config-time validation
    _clean_settings.engines = {"x": bad_engine}
    with pytest.raises(KeyError, match="Unknown search family 'ghost'"):
        dyn.register_search_tools(fresh_mcp)


def test_idempotent_registration_with_identical_settings(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"md": _make_engine("document")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    dyn.register_search_tools(fresh_mcp)
    count_after_first = len(fresh_mcp._tool_manager.list_tools())
    dyn.register_search_tools(fresh_mcp)
    count_after_second = len(fresh_mcp._tool_manager.list_tools())

    assert count_after_first == count_after_second


def test_stale_registration_with_different_family_raises(fresh_mcp, _clean_settings):
    _clean_settings.engines = {"x": _make_engine("document")}
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }
    dyn.register_search_tools(fresh_mcp)

    # Now mutate to a different family for the same engine name
    _clean_settings.engines["x"] = _make_engine("sql")
    with pytest.raises(RuntimeError, match="Stale tool registration"):
        dyn.register_search_tools(fresh_mcp)


def test_pre_flight_atomicity_no_partial_registration(fresh_mcp, _clean_settings):
    """If the LAST engine has a bad family, NONE of the earlier engines' tools
    should be registered."""
    bad_engine = _make_engine("document")
    bad_engine.family = "ghost"
    _clean_settings.engines = {
        "md": _make_engine("document"),
        "sql": _make_engine("sql"),
        "broken": bad_engine,
    }
    _clean_settings.profiles = {
        "p": TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=1)
    }

    with pytest.raises(KeyError):
        dyn.register_search_tools(fresh_mcp)

    tools = fresh_mcp._tool_manager.list_tools()
    assert tools == []
    assert getattr(fresh_mcp, "_dbs_vector_registrations", {}) == {}
