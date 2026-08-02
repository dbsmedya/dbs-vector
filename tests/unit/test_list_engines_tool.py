"""Tests for list_engines MCP tool and register_discovery_tool."""

import json

import pytest
from mcp.server.fastmcp import FastMCP

import dbs_vector.config as config_mod
import dbs_vector.mcp.discovery as discovery_mod
import dbs_vector.mcp.state as state_mod
from dbs_vector.config import EngineConfig, Settings, TuningProfile


def _make_engine(mapper: str = "document", desc: str = "test") -> EngineConfig:
    return EngineConfig(
        description=desc,
        model="gemma-bf16",
        mapper_type=mapper,
        chunker_type=mapper,
        table_name=f"{mapper}_table",
        workflow="w",
        tuning_profile="p",
    )


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    s = Settings()
    s.engines = {"md": _make_engine("document"), "sql": _make_engine("sql")}
    s.profiles = {"p": TuningProfile(max_token_length=2048, chunk_max_chars=1000, batch_size=64)}
    monkeypatch.setattr(config_mod, "settings", s)
    monkeypatch.setattr(discovery_mod, "settings", s)
    monkeypatch.setattr(state_mod, "_services", {})
    yield s


@pytest.fixture
def fresh_mcp() -> FastMCP:
    return FastMCP("test-dbs-vector")


@pytest.mark.asyncio
async def test_list_engines_returns_metadata_per_engine():
    out_str = await discovery_mod._list_engines()
    out = json.loads(out_str)
    names = {e["name"] for e in out}
    assert names == {"md", "sql"}


@pytest.mark.asyncio
async def test_list_engines_includes_profile_knobs():
    out = json.loads(await discovery_mod._list_engines())
    md_entry = next(e for e in out if e["name"] == "md")
    assert md_entry["profile"]["max_token_length"] == 2048
    assert md_entry["profile"]["chunk_max_chars"] == 1000
    assert md_entry["profile"]["batch_size"] == 64
    assert md_entry["profile"]["name"] == "p"


@pytest.mark.asyncio
async def test_list_engines_marks_unloaded_engines(_clean_state):
    out = json.loads(await discovery_mod._list_engines())
    for entry in out:
        assert entry["loaded"] is False


@pytest.mark.asyncio
async def test_list_engines_marks_loaded_engines(_clean_state, monkeypatch):
    monkeypatch.setattr(state_mod, "_services", {"md": object()})
    out = json.loads(await discovery_mod._list_engines())
    md = next(e for e in out if e["name"] == "md")
    sql = next(e for e in out if e["name"] == "sql")
    assert md["loaded"] is True
    assert sql["loaded"] is False


@pytest.mark.asyncio
async def test_list_engines_works_with_partial_services_map(_clean_state, monkeypatch):
    """list_engines tolerates _services missing entries — does not crash."""
    monkeypatch.setattr(state_mod, "_services", {"md": object()})
    out = json.loads(await discovery_mod._list_engines())
    assert len(out) == 2  # both still listed; sql.loaded == False


@pytest.mark.asyncio
async def test_list_engines_includes_mcp_tool_name():
    out = json.loads(await discovery_mod._list_engines())
    md = next(e for e in out if e["name"] == "md")
    assert md["mcp_tool"] == "search_md"


@pytest.mark.asyncio
async def test_list_engines_advertises_read_tool_only_for_documents():
    out = json.loads(await discovery_mod._list_engines())
    md = next(e for e in out if e["name"] == "md")
    sql = next(e for e in out if e["name"] == "sql")

    assert md["read_tool"] == "read_md"
    assert sql["read_tool"] is None


def test_register_discovery_tool_registers_list_engines(fresh_mcp):
    discovery_mod.register_discovery_tool(fresh_mcp)
    tool_names = {t.name for t in fresh_mcp._tool_manager.list_tools()}
    assert "list_engines" in tool_names


def test_register_discovery_tool_idempotent(fresh_mcp):
    discovery_mod.register_discovery_tool(fresh_mcp)
    count_first = len(fresh_mcp._tool_manager.list_tools())
    discovery_mod.register_discovery_tool(fresh_mcp)
    count_second = len(fresh_mcp._tool_manager.list_tools())
    assert count_first == count_second


def test_register_discovery_tool_raises_on_name_clash(fresh_mcp):
    """If something else registered list_engines under a different sentinel,
    the discovery registrar refuses to silently overwrite."""
    fresh_mcp._dbs_vector_registrations = {"list_engines": ("foo", "bar")}
    with pytest.raises(RuntimeError, match="non-discovery sentinel"):
        discovery_mod.register_discovery_tool(fresh_mcp)
