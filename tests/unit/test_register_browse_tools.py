import pytest
from mcp.server.fastmcp import FastMCP

import dbs_vector.mcp.dynamic_tools as dyn
from dbs_vector.config import EngineConfig


def _engine(**over) -> EngineConfig:
    base = dict(
        description="short summary",
        model="gemma-bf16",
        mapper_type="sql",
        chunker_type="api",
        table_name="query_vault",
        workflow="sql_clustering",
        tuning_profile="gemma-sql-atomic",
    )
    base.update(over)
    return EngineConfig(**base)


@pytest.fixture
def patched(monkeypatch):
    engines = {
        "sql-api": _engine(chunker_type="api"),
        "md": _engine(mapper_type="document", chunker_type="document", tuning_profile="gemma-md"),
    }

    class _S:
        pass

    s = _S()
    s.engines = engines
    monkeypatch.setattr(dyn, "settings", s)
    return engines


@pytest.mark.asyncio
async def test_search_tools_use_family_description(patched):
    mcp = FastMCP("t")
    dyn.register_search_tools(mcp)
    tool = next(t for t in await mcp.list_tools() if t.name == "search_sql_api")
    assert "min_time" in tool.description  # family prose, not config's "short summary"
    assert tool.description != "short summary"


@pytest.mark.asyncio
async def test_register_browse_tools_only_sql_engines(patched):
    mcp = FastMCP("t")
    dyn.register_browse_tools(mcp, allow_raw_queries=False)
    tools = {t.name for t in await mcp.list_tools()}
    assert "browse_sql_api" in tools
    assert "browse_md" not in tools  # md is not the sql family


@pytest.mark.asyncio
async def test_register_browse_tools_idempotent(patched):
    mcp = FastMCP("t")
    dyn.register_browse_tools(mcp, allow_raw_queries=False)
    dyn.register_browse_tools(mcp, allow_raw_queries=False)  # no raise
    tools = {t.name for t in await mcp.list_tools()}
    assert "browse_sql_api" in tools


@pytest.mark.asyncio
async def test_register_browse_tools_flag_change_raises(patched):
    mcp = FastMCP("t")
    dyn.register_browse_tools(mcp, allow_raw_queries=False)
    with pytest.raises(RuntimeError):
        dyn.register_browse_tools(mcp, allow_raw_queries=True)
