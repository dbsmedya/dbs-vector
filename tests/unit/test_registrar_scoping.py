"""engine_names=None keeps stdio identical; a subset registers only itself."""

import json

import pytest
from mcp.server.fastmcp import FastMCP

from dbs_vector.config import EngineConfig, TuningProfile, settings
from dbs_vector.mcp.discovery import register_discovery_tool
from dbs_vector.mcp.dynamic_tools import register_read_tools, register_search_tools


def _engine(table: str, token: str | None = None) -> EngineConfig:
    return EngineConfig(
        description="t",
        model="gemma-bf16",
        mapper_type="document",
        chunker_type="document",
        table_name=table,
        workflow="w",
        tuning_profile="p",
        token=token,
    )


@pytest.fixture()
def two_engines(monkeypatch):
    monkeypatch.setattr(
        settings,
        "profiles",
        {
            "p": TuningProfile(
                max_token_length=2048,
                chunk_max_chars=1000,
                batch_size=8,
                chunk_target_tokens=256,
                chunk_max_tokens=512,
            )
        },
    )
    monkeypatch.setattr(
        settings,
        "engines",
        {"alpha-md": _engine("ta", "x" * 32), "beta-md": _engine("tb")},
    )


async def _tool_names(mcp: FastMCP) -> set[str]:
    return {t.name for t in await mcp.list_tools()}


@pytest.mark.asyncio
async def test_none_registers_everything_even_with_tokens(two_engines):
    # stdio orthogonality: tokens present, engine_names=None -> ALL engines.
    mcp = FastMCP("t")
    register_search_tools(mcp)
    register_read_tools(mcp)
    register_discovery_tool(mcp)
    names = await _tool_names(mcp)
    assert {
        "search_alpha_md",
        "search_beta_md",
        "read_alpha_md",
        "read_beta_md",
        "list_engines",
    } <= names


@pytest.mark.asyncio
async def test_subset_registers_only_named_engines(two_engines):
    mcp = FastMCP("t")
    register_search_tools(mcp, engine_names={"alpha-md"})
    register_read_tools(mcp, engine_names={"alpha-md"})
    register_discovery_tool(mcp, engine_names={"alpha-md"})
    names = await _tool_names(mcp)
    assert "search_alpha_md" in names and "read_alpha_md" in names
    assert "search_beta_md" not in names and "read_beta_md" not in names


@pytest.mark.asyncio
async def test_scoped_list_engines_hides_other_scope(two_engines):
    mcp = FastMCP("t")
    register_search_tools(mcp, engine_names={"alpha-md"})
    register_discovery_tool(mcp, engine_names={"alpha-md"})
    result = await mcp.call_tool("list_engines", {})
    text = result[0][0].text if isinstance(result, tuple) else result[0].text
    listed = {e["name"] for e in json.loads(text)}
    assert listed == {"alpha-md"}
