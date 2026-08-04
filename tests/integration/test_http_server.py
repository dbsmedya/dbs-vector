"""HTTP MCP server: auth, structural scoping, structured output — via ASGI
TestClient (no real port, lifespan runs the session managers)."""

import pytest
from starlette.testclient import TestClient

from dbs_vector.config import EngineConfig, ServerConfig, TuningProfile, settings
from dbs_vector.mcp.http_config import build_http_plan
from dbs_vector.mcp.server import build_http_app

TOKEN_A = "a" * 32
TOKEN_B = "b" * 32
ACCEPT = "application/json, text/event-stream"


def _engine(table: str, token: str | None) -> EngineConfig:
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


def _sql_engine(table: str, token: str | None) -> EngineConfig:
    return EngineConfig(
        description="t",
        model="gemma-bf16",
        mapper_type="sql",
        chunker_type="sql",
        table_name=table,
        workflow="w",
        tuning_profile="p",
        token=token,
    )


@pytest.fixture()
def client(monkeypatch):
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
        {
            "alpha-md": _engine("t_alpha", TOKEN_A),
            "beta-md": _engine("t_beta", TOKEN_B),
            "gamma-sql": _sql_engine("t_gamma", TOKEN_A),  # same scope as alpha
            "hidden-md": _engine("t_hidden", None),
        },
    )
    monkeypatch.setattr(settings, "server", ServerConfig())
    app = build_http_app(build_http_plan(settings))
    # base_url uses 127.0.0.1 so the SDK's Host validation (DNS-rebinding
    # defense, auto-enabled for loopback) accepts the requests.
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


def _post(client, token, payload, session=None, extra_headers=None):
    headers = {"Accept": ACCEPT, "Authorization": f"Bearer {token}"}
    if session:
        headers["mcp-session-id"] = session
    if extra_headers:
        headers.update(extra_headers)
    return client.post("/mcp", json=payload, headers=headers)


def _initialize(client, token) -> str:
    r = _post(
        client,
        token,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
    )
    assert r.status_code == 200, r.text
    session = r.headers.get("mcp-session-id")
    _post(client, token, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    return session


def _tools(client, token):
    session = _initialize(client, token)
    r = _post(
        client, token, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, session
    )
    assert r.status_code == 200, r.text
    return {t["name"]: t for t in r.json()["result"]["tools"]}


def test_scopes_are_disjoint(client):
    tools_a = _tools(client, TOKEN_A)
    tools_b = _tools(client, TOKEN_B)
    assert "search_alpha_md" in tools_a and "search_beta_md" not in tools_a
    assert "search_beta_md" in tools_b and "search_alpha_md" not in tools_b
    # fail-closed: the untokened engine exists in NO scope
    assert "search_hidden_md" not in tools_a and "search_hidden_md" not in tools_b


def test_wrong_or_missing_token_is_401(client):
    r = _post(client, "wrong" * 8, {"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert r.status_code == 401
    r = client.post("/mcp", json={}, headers={"Accept": ACCEPT})
    assert r.status_code == 401


def test_browser_origin_is_rejected(client):
    r = _post(
        client,
        TOKEN_A,
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        extra_headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_unknown_path_is_404(client):
    r = client.get("/anything", headers={"Authorization": f"Bearer {TOKEN_A}"})
    assert r.status_code == 404


def test_search_declares_output_schema(client):
    tool = _tools(client, TOKEN_A)["search_alpha_md"]
    assert "outputSchema" in tool
    assert "results" in tool["outputSchema"].get("properties", {})


def test_uninitialized_search_returns_structured_error(client):
    session = _initialize(client, TOKEN_A)
    r = _post(
        client,
        TOKEN_A,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_alpha_md", "arguments": {"query": "x"}},
        },
        session,
    )
    result = r.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"]["results"] == []


def _call_tool(client, token, name, arguments, extra_headers=None):
    session = _initialize(client, token)
    r = _post(
        client,
        token,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        session,
        extra_headers,
    )
    assert r.status_code == 200, r.text
    return r.json()["result"]


def _sql_response():
    # Before running: confirm SqlSearchResult's field names against
    # src/dbs_vector/core/models.py (mirrors SearchResult with a SqlChunk).
    from datetime import UTC, datetime

    from dbs_vector.core.models import SearchResponse, SqlChunk, SqlSearchResult

    chunk = SqlChunk(
        id="q1",
        text="SELECT ?",
        raw_query="SELECT 'RAW-NEEDLE'",
        source="db1",
        execution_time_ms=12.5,
        calls=3,
        content_hash="deadbeefdeadbeef",
        tables=["t"],
        latest_ts=datetime(2026, 8, 4, tzinfo=UTC),
    )
    return SearchResponse(
        results=[SqlSearchResult(chunk=chunk, similarity=0.8, retrieved_by="vector")],
        floor=None,
        inspected=1,
        best_rejected=None,
    )


def _fake_sql_service():
    # Align with SqlFamily.make_handler's actual service calls (read
    # families/sql.py:335-405): execute_query for results, count_matching
    # for the total shown in the header.
    from types import SimpleNamespace

    return SimpleNamespace(
        execute_query=lambda *a, **k: _sql_response(),
        count_matching=lambda *a, **k: 1,
    )


def test_raw_header_toggles_sql_payload(client, monkeypatch):
    # Blocker-1 regression test: header ON must produce raw_query end-to-end
    # through select AND BrowseService/formatting with ONE effective value.
    from dbs_vector.mcp import state

    monkeypatch.setitem(state._services, "gamma-sql", _fake_sql_service())
    args = {"query": "x", "include_raw": True}

    off = _call_tool(client, TOKEN_A, "search_gamma_sql", args)
    assert "RAW-NEEDLE" not in off["content"][0]["text"]

    on = _call_tool(
        client,
        TOKEN_A,
        "search_gamma_sql",
        args,
        extra_headers={"X-DBS-Allow-Raw-Queries": "true"},
    )
    assert "RAW-NEEDLE" in on["content"][0]["text"]


def test_raw_header_toggles_triage_both_sites(client, monkeypatch):
    # Blocker-1 regression test for TRIAGE specifically: the handler has TWO
    # raw sites (the select append and the build_and_run kwarg) and they must
    # move together. A recording fake catches a mixed-value regression that a
    # rendering-based fake would miss (formatting only sees the frame).
    from types import SimpleNamespace

    from dbs_vector.mcp import state
    from dbs_vector.services.browse import BrowseValidationError

    calls: list[tuple[str, bool]] = []

    class FakeBrowseService:
        def __init__(self, vector_store, frame_alias):
            pass

        def build_and_run(
            self, *, filters, group_by, order_by, select, limit, allow_raw_queries=False
        ):
            calls.append((select, allow_raw_queries))
            raise BrowseValidationError("probe")  # handler catches; returns str(e)

    monkeypatch.setattr("dbs_vector.mcp.families.sql.BrowseService", FakeBrowseService)
    monkeypatch.setitem(state._services, "gamma-sql", SimpleNamespace(vector_store=None))
    args = {"include_raw": True}

    off = _call_tool(client, TOKEN_A, "top_impacting_gamma_sql", args)
    assert "probe" in off["content"][0]["text"]
    select_off, kwarg_off = calls[-1]
    assert "raw_query" not in select_off and kwarg_off is False

    on = _call_tool(
        client,
        TOKEN_A,
        "top_impacting_gamma_sql",
        args,
        extra_headers={"X-DBS-Allow-Raw-Queries": "true"},
    )
    assert "probe" in on["content"][0]["text"]
    select_on, kwarg_on = calls[-1]
    assert "raw_query" in select_on and kwarg_on is True  # ONE effective value, both sites


def test_structured_success_over_http(client, monkeypatch):
    from types import SimpleNamespace

    from dbs_vector.core.models import Chunk, SearchResponse, SearchResult
    from dbs_vector.mcp import state

    chunk = Chunk(id="c1", text="hello", source="a.md", content_hash="deadbeefdeadbeef")
    resp = SearchResponse(
        results=[SearchResult(chunk=chunk, similarity=0.9, retrieved_by="vector")],
        floor=None,
        inspected=1,
    )
    monkeypatch.setitem(
        state._services, "alpha-md", SimpleNamespace(execute_query=lambda *a, **k: resp)
    )
    result = _call_tool(client, TOKEN_A, "search_alpha_md", {"query": "greeting"})
    assert result.get("isError") is not True
    sc = result["structuredContent"]
    assert sc["inspected"] == 1
    assert sc["results"][0]["chunk"]["source"] == "a.md"


def test_untokened_engine_still_served_on_stdio_registration(client):
    # Spec: "no-token engine absent over HTTP but present via stdio."
    # Same settings as the HTTP fixture; engine_names=None is the stdio path.
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from dbs_vector.mcp.dynamic_tools import register_search_tools

    stdio_mcp = FastMCP("stdio-check")
    register_search_tools(stdio_mcp)
    names = {t.name for t in asyncio.run(stdio_mcp.list_tools())}
    assert "search_hidden_md" in names
