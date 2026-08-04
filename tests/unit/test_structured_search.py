"""Document search returns prose text + structuredContent envelope."""

import asyncio
from types import SimpleNamespace

from mcp.types import CallToolResult

from dbs_vector.core.models import Chunk, SearchResponse, SearchResult
from dbs_vector.mcp.families.document import DocumentFamily
from dbs_vector.services.search import envelope_payload


def _response() -> SearchResponse:
    chunk = Chunk(id="c1", text="hello", source="a.md", content_hash="deadbeefdeadbeef")
    return SearchResponse(
        results=[SearchResult(chunk=chunk, similarity=0.9, retrieved_by="vector")],
        floor=None,
        inspected=1,
        best_rejected=None,
    )


def _handler_with_fake_service(monkeypatch, response: SearchResponse):
    from dbs_vector.mcp import state

    fake = SimpleNamespace(execute_query=lambda *a, **k: response)
    monkeypatch.setitem(state._services, "alpha-md", fake)
    return DocumentFamily().make_handler("alpha-md")


def test_result_carries_prose_and_envelope(monkeypatch) -> None:
    response = _response()
    handler = _handler_with_fake_service(monkeypatch, response)
    result = asyncio.run(handler("greeting"))
    assert isinstance(result, CallToolResult)
    # Prose is byte-identical to the pre-change rendering.
    assert result.content[0].text == DocumentFamily().format_results(response, "greeting")
    # Envelope is THE shared serializer's output.
    assert result.structuredContent == envelope_payload(response)
    assert not result.isError


def test_error_is_iserror_with_schema_valid_envelope() -> None:
    handler = DocumentFamily().make_handler("not-initialized-engine")
    result = asyncio.run(handler("q"))
    assert isinstance(result, CallToolResult)
    assert result.isError
    assert "not initialized" in result.content[0].text
    # structuredContent still validates against the declared SearchResponse
    # schema: empty results, zero inspected.
    assert result.structuredContent["results"] == []
    assert result.structuredContent["inspected"] == 0


def test_structured_search_flag() -> None:
    from dbs_vector.mcp.families.sql import SqlFamily

    assert DocumentFamily.structured_search is True
    assert getattr(SqlFamily, "structured_search", False) is False
