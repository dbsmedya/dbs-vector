"""Tests for DocumentFamily run_search / format_results / make_handler."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dbs_vector.core.models import Chunk, SearchResult
from dbs_vector.mcp.families.document import DocumentFamily


def _fake_doc_result(source: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        distance=None,
        score=0.9,
        chunk=SimpleNamespace(source=source, text=text),
    )


def test_run_search_calls_service_with_kwargs():
    fam = DocumentFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="hello", limit=3, source_filter="docs/")

    service.execute_query.assert_called_once_with("hello", "docs/", 3, extra_filters={})


def test_format_results_includes_source_and_text():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="hello world", source="doc.md", content_hash="abc"),
            score=None,
            distance=0.1234,
            is_fts_match=False,
        ),
    ]
    out = fam.format_results(results, query="q")
    assert "Found 1 results for 'q'" in out
    assert "Source: doc.md" in out
    assert "hello world" in out
    assert "0.1234" in out


def test_format_results_uses_score_when_distance_none():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="t", source="s.md", content_hash="h"),
            score=0.0325,
            distance=None,
            is_fts_match=False,
        ),
    ]
    out = fam.format_results(results, query="q")
    assert "0.0325" in out
    assert "FTS" not in out


def test_format_results_marks_fts_match_with_no_score_or_distance():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="t", source="s.md", content_hash="h"),
            score=None,
            distance=None,
            is_fts_match=True,
        ),
    ]
    out = fam.format_results(results, query="q")
    assert "FTS" in out


def test_format_results_empty_returns_no_results_message():
    fam = DocumentFamily()
    out = fam.format_results([], query="zzz")
    assert out == "No results found for query: 'zzz'"


def test_make_handler_signature_has_expected_parameters():
    """FastMCP introspects this signature to build the tool schema."""
    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    sig = inspect.signature(handler)
    params = sig.parameters
    assert list(params) == ["query", "limit", "source_filter"]
    assert params["query"].annotation is str
    assert params["limit"].default == 5
    assert params["source_filter"].default is None


@pytest.mark.asyncio
async def test_make_handler_returns_error_when_service_missing(monkeypatch):
    """Handler reports a clear error if _services has no entry for the engine."""
    import dbs_vector.mcp.state as state_mod

    monkeypatch.setattr(state_mod, "_services", {})
    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    out = await handler(query="x")
    assert "search service 'md-test' is not initialized" in out


@pytest.mark.asyncio
async def test_make_handler_runs_search_and_formats(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [
        SearchResult(
            chunk=Chunk(id="x_0", text="content", source="f.md", content_hash="h"),
            score=None,
            distance=0.5,
            is_fts_match=False,
        ),
    ]
    monkeypatch.setattr(state_mod, "_services", {"md-test": service})

    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    out = await handler(query="q", limit=1)

    service.execute_query.assert_called_once_with("q", None, 1, extra_filters={})
    assert "Found 1 results for 'q'" in out
    assert "Source: f.md" in out


def test_document_family_caps_oversized_response():
    family = DocumentFamily()
    big_text = "z" * 600_000
    results = [_fake_doc_result(source=f"f{i}.md", text=big_text) for i in range(3)]
    out = family.format_results(results, query="q")
    assert len(out.encode("utf-8")) <= 1_000_000
    assert "results elided due to MCP response size cap" in out


def test_document_family_under_budget_unchanged():
    family = DocumentFamily()
    results = [_fake_doc_result(source="a.md", text="hello world")]
    out = family.format_results(results, query="q")
    assert out.startswith("Found 1 results for 'q':")
    assert "Source: a.md" in out
    assert "hello world" in out
    assert "elided" not in out
