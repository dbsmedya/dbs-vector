"""Tests for DocumentFamily run_search / format_results / make_handler."""

import inspect
from unittest.mock import MagicMock

import pytest

from dbs_vector.core.models import Chunk, RejectedCandidate, SearchResponse, SearchResult
from dbs_vector.mcp.families.document import DocumentFamily


def _fake_doc_result(source: str = "tests/x.py", text: str = "hello world") -> SearchResult:
    # Real domain models (not SimpleNamespace) so a field rename in
    # SearchResult/Chunk breaks these tests the same way it breaks production.
    return SearchResult(
        chunk=Chunk(id=f"{source}_chunk_0", text=text, source=source, content_hash="h"),
        similarity=0.9,
        retrieved_by="vector",
        rrf_score=0.0328,
    )


def _floor_response(results, floor, inspected, best=None):
    return SearchResponse(results=results, floor=floor, inspected=inspected, best_rejected=best)


def test_run_search_calls_service_with_kwargs():
    fam = DocumentFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="hello", limit=3, source_filter="docs/")

    service.execute_query.assert_called_once_with(
        "hello", "docs/", 3, extra_filters={}, min_similarity=None, disable_similarity_floor=False
    )


def test_format_results_includes_source_and_text():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="hello world", source="doc.md", content_hash="abc"),
            similarity=0.12,
            retrieved_by="vector",
            rrf_score=None,
        ),
    ]
    out = fam.format_results(SearchResponse(results=results, inspected=len(results)), query="q")
    assert "Found 1 results for 'q'" in out
    assert "Source: doc.md" in out
    assert "hello world" in out
    assert "similarity 0.12" in out
    assert "retrieved by: vector-only" in out


def test_format_results_renders_both_channel_label():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="t", source="s.md", content_hash="h"),
            similarity=0.0325,
            retrieved_by="both",
            rrf_score=0.014,
        ),
    ]
    out = fam.format_results(SearchResponse(results=results, inspected=len(results)), query="q")
    assert "similarity 0.03" in out
    assert "vector+fts" in out


def test_format_results_renders_fts_only_label():
    fam = DocumentFamily()
    results = [
        SearchResult(
            chunk=Chunk(id="x_0", text="t", source="s.md", content_hash="h"),
            similarity=0.05,
            retrieved_by="fts",
            rrf_score=0.014,
        ),
    ]
    out = fam.format_results(SearchResponse(results=results, inspected=len(results)), query="q")
    assert "fts-only" in out
    assert "0.014" not in out


def test_format_results_empty_returns_no_results_message():
    fam = DocumentFamily()
    out = fam.format_results(SearchResponse(results=[], inspected=0), query="zzz")
    assert out == "No results found for query: 'zzz'"


def test_make_handler_signature_has_expected_parameters():
    """FastMCP introspects this signature to build the tool schema."""
    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    sig = inspect.signature(handler)
    params = sig.parameters
    assert list(params) == [
        "query",
        "limit",
        "source_filter",
        "min_similarity",
        "disable_similarity_floor",
    ]
    assert params["query"].annotation is str
    assert params["limit"].default == 5
    assert params["source_filter"].default is None
    assert params["min_similarity"].default is None
    assert params["disable_similarity_floor"].default is False


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
    service.execute_query.return_value = SearchResponse(
        results=[
            SearchResult(
                chunk=Chunk(id="x_0", text="content", source="f.md", content_hash="h"),
                similarity=0.5,
                retrieved_by="vector",
            ),
        ],
        inspected=1,
    )
    monkeypatch.setattr(state_mod, "_services", {"md-test": service})

    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    out = await handler(query="q", limit=1)

    service.execute_query.assert_called_once_with(
        "q", None, 1, extra_filters={}, min_similarity=None, disable_similarity_floor=False
    )
    assert "Found 1 results for 'q'" in out
    assert "Source: f.md" in out


@pytest.mark.asyncio
async def test_make_handler_rejects_out_of_range_min_similarity(monkeypatch):
    """Validation happens before the service is called (no execute_query call)."""
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    monkeypatch.setattr(state_mod, "_services", {"md-test": service})

    fam = DocumentFamily()
    handler = fam.make_handler("md-test")
    out = await handler(query="q", min_similarity=2.0)

    assert out == "min_similarity must be within [-1, 1]; got 2.0."
    service.execute_query.assert_not_called()


def test_document_family_caps_oversized_response():
    family = DocumentFamily()
    big_text = "z" * 600_000
    results = [_fake_doc_result(source=f"f{i}.md", text=big_text) for i in range(3)]
    out = family.format_results(SearchResponse(results=results, inspected=len(results)), query="q")
    assert len(out.encode("utf-8")) <= 1_000_000
    assert "results elided due to MCP response size cap" in out


def test_document_family_under_budget_unchanged():
    family = DocumentFamily()
    results = [_fake_doc_result(source="a.md", text="hello world")]
    out = family.format_results(SearchResponse(results=results, inspected=len(results)), query="q")
    assert out.startswith("Found 1 results for 'q' (hybrid-ranked):")
    assert "Source: a.md" in out
    assert "hello world" in out
    assert "elided" not in out


class TestFloorPresentation:
    def test_no_floor_header_says_hybrid_ranked(self):
        out = DocumentFamily().format_results(_floor_response([_fake_doc_result()], None, 1), "q")
        assert "Found 1 results for 'q' (hybrid-ranked):" in out

    def test_floor_header_carries_admission_phrase(self):
        out = DocumentFamily().format_results(_floor_response([_fake_doc_result()], 0.55, 3), "q")
        assert "(hybrid-ranked, admission: similarity >= 0.55 or all query terms verbatim)" in out

    def test_admission_empty_leads_with_low_confidence_not_absence(self):
        best = RejectedCandidate(similarity=0.38, source="tests/x.py", retrieved_by="fts")
        out = DocumentFamily().format_results(
            _floor_response([], 0.55, 15, best), "beehive maintenance"
        )
        assert "No inspected candidate passed admission" in out
        assert "similarity >= 0.55 or all query terms verbatim" in out
        assert "'beehive maintenance'" in out
        assert "Inspected 15 hybrid-ranked candidates" in out
        assert "0.38" in out and "tests/x.py" in out and "fts-only" in out
        assert "does not establish corpus-level absence" in out

    def test_empty_with_no_candidates_keeps_current_message(self):
        out = DocumentFamily().format_results(_floor_response([], 0.55, 0), "q")
        assert out == "No results found for query: 'q'"

    def test_rrf_score_never_rendered(self):
        res = _fake_doc_result()  # builder sets rrf_score=0.0328
        out = DocumentFamily().format_results(_floor_response([res], None, 1), "q")
        assert "0.0328" not in out
