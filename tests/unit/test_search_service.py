"""Unit tests for the SearchService."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from dbs_vector.core.models import Chunk, RejectedCandidate, SearchResponse, SearchResult
from dbs_vector.core.source_scope import SourceResolution
from dbs_vector.services.search import (
    SearchService,
    format_admission_empty,
    format_unmatched_source,
)


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns predictable vectors."""
    embedder = MagicMock()
    embedder.embed_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store that returns predictable results."""
    store = MagicMock()
    # Default: any source_filter resolves. Tests that care about an unmatched
    # filter override this explicitly — a bare MagicMock would make every
    # filter look unresolved and silently skip the search.
    store.resolve_source_filter.side_effect = lambda value: SourceResolution(
        value=value, kind="exact", matched=[value]
    )
    return store


@pytest.fixture
def search_service(mock_embedder, mock_vector_store):
    """Create a SearchService with mocked dependencies."""
    return SearchService(embedder=mock_embedder, vector_store=mock_vector_store)


class TestExecuteQuery:
    """Tests for the execute_query method."""

    def test_basic_query_execution(self, search_service, mock_embedder, mock_vector_store):
        """Test basic query execution with default parameters."""
        # Arrange
        expected_results = [
            SearchResult(
                chunk=Chunk(
                    id="chunk_0",
                    text="test content",
                    source="test.md",
                    content_hash="hash1",
                ),
                similarity=0.9,
                retrieved_by="both",
                rrf_score=0.0125,
            )
        ]
        mock_vector_store.search.return_value = expected_results

        # Act
        response = search_service.execute_query(query="test query")

        # Assert
        mock_embedder.embed_query.assert_called_once_with("test query")
        # Check call arguments individually due to numpy array comparison issues
        assert mock_vector_store.search.call_count == 1
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["query"] == "test query"
        np.testing.assert_array_equal(
            call_args.kwargs["query_vector"],
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
        assert call_args.kwargs["source_filter"] is None
        assert call_args.kwargs["limit"] == 5
        assert response.results == expected_results
        assert response.floor is None
        assert response.inspected == len(expected_results)
        assert response.best_rejected is None

    def test_query_with_source_filter(self, search_service, mock_embedder, mock_vector_store):
        """Test query with source filter parameter."""
        # Arrange
        mock_vector_store.search.return_value = []

        # Act
        search_service.execute_query(
            query="test query",
            source_filter="docs/specific.md",
            limit=10,
        )

        # Assert
        assert mock_vector_store.search.call_count == 1
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["query"] == "test query"
        np.testing.assert_array_equal(
            call_args.kwargs["query_vector"],
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
        assert call_args.kwargs["source_filter"] == "docs/specific.md"
        assert call_args.kwargs["limit"] == 10

    def test_query_with_extra_filters(self, search_service, mock_embedder, mock_vector_store):
        """Test query with extra filters passed through."""
        # Arrange
        mock_vector_store.search.return_value = []
        extra_filters = {"min_time": 100.0, "custom_key": "value"}

        # Act
        search_service.execute_query(
            query="slow query",
            extra_filters=extra_filters,
        )

        # Assert
        assert mock_vector_store.search.call_count == 1
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs["query"] == "slow query"
        np.testing.assert_array_equal(
            call_args.kwargs["query_vector"],
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
        assert call_args.kwargs["source_filter"] is None
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["min_time"] == 100.0
        assert call_args.kwargs["custom_key"] == "value"

    def test_empty_extra_filters_default(self, search_service, mock_vector_store):
        """Test that empty extra_filters dict is used by default."""
        # Arrange
        mock_vector_store.search.return_value = []

        # Act - call without extra_filters
        search_service.execute_query(query="test")

        # Assert - verify search was called with default empty filters
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs.get("extra_filters") is None


def test_count_matching_delegates_to_store():
    embedder = MagicMock()
    vector_store = MagicMock()
    vector_store.count_matching.return_value = 42

    service = SearchService(embedder=embedder, vector_store=vector_store)
    result = service.count_matching(
        source_filter="db",
        extra_filters={"table_filter": "OrderShipment", "min_time": 100},
    )

    assert result == 42
    vector_store.count_matching.assert_called_once_with(
        source_filter="db",
        table_filter="OrderShipment",
        min_time=100,
    )


class TestUnmatchedSourceFilter:
    """A filter that names nothing must never look like an empty corpus."""

    @pytest.fixture
    def unmatched(self, mock_vector_store):
        mock_vector_store.resolve_source_filter.side_effect = None
        mock_vector_store.resolve_source_filter.return_value = SourceResolution(
            value="specs",
            kind="none",
            matched=[],
            suggestions=["/root/docs/spec.md"],
        )
        return mock_vector_store

    def test_unmatched_filter_short_circuits_before_embedding(
        self, search_service, mock_embedder, unmatched
    ):
        """No model load and no search for a filter that cannot match."""
        response = search_service.execute_query(query="anything", source_filter="specs")

        assert response.results == []
        assert response.source_resolution is not None
        assert response.source_resolution.is_unmatched
        mock_embedder.embed_query.assert_not_called()
        unmatched.search.assert_not_called()

    def test_unmatched_filter_is_not_reported_as_an_absent_corpus(self, search_service, unmatched):
        message = format_unmatched_source(
            search_service.execute_query(query="anything", source_filter="specs")
        )

        assert "matched no indexed source" in message
        assert "/root/docs/spec.md" in message
        # The distinguishing property: it must not assert anything about the
        # corpus, which is exactly what "No results found" wrongly implied.
        assert "No results found" not in message

    def test_a_resolved_filter_is_reported_on_a_successful_response(
        self, search_service, mock_vector_store
    ):
        mock_vector_store.search.return_value = []
        response = search_service.execute_query(query="q", source_filter="docs/api.md")

        assert response.source_resolution is not None
        assert not response.source_resolution.is_unmatched
        assert response.source_resolution.matched == ["docs/api.md"]

    def test_no_filter_leaves_the_resolution_unset(self, search_service, mock_vector_store):
        mock_vector_store.search.return_value = []
        response = search_service.execute_query(query="q")

        assert response.source_resolution is None
        mock_vector_store.resolve_source_filter.assert_not_called()


class TestPrintResults:
    """Tests for the print_results method."""

    def test_print_empty_results(self, search_service, caplog):
        """Test printing empty results."""
        search_service.print_results(SearchResponse(results=[], inspected=0), "some query")

        assert "No results found" in caplog.text

    def test_print_admission_empty_logs_format_admission_empty(self, search_service, caplog):
        """When floor is active and candidates were inspected, print_results
        logs the honest admission-empty message, not the generic one."""
        response = SearchResponse(
            results=[],
            floor=0.55,
            inspected=4,
            best_rejected=RejectedCandidate(similarity=0.3, source="doc.md", retrieved_by="vector"),
        )

        search_service.print_results(response, "some query")

        assert format_admission_empty("some query", response) in caplog.text

    def test_print_document_results(self, search_service, caplog):
        """Test printing document (non-SQL) results."""
        results = [
            SearchResult(
                chunk=Chunk(
                    id="doc_chunk_0",
                    text="This is the document content that should be displayed.",
                    source="docs/readme.md",
                    content_hash="abc123",
                    node_type="paragraph",
                    parent_scope="# Section",
                    line_range="10-20",
                ),
                similarity=0.95,
                retrieved_by="both",
                rrf_score=0.0125,
            )
        ]

        search_service.print_results(
            SearchResponse(results=results, inspected=len(results)), "some query"
        )

        assert "Top Results:" in caplog.text
        assert "docs/readme.md" in caplog.text
        assert "abc123" in caplog.text
        assert "Similarity: 0.95 (vector+fts)" in caplog.text
        assert "This is the document content" in caplog.text

    def test_print_sql_results(self, search_service, caplog):
        """Test printing SQL query results."""
        from dbs_vector.core.models import SqlChunk, SqlSearchResult

        results = [
            SqlSearchResult(
                chunk=SqlChunk(
                    id="sql_chunk_0",
                    text="SELECT * FROM users WHERE id = 1",
                    raw_query="SELECT * FROM users WHERE id = 1",
                    source="production_db",
                    execution_time_ms=150.5,
                    calls=42,
                    content_hash="sql_hash_123",
                    latest_ts=datetime.now(),
                ),
                similarity=0.88,
                retrieved_by="both",
                rrf_score=0.0125,
            )
        ]

        search_service.print_results(
            SearchResponse(results=results, inspected=len(results)), "some query"
        )

        assert "Top Results:" in caplog.text
        assert "production_db" in caplog.text
        assert "Calls: 42" in caplog.text
        assert "Time: 150.5ms" in caplog.text
        assert "SELECT * FROM users" in caplog.text

    def test_print_result_shows_similarity_and_channel(self, search_service, caplog):
        """print_results renders the exact cosine similarity and the
        retrieval-channel label, not a legacy score/distance fallback chain."""
        results = [
            SearchResult(
                chunk=Chunk(
                    id="hyb_chunk",
                    text="Hybrid search result",
                    source="docs/file.md",
                    content_hash="hyb_hash",
                ),
                similarity=0.78,
                retrieved_by="both",
                rrf_score=0.0325,
            )
        ]

        search_service.print_results(
            SearchResponse(results=results, inspected=len(results)), "some query"
        )

        assert "Similarity: 0.78 (vector+fts)" in caplog.text

    def test_print_fts_only_result_labels_channel(self, search_service, caplog):
        """A pure-FTS-channel result is labeled 'fts-only', with no 'N/A'
        fallback text (the legacy FTS-match placeholder is gone)."""
        results = [
            SearchResult(
                chunk=Chunk(
                    id="fts_chunk",
                    text="Full text search result",
                    source="docs/file.md",
                    content_hash="fts_hash",
                ),
                similarity=0.05,
                retrieved_by="fts",
                rrf_score=0.014,
            )
        ]

        search_service.print_results(
            SearchResponse(results=results, inspected=len(results)), "some query"
        )

        assert "(fts-only)" in caplog.text
        assert "N/A" not in caplog.text
        assert "Full text search result" in caplog.text

    def test_print_multiple_results(self, search_service, caplog):
        """Test printing multiple results."""
        results = [
            SearchResult(
                chunk=Chunk(
                    id="chunk_0",
                    text="First result content here.",
                    source="docs/a.md",
                    content_hash="hash_a",
                ),
                similarity=0.9,
                retrieved_by="both",
                rrf_score=0.0125,
            ),
            SearchResult(
                chunk=Chunk(
                    id="chunk_1",
                    text="Second result content here.",
                    source="docs/b.md",
                    content_hash="hash_b",
                ),
                similarity=0.8,
                retrieved_by="vector",
                rrf_score=None,
            ),
        ]

        search_service.print_results(
            SearchResponse(results=results, inspected=len(results)), "some query"
        )

        assert caplog.text.count("Source:") == 2
        assert "hash_a" in caplog.text
        assert "hash_b" in caplog.text


class TestResultsToJson:
    """Tests for the results_to_json method (full-fidelity JSON dump)."""

    def test_empty_response_serializes_to_envelope(self, search_service):
        payload = json.loads(
            search_service.results_to_json(SearchResponse(results=[], floor=None, inspected=0))
        )
        assert payload == {
            "floor": None,
            "inspected": 0,
            "best_rejected": None,
            "source_resolution": None,
            "results": [],
        }

    def test_unmatched_source_filter_is_visible_in_the_envelope(self, search_service):
        payload = json.loads(
            search_service.results_to_json(
                SearchResponse(
                    results=[],
                    floor=None,
                    inspected=0,
                    source_resolution=SourceResolution(
                        value="specs", kind="none", suggestions=["/root/spec.md"]
                    ),
                )
            )
        )
        assert payload["source_resolution"] == {
            "value": "specs",
            "kind": "none",
            "matched": [],
            "suggestions": ["/root/spec.md"],
        }

    def test_document_result_includes_score_source_full_text_and_metadata(self, search_service):
        results = [
            SearchResult(
                chunk=Chunk(
                    id="doc_chunk_0",
                    text="This is the full document content that should be present verbatim.",
                    source="docs/readme.md",
                    content_hash="abc123",
                    node_type="paragraph",
                    parent_scope="# Section",
                    line_range="10-20",
                ),
                similarity=0.95,
                retrieved_by="both",
                rrf_score=0.42,
            )
        ]

        payload = json.loads(
            search_service.results_to_json(SearchResponse(results=results, inspected=1))
        )

        assert len(payload["results"]) == 1
        item = payload["results"][0]
        assert item["similarity"] == 0.95
        assert item["retrieved_by"] == "both"
        assert item["rrf_score"] == 0.42
        # Full text is present verbatim (not truncated like print_results).
        assert item["chunk"]["text"] == (
            "This is the full document content that should be present verbatim."
        )
        assert item["chunk"]["source"] == "docs/readme.md"
        assert item["chunk"]["content_hash"] == "abc123"
        # Metadata fields survive the round-trip.
        assert item["chunk"]["node_type"] == "paragraph"
        assert item["chunk"]["parent_scope"] == "# Section"
        assert item["chunk"]["line_range"] == "10-20"

    def test_sql_result_includes_raw_query_and_sql_metadata(self, search_service):
        from dbs_vector.core.models import SqlChunk, SqlSearchResult

        results = [
            SqlSearchResult(
                chunk=SqlChunk(
                    id="sql_chunk_0",
                    text="SELECT * FROM users WHERE id = ?",
                    raw_query="SELECT * FROM users WHERE id = 1",
                    source="production_db",
                    execution_time_ms=150.5,
                    calls=42,
                    content_hash="sql_hash_123",
                    tables=["users"],
                    latest_ts=datetime(2026, 1, 1, 12, 0, 0),
                ),
                similarity=0.88,
                retrieved_by="vector",
                rrf_score=None,
            )
        ]

        payload = json.loads(
            search_service.results_to_json(SearchResponse(results=results, inspected=1))
        )

        item = payload["results"][0]
        assert item["similarity"] == 0.88
        assert item["retrieved_by"] == "vector"
        assert item["rrf_score"] is None
        assert item["chunk"]["raw_query"] == "SELECT * FROM users WHERE id = 1"
        assert item["chunk"]["source"] == "production_db"
        assert item["chunk"]["execution_time_ms"] == 150.5
        assert item["chunk"]["calls"] == 42
        assert item["chunk"]["tables"] == ["users"]
        # datetime serializes to an ISO-8601 string in JSON mode.
        assert item["chunk"]["latest_ts"].startswith("2026-01-01T12:00:00")


def _floor_result(sim, rb="vector", text="body text", source="doc.md"):
    return SearchResult(
        chunk=Chunk(id="c1", text=text, source=source, content_hash="h1"),
        similarity=sim,
        retrieved_by=rb,
        rrf_score=None,
    )


class TestFloorPolicy:
    def _service(self, results, floor=None):
        embedder = MagicMock()
        embedder.embed_query.return_value = np.zeros(4, dtype=np.float32)
        store = MagicMock()
        store.search.return_value = results
        return SearchService(embedder, store, similarity_floor=floor), store

    def test_no_floor_returns_everything_unchanged(self):
        svc, store = self._service([_floor_result(0.1), _floor_result(-0.5)])
        resp = svc.execute_query("q", limit=5)
        assert resp.floor is None
        assert len(resp.results) == 2
        assert resp.inspected == 2
        assert resp.best_rejected is None
        assert store.search.call_args.kwargs["limit"] == 5

    def test_engine_floor_oversamples_and_filters(self):
        svc, store = self._service([_floor_result(0.9), _floor_result(0.2)], floor=0.5)
        resp = svc.execute_query("q", limit=5)
        assert store.search.call_args.kwargs["limit"] == 15  # limit * FLOOR_OVERSAMPLE
        assert [r.similarity for r in resp.results] == [0.9]
        assert resp.floor == 0.5
        assert resp.inspected == 2
        assert resp.best_rejected is not None
        assert resp.best_rejected.similarity == 0.2
        assert resp.best_rejected.source == "doc.md"

    def test_per_call_min_similarity_overrides_engine_floor(self):
        svc, _ = self._service([_floor_result(0.4)], floor=0.9)
        resp = svc.execute_query("q", min_similarity=0.3)
        assert resp.floor == 0.3
        assert len(resp.results) == 1

    def test_min_similarity_zero_is_a_real_floor_not_disable(self):
        svc, store = self._service([_floor_result(-0.2)])
        resp = svc.execute_query("q", limit=5, min_similarity=0.0)
        assert resp.floor == 0.0
        assert resp.results == []
        assert store.search.call_args.kwargs["limit"] == 15  # still oversampled

    def test_disable_flag_beats_everything_and_keeps_original_pool(self):
        svc, store = self._service([_floor_result(-0.9)], floor=0.9)
        resp = svc.execute_query("q", limit=5, min_similarity=0.8, disable_similarity_floor=True)
        assert resp.floor is None
        assert len(resp.results) == 1
        assert store.search.call_args.kwargs["limit"] == 5  # exact-baseline pool

    def test_lexical_gate_admits_fts_verbatim_row_below_floor(self):
        row = _floor_result(0.0, rb="fts", text="def delete_by_source(): ...")
        svc, _ = self._service([row], floor=0.5)
        resp = svc.execute_query("delete_by_source")
        assert resp.results == [row]
        assert resp.best_rejected is None

    def test_lexical_gate_requires_fts_channel(self):
        row = _floor_result(0.0, rb="vector", text="def delete_by_source(): ...")
        svc, _ = self._service([row], floor=0.5)
        assert svc.execute_query("delete_by_source").results == []

    def test_truncation_happens_after_admission(self):
        rows = [
            _floor_result(0.9),
            _floor_result(0.2),
            _floor_result(0.8),
            _floor_result(0.7),
        ]
        svc, _ = self._service(rows, floor=0.5)
        resp = svc.execute_query("q", limit=2)
        assert [r.similarity for r in resp.results] == [0.9, 0.8]  # RRF order kept, gaps dropped
        assert resp.inspected == 4

    def test_out_of_range_min_similarity_raises(self):
        svc, _ = self._service([])
        with pytest.raises(ValueError, match="min_similarity"):
            svc.execute_query("q", min_similarity=1.5)

    def test_range_validation_is_unconditional_even_with_disable_flag(self):
        # The chosen rule for conflicting controls: input validation always
        # runs; disable_similarity_floor only wins the FLOOR resolution.
        # Garbage input fails loudly instead of being masked by the flag.
        svc, _ = self._service([])
        with pytest.raises(ValueError, match="min_similarity"):
            svc.execute_query("q", min_similarity=2.0, disable_similarity_floor=True)

    def test_invalid_limit_raises(self):
        # LanceDB treats a non-positive limit as "no limit" (unbounded fetch)
        # on an LLM-callable surface; floor mode also multiplies limit by 3.
        svc, _ = self._service([])
        for bad in (0, -1, 101):
            with pytest.raises(ValueError, match="limit"):
                svc.execute_query("q", limit=bad)

    def test_best_rejected_is_highest_similarity_among_all_rejected(self):
        rows = [
            _floor_result(0.1),
            _floor_result(0.4, source="close.md"),
            _floor_result(0.2),
        ]
        svc, _ = self._service(rows, floor=0.5)
        resp = svc.execute_query("q")
        assert resp.best_rejected is not None
        assert resp.best_rejected.similarity == 0.4
        assert resp.best_rejected.source == "close.md"
