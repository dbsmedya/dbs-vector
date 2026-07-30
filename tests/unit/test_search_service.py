"""Unit tests for the SearchService."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from dbs_vector.core.models import Chunk, SearchResult
from dbs_vector.services.search import SearchService


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
        results = search_service.execute_query(query="test query")

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
        assert results == expected_results

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


class TestPrintResults:
    """Tests for the print_results method."""

    def test_print_empty_results(self, search_service, caplog):
        """Test printing empty results."""
        search_service.print_results([])

        assert "No results found" in caplog.text

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

        search_service.print_results(results)

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

        search_service.print_results(results)

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

        search_service.print_results(results)

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

        search_service.print_results(results)

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

        search_service.print_results(results)

        assert caplog.text.count("Source:") == 2
        assert "hash_a" in caplog.text
        assert "hash_b" in caplog.text


class TestResultsToJson:
    """Tests for the results_to_json method (full-fidelity JSON dump)."""

    def test_empty_results_is_empty_array(self, search_service):
        assert json.loads(search_service.results_to_json([])) == []

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

        payload = json.loads(search_service.results_to_json(results))

        assert len(payload) == 1
        item = payload[0]
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

        payload = json.loads(search_service.results_to_json(results))

        item = payload[0]
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
