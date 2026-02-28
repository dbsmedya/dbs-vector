"""Unit tests for the SearchService."""

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
                score=0.9,
                distance=0.9,
                is_fts_match=False,
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
                score=0.95,
                distance=0.95,
                is_fts_match=False,
            )
        ]

        search_service.print_results(results)

        assert "Top Results:" in caplog.text
        assert "docs/readme.md" in caplog.text
        assert "abc123" in caplog.text
        assert "Score/Dist: 0.9500" in caplog.text
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
                ),
                score=0.88,
                distance=0.88,
                is_fts_match=False,
            )
        ]

        search_service.print_results(results)

        assert "Top Results:" in caplog.text
        assert "production_db" in caplog.text
        assert "Calls: 42" in caplog.text
        assert "Time: 150.5ms" in caplog.text
        assert "SELECT * FROM users" in caplog.text

    def test_print_fts_match_result(self, search_service, caplog):
        """Test printing FTS match result (no distance score)."""
        results = [
            SearchResult(
                chunk=Chunk(
                    id="fts_chunk",
                    text="Full text search result",
                    source="docs/file.md",
                    content_hash="fts_hash",
                ),
                score=None,
                distance=None,
                is_fts_match=True,
            )
        ]

        search_service.print_results(results)

        assert "N/A (FTS Match)" in caplog.text
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
                score=0.9,
                distance=0.9,
                is_fts_match=False,
            ),
            SearchResult(
                chunk=Chunk(
                    id="chunk_1",
                    text="Second result content here.",
                    source="docs/b.md",
                    content_hash="hash_b",
                ),
                score=0.8,
                distance=0.8,
                is_fts_match=False,
            ),
        ]

        search_service.print_results(results)

        assert caplog.text.count("Source:") == 2
        assert "hash_a" in caplog.text
        assert "hash_b" in caplog.text
