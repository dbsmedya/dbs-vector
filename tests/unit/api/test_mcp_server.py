from unittest.mock import MagicMock

import pytest

from dbs_vector.api.mcp_server import search_documents, search_sql_logs
from dbs_vector.api.state import _services
from dbs_vector.core.models import Chunk, SearchResult, SqlChunk, SqlSearchResult


@pytest.fixture
def mock_services():
    """Set up and tear down mock services in the global state."""
    mock_md_service = MagicMock()
    mock_sql_service = MagicMock()

    _services["md"] = mock_md_service
    _services["sql"] = mock_sql_service

    yield _services

    _services.clear()


@pytest.mark.asyncio
async def test_search_documents_success(mock_services):
    """Test that search_documents formats results correctly."""
    mock_service = mock_services["md"]
    mock_service.execute_query.return_value = [
        SearchResult(
            chunk=Chunk(
                id="1_0",
                source="doc1.md",
                text="mock content",
                content_hash="123",
                document_id="1",
                index=0,
            ),
            distance=0.1234,
        )
    ]

    result_str = await search_documents(query="test", limit=1)

    mock_service.execute_query.assert_called_once_with(
        query="test", source_filter=None, limit=1, extra_filters={}
    )

    assert "Found 1 results for 'test'" in result_str
    assert "Score: 0.1234" in result_str
    assert "Source: doc1.md" in result_str
    assert "mock content" in result_str


@pytest.mark.asyncio
async def test_search_documents_no_results(mock_services):
    """Test that search_documents handles empty results gracefully."""
    mock_service = mock_services["md"]
    mock_service.execute_query.return_value = []

    result_str = await search_documents(query="test empty")

    assert result_str == "No results found for query: 'test empty'"


@pytest.mark.asyncio
async def test_search_documents_not_initialized():
    """Test behavior when the md engine is not loaded."""
    _services.clear()  # Ensure state is empty
    result_str = await search_documents(query="test")
    assert "Error: Document search service" in result_str


@pytest.mark.asyncio
async def test_search_sql_logs_success(mock_services):
    """Test that search_sql_logs formats SQL results correctly."""
    mock_service = mock_services["sql"]
    mock_service.execute_query.return_value = [
        SqlSearchResult(
            chunk=SqlChunk(
                id="sql_0",
                source="test_db",
                raw_query="SELECT * FROM test;",
                text="SELECT * FROM test;",
                content_hash="hash",
                document_id="doc1",
                execution_time_ms=500.5,
                calls=2,
                query_hash="abc",
                index=0,
            ),
            distance=0.5678,
        )
    ]

    result_str = await search_sql_logs(query="select test", limit=2, min_time=100.0)

    mock_service.execute_query.assert_called_once_with(
        query="select test", source_filter=None, limit=2, extra_filters={"min_time": 100.0}
    )

    assert "Found 1 results for 'select test'" in result_str
    assert "Score: 0.5678" in result_str
    assert "Source Database: test_db" in result_str
    assert "Execution Time: 500.5ms (Calls: 2)" in result_str
    assert "SELECT * FROM test;" in result_str


@pytest.mark.asyncio
async def test_search_sql_logs_exception(mock_services):
    """Test that search_sql_logs handles exceptions gracefully."""
    mock_service = mock_services["sql"]
    mock_service.execute_query.side_effect = Exception("DB Connection Failed")

    result_str = await search_sql_logs(query="crash test")

    assert "Search execution failed: DB Connection Failed" in result_str
