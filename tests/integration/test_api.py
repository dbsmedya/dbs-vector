"""Integration tests for FastAPI endpoints."""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@contextmanager
def mocked_client():
    """Context manager that yields (client, mock_md_service, mock_sql_service) with proper cleanup."""
    patches = []

    try:
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.engines = {
            "md": MagicMock(model_name="test-model", vector_dimension=384),
            "sql": MagicMock(model_name="sql-model", vector_dimension=768),
        }
        mock_settings.db_path = "./test_db"
        mock_settings.batch_size = 64
        mock_settings.nprobes = 20

        # Create mock services
        mock_md_service = MagicMock()
        mock_sql_service = MagicMock()

        def mock_build_deps(engine_name):
            return MagicMock(embedder=MagicMock(), store=MagicMock())

        # Start patches
        patches.append(patch("dbs_vector.api.main.settings", mock_settings))
        patches.append(patch("dbs_vector.api.main._build_dependencies", side_effect=mock_build_deps))
        patches.append(patch("dbs_vector.api.main._services", {"md": mock_md_service, "sql": mock_sql_service}))

        for p in patches:
            p.start()

        from dbs_vector.api.main import app

        client = TestClient(app)
        yield client, mock_md_service, mock_sql_service

    finally:
        # Always stop patches in reverse order
        for p in reversed(patches):
            p.stop()


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_healthy(self):
        """Test health check when services are initialized."""
        with mocked_client() as (client, _, _):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "md_model" in data


class TestSearchMdEndpoint:
    """Tests for the /search/md endpoint."""

    def test_search_md_success(self):
        """Test successful markdown search."""
        from dbs_vector.core.models import Chunk, SearchResult

        with mocked_client() as (client, mock_md_service, _):
            mock_results = [
                SearchResult(
                    chunk=Chunk(
                        id="chunk_0",
                        text="Test content",
                        source="docs/test.md",
                        content_hash="hash1",
                    ),
                    score=0.95,
                    distance=0.95,
                    is_fts_match=False,
                )
            ]
            mock_md_service.execute_query.return_value = mock_results

            response = client.post("/search/md", json={"query": "test query", "limit": 5})

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test query"
            assert len(data["results"]) == 1
            mock_md_service.execute_query.assert_called_once()

    def test_search_md_with_source_filter(self):
        """Test markdown search with source filter."""
        with mocked_client() as (client, mock_md_service, _):
            mock_md_service.execute_query.return_value = []

            response = client.post(
                "/search/md",
                json={"query": "test", "limit": 10, "source_filter": "docs/specific.md"},
            )

            assert response.status_code == 200
            call_args = mock_md_service.execute_query.call_args
            assert call_args[0][1] == "docs/specific.md"

    def test_search_md_validation_error(self):
        """Test validation error for invalid request data."""
        with mocked_client() as (client, _, _):
            response = client.post("/search/md", json={"limit": 5})
            assert response.status_code == 422

    def test_search_md_limit_validation(self):
        """Test limit parameter validation."""
        with mocked_client() as (client, _, _):
            response = client.post("/search/md", json={"query": "test", "limit": 200})
            assert response.status_code == 422

            response = client.post("/search/md", json={"query": "test", "limit": 0})
            assert response.status_code == 422

    def test_search_md_execution_error(self):
        """Test handling of search execution errors."""
        with mocked_client() as (client, mock_md_service, _):
            mock_md_service.execute_query.side_effect = Exception("Search failed")

            response = client.post("/search/md", json={"query": "test"})

            assert response.status_code == 500
            assert "Search execution failed" in response.json()["detail"]


class TestSearchSqlEndpoint:
    """Tests for the /search/sql endpoint."""

    def test_search_sql_success(self):
        """Test successful SQL search."""
        from dbs_vector.core.models import SqlChunk, SqlSearchResult

        with mocked_client() as (client, _, mock_sql_service):
            mock_results = [
                SqlSearchResult(
                    chunk=SqlChunk(
                        id="sql_0",
                        text="SELECT * FROM users",
                        raw_query="SELECT * FROM users WHERE id = 1",
                        source="production_db",
                        execution_time_ms=150.5,
                        calls=42,
                        content_hash="hash1",
                    ),
                    score=0.88,
                    distance=0.88,
                    is_fts_match=False,
                )
            ]
            mock_sql_service.execute_query.return_value = mock_results

            response = client.post("/search/sql", json={"query": "SELECT users", "limit": 5})

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "SELECT users"
            assert len(data["results"]) == 1

    def test_search_sql_validation_error(self):
        """Test validation error for invalid SQL search request."""
        with mocked_client() as (client, _, _):
            response = client.post("/search/sql", json={})
            assert response.status_code == 422


class TestRequestModels:
    """Tests for request model validation."""

    def test_search_request_defaults(self):
        """Test that SearchRequest has correct defaults."""
        with mocked_client() as (client, mock_md_service, _):
            mock_md_service.execute_query.return_value = []

            response = client.post("/search/md", json={"query": "test"})

            assert response.status_code == 200
            call_args = mock_md_service.execute_query.call_args
            assert call_args[0][2] == 5  # default limit

    def test_sql_search_request_defaults(self):
        """Test that SqlSearchRequest has correct defaults."""
        with mocked_client() as (client, _, mock_sql_service):
            mock_sql_service.execute_query.return_value = []

            response = client.post("/search/sql", json={"query": "test"})

            assert response.status_code == 200
            call_args = mock_sql_service.execute_query.call_args
            assert call_args[0][2] == 5  # default limit


class TestResponseModels:
    """Tests for response model structure."""

    def test_search_response_structure(self):
        """Test SearchResponse model structure."""
        from dbs_vector.core.models import Chunk, SearchResult

        with mocked_client() as (client, mock_md_service, _):
            mock_md_service.execute_query.return_value = [
                SearchResult(
                    chunk=Chunk(
                        id="test_chunk",
                        text="Test content",
                        source="test.md",
                        content_hash="abc123",
                    ),
                    score=0.9,
                    distance=0.9,
                    is_fts_match=False,
                )
            ]

            response = client.post("/search/md", json={"query": "test"})

            assert response.status_code == 200
            data = response.json()
            assert "query" in data
            assert "results" in data
            assert data["results"][0]["chunk"]["id"] == "test_chunk"

    def test_sql_search_response_structure(self):
        """Test SqlSearchResponse model structure."""
        from dbs_vector.core.models import SqlChunk, SqlSearchResult

        with mocked_client() as (client, _, mock_sql_service):
            mock_sql_service.execute_query.return_value = [
                SqlSearchResult(
                    chunk=SqlChunk(
                        id="sql_chunk",
                        text="SELECT 1",
                        raw_query="SELECT 1 FROM table",
                        source="db",
                        execution_time_ms=100.0,
                        calls=50,
                        content_hash="xyz789",
                    ),
                    score=None,
                    distance=None,
                    is_fts_match=True,
                )
            ]

            response = client.post("/search/sql", json={"query": "SELECT"})

            assert response.status_code == 200
            data = response.json()
            result = data["results"][0]
            assert result["score"] is None
            assert result["is_fts_match"] is True
            assert result["chunk"]["raw_query"] == "SELECT 1 FROM table"


class TestServiceUnavailable:
    """Tests for 503 Service Unavailable responses."""

    def test_search_md_service_not_initialized(self):
        """Test search returns 503 when md service is not available."""
        with patch("dbs_vector.api.main.settings") as mock_settings:
            mock_settings.engines = {"sql": MagicMock()}
            mock_settings.db_path = "./test_db"

            with patch("dbs_vector.api.main._services", {"sql": MagicMock()}):
                with patch("dbs_vector.api.main._build_dependencies"):
                    from dbs_vector.api.main import app

                    with TestClient(app) as client:
                        response = client.post("/search/md", json={"query": "test"})

                        assert response.status_code == 503
                        assert "not initialized" in response.json()["detail"]

    def test_search_sql_service_not_initialized(self):
        """Test search returns 503 when sql service is not available."""
        with patch("dbs_vector.api.main.settings") as mock_settings:
            mock_settings.engines = {"md": MagicMock()}

            with patch("dbs_vector.api.main._services", {"md": MagicMock()}):
                with patch("dbs_vector.api.main._build_dependencies"):
                    from dbs_vector.api.main import app

                    with TestClient(app) as client:
                        response = client.post("/search/sql", json={"query": "test"})

                        assert response.status_code == 503
                        assert "not initialized" in response.json()["detail"]


class TestLifespan:
    """Tests for application lifespan (startup/shutdown)."""

    def test_lifespan_is_async_context_manager(self):
        """Test that lifespan returns an async context manager."""
        with patch("dbs_vector.api.main.settings") as mock_settings:
            mock_settings.engines = {"md": MagicMock()}

            with patch("dbs_vector.api.main._build_dependencies"):
                from contextlib import AbstractAsyncContextManager

                from dbs_vector.api.main import lifespan

                result = lifespan(MagicMock())
                assert isinstance(result, AbstractAsyncContextManager)
