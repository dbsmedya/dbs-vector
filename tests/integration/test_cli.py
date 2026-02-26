"""Integration tests for CLI commands."""
import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Need to import after setting up mocks
runner = CliRunner()


@pytest.fixture
def mock_settings():
    """Create mock settings with engines."""
    with patch("dbs_vector.cli.settings") as mock:
        mock.engines = {
            "md": MagicMock(
                model_name="test-model",
                vector_dimension=384,
                max_token_length=512,
                table_name="md_table",
                mapper_type="document",
                chunker_type="document",
                chunk_max_chars=1000,
            ),
            "sql": MagicMock(
                model_name="sql-model",
                vector_dimension=768,
                max_token_length=256,
                table_name="sql_table",
                mapper_type="sql",
                chunker_type="sql",
                chunk_max_chars=0,
            ),
        }
        mock.db_path = "./test_db"
        mock.batch_size = 64
        mock.nprobes = 20
        yield mock


@pytest.fixture
def mock_embedder():
    """Mock MLXEmbedder to avoid loading actual models."""
    with patch("dbs_vector.cli.MLXEmbedder") as mock:
        mock_instance = MagicMock()
        mock_instance.dimension = 384
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_store():
    """Mock LanceDBStore."""
    with patch("dbs_vector.cli.LanceDBStore") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_chunker():
    """Mock chunker classes."""
    with patch("dbs_vector.cli.ComponentRegistry.get_chunker") as mock_get:
        mock_chunker_class = MagicMock()
        mock_chunker_instance = MagicMock()
        mock_chunker_class.return_value = mock_chunker_instance
        mock_get.return_value = mock_chunker_class
        yield mock_get, mock_chunker_instance


@pytest.fixture
def mock_mapper():
    """Mock mapper classes."""
    with patch("dbs_vector.cli.ComponentRegistry.get_mapper") as mock_get:
        mock_mapper_class = MagicMock()
        mock_mapper_instance = MagicMock()
        mock_mapper_class.return_value = mock_mapper_instance
        mock_get.return_value = mock_mapper_class
        yield mock_get, mock_mapper_instance


@pytest.fixture
def mock_ingestion_service():
    """Mock IngestionService."""
    with patch("dbs_vector.cli.IngestionService") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_search_service():
    """Mock SearchService."""
    with patch("dbs_vector.cli.SearchService") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


class TestMainCallback:
    """Tests for the main callback/config loading."""

    def test_default_config_file(self, mock_settings, mock_embedder, mock_store):
        """Test that default config file is used."""
        from dbs_vector.cli import app

        with patch.dict(os.environ, {}, clear=True):
            runner.invoke(app, ["search", "test"])
            # Should use default config.yaml
            assert "DBS_CONFIG_FILE" in os.environ
            assert os.environ["DBS_CONFIG_FILE"] == "config.yaml"

    def test_custom_config_file(self, mock_settings, mock_embedder, mock_store):
        """Test that custom config file can be specified."""
        from dbs_vector.cli import app

        with patch.dict(os.environ, {}, clear=True):
            runner.invoke(app, ["-c", "custom.yaml", "search", "test"])
            assert os.environ["DBS_CONFIG_FILE"] == "custom.yaml"

    def test_config_file_short_option(self, mock_settings, mock_embedder, mock_store):
        """Test -c short option for config file."""
        from dbs_vector.cli import app

        with patch.dict(os.environ, {}, clear=True):
            runner.invoke(app, ["--config-file", "other.yaml", "search", "test"])
            assert os.environ["DBS_CONFIG_FILE"] == "other.yaml"


class TestIngestCommand:
    """Tests for the ingest command."""

    def test_ingest_basic(self, mock_settings, mock_embedder, mock_store, mock_chunker,
                          mock_mapper, mock_ingestion_service):
        """Test basic ingest command."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "docs/*.md"])

        assert result.exit_code == 0
        mock_ingestion_service.return_value.ingest_directory.assert_called_once_with(
            "docs/*.md", rebuild=False
        )

    def test_ingest_with_engine_type(self, mock_settings, mock_embedder, mock_store,
                                     mock_chunker, mock_mapper, mock_ingestion_service):
        """Test ingest with specific engine type."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "queries.json", "--type", "sql"])

        assert result.exit_code == 0
        # Verify SQL engine was used (via embedder call with sql model)
        mock_embedder.assert_called_once()
        call_kwargs = mock_embedder.call_args.kwargs
        assert call_kwargs["model_name"] == "sql-model"

    def test_ingest_unknown_engine(self, mock_settings):
        """Test ingest with unknown engine type."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "path", "--type", "unknown"])

        assert result.exit_code == 1
        assert "Unknown engine type" in result.output

    def test_ingest_rebuild_without_force(self, mock_settings, mock_embedder, mock_store,
                                          mock_chunker, mock_mapper, mock_ingestion_service):
        """Test rebuild flag triggers confirmation."""
        from dbs_vector.cli import app

        # Without --force, should prompt for confirmation
        result = runner.invoke(app, ["ingest", "path", "--rebuild"], input="n\n")

        # Should abort when user says no
        assert result.exit_code != 0 or "Aborted" in result.output

    def test_ingest_rebuild_with_force(self, mock_settings, mock_embedder, mock_store,
                                       mock_chunker, mock_mapper, mock_ingestion_service):
        """Test rebuild with force flag bypasses confirmation."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "path", "--rebuild", "--force"])

        assert result.exit_code == 0
        mock_ingestion_service.return_value.ingest_directory.assert_called_once_with(
            "path", rebuild=True
        )

    def test_ingest_short_options(self, mock_settings, mock_embedder, mock_store,
                                  mock_chunker, mock_mapper, mock_ingestion_service):
        """Test short options for ingest command."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "path", "-t", "sql", "-r", "-f"])

        assert result.exit_code == 0
        mock_ingestion_service.return_value.ingest_directory.assert_called_once_with(
            "path", rebuild=True
        )


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_basic(self, mock_settings, mock_embedder, mock_store, mock_chunker,
                          mock_mapper, mock_search_service):
        """Test basic search command."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 0
        mock_search_service.return_value.execute_query.assert_called_once()
        call_args = mock_search_service.return_value.execute_query.call_args
        assert call_args[0][0] == "test query"  # query
        assert call_args[1]["source_filter"] is None
        assert call_args[1]["limit"] == 5

    def test_search_with_options(self, mock_settings, mock_embedder, mock_store,
                                 mock_chunker, mock_mapper, mock_search_service):
        """Test search with all options."""
        from dbs_vector.cli import app

        result = runner.invoke(
            app,
            ["search", "my query", "--type", "sql", "--source", "mydb", "--limit", "10"],
        )

        assert result.exit_code == 0
        call_args = mock_search_service.return_value.execute_query.call_args
        assert call_args[0][0] == "my query"
        assert call_args[1]["source_filter"] == "mydb"
        assert call_args[1]["limit"] == 10

    def test_search_sql_with_min_time(self, mock_settings, mock_embedder, mock_store,
                                      mock_chunker, mock_mapper, mock_search_service):
        """Test SQL search with min_time filter."""
        from dbs_vector.cli import app

        result = runner.invoke(
            app,
            ["search", "slow query", "--type", "sql", "--min-time", "100.5"],
        )

        assert result.exit_code == 0
        call_args = mock_search_service.return_value.execute_query.call_args
        assert call_args[1]["extra_filters"] == {"min_time": 100.5}

    def test_search_md_ignores_min_time(self, mock_settings, mock_embedder, mock_store,
                                        mock_chunker, mock_mapper, mock_search_service):
        """Test that min_time is ignored for non-sql engines."""
        from dbs_vector.cli import app

        result = runner.invoke(
            app,
            ["search", "query", "--type", "md", "--min-time", "100"],
        )

        assert result.exit_code == 0
        call_args = mock_search_service.return_value.execute_query.call_args
        # min_time should not be in extra_filters for md engine
        assert call_args[1]["extra_filters"] == {}

    def test_search_unknown_engine(self, mock_settings):
        """Test search with unknown engine type."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "query", "--type", "unknown"])

        assert result.exit_code == 1
        assert "Unknown engine type" in result.output

    def test_search_results_printed(self, mock_settings, mock_embedder, mock_store,
                                    mock_chunker, mock_mapper, mock_search_service):
        """Test that search results are printed."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "test"])

        assert result.exit_code == 0
        mock_search_service.return_value.print_results.assert_called_once()


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_default_options(self, mock_settings):
        """Test serve with default options."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            result = runner.invoke(app, ["serve"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="127.0.0.1",
                port=8000,
                reload=False,
            )

    def test_serve_custom_host_port(self, mock_settings):
        """Test serve with custom host and port."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9000"])

            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="0.0.0.0",
                port=9000,
                reload=False,
            )

    def test_serve_with_reload(self, mock_settings):
        """Test serve with reload option."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            runner.invoke(app, ["serve", "--reload"])

            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="127.0.0.1",
                port=8000,
                reload=True,
            )

    def test_serve_short_options(self, mock_settings):
        """Test short options for serve command."""
        from dbs_vector.cli import app

        with patch("uvicorn.run") as mock_uvicorn:
            runner.invoke(app, ["serve", "-h", "0.0.0.0", "-p", "8080"])

            mock_uvicorn.assert_called_once_with(
                "dbs_vector.api.main:app",
                host="0.0.0.0",
                port=8080,
                reload=False,
            )


class TestBuildDependencies:
    """Tests for the _build_dependencies function."""

    def test_build_dependencies_success(self, mock_settings, mock_embedder, mock_store, mock_chunker, mock_mapper):
        """Test successful dependency building."""
        from dbs_vector.cli import _build_dependencies

        deps = _build_dependencies("md")

        assert deps.embedder is mock_embedder.return_value
        assert deps.store is mock_store.return_value
        mock_embedder.assert_called_once_with(
            model_name="test-model",
            max_token_length=512,
            dimension=384,
        )

    def test_build_dependencies_unknown_engine(self, mock_settings):
        """Test error for unknown engine."""
        from dbs_vector.cli import _build_dependencies

        with pytest.raises(ValueError, match="Unknown engine: 'unknown'"):
            _build_dependencies("unknown")

    def test_build_dependencies_chunker_with_max_chars(self, mock_settings, mock_embedder,
                                                        mock_store, mock_chunker, mock_mapper):
        """Test chunker gets max_chars when > 0."""
        from dbs_vector.cli import _build_dependencies

        _build_dependencies("md")

        mock_get_chunker, _ = mock_chunker
        mock_chunker_class = mock_get_chunker.return_value
        mock_chunker_class.assert_called_once_with(max_chars=1000)

    def test_build_dependencies_chunker_without_max_chars(self, mock_settings, mock_embedder,
                                                          mock_store, mock_chunker, mock_mapper):
        """Test chunker gets no max_chars when = 0 (SQL)."""
        from dbs_vector.cli import _build_dependencies

        _build_dependencies("sql")

        mock_get_chunker, _ = mock_chunker
        mock_chunker_class = mock_get_chunker.return_value
        mock_chunker_class.assert_called_once_with()  # No kwargs

    def test_store_initialized_correctly(self, mock_settings, mock_embedder, mock_store,
                                         mock_chunker, mock_mapper):
        """Test that store is initialized with correct parameters."""
        from dbs_vector.cli import _build_dependencies

        _build_dependencies("md")

        mock_store.assert_called_once_with(
            db_path="./test_db",
            table_name="md_table",
            vector_dimension=384,
            mapper=mock_mapper[1],
            nprobes=20,
        )


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_no_args_shows_help(self):
        """Test that running with no args shows help."""
        from dbs_vector.cli import app

        result = runner.invoke(app, [])

        # Typer exits with code 0 when showing help via no_args_is_help=True
        # The output should contain help text
        assert result.exit_code == 0 or "Usage:" in result.output
        assert "Usage:" in result.output


class TestHelpOutput:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Test main help output."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "dbs-vector" in result.output
        assert "--config-file" in result.output

    def test_ingest_help(self):
        """Test ingest command help."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "ingest" in result.output.lower()
        assert "--type" in result.output
        assert "--rebuild" in result.output

    def test_search_help(self):
        """Test search command help."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0
        assert "search" in result.output.lower()
        assert "--source" in result.output
        assert "--limit" in result.output

    def test_serve_help(self):
        """Test serve command help."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "serve" in result.output.lower()
        assert "--host" in result.output
        assert "--port" in result.output
