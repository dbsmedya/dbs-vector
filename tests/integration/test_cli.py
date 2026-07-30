"""Integration tests for CLI commands."""

import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from dbs_vector.config import EngineConfig, TuningProfile

# Need to import after setting up mocks
runner = CliRunner()


@pytest.fixture
def mock_settings():
    """Create mock settings with engines on the new schema.

    Both engines use the registered `gemma-bf16` ModelContract so we don't
    need fresh ModelRegistry registrations per test. Sizing knobs come from
    profiles.

    Patches both `dbs_vector.cli.settings` (used for CLI-level engine validation)
    and `dbs_vector.services.bootstrap.settings` (used by the DI factory).
    """
    profiles = {
        "test-md-profile": TuningProfile(
            max_token_length=512,
            chunk_max_chars=1000,
            chunk_target_tokens=512,
            chunk_max_tokens=1024,
            batch_size=64,
        ),
        "test-sql-profile": TuningProfile(max_token_length=256, chunk_max_chars=0, batch_size=64),
    }
    engines = {
        "md": EngineConfig(
            description="Markdown Engine",
            model="gemma-bf16",
            mapper_type="document",
            chunker_type="document",
            table_name="md_table",
            workflow="test_md",
            passage_prefix="passage: ",
            query_prefix="query: ",
            tuning_profile="test-md-profile",
        ),
        "sql": EngineConfig(
            description="SQL Engine",
            model="gemma-bf16",
            mapper_type="sql",
            chunker_type="sql",
            table_name="sql_table",
            workflow="test_sql",
            passage_prefix="",
            query_prefix="",
            tuning_profile="test-sql-profile",
        ),
    }
    with (
        patch("dbs_vector.cli.settings") as cli_mock,
        patch("dbs_vector.services.bootstrap.settings") as bootstrap_mock,
    ):
        for mock in (cli_mock, bootstrap_mock):
            mock.engines = engines
            mock.profiles = profiles
            mock.db_path = "./test_db"
            mock.nprobes = 20
            mock.memory_budget_gb = 22.0
        yield cli_mock


@pytest.fixture
def mock_embedder():
    """Mock MLXEmbedder to avoid loading actual models."""
    with patch("dbs_vector.services.bootstrap.MLXEmbedder") as mock:
        mock_instance = MagicMock()
        mock_instance.dimension = 384
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_store():
    """Mock LanceDBStore."""
    with patch("dbs_vector.services.bootstrap.LanceDBStore") as mock:
        mock_instance = MagicMock()
        mock_instance.mapper = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_chunker():
    """Mock chunker classes."""
    with patch("dbs_vector.services.bootstrap.ComponentRegistry.get_chunker") as mock_get:
        mock_chunker_class = MagicMock()
        mock_chunker_instance = MagicMock()
        mock_chunker_class.return_value = mock_chunker_instance
        mock_get.return_value = mock_chunker_class
        yield mock_get, mock_chunker_instance


@pytest.fixture
def mock_mapper():
    """Mock mapper classes."""
    with patch("dbs_vector.services.bootstrap.ComponentRegistry.get_mapper") as mock_get:
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
    """Mock the build_search_service factory used by the CLI's search command."""
    with patch("dbs_vector.cli.build_search_service") as mock:
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

    def test_ingest_basic(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
    ):
        """Test basic ingest command."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "docs/*.md"])

        assert result.exit_code == 0
        mock_ingestion_service.assert_called_once()
        call_args = mock_ingestion_service.call_args.args
        assert call_args[3] == "test_md"
        mock_ingestion_service.return_value.ingest_directory.assert_called_once_with(
            "docs/*.md", rebuild=False
        )

    def test_ingest_with_engine_type(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
    ):
        """Test ingest with specific engine type."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "queries.json", "--type", "sql"])

        assert result.exit_code == 0
        # Verify SQL engine was used (via embedder call with sql model)
        mock_embedder.assert_called_once()
        call_kwargs = mock_embedder.call_args.kwargs
        assert call_kwargs["model_name"] == "mlx-community/embeddinggemma-300m-bf16"

    def test_ingest_unknown_engine(self, mock_settings):
        """Test ingest with unknown engine type."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "path", "--type", "unknown"])

        assert result.exit_code == 1
        assert "Unknown engine type" in result.output

    def test_ingest_rebuild_without_force(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
    ):
        """Test rebuild flag triggers confirmation."""
        from dbs_vector.cli import app

        # Without --force, should prompt for confirmation
        result = runner.invoke(app, ["ingest", "path", "--rebuild"], input="n\n")

        # Should abort when user says no
        assert result.exit_code != 0 or "Aborted" in result.output

    def test_ingest_rebuild_with_force(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
    ):
        """Test rebuild with force flag bypasses confirmation."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "path", "--rebuild", "--force"])

        assert result.exit_code == 0
        mock_ingestion_service.return_value.ingest_directory.assert_called_once_with(
            "path", rebuild=True
        )

    def test_ingest_short_options(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
    ):
        """Test short options for ingest command."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "path", "-t", "sql", "-r", "-f"])

        assert result.exit_code == 0
        mock_ingestion_service.return_value.ingest_directory.assert_called_once_with(
            "path", rebuild=True
        )

    def test_ingest_without_path_uses_configured_roots(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
        tmp_path,
    ):
        """No positional path -> ingest every configured root in ONE run."""
        from dbs_vector.cli import app

        roots = [str(tmp_path / "one"), str(tmp_path / "two")]
        mock_settings.engines["md"] = mock_settings.engines["md"].model_copy(
            update={"paths": roots}
        )

        result = runner.invoke(app, ["ingest", "--type", "md"])

        assert result.exit_code == 0
        mock_ingestion_service.return_value.ingest_directory.assert_called_once_with(
            roots, rebuild=False
        )

    def test_ingest_without_path_and_without_roots_is_a_usage_error(
        self, mock_settings, mock_embedder, mock_store, mock_chunker, mock_mapper
    ):
        from dbs_vector.cli import app

        result = runner.invoke(app, ["ingest", "--type", "md"])

        assert result.exit_code == 1
        assert "no `paths:` configured" in result.output

    def test_explicit_path_outside_configured_roots_prints_a_notice(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
        tmp_path,
    ):
        from dbs_vector.cli import app
        from dbs_vector.config import WatchConfig

        vault = tmp_path / "vault"
        vault.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        mock_settings.engines["md"] = mock_settings.engines["md"].model_copy(
            update={"paths": [str(vault)], "watch": WatchConfig(enabled=True)}
        )

        result = runner.invoke(app, ["ingest", str(outside), "--type", "md"])

        assert result.exit_code == 0
        assert "will not be watched" in result.output

    def test_explicit_path_inside_configured_roots_prints_no_notice(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_ingestion_service,
        tmp_path,
    ):
        from dbs_vector.cli import app
        from dbs_vector.config import WatchConfig

        vault = tmp_path / "vault"
        (vault / "sub").mkdir(parents=True)
        mock_settings.engines["md"] = mock_settings.engines["md"].model_copy(
            update={"paths": [str(vault)], "watch": WatchConfig(enabled=True)}
        )

        result = runner.invoke(app, ["ingest", str(vault / "sub"), "--type", "md"])

        assert result.exit_code == 0
        assert "will not be watched" not in result.output


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_basic(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_search_service,
    ):
        """Test basic search command."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "test query"])

        assert result.exit_code == 0
        mock_search_service.return_value.execute_query.assert_called_once()
        call_args = mock_search_service.return_value.execute_query.call_args
        assert call_args[0][0] == "test query"  # query
        assert call_args[1]["source_filter"] is None
        assert call_args[1]["limit"] == 5

    def test_search_with_options(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_search_service,
    ):
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

    def test_search_sql_with_min_time(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_search_service,
    ):
        """Test SQL search with min_time filter."""
        from dbs_vector.cli import app

        result = runner.invoke(
            app,
            ["search", "slow query", "--type", "sql", "--min-time", "100.5"],
        )

        assert result.exit_code == 0
        call_args = mock_search_service.return_value.execute_query.call_args
        assert call_args[1]["extra_filters"] == {"min_time": 100.5}

    def test_search_md_ignores_min_time(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_search_service,
    ):
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

    def test_search_forwards_similarity_flags(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_search_service,
    ):
        """--min-similarity and --no-similarity-floor forward to execute_query."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "query", "--min-similarity", "0.4"])

        assert result.exit_code == 0
        call_args = mock_search_service.return_value.execute_query.call_args
        assert call_args[1]["min_similarity"] == 0.4
        assert call_args[1]["disable_similarity_floor"] is False

        result = runner.invoke(app, ["search", "query", "--no-similarity-floor"])

        assert result.exit_code == 0
        call_args = mock_search_service.return_value.execute_query.call_args
        assert call_args[1]["min_similarity"] is None
        assert call_args[1]["disable_similarity_floor"] is True

    def test_search_reports_value_error_cleanly(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_search_service,
    ):
        """A ValueError from execute_query (e.g. out-of-range --limit) becomes
        a clean CLI error message, not an uncaught traceback."""
        from dbs_vector.cli import app

        mock_search_service.return_value.execute_query.side_effect = ValueError(
            "limit must be within [1, 100]; got 200"
        )

        result = runner.invoke(app, ["search", "query", "--limit", "200"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "limit must be within" in result.output

    def test_search_unknown_engine(self, mock_settings):
        """Test search with unknown engine type."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "query", "--type", "unknown"])

        assert result.exit_code == 1
        assert "Unknown engine type" in result.output

    def test_search_results_printed(
        self,
        mock_settings,
        mock_embedder,
        mock_store,
        mock_chunker,
        mock_mapper,
        mock_search_service,
    ):
        """Test that search results are printed."""
        from dbs_vector.cli import app

        result = runner.invoke(app, ["search", "test"])

        assert result.exit_code == 0
        mock_search_service.return_value.print_results.assert_called_once()


class TestBuildDependencies:
    """Tests for the _build_dependencies function."""

    def test_build_dependencies_success(
        self, mock_settings, mock_embedder, mock_store, mock_chunker, mock_mapper
    ):
        """Test successful dependency building."""
        from dbs_vector.cli import _build_dependencies

        deps = _build_dependencies("md")

        assert deps.embedder is mock_embedder.return_value
        assert deps.store is mock_store.return_value
        mock_embedder.assert_called_once_with(
            model_name="mlx-community/embeddinggemma-300m-bf16",
            max_token_length=512,  # from test-md-profile
            dimension=768,  # from gemma-bf16 contract
            passage_prefix="passage: ",
            query_prefix="query: ",
            attention_mask_dtype="float16",  # from gemma-bf16 contract
        )

    def test_build_dependencies_unknown_engine(self, mock_settings):
        """Test error for unknown engine."""
        from dbs_vector.cli import _build_dependencies

        with pytest.raises(ValueError, match="Unknown engine: 'unknown'"):
            _build_dependencies("unknown")

    def test_build_dependencies_document_chunker_wired_with_token_budgets(
        self, mock_settings, mock_embedder, mock_store, mock_chunker, mock_mapper
    ):
        """Test document chunker is built with token budgets, length_fn, and filters
        (not via chunker_kwargs) — the explicit wiring path from Task 6."""
        from dbs_vector.cli import _build_dependencies

        _build_dependencies("md")

        mock_get_chunker, _ = mock_chunker
        mock_chunker_class = mock_get_chunker.return_value
        call_kwargs = mock_chunker_class.call_args.kwargs
        # Token budgets flow from the profile (test-md-profile has non-zero values).
        assert call_kwargs["max_chars"] == 1000
        assert call_kwargs["target_tokens"] == 512
        assert call_kwargs["max_tokens"] == 1024
        assert "length_fn" in call_kwargs
        assert "filters" in call_kwargs
        # length_fn must be the embedder's count_tokens, not built-in len
        assert call_kwargs["length_fn"] is mock_embedder.return_value.count_tokens

    def test_build_dependencies_chunker_without_max_chars(
        self, mock_settings, mock_embedder, mock_store, mock_chunker, mock_mapper
    ):
        """Test SQL chunker is built via chunker_kwargs (no explicit token-budget wiring)."""
        from dbs_vector.cli import _build_dependencies

        _build_dependencies("sql")

        mock_get_chunker, _ = mock_chunker
        mock_chunker_class = mock_get_chunker.return_value
        mock_chunker_class.assert_called_once_with()  # No kwargs

    def test_store_initialized_correctly(
        self, mock_settings, mock_embedder, mock_store, mock_chunker, mock_mapper
    ):
        """Test that store is initialized with correct parameters."""
        from dbs_vector.cli import _build_dependencies

        _build_dependencies("md")

        mock_store.assert_called_once_with(
            db_path="./test_db",
            table_name="md_table",
            vector_dimension=768,  # from gemma-bf16 contract
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
