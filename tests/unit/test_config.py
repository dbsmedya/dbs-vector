"""Unit tests for the configuration module."""

import os
import tempfile
from pathlib import Path

import pytest

from dbs_vector.config import Settings, load_settings


class TestSettingsDefaults:
    """Tests for Settings default values."""

    def test_default_settings(self):
        """Test Settings with default values."""
        settings = Settings()

        assert settings.db_path == "./lancedb_dbs_vector"
        assert settings.nprobes == 20
        assert settings.memory_budget_gb is None
        assert settings.mlx_memory_limit_gb is None
        assert settings.mlx_cache_limit_gb is None
        assert settings.profiles == {}
        assert settings.engines == {}

    def test_settings_custom_values(self):
        """Test Settings with custom values."""
        settings = Settings(
            db_path="/custom/path",
            nprobes=50,
            memory_budget_gb=22.0,
            mlx_memory_limit_gb=16.0,
            mlx_cache_limit_gb=2.0,
        )

        assert settings.db_path == "/custom/path"
        assert settings.nprobes == 50
        assert settings.memory_budget_gb == 22.0
        assert settings.mlx_memory_limit_gb == 16.0
        assert settings.mlx_cache_limit_gb == 2.0


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_settings_with_no_config_file(self):
        """Test loading settings when config file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_config = os.path.join(tmp_dir, "non_existent.yaml")
            settings = load_settings(non_existent_config)

            assert isinstance(settings, Settings)
            assert settings.db_path == "./lancedb_dbs_vector"
            assert settings.engines == {}

    def test_load_settings_with_empty_config_file(self):
        """Test loading settings with empty config file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            Path(config_path).write_text("")

            settings = load_settings(config_path)

            assert isinstance(settings, Settings)
            assert settings.db_path == "./lancedb_dbs_vector"

    def test_load_settings_with_system_config(self):
        """Test loading settings with system configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            config_content = """
system:
  db_path: "./custom_db"
  nprobes: 100
"""
            Path(config_path).write_text(config_content)

            settings = load_settings(config_path)

            assert settings.db_path == "./custom_db"
            assert settings.nprobes == 100

    def test_load_settings_with_optional_mlx_limits(self):
        """MLX limits are optional system settings and accept fractional GiB."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            Path(config_path).write_text(
                """
system:
  memory_budget_gb: 22.0
  mlx_memory_limit_gb: 16.0
  mlx_cache_limit_gb: 1.5
"""
            )

            settings = load_settings(config_path)

            assert settings.memory_budget_gb == 22.0
            assert settings.mlx_memory_limit_gb == 16.0
            assert settings.mlx_cache_limit_gb == 1.5

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("mlx_memory_limit_gb", 0),
            ("mlx_memory_limit_gb", -1),
            ("mlx_cache_limit_gb", -1),
        ],
    )
    def test_load_settings_rejects_invalid_mlx_limits(self, key, value):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            Path(config_path).write_text(f"system:\n  {key}: {value}\n")

            with pytest.raises(ValueError):
                load_settings(config_path)

    def test_load_settings_with_engines(self):
        """Test loading settings with engine configurations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            config_content = """
system:
  db_path: "./test_db"

profiles:
  gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
  gemma-sql: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 64}

engines:
  md:
    description: "Markdown Engine"
    model: "gemma-bf16"
    table_name: "md_table"
    mapper_type: "document"
    chunker_type: "document"
    workflow: "md_search"
    tuning_profile: "gemma-md"
  sql:
    description: "SQL Engine"
    model: "gemma-bf16"
    table_name: "sql_table"
    mapper_type: "sql"
    chunker_type: "duckdb"
    workflow: "sql_clustering"
    tuning_profile: "gemma-sql"
"""
            Path(config_path).write_text(config_content)

            settings = load_settings(config_path)

            assert "md" in settings.engines
            assert "sql" in settings.engines

            md_config = settings.engines["md"]
            assert md_config.description == "Markdown Engine"
            assert md_config.model == "gemma-bf16"
            assert md_config.tuning_profile == "gemma-md"

            sql_config = settings.engines["sql"]
            assert sql_config.description == "SQL Engine"
            assert sql_config.model == "gemma-bf16"

    def test_load_settings_raises_for_unknown_system_keys(self):
        """Test that unknown system keys raise a ValueError with the allow-list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            config_content = """
system:
  db_path: "./custom_db"
  unknown_key: "should_be_ignored"
  another_unknown: 123
"""
            Path(config_path).write_text(config_content)

            with pytest.raises(ValueError, match="Unknown keys.*unknown_key"):
                load_settings(config_path)

    def test_load_settings_from_env_var(self, monkeypatch):
        """Test loading settings from config file specified in env var."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "env_config.yaml")
            config_content = """
system:
  db_path: "./env_db"
"""
            Path(config_path).write_text(config_content)

            monkeypatch.setenv("DBS_CONFIG_FILE", config_path)

            # Call without argument should use env var
            settings = load_settings()

            assert settings.db_path == "./env_db"

    def test_load_settings_env_var_overrides_default(self, monkeypatch):
        """Test that DBS_CONFIG_FILE env var overrides default config path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "from_env.yaml")
            Path(config_path).write_text('system:\n  db_path: "./from_env_db"\n')

            monkeypatch.setenv("DBS_CONFIG_FILE", config_path)

            settings = load_settings()

            assert settings.db_path == "./from_env_db"

    def test_load_settings_explicit_path_overrides_env(self, monkeypatch):
        """Test that explicit config_file parameter overrides env var."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_config = os.path.join(tmp_dir, "env.yaml")
            explicit_config = os.path.join(tmp_dir, "explicit.yaml")

            Path(env_config).write_text('system:\n  db_path: "./env_db"\n')
            Path(explicit_config).write_text('system:\n  db_path: "./explicit_db"\n')

            monkeypatch.setenv("DBS_CONFIG_FILE", env_config)

            settings = load_settings(explicit_config)

            assert settings.db_path == "./explicit_db"

    def test_load_settings_preserves_defaults_for_unspecified(self):
        """Test that unspecified settings keep their defaults."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            config_content = """
system:
  nprobes: 50
"""
            Path(config_path).write_text(config_content)

            settings = load_settings(config_path)

            assert settings.nprobes == 50
            # These should retain defaults
            assert settings.db_path == "./lancedb_dbs_vector"
            assert settings.memory_budget_gb is None

    def test_load_settings_with_null_data(self):
        """Test loading settings when yaml returns None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            # Just a comment, yaml.safe_load returns None
            Path(config_path).write_text("# Just a comment")

            settings = load_settings(config_path)

            # Should use defaults
            assert settings.db_path == "./lancedb_dbs_vector"
            assert settings.engines == {}


def test_every_settings_field_has_a_propagation_decision():
    """Adding a Settings field without deciding propagation must fail CI."""
    from dbs_vector.config import (
        _NOT_PROPAGATED_SETTINGS_FIELDS,  # noqa: PLC2701
        _PROPAGATED_SETTINGS_FIELDS,  # noqa: PLC2701
        Settings,
    )

    decided = _PROPAGATED_SETTINGS_FIELDS | _NOT_PROPAGATED_SETTINGS_FIELDS
    actual = set(Settings.model_fields)
    missing = actual - decided
    stale = decided - actual
    assert not missing, f"Settings fields with no propagation decision: {missing}"
    assert not stale, f"Propagation sets reference non-existent fields: {stale}"
