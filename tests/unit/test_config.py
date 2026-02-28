"""Unit tests for the configuration module."""

import os
import tempfile
from pathlib import Path

from dbs_vector.config import EngineConfig, Settings, load_settings


class TestEngineConfig:
    """Tests for EngineConfig model."""

    def test_engine_config_creation(self):
        """Test creating an EngineConfig with all required fields."""
        config = EngineConfig(
            description="Test Engine",
            model_name="test-model",
            vector_dimension=384,
            max_token_length=512,
            table_name="test_table",
            mapper_type="document",
            chunker_type="document",
            chunk_max_chars=1000,
        )

        assert config.description == "Test Engine"
        assert config.model_name == "test-model"
        assert config.vector_dimension == 384
        assert config.max_token_length == 512
        assert config.table_name == "test_table"
        assert config.mapper_type == "document"
        assert config.chunker_type == "document"
        assert config.chunk_max_chars == 1000


class TestSettingsDefaults:
    """Tests for Settings default values."""

    def test_default_settings(self):
        """Test Settings with default values."""
        settings = Settings()

        assert settings.db_path == "./lancedb_dbs_vector"
        assert settings.batch_size == 64
        assert settings.nprobes == 20
        assert settings.engines == {}

    def test_settings_custom_values(self):
        """Test Settings with custom values."""
        settings = Settings(
            db_path="/custom/path",
            batch_size=128,
            nprobes=50,
        )

        assert settings.db_path == "/custom/path"
        assert settings.batch_size == 128
        assert settings.nprobes == 50


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
  batch_size: 128
  nprobes: 100
"""
            Path(config_path).write_text(config_content)

            settings = load_settings(config_path)

            assert settings.db_path == "./custom_db"
            assert settings.batch_size == 128
            assert settings.nprobes == 100

    def test_load_settings_with_engines(self):
        """Test loading settings with engine configurations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            config_content = """
system:
  db_path: "./test_db"

engines:
  md:
    description: "Markdown Engine"
    model_name: "test-model"
    vector_dimension: 384
    max_token_length: 512
    table_name: "md_table"
    mapper_type: "document"
    chunker_type: "document"
    chunk_max_chars: 1000
  sql:
    description: "SQL Engine"
    model_name: "sql-model"
    vector_dimension: 768
    max_token_length: 256
    table_name: "sql_table"
    mapper_type: "sql"
    chunker_type: "sql"
    chunk_max_chars: 0
"""
            Path(config_path).write_text(config_content)

            settings = load_settings(config_path)

            assert "md" in settings.engines
            assert "sql" in settings.engines

            md_config = settings.engines["md"]
            assert md_config.description == "Markdown Engine"
            assert md_config.model_name == "test-model"
            assert md_config.vector_dimension == 384

            sql_config = settings.engines["sql"]
            assert sql_config.description == "SQL Engine"
            assert sql_config.vector_dimension == 768

    def test_load_settings_ignores_unknown_system_keys(self):
        """Test that unknown system keys are ignored gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.yaml")
            config_content = """
system:
  db_path: "./custom_db"
  unknown_key: "should_be_ignored"
  another_unknown: 123
"""
            Path(config_path).write_text(config_content)

            settings = load_settings(config_path)

            assert settings.db_path == "./custom_db"
            # Should not raise an error for unknown keys

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
  batch_size: 256
"""
            Path(config_path).write_text(config_content)

            settings = load_settings(config_path)

            assert settings.batch_size == 256
            # These should retain defaults
            assert settings.db_path == "./lancedb_dbs_vector"
            assert settings.nprobes == 20

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
