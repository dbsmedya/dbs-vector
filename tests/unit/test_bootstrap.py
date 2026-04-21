"""Unit tests for the bootstrap (DI factory) module."""

from unittest.mock import MagicMock, patch

import pytest

from dbs_vector.services.bootstrap import EngineDeps, build_dependencies


@pytest.fixture
def mock_settings():
    """Minimal settings fixture with one engine registered."""
    engine_config = MagicMock()
    engine_config.model_name = "test-model"
    engine_config.max_token_length = 2048
    engine_config.vector_dimension = 3
    engine_config.passage_prefix = ""
    engine_config.query_prefix = ""
    engine_config.mapper_type = "document"
    engine_config.chunker_type = "document"
    engine_config.table_name = "t"
    engine_config.workflow = "default"
    engine_config.chunker_kwargs.return_value = {"max_chars": 500}

    with patch("dbs_vector.services.bootstrap.settings") as s:
        s.engines = {"md": engine_config}
        s.db_path = "./test.db"
        s.nprobes = 10
        yield s


def test_build_dependencies_unknown_engine_raises(mock_settings):
    """Requesting an unconfigured engine raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        build_dependencies("no-such-engine")


def test_build_dependencies_returns_engine_deps(mock_settings):
    """Happy path: returns a fully-populated EngineDeps tuple."""
    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder") as MockEmbedder,
        patch("dbs_vector.services.bootstrap.LanceDBStore") as MockStore,
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockRegistry.get_mapper.return_value = MagicMock()
        MockRegistry.get_chunker.return_value = MagicMock()

        deps = build_dependencies("md")

    assert isinstance(deps, EngineDeps)
    assert deps.embedder == MockEmbedder.return_value
    assert deps.store == MockStore.return_value
    assert deps.workflow == "default"
