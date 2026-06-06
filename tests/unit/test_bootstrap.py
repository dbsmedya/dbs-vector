# tests/unit/test_bootstrap.py
from unittest.mock import MagicMock, patch

import pytest

from dbs_vector.services.bootstrap import EngineDeps, build_dependencies


@pytest.fixture
def mock_settings():
    """Minimal settings fixture: one engine + one profile registered."""
    engine_config = MagicMock()
    engine_config.model = "gemma-bf16"
    engine_config.mapper_type = "document"
    engine_config.chunker_type = "document"
    engine_config.table_name = "t"
    engine_config.workflow = "default"
    engine_config.tuning_profile = "test-profile"
    engine_config.passage_prefix = "P:"
    engine_config.query_prefix = "Q:"
    engine_config.exclusion_filters = []
    engine_config.chunker_kwargs.return_value = {"max_chars": 500}

    profile = MagicMock()
    profile.max_token_length = 2048
    profile.chunk_max_chars = 500
    profile.chunk_target_tokens = 512
    profile.chunk_max_tokens = 1024
    profile.batch_size = 64

    with patch("dbs_vector.services.bootstrap.settings") as s:
        s.engines = {"md": engine_config}
        s.profiles = {"test-profile": profile}
        s.db_path = "./test.db"
        s.nprobes = 10
        s.memory_budget_gb = 22.0
        yield s, engine_config, profile


def test_unknown_engine_raises(mock_settings):
    with pytest.raises(ValueError, match="Unknown engine"):
        build_dependencies("no-such-engine")


def test_returns_engine_deps_with_batch_size(mock_settings):
    _, _, profile = mock_settings
    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder"),
        patch("dbs_vector.services.bootstrap.LanceDBStore"),
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockRegistry.get_mapper.return_value = MagicMock()
        MockRegistry.get_chunker.return_value = MagicMock()
        deps = build_dependencies("md")
    assert isinstance(deps, EngineDeps)
    assert deps.batch_size == 64
    assert deps.workflow == "default"


def test_resolves_via_model_registry(mock_settings):
    """MLXEmbedder is constructed with model_name etc. from ModelRegistry, not engine."""
    _, engine_config, profile = mock_settings
    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder") as MockEmbedder,
        patch("dbs_vector.services.bootstrap.LanceDBStore"),
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockRegistry.get_mapper.return_value = MagicMock()
        MockRegistry.get_chunker.return_value = MagicMock()
        build_dependencies("md")
    _, kwargs = MockEmbedder.call_args
    assert kwargs["model_name"] == "mlx-community/embeddinggemma-300m-bf16"
    assert kwargs["max_token_length"] == 2048
    assert kwargs["dimension"] == 768
    assert kwargs["passage_prefix"] == "P:"
    assert kwargs["query_prefix"] == "Q:"
    assert kwargs["attention_mask_dtype"] == "float16"


def test_document_chunker_receives_token_budgets_and_length_fn(mock_settings):
    """Document chunker is built with profile token budgets and embedder.count_tokens."""
    _, engine_config, profile = mock_settings
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.count_tokens = MagicMock(return_value=42)

    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder") as MockEmbedder,
        patch("dbs_vector.services.bootstrap.LanceDBStore"),
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockEmbedder.return_value = mock_embedder_instance
        MockRegistry.get_mapper.return_value = MagicMock()

        # Use the real DocumentChunker so we can inspect its attributes
        from dbs_vector.infrastructure.chunking.document import DocumentChunker

        MockRegistry.get_chunker.return_value = DocumentChunker

        deps = build_dependencies("md")

    ch = deps.chunker
    assert ch.target_tokens == 512   # profile.chunk_target_tokens
    assert ch.max_tokens == 1024      # profile.chunk_max_tokens
    # length_fn is the embedder's count_tokens (not the default `len`)
    assert ch._len is mock_embedder_instance.count_tokens


def test_chunker_kwargs_not_called_for_document_engine(mock_settings):
    """bootstrap no longer calls chunker_kwargs for document engines; the
    document chunker is wired explicitly via token budgets / filters in Task 6."""
    _, engine_config, _ = mock_settings
    with (
        patch("dbs_vector.services.bootstrap.MLXEmbedder"),
        patch("dbs_vector.services.bootstrap.LanceDBStore"),
        patch("dbs_vector.services.bootstrap.ComponentRegistry") as MockRegistry,
    ):
        MockRegistry.get_mapper.return_value = MagicMock()
        MockRegistry.get_chunker.return_value = MagicMock()
        build_dependencies("md")
    # chunker_kwargs must NOT be called for document engines
    engine_config.chunker_kwargs.assert_not_called()
