# tests/unit/test_bootstrap.py
from unittest.mock import MagicMock, patch

import pytest

from dbs_vector.services.bootstrap import build_dependencies


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
    engine_config.passage_prefix = ""
    engine_config.query_prefix = ""
    engine_config.chunker_kwargs.return_value = {"max_chars": 500}

    profile = MagicMock()
    profile.max_token_length = 2048
    profile.chunk_max_chars = 500
    profile.batch_size = 64

    with patch("dbs_vector.services.bootstrap.settings") as s:
        s.engines = {"md": engine_config}
        s.profiles = {"test-profile": profile}
        s.db_path = "./test.db"
        s.nprobes = 10
        s.memory_budget_gb = 22.0
        yield s


def test_build_dependencies_unknown_engine_raises(mock_settings):
    with pytest.raises(ValueError, match="Unknown engine"):
        build_dependencies("no-such-engine")
