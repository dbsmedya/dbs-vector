from unittest.mock import MagicMock, patch

import dbs_vector.services.bootstrap as boot


def test_build_store_does_not_construct_embedder(monkeypatch):
    calls = {"embedder": 0, "store": 0}

    def _fake_embedder(*a, **k):
        calls["embedder"] += 1
        raise AssertionError("build_store must NOT construct an embedder")

    captured = {}

    class _FakeStore:
        def __init__(self, **kwargs):
            calls["store"] += 1
            captured.update(kwargs)

    engine_config = MagicMock()
    engine_config.model = "gemma-bf16"
    engine_config.mapper_type = "sql"
    engine_config.table_name = "query_vault"

    monkeypatch.setattr(boot, "MLXEmbedder", _fake_embedder)
    monkeypatch.setattr(boot, "LanceDBStore", _FakeStore)

    with patch("dbs_vector.services.bootstrap.settings") as mock_settings:
        mock_settings.engines = {"sql-api": engine_config}
        mock_settings.db_path = "./test.db"
        mock_settings.nprobes = 10

        store = boot.build_store("sql-api")     # sql-api exists in config.yaml

    assert calls["embedder"] == 0
    assert calls["store"] == 1
    assert captured["table_name"] == "query_vault"
    assert isinstance(store, _FakeStore)


def test_build_store_unknown_engine_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown engine"):
        boot.build_store("does-not-exist")
