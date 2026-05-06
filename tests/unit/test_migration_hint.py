import textwrap

import pytest

from dbs_vector.config import load_settings


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


def test_legacy_system_batch_size_raises_migration_hint(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        system:
          db_path: "./lance"
          batch_size: 8
        """,
    )
    with pytest.raises(ValueError, match="Legacy keys found.*batch_size"):
        load_settings(yaml_path)


def test_unknown_system_key_raises_with_allowlist(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        system:
          db_path: "./lance"
          unknown_key: true
        """,
    )
    with pytest.raises(ValueError, match="Unknown keys.*unknown_key"):
        load_settings(yaml_path)


def test_known_system_keys_pass_through(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        system:
          db_path: "/tmp/lance"
          nprobes: 30
          memory_budget_gb: 16.0
        """,
    )
    s = load_settings(yaml_path)
    assert s.db_path == "/tmp/lance"
    assert s.nprobes == 30
    assert s.memory_budget_gb == 16.0


def test_known_system_keys_matches_settings_fields():
    """Drift guard: _KNOWN_SYSTEM_KEYS must equal the public Settings fields
    minus the dict-blocks (`profiles`, `engines`). If someone adds a new
    field to Settings without updating the allow-list, this fails immediately
    rather than rejecting valid config at runtime."""
    from dbs_vector.config import _KNOWN_SYSTEM_KEYS, Settings

    declared = set(Settings.model_fields) - {"profiles", "engines"}
    assert _KNOWN_SYSTEM_KEYS == declared, (
        f"_KNOWN_SYSTEM_KEYS drifted from Settings fields. "
        f"Missing from allow-list: {declared - _KNOWN_SYSTEM_KEYS}. "
        f"Stale in allow-list: {_KNOWN_SYSTEM_KEYS - declared}."
    )


def test_legacy_engine_field_raises_migration_hint(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "x"
            model_name: "mlx-community/embeddinggemma-300m-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """,
    )
    with pytest.raises(ValueError, match="Legacy per-engine fields found.*model_name"):
        load_settings(yaml_path)


def test_missing_required_engine_fields_raises_migration_hint(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "x"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
        """,
    )
    with pytest.raises(ValueError, match="Missing new required fields.*model"):
        load_settings(yaml_path)


def test_genuine_validation_error_propagates_unchanged(tmp_path):
    """A genuine validation bug in new schema should NOT be wrapped as migration."""
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: 12345  # wrong type — should be string
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """,
    )
    with pytest.raises(ValueError, match="mapper_type"):
        load_settings(yaml_path)
    # Ensure the migration hint phrase is NOT present
    try:
        load_settings(yaml_path)
    except ValueError as e:
        assert "Legacy per-engine fields" not in str(e)
        assert "Missing new required fields" not in str(e)


def test_valid_new_schema_loads(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        """
        profiles:
          gemma-md: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}
        engines:
          md:
            description: "Gemma markdown"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "md_search"
            passage_prefix: "title: none | text: "
            query_prefix: "task: search result | query: "
            tuning_profile: "gemma-md"
        """,
    )
    s = load_settings(yaml_path)
    assert s.engines["md"].model == "gemma-bf16"
    assert s.engines["md"].tuning_profile == "gemma-md"
    assert s.engines["md"].passage_prefix == "title: none | text: "
