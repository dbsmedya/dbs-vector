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
