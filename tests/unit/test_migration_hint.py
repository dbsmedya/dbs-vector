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
