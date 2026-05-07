# tests/unit/test_cli_callback.py
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(
    args: list[str], cwd: Path, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "dbs_vector.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_help_works_with_no_config(tmp_path):
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0
    assert "dbs-vector" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_help_works_with_malformed_config(tmp_path):
    (tmp_path / "config.yaml").write_text("system: : :\nthis: is: not: yaml")
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0


def test_help_works_with_old_schema_config(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
system:
  db_path: "./lance"
  batch_size: 64
engines:
  md:
    description: "Old"
    model_name: "mlx-community/embeddinggemma-300m-bf16"
    vector_dimension: 768
    max_token_length: 2048
    chunk_max_chars: 1000
    table_name: "t"
    mapper_type: "document"
    chunker_type: "document"
    workflow: "w"
"""
    )
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0


def test_version_works_with_malformed_config(tmp_path):
    (tmp_path / "config.yaml").write_text("not: valid: yaml: at: all: : :")
    result = _run_cli(["--version"], cwd=tmp_path)
    assert result.returncode == 0
    assert "dbs-vector" in result.stdout.lower()


# Direct unit tests of the singleton-mutation helper.


def _make_fake_new_settings():
    fake = MagicMock()
    fake.db_path = "/tmp/lance"
    fake.nprobes = 30
    fake.engines = {"md": object()}
    fake.profiles = {"gemma-md": object()}
    fake.memory_budget_gb = 22.0
    fake.log_level = "DEBUG"
    fake.log_serialize = True
    return fake


def test_populate_singleton_copies_profiles_and_memory_budget():
    """Helper extracted from main() callback must copy profiles + memory_budget_gb."""
    from dbs_vector.cli import _populate_singleton_from
    from dbs_vector.config import settings

    new = _make_fake_new_settings()
    _populate_singleton_from(new)

    assert settings.db_path == "/tmp/lance"
    assert settings.nprobes == 30
    assert settings.engines == new.engines
    assert settings.profiles == new.profiles
    assert settings.memory_budget_gb == 22.0
    assert settings.log_level == "DEBUG"
    assert settings.log_serialize is True


def test_populate_singleton_does_not_set_legacy_batch_size():
    """The new schema has no Settings.batch_size; the helper must not set it."""
    from dbs_vector.cli import _populate_singleton_from
    from dbs_vector.config import Settings, settings

    new = _make_fake_new_settings()
    _populate_singleton_from(new)
    # Settings has no batch_size field after the schema change.
    assert "batch_size" not in Settings.model_fields
    # And the singleton instance does not have one either.
    assert not hasattr(settings, "batch_size")
