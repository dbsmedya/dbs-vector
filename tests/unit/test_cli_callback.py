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
    from dbs_vector.config import _populate_singleton_from, settings

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
    from dbs_vector.config import Settings, _populate_singleton_from, settings

    new = _make_fake_new_settings()
    _populate_singleton_from(new)
    # Settings has no batch_size field after the schema change.
    assert "batch_size" not in Settings.model_fields
    # And the singleton instance does not have one either.
    assert not hasattr(settings, "batch_size")


def test_mcp_uses_global_config_when_no_subcommand_override(tmp_path, monkeypatch):
    """`dbs-vector mcp` (no -c at any level) uses the cwd's config.yaml.

    The CLI callback's `--config-file` default is the literal string
    "config.yaml", which resolves relative to the current working
    directory. We monkeypatch chdir to make tmp_path the cwd so the
    callback's default points at our test config.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'system:\n  db_path: "./global_db"\n'
        "profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n"
        "engines: {}\n"
    )
    monkeypatch.chdir(tmp_path)

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod

        called = {"yes": False}

        def fake_start():
            called["yes"] = True

        ctx.setattr(server_mod, "start_stdio_server", fake_start)
        result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0
    assert called["yes"] is True

    from dbs_vector.config import settings

    assert settings.db_path == "./global_db"


def test_mcp_uses_global_callback_config(tmp_path, monkeypatch):
    """`dbs-vector -c X mcp` uses global config X."""
    config_path = tmp_path / "global.yaml"
    config_path.write_text(
        'system:\n  db_path: "./X_db"\n'
        "profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n"
        "engines: {}\n"
    )

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod

        ctx.setattr(server_mod, "start_stdio_server", lambda: None)
        result = runner.invoke(app, ["-c", str(config_path), "mcp"])
    assert result.exit_code == 0

    from dbs_vector.config import settings

    assert settings.db_path == "./X_db"


def test_mcp_subcommand_config_overrides_global(tmp_path, monkeypatch):
    """`dbs-vector mcp -c Y` reloads from Y AFTER the global callback ran."""
    global_path = tmp_path / "global.yaml"
    global_path.write_text(
        'system:\n  db_path: "./GLOBAL_db"\n'
        "profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n"
        "engines: {}\n"
    )
    sub_path = tmp_path / "sub.yaml"
    sub_path.write_text(
        'system:\n  db_path: "./SUB_db"\n'
        "profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n"
        "engines: {}\n"
    )

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod

        ctx.setattr(server_mod, "start_stdio_server", lambda: None)
        result = runner.invoke(app, ["-c", str(global_path), "mcp", "-c", str(sub_path)])
    assert result.exit_code == 0

    from dbs_vector.config import settings

    assert settings.db_path == "./SUB_db"


def test_mcp_subcommand_config_sets_env_var(tmp_path, monkeypatch):
    """`dbs-vector mcp -c Y` re-exports DBS_CONFIG_FILE so spawned subprocesses
    inherit Y, not the global value."""
    import os

    global_path = tmp_path / "global.yaml"
    global_path.write_text(
        'system:\n  db_path: "./G"\n'
        "profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n"
        "engines: {}\n"
    )
    sub_path = tmp_path / "sub.yaml"
    sub_path.write_text(
        'system:\n  db_path: "./S"\n'
        "profiles:\n  p: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 1}\n"
        "engines: {}\n"
    )

    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    with monkeypatch.context() as ctx:
        from dbs_vector.mcp import server as server_mod

        ctx.setattr(server_mod, "start_stdio_server", lambda: None)
        runner.invoke(app, ["-c", str(global_path), "mcp", "-c", str(sub_path)])

    assert os.environ.get("DBS_CONFIG_FILE") == str(sub_path)


def test_serve_subcommand_no_longer_exists():
    """The serve CLI command was deleted in the MCP-only revision."""
    from typer.testing import CliRunner

    from dbs_vector.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code != 0
