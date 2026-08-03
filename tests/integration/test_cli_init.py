"""Integration tests for `dbs-vector init`."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from dbs_vector.cli import app

runner = CliRunner()


def _answers(tmp_path, install_dir: str | None = None) -> str:
    """stdin for one full interview, in prompt order."""
    return (
        "\n".join(
            [
                "docs",  # Engine name
                "granite-r2",  # Embedding model
                "medium",  # Chunk granularity
                str(tmp_path / "notes"),  # first path
                "",  # blank ends the path loop
                "",  # no extra ignore patterns
                "excalidraw,compressed_json",  # exclusion filters
                "n",  # watch
                str(tmp_path / "lancedb"),  # db_path
                install_dir or str(tmp_path / "repo"),  # install dir
                str(tmp_path / "config.yaml"),  # config destination
                str(tmp_path / ".mcp.json"),  # mcp destination
            ]
        )
        + "\n"
    )


def _prepare(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")


def test_init_writes_a_loadable_config(tmp_path):
    _prepare(tmp_path)
    result = runner.invoke(app, ["init"], input=_answers(tmp_path))
    assert result.exit_code == 0, result.output

    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert list(config["engines"]) == ["docs"]

    mcp = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["dbs-vector"]["command"] == "uv"


def test_a_broken_config_really_does_raise_without_the_bypass(tmp_path):
    """Guard for the test below: prove the fixture config is genuinely fatal.

    `engines: [not, a, mapping]` is NOT - load_settings only processes
    `engines` when it is a dict, so a list is silently ignored and the test
    would pass with or without the bypass. An unknown system key raises.
    """
    from dbs_vector.config import load_settings

    broken = tmp_path / "broken.yaml"
    broken.write_text("system:\n  bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_settings(str(broken), validate=True)


def test_init_runs_with_a_malformed_config_present(tmp_path, monkeypatch):
    """The callback must not validate an existing config before init runs."""
    _prepare(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "broken.yaml").write_text("system:\n  bogus_key: 1\n", encoding="utf-8")

    result = runner.invoke(app, ["--config-file", "broken.yaml", "init"], input=_answers(tmp_path))
    assert result.exit_code == 0, result.output


def test_init_prints_a_runnable_next_step(tmp_path):
    """`ingest` has no --all flag and defaults to the `md` engine, so the hint
    must name the generated engine via --type or it fails on a fresh setup."""
    _prepare(tmp_path)
    result = runner.invoke(app, ["init"], input=_answers(tmp_path))
    assert "ingest --type docs" in result.output
    assert "--all" not in result.output


def test_an_illegal_engine_name_re_prompts_at_the_terminal(tmp_path):
    """Spec §7 row 1 is re-ask, not abort: a typo must be recoverable."""
    _prepare(tmp_path)
    result = runner.invoke(app, ["init"], input="Bad Name\n" + _answers(tmp_path))
    assert result.exit_code == 0, result.output
    assert "must match" in result.output
    assert (tmp_path / "config.yaml").exists()


def test_a_terminal_refusal_exits_one_without_a_traceback(tmp_path):
    """An install directory with no pyproject.toml is genuinely unrecoverable
    (unlike a bad engine name), so it must exit 1 with a readable message."""
    _prepare(tmp_path)
    result = runner.invoke(
        app, ["init"], input=_answers(tmp_path, install_dir=str(tmp_path / "notes"))
    )
    assert result.exit_code == 1
    assert "pyproject.toml" in result.output
    assert "Traceback" not in result.output
