import json

import pytest

from dbs_vector.services.initializer.render import (
    dump_mcp_json,
    merge_mcp_config,
    read_mcp_config,
)

EXISTING = {
    "mcpServers": {
        "oto-mysql": {
            "command": "/opt/homebrew/bin/mysql-mcp-server",
            "env": {"MYSQL_DSN": "user:pw@tcp(127.0.0.1:3306)/db"},
        }
    }
}


def test_merge_into_nothing_creates_the_server():
    merged = merge_mcp_config(None, "/repo", "/abs/config.yaml")
    assert list(merged["mcpServers"]) == ["dbs-vector"]


def test_config_file_precedes_the_subcommand():
    """--config-file must be the GLOBAL option, before `mcp`.

    Verified against the CLI: `dbs-vector mcp --config-file X` still runs the
    Typer callback first with its own default, so ./config.yaml in the spawned
    server's cwd is loaded AND VALIDATED before the subcommand override takes
    effect - a broken or unrelated config.yaml there tracebacks at cli.py:68.
    The global form bypasses that entirely.
    """
    merged = merge_mcp_config(None, "/repo", "/abs/config.yaml")
    args = merged["mcpServers"]["dbs-vector"]["args"]
    assert args == [
        "--directory",
        "/repo",
        "run",
        "dbs-vector",
        "--config-file",
        "/abs/config.yaml",
        "mcp",
    ]
    assert args.index("--config-file") < args.index("mcp")
    assert merged["mcpServers"]["dbs-vector"]["command"] == "uv"


def test_args_use_an_absolute_config_path():
    """A relative --config-file only works if the spawned server's cwd matches."""
    merged = merge_mcp_config(None, "/repo", "/abs/config.yaml")
    assert "/abs/config.yaml" in merged["mcpServers"]["dbs-vector"]["args"]


def test_allow_raw_queries_is_never_emitted():
    """It exposes literal PII from the SQL raw_query column and does nothing
    for a document engine."""
    merged = merge_mcp_config(None, "/repo", "/abs/config.yaml")
    assert "--allow-raw-queries" not in merged["mcpServers"]["dbs-vector"]["args"]


def test_other_servers_are_preserved_verbatim():
    merged = merge_mcp_config(EXISTING, "/repo", "/abs/config.yaml")
    assert merged["mcpServers"]["oto-mysql"] == EXISTING["mcpServers"]["oto-mysql"]
    assert "dbs-vector" in merged["mcpServers"]


def test_an_existing_dbs_vector_entry_is_replaced():
    existing = {"mcpServers": {"dbs-vector": {"command": "old", "args": ["stale"]}}}
    merged = merge_mcp_config(existing, "/repo", "/abs/config.yaml")
    assert merged["mcpServers"]["dbs-vector"]["command"] == "uv"
    assert "stale" not in merged["mcpServers"]["dbs-vector"]["args"]


def test_merge_does_not_mutate_the_input():
    original = json.loads(json.dumps(EXISTING))
    merge_mcp_config(EXISTING, "/repo", "/abs/config.yaml")
    assert EXISTING == original


def test_top_level_keys_other_than_mcpservers_survive():
    existing = {"mcpServers": {}, "someOtherKey": {"a": 1}}
    merged = merge_mcp_config(existing, "/repo", "/abs/config.yaml")
    assert merged["someOtherKey"] == {"a": 1}


def test_read_missing_file_returns_none(tmp_path):
    assert read_mcp_config(tmp_path / "absent.json") is None


def test_read_malformed_file_raises_rather_than_discarding_servers(tmp_path):
    bad = tmp_path / ".mcp.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be parsed"):
        read_mcp_config(bad)


def test_read_non_object_raises(tmp_path):
    bad = tmp_path / ".mcp.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_mcp_config(bad)


def test_read_a_directory_raises_valueerror(tmp_path):
    """Every failure must be ValueError - run_init's recovery prompt and the
    CLI handler catch nothing else. A bare read_text() raises
    IsADirectoryError here, which would reach the user as a traceback."""
    as_dir = tmp_path / ".mcp.json"
    as_dir.mkdir()
    with pytest.raises(ValueError, match="could not be read"):
        read_mcp_config(as_dir)


def test_read_invalid_utf8_raises_valueerror(tmp_path):
    """UnicodeDecodeError is not a JSONDecodeError and is not an OSError."""
    bad = tmp_path / ".mcp.json"
    bad.write_bytes(b'\xff\xfe{"mcpServers": {}}')
    with pytest.raises(ValueError, match="could not be read"):
        read_mcp_config(bad)


def test_read_unreadable_file_raises_valueerror(tmp_path):
    bad = tmp_path / ".mcp.json"
    bad.write_text("{}", encoding="utf-8")
    bad.chmod(0o000)
    try:
        with pytest.raises(ValueError, match="could not be read"):
            read_mcp_config(bad)
    finally:
        bad.chmod(0o600)


def test_read_rejects_a_non_object_mcpservers(tmp_path):
    """`{"mcpServers": []}` parses, but dict.setdefault would return the list
    and `servers["dbs-vector"] = ...` would raise TypeError - which no caller
    catches. Reject at read time."""
    bad = tmp_path / ".mcp.json"
    bad.write_text('{"mcpServers": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        read_mcp_config(bad)


def test_merge_survives_a_non_object_mcpservers(tmp_path):
    """Defence in depth: merge stays total even if a caller bypasses read."""
    merged = merge_mcp_config({"mcpServers": []}, "/repo", "/abs/config.yaml")
    assert merged["mcpServers"]["dbs-vector"]["command"] == "uv"


def test_dump_is_stable_and_newline_terminated():
    text = dump_mcp_json(merge_mcp_config(None, "/repo", "/abs/config.yaml"))
    assert text.endswith("\n")
    assert json.loads(text)["mcpServers"]["dbs-vector"]["command"] == "uv"
