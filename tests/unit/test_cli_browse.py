import json

import pyarrow as pa
import pytest
from typer.testing import CliRunner

import dbs_vector.cli as cli_mod
from dbs_vector.cli import app

runner = CliRunner()


class _FakeStore:
    def scan(self, columns=None):
        # Must include 'tables' (list column) for BrowseService._frames() to build t_by_table.
        return pa.table({
            "id": ["A", "B"],
            "calls": [10, 5],
            "tables": pa.array([["x"], ["y"]], type=pa.list_(pa.string())),
        })


@pytest.fixture
def patched(monkeypatch):
    class _E:
        def __init__(self, fam):
            self.resolved_family = fam
    monkeypatch.setattr(cli_mod.settings, "engines",
                        {"sql-api": _E("sql"), "md": _E("document")}, raising=False)
    monkeypatch.setattr(cli_mod, "_build_store", lambda name: _FakeStore())


def test_browse_table_output(patched):
    res = runner.invoke(app, ["browse", "--type", "sql-api",
                              "--sql", "SELECT id, calls FROM t ORDER BY calls DESC"])
    assert res.exit_code == 0
    assert "A" in res.stdout and "B" in res.stdout


def test_browse_json_output(patched):
    res = runner.invoke(app, ["browse", "--type", "sql-api",
                              "--sql", "SELECT id FROM t", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert {r["id"] for r in data} == {"A", "B"}


def test_browse_rejects_non_sql_engine(patched):
    res = runner.invoke(app, ["browse", "--type", "md", "--sql", "SELECT 1"])
    assert res.exit_code == 1
    assert "sql" in res.stdout.lower()
