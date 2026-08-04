"""mcp --http wiring: mode dispatch and the flag-combination guard."""

from typer.testing import CliRunner

from dbs_vector.cli import app

runner = CliRunner()


def test_http_with_allow_raw_queries_is_refused():
    result = runner.invoke(app, ["mcp", "--http", "--allow-raw-queries"])
    assert result.exit_code == 2
    assert "X-DBS-Allow-Raw-Queries" in result.output


def test_http_dispatches_to_http_server(monkeypatch):
    called = []
    monkeypatch.setattr("dbs_vector.mcp.server.start_http_server", lambda: called.append(True))
    result = runner.invoke(app, ["mcp", "--http"])
    assert called == [True]
    assert result.exit_code == 0


def test_default_still_dispatches_to_stdio(monkeypatch):
    called = []
    monkeypatch.setattr(
        "dbs_vector.mcp.server.start_stdio_server",
        lambda allow_raw_queries=False: called.append(allow_raw_queries),
    )
    result = runner.invoke(app, ["mcp"])
    assert called == [False]
    assert result.exit_code == 0
