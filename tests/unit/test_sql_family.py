"""Tests for SqlFamily run_search / format_results / make_handler — includes
the family-specific min_time filter."""

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from dbs_vector.core.models import SqlChunk, SqlSearchResult
from dbs_vector.mcp.families.sql import SqlFamily


def _make_sql_result() -> SqlSearchResult:
    return SqlSearchResult(
        chunk=SqlChunk(
            id="sql_0",
            text="SELECT * FROM t",
            raw_query="SELECT * FROM t WHERE id=1",
            source="prod_db",
            execution_time_ms=500.5,
            calls=2,
            content_hash="h",
            latest_ts=datetime.now(),
        ),
        score=None,
        distance=0.5678,
        is_fts_match=False,
    )


def test_run_search_passes_min_time_filter():
    fam = SqlFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="q", limit=2, source_filter=None, min_time=100.0)

    service.execute_query.assert_called_once_with("q", None, 2, extra_filters={"min_time": 100.0})


def test_run_search_omits_min_time_when_unset():
    fam = SqlFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="q", limit=2, source_filter=None)

    service.execute_query.assert_called_once_with("q", None, 2, extra_filters={})


def test_format_results_includes_execution_time_calls_and_raw_query():
    fam = SqlFamily()
    out = fam.format_results([_make_sql_result()], query="q")
    assert "Found 1 results for 'q'" in out
    assert "Source Database: prod_db" in out
    assert "Execution Time: 500.5ms (Calls: 2)" in out
    assert "SELECT * FROM t WHERE id=1" in out
    assert "0.5678" in out


def test_format_results_empty_returns_no_results_message():
    fam = SqlFamily()
    out = fam.format_results([], query="zzz")
    assert out == "No results found for query: 'zzz'"


def test_make_handler_signature_includes_min_time():
    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    sig = inspect.signature(handler)
    params = sig.parameters
    assert list(params) == ["query", "limit", "source_filter", "min_time"]
    assert params["min_time"].default is None


@pytest.mark.asyncio
async def test_make_handler_runs_search_and_formats(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [_make_sql_result()]
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    out = await handler(query="q", limit=1, min_time=200.0)

    service.execute_query.assert_called_once_with("q", None, 1, extra_filters={"min_time": 200.0})
    assert "Source Database: prod_db" in out


@pytest.mark.asyncio
async def test_make_handler_handles_exception(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.side_effect = Exception("DB down")
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    out = await handler(query="q")

    assert "Search execution failed: DB down" in out
