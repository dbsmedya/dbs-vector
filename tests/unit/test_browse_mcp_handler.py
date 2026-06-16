from datetime import UTC, datetime

import pyarrow as pa
import pytest

import dbs_vector.mcp.state as state
from dbs_vector.mcp.families.sql import (
    _RAW_QUERY_DISPLAY_LIMIT,
    _fmt_cell,
    format_browse,
    SqlFamily,
)
from dbs_vector.services.browse import BrowseResult


class _FakeStore:
    def __init__(self, table, raise_exc=None):
        self._t = table
        self._raise = raise_exc

    def scan(self, columns=None):
        if self._raise is not None:
            raise self._raise
        return self._t


class _FakeService:
    def __init__(self, store):
        self.vector_store = store


def _table() -> pa.Table:
    return pa.table(
        {
            "id": ["A", "B"],
            "content_hash": ["h1", "h2"],
            "text": ["s1", "s2"],
            "raw_query": ["RAW-A-SECRET", "RAW-B"],
            "source": ["db1", "db1"],
            "user": ["alice", "bob"],
            "host": ["h1", "h2"],
            "tables": [["orders"], ["items"]],
            "calls": [10, 5],
            "execution_time_ms": [100.0, 50.0],
            "lock_time_sec": [1.0, None],
            "rows_examined": [50, 20],
            "rows_sent": [5, 2],
            "latest_ts": [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)],
        }
    )


@pytest.fixture
def wired(monkeypatch):
    def _wire(store):
        monkeypatch.setitem(state._services, "sql-api", _FakeService(store))

    return _wire


@pytest.mark.asyncio
async def test_handler_groups_and_formats(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(group_by="user", order_by="execution_time_ms:desc")
    assert "alice" in out
    assert "Showing" in out


@pytest.mark.asyncio
async def test_handler_validation_error_returned_verbatim(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(select="raw_query")  # gated off
    assert "not exposed" in out
    assert "raw query text" in out.lower()


@pytest.mark.asyncio
async def test_handler_infra_exception_is_sanitized(wired):
    secret_path = "/Users/op/.ssh/id_rsa"
    wired(_FakeStore(_table(), raise_exc=RuntimeError(f"lance IO error at {secret_path}")))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(order_by="execution_time_ms:desc")
    assert secret_path not in out
    assert "browse execution failed" in out.lower()


@pytest.mark.asyncio
async def test_handler_raw_query_gated_off_not_in_output(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_browse_handler("sql-api", allow_raw_queries=False)
    out = await handler(order_by="execution_time_ms:desc")
    assert "RAW-A-SECRET" not in out


def test_fmt_cell_truncates_long_string():
    long = "x" * (_RAW_QUERY_DISPLAY_LIMIT + 500)
    out = _fmt_cell(long)
    assert len(out) < len(long)
    assert "more chars elided" in out


def test_fmt_cell_short_string_unchanged():
    assert _fmt_cell("SELECT 1") == "SELECT 1"


def test_format_browse_truncates_raw_query_cell():
    big = "SELECT " + "a," * 3000  # > 2000 chars
    result = BrowseResult(
        rows=[{"id": "A", "raw_query": big}],
        columns=["id", "raw_query"],
        total_matching=1,
        grouped=False,
        limit_applied=False,
    )
    out = format_browse(result)
    assert "more chars elided" in out
    assert len(out.encode("utf-8")) < 1_000_000
