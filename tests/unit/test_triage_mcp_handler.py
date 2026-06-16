from datetime import UTC, datetime

import pyarrow as pa
import pytest

import dbs_vector.mcp.state as state
from dbs_vector.mcp.families.base import TriageFamily
from dbs_vector.mcp.families.sql import SqlFamily


class _FakeStore:
    def __init__(self, table):
        self._t = table

    def scan(self, columns=None):
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


def test_sqlfamily_is_triage_family():
    assert isinstance(SqlFamily(), TriageFamily)


@pytest.mark.asyncio
async def test_triage_default_ranks_by_impact_score(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler()
    assert "impact_score" in out
    assert out.index("A") < out.index("B")  # A (impact 1000) before B (250)


@pytest.mark.asyncio
async def test_triage_rejects_bad_order_by(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(order_by="user:desc")
    assert "order_by must be one of" in out


@pytest.mark.asyncio
async def test_triage_raw_query_silently_downgraded(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(include_raw=True)  # flag off -> no raw_query
    assert "RAW-A-SECRET" not in out


@pytest.mark.asyncio
async def test_triage_raw_query_present_when_allowed(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=True)
    out = await handler(include_raw=True)
    assert "RAW-A-SECRET" in out
    assert "Raw SQL:" in out  # exemplar rendered on its own block, not an inline cell


@pytest.mark.asyncio
async def test_triage_table_filter(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(table="items")
    assert "Showing 1 of 1 fingerprints" in out
    assert "items" in out


@pytest.mark.asyncio
async def test_triage_min_calls_filter(wired):
    wired(_FakeStore(_table()))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler(min_calls=8)  # only A (calls=10)
    assert "Showing 1 of 1 fingerprints" in out


@pytest.mark.asyncio
async def test_triage_uninitialized_service():
    handler = SqlFamily().make_triage_handler("missing-engine", allow_raw_queries=False)
    out = await handler()
    assert "not initialized" in out
