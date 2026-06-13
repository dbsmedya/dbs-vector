from datetime import UTC, datetime

import pyarrow as pa
import pytest

from dbs_vector.services.browse import (
    BrowseError,
    BrowseResult,
    BrowseService,
    BrowseValidationError,  # noqa: F401 — verify it is exported
)


class FakeStore:
    """Minimal IVectorStore.scan stand-in returning a fixed Arrow table."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table
        self.scanned = 0

    def scan(self, columns=None):
        self.scanned += 1
        return self._table


def _table() -> pa.Table:
    return pa.table(
        {
            "id": ["A", "B", "C"],
            "content_hash": ["h1", "h2", "h3"],
            "text": ["select 1", "select 2", "select 3"],
            "raw_query": ["SELECT 1 /*a@x.com*/", "SELECT 2", "SELECT 3"],
            "source": ["db1", "db1", "db2"],
            "user": ["alice", "bob", None],
            "host": ["h1", "h2", None],
            "tables": [["orders", "items"], ["orders"], []],
            "calls": [10, 5, 1],
            "execution_time_ms": [100.0, 50.0, 25.0],
            "lock_time_sec": [None, None, None],
            "rows_examined": [50, 20, 5],
            "rows_sent": [5, 2, 1],
            "latest_ts": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 3, tzinfo=UTC),
            ],
        }
    )


def _svc() -> BrowseService:
    return BrowseService(FakeStore(_table()), frame_alias="sql_api")


def test_run_sql_returns_all_rows_uncapped():
    result = _svc().run_sql("SELECT id, calls FROM t ORDER BY calls DESC")
    assert isinstance(result, BrowseResult)
    assert result.total_matching == 3
    assert result.limit_applied is False
    assert result.grouped is False
    assert [r["id"] for r in result.rows] == ["A", "B", "C"]


def test_run_sql_respects_authors_own_limit():
    result = _svc().run_sql("SELECT id FROM t ORDER BY calls DESC LIMIT 1")
    assert result.total_matching == 1
    assert [r["id"] for r in result.rows] == ["A"]


def test_run_sql_quoted_user_and_alias_frame():
    result = _svc().run_sql('SELECT "user" FROM sql_api WHERE "user" = \'alice\'')
    assert [r["user"] for r in result.rows] == ["alice"]


def test_run_sql_t_by_table_explodes():
    result = _svc().run_sql("SELECT id FROM t_by_table WHERE tables = 'orders'")
    assert sorted(r["id"] for r in result.rows) == ["A", "B"]


def test_run_sql_bad_column_raises_browse_error():
    with pytest.raises(BrowseError):
        _svc().run_sql("SELECT nonexistent_col FROM t")
