from datetime import UTC, datetime

import pyarrow as pa
import pytest

from dbs_vector.services.browse import (
    BrowseError,
    BrowseResult,
    BrowseService,
    BrowseValidationError,
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


def test_build_and_run_grouped_aggregates_by_user():
    result = _svc().build_and_run(
        filters={},
        group_by="user",
        order_by="execution_time_ms:desc",
        select=None,
        limit=10,
    )
    assert result.grouped is True
    # one row per distinct user incl. the null group
    assert result.total_matching == 3
    cols = result.columns
    for expected in [
        "fingerprints",
        "calls",
        "execution_time_ms",
        "avg_ms_per_fingerprint",
        "avg_ms_per_call",
        "latest_ts",
    ]:
        assert expected in cols
    top = result.rows[0]
    assert top["user"] == "alice"
    assert top["fingerprints"] == 1
    assert top["execution_time_ms"] == 100.0


def test_build_and_run_caps_to_limit_but_reports_total():
    result = _svc().build_and_run(
        filters={},
        group_by="user",
        order_by="execution_time_ms:desc",
        select=None,
        limit=1,
    )
    assert result.total_matching == 3
    assert len(result.rows) == 1
    assert result.limit_applied is True


def test_build_and_run_filter_binds_value_correctly():
    result = _svc().build_and_run(
        filters={"user": "alice"},
        group_by=None,
        order_by="execution_time_ms:desc",
        select=None,
        limit=10,
    )
    assert [r["id"] for r in result.rows] == ["A"]  # only alice's row


def test_build_and_run_injection_value_is_inert():
    payload = "x') UNION SELECT 1 FROM read_csv('/etc/passwd')--"
    result = _svc().build_and_run(
        filters={"source": payload},
        group_by=None,
        order_by="execution_time_ms:desc",
        select=None,
        limit=10,
    )
    assert result.rows == []  # treated as opaque data; no read_csv, no error


def test_build_sql_carries_no_user_values():
    # _build_sql has no parameter through which a filter value could enter;
    # the emitted SQL is built solely from allowlisted identifiers + aggregates.
    sql = _svc()._build_sql(
        group_cols=[],
        order_by="execution_time_ms:desc",
        select=None,
        allow_raw_queries=False,
    )
    assert "FROM data" in sql and "ORDER BY" in sql
    assert "read_csv" not in sql


def test_build_and_run_div_by_zero_average_is_null():
    # craft a frame whose only group has SUM(calls)=0 → NULLIF → null
    tbl = pa.table(
        {
            "id": ["A"],
            "content_hash": ["h"],
            "text": ["x"],
            "raw_query": ["x"],
            "source": ["db1"],
            "user": ["alice"],
            "host": ["h"],
            "tables": [["orders"]],
            "calls": [0],
            "execution_time_ms": [10.0],
            "lock_time_sec": [None],
            "rows_examined": [1],
            "rows_sent": [1],
            "latest_ts": [datetime(2026, 1, 1, tzinfo=UTC)],
        }
    )
    svc = BrowseService(FakeStore(tbl), frame_alias="sql_api")
    result = svc.build_and_run(
        filters={},
        group_by="user",
        order_by="execution_time_ms:desc",
        select=None,
        limit=10,
    )
    assert result.rows[0]["avg_ms_per_call"] is None  # rendered n/a downstream


def test_build_and_run_group_by_tables_uses_exploded_frame():
    result = _svc().build_and_run(
        filters={},
        group_by="tables",
        order_by="fingerprints:desc",
        select=None,
        limit=10,
    )
    counts = {r["tables"]: r["fingerprints"] for r in result.rows}
    assert counts.get("orders") == 2


def test_build_and_run_table_filter_uses_list_contains():
    result = _svc().build_and_run(
        filters={"table": "orders"},
        group_by=None,
        order_by="calls:desc",
        select=None,
        limit=10,
    )
    assert sorted(r["id"] for r in result.rows) == ["A", "B"]


def test_build_and_run_rejects_unknown_column():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by="nonsense",
            order_by="execution_time_ms:desc",
            select=None,
            limit=10,
        )


def test_build_and_run_rejects_order_by_tables_scalar():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by=None,
            order_by="tables:desc",
            select=None,
            limit=10,
        )


def test_build_and_run_rejects_grouped_select_of_raw_field():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by="user",
            order_by="execution_time_ms:desc",
            select="id",
            limit=10,
        )


def test_build_and_run_raw_query_gated_off():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by=None,
            order_by="execution_time_ms:desc",
            select="id,raw_query",
            limit=10,
            allow_raw_queries=False,
        )


def test_build_and_run_raw_query_allowed_on():
    result = _svc().build_and_run(
        filters={},
        group_by=None,
        order_by="execution_time_ms:desc",
        select="id,raw_query",
        limit=10,
        allow_raw_queries=True,
    )
    assert "raw_query" in result.columns


def test_build_and_run_empty_group_by_string_is_not_grouped():
    result = _svc().build_and_run(
        filters={},
        group_by="",
        order_by="execution_time_ms:desc",
        select=None,
        limit=10,
    )
    assert result.grouped is False
    assert result.total_matching == 3  # flat rows, not aggregated


def test_build_and_run_grouped_order_by_excluded_select_raises():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by="user",
            select="fingerprints",
            order_by="calls:desc",
            limit=10,  # calls not in the select projection
        )


def test_build_and_run_rejects_non_positive_limit():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by=None,
            order_by="execution_time_ms:desc",
            select=None,
            limit=0,
        )


def test_build_and_run_empty_store_returns_no_rows():
    empty = _table().slice(0, 0)  # correct schema, zero rows
    svc = BrowseService(FakeStore(empty), frame_alias="sql_api")
    result = svc.build_and_run(
        filters={},
        group_by=None,
        order_by="execution_time_ms:desc",
        select=None,
        limit=10,
    )
    assert result.rows == []
    assert result.total_matching == 0


def test_flat_select_derived_columns():
    result = _svc().build_and_run(
        filters={},
        group_by=None,
        order_by="impact_score:desc",
        select="id,impact_score,avg_ms_per_call,selectivity",
        limit=10,
    )
    assert "impact_score" in result.columns
    top = result.rows[0]
    # row A: calls=10, exec=100 -> impact=1000, avg=10; examined=50,sent=5 -> sel=10
    assert top["id"] == "A"
    assert top["impact_score"] == 1000.0
    assert top["avg_ms_per_call"] == 10.0
    assert top["selectivity"] == 10.0


def test_flat_order_by_impact_score_without_selecting_it():
    result = _svc().build_and_run(
        filters={},
        group_by=None,
        order_by="impact_score:desc",
        select="id",  # impact_score NOT in the projection
        limit=10,
    )
    # impacts: A=1000, B=250, C=25 -> descending order A,B,C
    assert [r["id"] for r in result.rows] == ["A", "B", "C"]


def test_flat_select_rejects_unknown_derived():
    with pytest.raises(BrowseValidationError):
        _svc().build_and_run(
            filters={},
            group_by=None,
            order_by="execution_time_ms:desc",
            select="id,bogus_score",
            limit=10,
        )


def test_flat_selectivity_div_by_zero_is_null():
    tbl = pa.table(
        {
            "id": ["A"],
            "content_hash": ["h"],
            "text": ["x"],
            "raw_query": ["x"],
            "source": ["db1"],
            "user": ["alice"],
            "host": ["h"],
            "tables": [["orders"]],
            "calls": [10],
            "execution_time_ms": [100.0],
            "lock_time_sec": [None],
            "rows_examined": [50],
            "rows_sent": [0],  # division by zero -> NULLIF -> null
            "latest_ts": [datetime(2026, 1, 1, tzinfo=UTC)],
        }
    )
    svc = BrowseService(FakeStore(tbl), frame_alias="sql_api")
    result = svc.build_and_run(
        filters={},
        group_by=None,
        order_by="execution_time_ms:desc",
        select="id,selectivity",
        limit=10,
    )
    assert result.rows[0]["selectivity"] is None
