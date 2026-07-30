"""Tests for SqlFamily run_search / format_results / make_handler — includes
the family-specific min_time filter."""

import inspect
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from dbs_vector.core.models import SqlChunk, SqlSearchResult
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import SqlMapper
from dbs_vector.mcp.families.base import RESPONSE_BUDGET_BYTES
from dbs_vector.mcp.families.sql import (
    _RAW_QUERY_DISPLAY_LIMIT,
    SqlFamily,
)


def _make_sql_result(
    raw_query: str = "SELECT * FROM t WHERE id=1", id_: str = "sql_0"
) -> SqlSearchResult:
    return SqlSearchResult(
        chunk=SqlChunk(
            id=id_,
            text="SELECT * FROM t",
            raw_query=raw_query,
            source="prod_db",
            execution_time_ms=500.5,
            calls=2,
            content_hash="h",
            latest_ts=datetime.now(),
        ),
        similarity=0.5678,
        retrieved_by="vector",
        rrf_score=None,
    )


def _make_full_sql_result(**overrides) -> SqlSearchResult:
    """SqlSearchResult with every field populated, used for field-rendering tests."""
    chunk_kwargs: dict = {
        "id": "AD90D81BD56976A3",
        "text": "select count(?) as count from magentoorders",
        "raw_query": "SELECT count(1) as count FROM MagentoOrders",
        "source": "prod_db",
        "execution_time_ms": 131_758_096.586,
        "calls": 312,
        "content_hash": "h",
        "tables": ["TryOTODyn.MagentoOrders", "TryOTODyn.DeliveryCompanySettings"],
        "latest_ts": datetime(2026, 5, 21, 14, 51, 44, tzinfo=UTC),
        "user": "s_web_api",
        "host": "10.132.0.26",
        "rows_sent": 1,
        "rows_examined": 853_831,
        "lock_time_sec": 0.000008,
    }
    chunk_kwargs.update(overrides)
    return SqlSearchResult(
        chunk=SqlChunk(**chunk_kwargs),
        similarity=0.0328,
        retrieved_by="vector",
        rrf_score=None,
    )


@pytest.fixture
def populated_store(tmp_path: Path):
    mapper = SqlMapper(vector_dimension=4)
    store = LanceDBStore(
        db_path=str(tmp_path / "lance"),
        table_name="test_table",
        vector_dimension=4,
        mapper=mapper,
    )
    rng = np.random.default_rng(42)
    now = datetime.now(UTC)
    chunks = [
        SqlChunk(
            id="c1",
            text="SELECT * FROM OrderShipment",
            raw_query="",
            source="db",
            execution_time_ms=100.0,
            calls=1,
            content_hash="h1",
            tables=["ordershipment"],
            latest_ts=now,
        ),
        SqlChunk(
            id="c2",
            text="SELECT * FROM Clients",
            raw_query="",
            source="db",
            execution_time_ms=50.0,
            calls=1,
            content_hash="h2",
            tables=["clients"],
            latest_ts=now,
        ),
        SqlChunk(
            id="c3",
            text="SELECT * FROM OrderShipment JOIN Clients",
            raw_query="",
            source="db",
            execution_time_ms=200.0,
            calls=1,
            content_hash="h3",
            tables=["ordershipment", "clients"],
            latest_ts=now,
        ),
    ]
    vectors = rng.standard_normal((3, 4)).astype(np.float32)
    store.ingest_chunks(chunks, vectors, workflow="test")
    return store


class TestTableFilter:
    def test_filter_exact_match_returns_only_matching(self, populated_store):
        q = np.ones(4, dtype=np.float32)
        results = populated_store.search(
            query="anything",
            query_vector=q,
            table_filter="OrderShipment",
            limit=10,
        )
        ids = sorted(r.chunk.id for r in results)
        assert ids == ["c1", "c3"]

    def test_filter_case_insensitive(self, populated_store):
        q = np.ones(4, dtype=np.float32)
        results = populated_store.search(
            query="anything",
            query_vector=q,
            table_filter="ORDERSHIPMENT",
            limit=10,
        )
        ids = sorted(r.chunk.id for r in results)
        assert ids == ["c1", "c3"]

    def test_filter_schema_qualified_input(self, populated_store):
        q = np.ones(4, dtype=np.float32)
        results = populated_store.search(
            query="anything",
            query_vector=q,
            table_filter="TryOTODyn.OrderShipment",
            limit=10,
        )
        ids = sorted(r.chunk.id for r in results)
        assert ids == ["c1", "c3"]

    def test_filter_nonexistent_table_returns_empty(self, populated_store):
        q = np.ones(4, dtype=np.float32)
        results = populated_store.search(
            query="anything",
            query_vector=q,
            table_filter="DoesNotExist",
            limit=10,
        )
        assert results == []


class TestCountMatching:
    def test_count_with_no_filters_returns_total(self, populated_store):
        assert populated_store.count_matching() == 3

    def test_count_with_table_filter(self, populated_store):
        assert populated_store.count_matching(table_filter="OrderShipment") == 2

    def test_count_with_min_time(self, populated_store):
        assert populated_store.count_matching(min_time=100) == 2

    def test_count_combined_filters(self, populated_store):
        assert populated_store.count_matching(table_filter="OrderShipment", min_time=200) == 1


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


def test_run_search_passes_min_lock_time_filter():
    fam = SqlFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(service, query="q", limit=2, source_filter=None, min_lock_time=50.0)

    service.execute_query.assert_called_once_with(
        "q", None, 2, extra_filters={"min_lock_time": 50.0}
    )


def test_run_search_passes_table_filter():
    fam = SqlFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(
        service,
        query="q",
        limit=2,
        source_filter=None,
        table_filter="dt_customer_performance_report",
    )

    service.execute_query.assert_called_once_with(
        "q", None, 2, extra_filters={"table_filter": "dt_customer_performance_report"}
    )


def test_run_search_combines_all_three_filters():
    fam = SqlFamily()
    service = MagicMock()
    service.execute_query.return_value = []

    fam.run_search(
        service,
        query="q",
        limit=2,
        source_filter=None,
        min_time=100.0,
        min_lock_time=10.0,
        table_filter="tx_process",
    )

    service.execute_query.assert_called_once_with(
        "q",
        None,
        2,
        extra_filters={"min_time": 100.0, "min_lock_time": 10.0, "table_filter": "tx_process"},
    )


def test_make_handler_signature_includes_new_filters():
    """FastMCP introspects this signature for the tool schema."""
    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    sig = inspect.signature(handler)
    params = list(sig.parameters)
    assert params == [
        "query",
        "limit",
        "source_filter",
        "min_time",
        "min_lock_time",
        "table_filter",
        "include_raw",
    ]
    assert sig.parameters["min_lock_time"].default is None
    assert sig.parameters["table_filter"].default is None
    assert sig.parameters["include_raw"].default is False


def test_format_results_includes_execution_time_calls_and_normalized_sql():
    fam = SqlFamily()
    out = fam.format_results([_make_sql_result()], query="q", total_matching=1)
    assert "Showing 1 of 1 results that matched your filters for 'q'" in out
    assert "Source Database: prod_db" in out
    assert "Execution Time: 500.500ms (Calls: 2)" in out
    # Default renders normalized text, not the raw_query
    assert "SELECT * FROM t" in out
    assert "similarity 0.57" in out


def test_format_results_truncates_long_text():
    fam = SqlFamily()
    long_text = "SELECT " + ("x" * 100_000)
    result = _make_sql_result()
    result.chunk.text = long_text

    out = fam.format_results([result], query="q", total_matching=1)

    assert long_text not in out
    assert "more chars elided" in out
    assert len(out) < _RAW_QUERY_DISPLAY_LIMIT + 500


def test_format_results_truncates_long_raw_query_when_include_raw():
    fam = SqlFamily()
    raw_query = "SELECT " + ("x" * 100_000)

    out = fam.format_results(
        [_make_sql_result(raw_query=raw_query)],
        query="q",
        total_matching=1,
        include_raw=True,
    )

    assert raw_query not in out
    assert "more chars elided" in out


def test_format_results_caps_total_response_size():
    fam = SqlFamily()
    results = []
    for idx in range(1_000):
        res = _make_sql_result(id_=f"sql_{idx}")
        res.chunk.text = f"SELECT {idx} " + ("x" * 2_000)
        results.append(res)

    out = fam.format_results(results, query="q", total_matching=1_000)

    assert len(out.encode("utf-8")) <= RESPONSE_BUDGET_BYTES
    assert "results elided due to MCP response size cap" in out


def test_format_results_flags_anomaly_when_search_exceeds_count(caplog):
    """Loguru warnings are captured by tests/conftest.py's caplog bridge."""
    fam = SqlFamily()
    results = []
    for i in range(3):
        r = MagicMock()
        r.similarity = 0.5
        r.retrieved_by = "vector"
        r.chunk = MagicMock(
            id=f"fp_{i}",
            source="s",
            tables=["t"],
            host="h",
            user="u",
            latest_ts=None,
            execution_time_ms=100.0,
            calls=1,
            rows_examined=10,
            rows_sent=1,
            lock_time_sec=0.0,
            text="select 1",
            raw_query="select 1",
        )
        results.append(r)

    output = fam.format_results(results, query="q", total_matching=1)
    first_line = output.split("\n")[0]

    assert "Showing 3 results" in first_line
    assert "count_matching reported only 1" in first_line
    assert "WARNING" in first_line
    assert any(
        "len(results)=3 exceeds total_matching=1" in rec.message for rec in caplog.records
    ), f"Expected anomaly warning in logs; got: {[r.message for r in caplog.records]}"


def test_format_results_normal_header_when_counts_agree():
    fam = SqlFamily()
    results = []
    for i in range(2):
        r = MagicMock()
        r.similarity = 0.5
        r.retrieved_by = "vector"
        r.chunk = MagicMock(
            id=f"fp_{i}",
            source="s",
            tables=["t"],
            host="h",
            user="u",
            latest_ts=None,
            execution_time_ms=100.0,
            calls=1,
            rows_examined=10,
            rows_sent=1,
            lock_time_sec=0.0,
            text="select 1",
            raw_query="select 1",
        )
        results.append(r)

    output = fam.format_results(results, query="q", total_matching=5)
    first_line = output.split("\n")[0]
    assert "Showing 2 of 5 results" in first_line
    assert "WARNING" not in first_line


# ---------------------------------------------------------------------------
# Extended field surface (claude_todo.md #2)
# ---------------------------------------------------------------------------


def test_format_results_includes_fingerprint_id():
    fam = SqlFamily()
    out = fam.format_results([_make_full_sql_result()], query="q", total_matching=1)
    assert "Fingerprint ID: AD90D81BD56976A3" in out


def test_format_results_includes_tables_list():
    fam = SqlFamily()
    out = fam.format_results([_make_full_sql_result()], query="q", total_matching=1)
    assert "Tables: TryOTODyn.MagentoOrders, TryOTODyn.DeliveryCompanySettings" in out


def test_format_results_includes_host_and_user():
    fam = SqlFamily()
    out = fam.format_results([_make_full_sql_result()], query="q", total_matching=1)
    assert "Host: 10.132.0.26" in out
    assert "User: s_web_api" in out


def test_format_results_includes_latest_ts():
    fam = SqlFamily()
    out = fam.format_results([_make_full_sql_result()], query="q", total_matching=1)
    assert "Last Seen: 2026-05-21T14:51:44" in out


def test_format_results_renders_rows_examined_sent_with_selectivity():
    fam = SqlFamily()
    out = fam.format_results([_make_full_sql_result()], query="q", total_matching=1)
    assert "Rows Examined / Sent: 853,831 / 1" in out
    assert "selectivity 853,831:1" in out


def test_format_results_includes_lock_time():
    fam = SqlFamily()
    out = fam.format_results([_make_full_sql_result()], query="q", total_matching=1)
    assert "Lock Time:" in out
    assert "s" in out  # rendered with seconds suffix


def test_format_results_shows_normalized_by_default_not_raw():
    fam = SqlFamily()
    out = fam.format_results([_make_full_sql_result()], query="q", total_matching=1)
    assert "Normalized SQL:" in out
    assert "select count(?) as count from magentoorders" in out
    # raw_query NOT shown by default
    assert "SELECT count(1) as count FROM MagentoOrders" not in out
    assert "Raw SQL:" not in out


def test_format_results_include_raw_renders_both_sections():
    fam = SqlFamily()
    out = fam.format_results(
        [_make_full_sql_result()], query="q", total_matching=1, include_raw=True
    )
    assert "Normalized SQL:" in out
    assert "Raw SQL:" in out
    assert "SELECT count(1) as count FROM MagentoOrders" in out


def test_format_results_optional_fields_render_na_without_crashing():
    """When optional fields (host, user, rows_*, lock_time, tables) are
    None / [], render placeholder 'n/a' instead of crashing or showing 'None'."""
    fam = SqlFamily()
    result = _make_full_sql_result(
        host=None,
        user=None,
        rows_sent=None,
        rows_examined=None,
        lock_time_sec=None,
        tables=[],
    )
    out = fam.format_results([result], query="q", total_matching=1)
    assert "Host: n/a" in out
    assert "User: n/a" in out
    assert "Tables: n/a" in out
    assert "Rows Examined / Sent: n/a / n/a" in out
    assert "selectivity n/a" in out
    assert "Lock Time: n/a" in out


def test_format_results_selectivity_na_when_rows_sent_zero():
    """rows_sent=0 would divide-by-zero; must render as n/a."""
    fam = SqlFamily()
    result = _make_full_sql_result(rows_sent=0, rows_examined=1000)
    out = fam.format_results([result], query="q", total_matching=1)
    assert "selectivity n/a" in out


@pytest.mark.asyncio
async def test_make_handler_passes_include_raw_to_formatter(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [_make_full_sql_result()]
    service.count_matching.return_value = 1
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test", allow_raw_queries=True)
    out = await handler(query="q", include_raw=True)

    assert "Raw SQL:" in out
    assert "SELECT count(1) as count FROM MagentoOrders" in out


@pytest.mark.asyncio
async def test_make_handler_gates_raw_off_even_when_include_raw_true(monkeypatch):
    """allow_raw_queries=False downgrades include_raw=True: no raw SQL leaks."""
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [_make_full_sql_result()]
    service.count_matching.return_value = 1
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test", allow_raw_queries=False)
    out = await handler(query="q", include_raw=True)

    assert "Raw SQL:" not in out
    assert "SELECT count(1) as count FROM MagentoOrders" not in out


@pytest.mark.asyncio
async def test_make_handler_no_raw_when_flag_on_but_include_raw_false(monkeypatch):
    """include_raw=False suppresses raw SQL even with the server flag on."""
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [_make_full_sql_result()]
    service.count_matching.return_value = 1
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test", allow_raw_queries=True)
    out = await handler(query="q", include_raw=False)

    assert "Raw SQL:" not in out


def test_format_results_empty_with_matching_filters_signals_total():
    fam = SqlFamily()
    out = fam.format_results([], query="zzz", total_matching=37)
    assert "37 rows matched your filters" in out
    assert "none ranked above the similarity/FTS threshold" in out


def test_format_results_empty_returns_no_results_message():
    fam = SqlFamily()
    out = fam.format_results([], query="zzz")
    assert out == "No results found for query: 'zzz'"


@pytest.mark.asyncio
async def test_make_handler_runs_search_and_formats(monkeypatch):
    import dbs_vector.mcp.state as state_mod

    service = MagicMock()
    service.execute_query.return_value = [_make_sql_result()]
    service.count_matching.return_value = 1
    monkeypatch.setattr(state_mod, "_services", {"sql-test": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql-test")
    out = await handler(query="q", limit=1, min_time=200.0)

    service.execute_query.assert_called_once_with("q", None, 1, extra_filters={"min_time": 200.0})
    service.count_matching.assert_called_once_with(None, {"min_time": 200.0})
    assert "Source Database: prod_db" in out


@pytest.mark.asyncio
async def test_handler_handles_50_concurrent_search_count_pairs(monkeypatch):
    """Stress guard: the handler must fan out search+count correctly under
    50 concurrent invocations on the same (shared) service handle. Pins the
    invariant across the sequential-in-one-thread refactor — no exceptions,
    50 string results, both service methods called exactly 50 times."""
    import asyncio
    import threading

    import dbs_vector.mcp.state as state_mod

    calls = {"search": 0, "count": 0}
    lock = threading.Lock()

    def fake_search(query, source_filter, limit, *, extra_filters):
        with lock:
            calls["search"] += 1
        return []  # empty results → formatter "no results" branch

    def fake_count(source_filter, extra_filters):
        with lock:
            calls["count"] += 1
        return 0

    service = MagicMock()
    service.execute_query.side_effect = fake_search
    service.count_matching.side_effect = fake_count
    monkeypatch.setattr(state_mod, "_services", {"sql_stress": service})

    fam = SqlFamily()
    handler = fam.make_handler("sql_stress")
    outs = await asyncio.gather(*[handler(query="q", limit=3) for _ in range(50)])

    assert len(outs) == 50
    assert calls["search"] == 50
    assert calls["count"] == 50
    assert all(isinstance(o, str) for o in outs)


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
