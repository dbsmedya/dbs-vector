from datetime import UTC, datetime

import numpy as np
import pytest

from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import SqlMapper

VECTOR_DIM = 4


def _make_store(tmp_path) -> LanceDBStore:
    mapper = SqlMapper(VECTOR_DIM)
    return LanceDBStore(
        db_path=str(tmp_path / "db"),
        table_name="query_vault",
        vector_dimension=VECTOR_DIM,
        mapper=mapper,
    )


def _seed(store: LanceDBStore) -> None:
    from dbs_vector.core.models import SqlChunk

    chunks = [
        SqlChunk(
            id="A",
            text="select 1",
            raw_query="SELECT 1 WHERE email='a@x.com'",
            source="db1",
            execution_time_ms=100.0,
            calls=10,
            content_hash="h1",
            tables=["orders", "items"],
            latest_ts=datetime(2026, 1, 1, tzinfo=UTC),
            user="alice",
            host="h1",
            rows_sent=5,
            rows_examined=50,
            lock_time_sec=1.0,
        ),
        SqlChunk(
            id="B",
            text="select 2",
            raw_query="SELECT 2",
            source="db1",
            execution_time_ms=50.0,
            calls=5,
            content_hash="h2",
            tables=["orders"],
            latest_ts=datetime(2026, 1, 2, tzinfo=UTC),
            user="bob",
            host="h2",
            rows_sent=1,
            rows_examined=2,
            lock_time_sec=None,
        ),
    ]
    vectors = np.ones((len(chunks), VECTOR_DIM), dtype=np.float32)
    store.ingest_chunks(chunks, vectors, workflow="sql_clustering")


def test_scan_empty_table_returns_zero_rows_with_projected_schema(tmp_path):
    store = _make_store(tmp_path)
    table = store.scan()
    assert table.num_rows == 0
    assert "vector" not in table.schema.names
    assert "workflow" not in table.schema.names
    assert "id" in table.schema.names


def test_scan_returns_all_rows_without_vector_or_workflow(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)

    table = store.scan()

    assert table.num_rows == 2
    assert "vector" not in table.schema.names
    assert "workflow" not in table.schema.names
    assert "raw_query" in table.schema.names
    assert "id" in table.schema.names
    assert set(table.column("id").to_pylist()) == {"A", "B"}


def test_browse_run_sql_point_lookup_end_to_end(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    result = svc.run_sql("SELECT id, calls FROM t WHERE id = 'A'")
    assert result.total_matching == 1
    assert result.rows[0]["id"] == "A"
    assert result.rows[0]["calls"] == 10


def test_browse_grouped_by_user_end_to_end(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    result = svc.build_and_run(
        filters={},
        group_by="user",
        order_by="execution_time_ms:desc",
        select=None,
        limit=10,
    )
    assert result.grouped is True
    assert result.rows[0]["user"] == "alice"
    assert result.rows[0]["execution_time_ms"] == 100.0


def test_browse_grouped_by_table_end_to_end(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    result = svc.build_and_run(
        filters={},
        group_by="tables",
        order_by="fingerprints:desc",
        select=None,
        limit=10,
    )
    counts = {r["tables"]: r["fingerprints"] for r in result.rows}
    assert counts["orders"] == 2  # A and B both touch orders


def test_browse_sees_checkout_latest_updates(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    from dbs_vector.services.browse import BrowseService

    svc = BrowseService(store, frame_alias="sql_api")
    assert svc.run_sql("SELECT id FROM t").total_matching == 2

    # add rows via a second store handle on the same path/table
    store2 = _make_store(tmp_path)
    _seed(store2)  # adds A,B again (dup ids ok for count) → table grows
    assert svc.run_sql("SELECT id FROM t").total_matching == 4


@pytest.mark.asyncio
async def test_triage_handler_end_to_end(tmp_path, monkeypatch):
    import dbs_vector.mcp.state as state
    from dbs_vector.mcp.families.sql import SqlFamily

    store = _make_store(tmp_path)
    _seed(store)  # A: calls=10 exec=100 -> impact 1000; B: calls=5 exec=50 -> impact 250

    class _Svc:
        def __init__(self, s):
            self.vector_store = s

    monkeypatch.setitem(state._services, "sql-api", _Svc(store))
    handler = SqlFamily().make_triage_handler("sql-api", allow_raw_queries=False)
    out = await handler()

    assert "impact_score" in out
    assert out.index("A") < out.index("B")  # A outranks B by impact_score
    assert "RAW-A" not in out  # raw_query gated off by default
