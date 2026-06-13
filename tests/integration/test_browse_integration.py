from datetime import UTC, datetime

import numpy as np

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
            id="A", text="select 1", raw_query="SELECT 1 WHERE email='a@x.com'",
            source="db1", execution_time_ms=100.0, calls=10, content_hash="h1",
            tables=["orders", "items"], latest_ts=datetime(2026, 1, 1, tzinfo=UTC),
            user="alice", host="h1", rows_sent=5, rows_examined=50, lock_time_sec=1.0,
        ),
        SqlChunk(
            id="B", text="select 2", raw_query="SELECT 2",
            source="db1", execution_time_ms=50.0, calls=5, content_hash="h2",
            tables=["orders"], latest_ts=datetime(2026, 1, 2, tzinfo=UTC),
            user="bob", host="h2", rows_sent=1, rows_examined=2, lock_time_sec=None,
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
