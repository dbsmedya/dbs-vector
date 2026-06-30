"""End-to-end: original-case display + exact case/schema-insensitive matching."""

from datetime import UTC, datetime

import numpy as np

from dbs_vector.core.models import sql_chunk_from_record
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import SqlMapper


def _records():
    base = {
        "text": "select",
        "source": "TryOTODyn",
        "execution_time_ms": 10.0,
        "calls": 1,
        "latest_ts": datetime(2026, 1, 1, tzinfo=UTC),
    }
    return [
        {**base, "id": "1", "tables": ["TryOTODyn.MagentoOrders", "TryOTODyn.Clients"]},
        {**base, "id": "2", "tables": ["MagentoOrders"]},
        {**base, "id": "3", "tables": ["TryOTODyn.MagentoOrdersAddress"]},
        {**base, "id": "4", "tables": ["address.CityTag"]},
    ]


def _store(tmp_path):
    mapper = SqlMapper(vector_dimension=4)
    store = LanceDBStore(str(tmp_path / "l.db"), "sql", 4, mapper, nprobes=5)
    chunks = [sql_chunk_from_record(r) for r in _records()]
    vectors = np.tile([0.1, 0.2, 0.3, 0.4], (4, 1)).astype(np.float32)
    store.ingest_chunks(chunks, vectors, workflow="sql")
    store.create_indices()
    return store


def test_stores_and_displays_original_case(tmp_path):
    store = _store(tmp_path)
    qv = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    res = store.search("select", qv, limit=25, table_filter="magentoorders")
    by_id = {r.chunk.id: r.chunk.tables for r in res}
    assert by_id["1"] == ["TryOTODyn.MagentoOrders", "TryOTODyn.Clients"]
    assert by_id["2"] == ["MagentoOrders"]


def test_lowercase_filter_matches_both_no_false_positive(tmp_path):
    store = _store(tmp_path)
    qv = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    res = store.search("select", qv, limit=25, table_filter="magentoorders")
    assert sorted(r.chunk.id for r in res) == ["1", "2"]


def test_count_matches_search(tmp_path):
    store = _store(tmp_path)
    assert store.count_matching(table_filter="MAGENTOORDERS") == 2
