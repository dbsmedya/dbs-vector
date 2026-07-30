import numpy as np
import pyarrow as pa

from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore


class _Mapper:
    def __init__(self):
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 3)),
                pa.field("tables", pa.list_(pa.string())),
                pa.field("workflow", pa.string()),
            ]
        )

    def to_record_batch(self, chunks, vectors, workflow):
        return pa.RecordBatch.from_pydict(
            {
                "id": [c["id"] for c in chunks],
                "text": [c["text"] for c in chunks],
                "vector": [v.tolist() for v in vectors],
                "tables": [c["tables"] for c in chunks],
                "workflow": [workflow] * len(chunks),
            },
            schema=self.schema,
        )

    def from_polars_row(self, row, *args, **kwargs):
        return {"id": row["id"], "tables": row.get("tables")}


def _store(tmp_path):
    store = LanceDBStore(str(tmp_path / "l.db"), "t", 3, _Mapper(), nprobes=5)
    chunks = [
        {
            "id": "1",
            "text": "select",
            "tables": ["TryOTODyn.MagentoOrders", "TryOTODyn.Clients"],
        },
        {"id": "2", "text": "update", "tables": ["MagentoOrders"]},
        {"id": "3", "text": "select", "tables": ["TryOTODyn.MagentoOrdersAddress"]},
        {"id": "4", "text": "select", "tables": ["address.CityTag"]},
    ]
    vectors = np.tile([0.1, 0.2, 0.3], (4, 1)).astype(np.float32)
    store.ingest_chunks(chunks, vectors, workflow="w")
    store.create_indices()
    return store


def test_lowercase_filter_finds_qualified_and_bare(tmp_path):
    store = _store(tmp_path)
    qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    res = store.search("select", qv, limit=25, table_filter="magentoorders")
    assert sorted(r["id"] for r in res) == ["1", "2"]


def test_filter_is_schema_insensitive(tmp_path):
    store = _store(tmp_path)
    qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    res = store.search("select", qv, limit=25, table_filter="TryOTODyn.MagentoOrders")
    assert sorted(r["id"] for r in res) == ["1", "2"]


def test_snake_case_does_not_overmatch(tmp_path):
    store = LanceDBStore(str(tmp_path / "s.db"), "t", 3, _Mapper(), nprobes=5)
    chunks = [
        {"id": "a", "text": "x", "tables": ["order_items"]},
        {"id": "b", "text": "y", "tables": ["orders"]},
    ]
    vectors = np.tile([0.1, 0.2, 0.3], (2, 1)).astype(np.float32)
    store.ingest_chunks(chunks, vectors, workflow="w")
    store.create_indices()
    qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert [r["id"] for r in store.search("x", qv, limit=25, table_filter="orders")] == ["b"]


def test_no_match_returns_empty(tmp_path):
    store = _store(tmp_path)
    qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert store.search("select", qv, limit=25, table_filter="nonexistent") == []


def test_oversized_fallback_is_exact(tmp_path, monkeypatch):
    from dbs_vector.infrastructure.storage import lancedb_engine

    monkeypatch.setattr(lancedb_engine, "_TABLE_FILTER_PREFILTER_CAP", 1)
    store = _store(tmp_path)
    qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    res = store.search("select", qv, limit=25, table_filter="magentoorders")
    assert sorted(r["id"] for r in res) == ["1", "2"]
