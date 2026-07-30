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
                pa.field("execution_time_ms", pa.float64()),
                pa.field("lock_time_sec", pa.float64()),
                pa.field("source", pa.string()),
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
                "execution_time_ms": [c.get("execution_time_ms", 0.0) for c in chunks],
                "lock_time_sec": [c.get("lock_time_sec", 0.0) for c in chunks],
                "source": [c.get("source", "db") for c in chunks],
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
            "execution_time_ms": 10.0,
        },
        {"id": "2", "text": "update", "tables": ["MagentoOrders"], "execution_time_ms": 10.0},
        {
            "id": "3",
            "text": "select",
            "tables": ["TryOTODyn.MagentoOrdersAddress"],
            "execution_time_ms": 10.0,
        },
        {"id": "4", "text": "select", "tables": ["address.CityTag"], "execution_time_ms": 10.0},
    ]
    vectors = np.tile([0.1, 0.2, 0.3], (4, 1)).astype(np.float32)
    store.ingest_chunks(chunks, vectors, workflow="w")
    store.create_indices()
    return store


def test_count_matching_is_case_insensitive(tmp_path):
    store = _store(tmp_path)
    assert store.count_matching(table_filter="magentoorders") == 2
    assert store.count_matching(table_filter="TryOTODyn.MagentoOrders") == 2
    assert store.count_matching(table_filter="citytag") == 1
    assert store.count_matching(table_filter="nonexistent") == 0


def test_count_matching_combines_table_and_scalar_filters(tmp_path):
    store = _store(tmp_path)
    assert store.count_matching(table_filter="magentoorders", min_time=11.0) == 0
