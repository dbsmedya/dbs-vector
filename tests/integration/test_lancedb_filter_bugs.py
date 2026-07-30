"""Integration tests for LanceDB table_filter with time prefilters."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore


class _IdentityMapper:
    """Minimal mapper that returns the raw polars row as a result."""

    def __init__(self) -> None:
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 3)),
                pa.field("execution_time_ms", pa.float64()),
                pa.field("lock_time_sec", pa.float64()),
                pa.field("tables", pa.list_(pa.string())),
                pa.field("workflow", pa.string()),
            ]
        )

    def to_record_batch(self, chunks, vectors, workflow):
        return pa.RecordBatch.from_pydict(
            {
                "id": [c["id"] for c in chunks],
                "text": [c["text"] for c in chunks],
                "source": [c.get("source", "") for c in chunks],
                "content_hash": [c["id"] for c in chunks],
                "vector": [v.tolist() for v in vectors],
                "execution_time_ms": [c["execution_time_ms"] for c in chunks],
                "lock_time_sec": [c["lock_time_sec"] for c in chunks],
                "tables": [c["tables"] for c in chunks],
                "workflow": [workflow] * len(chunks),
            },
            schema=self.schema,
        )

    def from_polars_row(self, row, *args, **kwargs):
        return {
            "id": row["id"],
            "lock_time_sec": row.get("lock_time_sec"),
            "execution_time_ms": row.get("execution_time_ms"),
            "tables": row.get("tables"),
        }


@pytest.fixture
def populated_store(tmp_path):
    """A LanceDBStore with mixed table and timing buckets."""
    mapper = _IdentityMapper()
    store = LanceDBStore(
        db_path=str(tmp_path / "lance.db"),
        table_name="t",
        vector_dimension=3,
        mapper=mapper,
        nprobes=10,
    )

    chunks = []
    for i in range(6):
        chunks.append(
            {
                "id": f"mo_{i}",
                "text": "update magentoorders set x = ?",
                "execution_time_ms": 1000.0 if i < 3 else 5.0,
                "lock_time_sec": 5.0 if i < 3 else 0.0001,
                "tables": ["magentoorders"],
            }
        )
    for i in range(6):
        chunks.append(
            {
                "id": f"ot_{i}",
                "text": "update other_table set y = ?",
                "execution_time_ms": 1000.0 if i < 3 else 5.0,
                "lock_time_sec": 5.0 if i < 3 else 0.0001,
                "tables": ["other_table"],
            }
        )

    vectors = np.array([[0.1, 0.2, 0.3]] * 12, dtype=np.float32)
    store.ingest_chunks(chunks, vectors, workflow="test")
    store.create_indices()
    return store


def test_min_lock_time_enforced_under_table_filter(populated_store):
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = populated_store.search(
        query="update",
        query_vector=query_vector,
        limit=25,
        min_lock_time=1.0,
        table_filter="magentoorders",
    )

    leaked = [r for r in results if r["lock_time_sec"] < 1.0]
    assert not leaked, (
        f"min_lock_time prefilter not enforced under table_filter: "
        f"{len(leaked)} rows leaked: {[r['id'] for r in leaked]}"
    )
    assert len(results) == 3
    assert all(r["lock_time_sec"] == 5.0 for r in results)
    assert all("mo_" in r["id"] for r in results)


def test_min_time_enforced_under_table_filter(populated_store):
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = populated_store.search(
        query="update",
        query_vector=query_vector,
        limit=25,
        min_time=100.0,
        table_filter="magentoorders",
    )

    leaked = [r for r in results if r["execution_time_ms"] < 100.0]
    assert not leaked, (
        f"min_time prefilter not enforced under table_filter: "
        f"{len(leaked)} rows leaked: {[r['id'] for r in leaked]}"
    )
    assert len(results) == 3
    assert all(r["execution_time_ms"] == 1000.0 for r in results)


def test_search_count_matches_count_matching(populated_store):
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    kwargs = {"min_lock_time": 1.0, "table_filter": "magentoorders"}
    results = populated_store.search(
        query="update",
        query_vector=query_vector,
        limit=25,
        **kwargs,
    )
    total = populated_store.count_matching(**kwargs)

    assert len(results) <= total, (
        f"search returned {len(results)} rows but count_matching reports {total}"
    )


def test_search_results_are_unique_by_id(populated_store):
    query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    results = populated_store.search(
        query="update magentoorders set",
        query_vector=query_vector,
        limit=25,
        table_filter="magentoorders",
    )

    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids)), (
        f"Duplicate fingerprint id(s) in result set: "
        f"{ {i: ids.count(i) for i in set(ids) if ids.count(i) > 1} }"
    )
