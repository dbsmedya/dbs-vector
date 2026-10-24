"""The IVF_PQ index must survive index maintenance with its partitions intact.

Ingestion and the watcher both run create_indices() and then compact(). From
lancedb 0.39, Table.optimize() (what compact() calls) rebuilds the IVF_PQ index
with a single partition, existing indexes included. Search keeps working on the
collapsed index, but every vector is quantized against one global centroid, so
the ANN leg loses recall. Nothing else in the suite looks at index geometry, so
this is the test that fails when a lancedb bump brings that back.
"""

from __future__ import annotations

import re

import numpy as np
import pyarrow as pa
import pytest

from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore

_DIM = 32
_ROWS = 2100  # > 256, so create_indices() builds the vector index


class _VectorMapper:
    def __init__(self) -> None:
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), _DIM)),
                pa.field("workflow", pa.string()),
            ]
        )

    def to_record_batch(self, chunks, vectors, workflow):
        return pa.RecordBatch.from_pydict(
            {
                "id": [c["id"] for c in chunks],
                "text": [c["text"] for c in chunks],
                "source": [c["source"] for c in chunks],
                "vector": [v.tolist() for v in vectors],
                "workflow": [workflow] * len(chunks),
            },
            schema=self.schema,
        )


def _ivf_partitions(store: LanceDBStore, query: np.ndarray) -> int:
    plan = store.table.search(query).metric("cosine").nprobes(20).limit(5).analyze_plan()
    match = re.search(r"partitions_ranked=(\d+)", plan)
    assert match, f"no IVF partition metric in the query plan:\n{plan}"
    return int(match.group(1))


@pytest.fixture
def indexed_store(tmp_path):
    store = LanceDBStore(
        db_path=str(tmp_path / "lance.db"),
        table_name="t",
        vector_dimension=_DIM,
        mapper=_VectorMapper(),
    )
    vectors = np.random.default_rng(0).standard_normal((_ROWS, _DIM)).astype(np.float32)
    chunks = [{"id": f"c{i}", "text": f"chunk {i}", "source": f"s{i % 50}"} for i in range(_ROWS)]
    store.ingest_chunks(chunks, vectors, workflow="test")
    return store, vectors


def test_ivf_partitions_survive_production_maintenance_order(indexed_store):
    store, vectors = indexed_store

    store.create_indices()
    built = _ivf_partitions(store, vectors[0])
    store.compact()

    assert built > 1
    assert _ivf_partitions(store, vectors[0]) > 1


def test_ivf_partitions_survive_a_second_maintenance_cycle(indexed_store):
    store, vectors = indexed_store

    store.create_indices()
    store.compact()
    store.delete_by_source("s0")  # a watcher re-ingest deletes before it adds
    store.create_indices()
    store.compact()

    assert _ivf_partitions(store, vectors[0]) > 1
