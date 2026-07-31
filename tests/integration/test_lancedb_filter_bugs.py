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


class _WideIdentityMapper(_IdentityMapper):
    """`_IdentityMapper` at a dimension that can carry an IVF_PQ index.

    `create_indices` uses `num_sub_vectors = min(16, dimension // 8)`, so a
    3-dimensional vector yields 0 and index creation fails outright.
    """

    DIM = 64

    def __init__(self) -> None:
        super().__init__()
        self.schema = pa.schema(
            [
                field if field.name != "vector" else pa.field("vector", pa.list_(pa.float32(), 64))
                for field in self.schema
            ]
        )

    def from_polars_row(self, row, *args, **kwargs):
        return {"id": row["id"], "source": row.get("source")}


@pytest.fixture
def indexed_multi_source_store(tmp_path):
    """A store large enough for a real IVF index, with one source's chunks
    deliberately scattered across partitions by giving them diverse vectors."""
    mapper = _WideIdentityMapper()
    store = LanceDBStore(
        db_path=str(tmp_path / "lance.db"),
        table_name="t",
        vector_dimension=_WideIdentityMapper.DIM,
        mapper=mapper,
        nprobes=10,
    )

    rng = np.random.default_rng(1234)
    chunks = []
    for i in range(600):
        chunks.append(
            {
                "id": f"c_{i}",
                # Deliberately shares no term with the query below: the FTS leg
                # is not partition-limited, so lexical hits would mask a vector
                # leg that never opened the right partitions.
                "text": f"paragraph {i} describing widgets and sprockets",
                "source": f"/abs/root/group_{i % 15 % 3}/doc_{i % 15}.md",
                "execution_time_ms": 1.0,
                "lock_time_sec": 0.0,
                "tables": [],
            }
        )

    vectors = rng.normal(size=(600, _WideIdentityMapper.DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.ingest_chunks(chunks, vectors, workflow="test")
    store.create_indices()
    assert store.table.list_indices(), "fixture drift: no index was created"
    return store


def test_source_filter_returns_all_matching_rows(indexed_multi_source_store):
    """Regression: source_filter used the approximate IVF path, so matching
    rows living in unprobed partitions were never scored. `prefilter=True`
    restricts which rows may be RETURNED, not which partitions are OPENED.
    """
    store = indexed_multi_source_store
    target = "/abs/root/group_0/doc_3.md"
    expected = store.count_matching(source_filter=target)
    assert expected == 40, f"fixture drift: expected 40 rows for {target}, got {expected}"

    query_vector = np.random.default_rng(99).normal(size=_WideIdentityMapper.DIM).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)
    results = store.search(
        query="quantum entanglement",
        query_vector=query_vector,
        source_filter=target,
        limit=500,
    )

    # Hybrid fuses two legs and dedupes after LanceDB applies `limit`, so a
    # couple of tail rows may drop; the bug this guards produced a small
    # fraction of `expected`, not a near-miss.
    assert len(results) >= expected - 2, (
        f"source_filter recall loss: {len(results)} of {expected} matching rows "
        f"returned. The IVF index is likely being used instead of an exact scan."
    )


def test_bare_filename_resolves_against_the_real_table(indexed_multi_source_store):
    store = indexed_multi_source_store
    query_vector = np.random.default_rng(7).normal(size=_WideIdentityMapper.DIM).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)

    results = store.search(
        query="widgets",
        query_vector=query_vector,
        source_filter="doc_3.md",
        limit=500,
    )

    assert len(results) == 40
    assert {r["source"] for r in results} == {"/abs/root/group_0/doc_3.md"}


def test_directory_name_scopes_to_every_source_beneath_it(indexed_multi_source_store):
    """'group_0' is the shape of input that used to return nothing at all."""
    store = indexed_multi_source_store
    query_vector = np.random.default_rng(7).normal(size=_WideIdentityMapper.DIM).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)

    results = store.search(
        query="widgets",
        query_vector=query_vector,
        source_filter="group_0",
        limit=500,
    )

    # doc_0, doc_3, doc_6, doc_9, doc_12 — 5 files of 40 chunks each.
    assert {r["source"] for r in results} == {
        f"/abs/root/group_0/doc_{i}.md" for i in (0, 3, 6, 9, 12)
    }
    assert len(results) == 200
    assert store.count_matching(source_filter="group_0") == 200


def test_unmatched_filter_returns_nothing_but_is_reported(indexed_multi_source_store):
    store = indexed_multi_source_store
    resolution = store.resolve_source_filter("nope_does_not_exist.md")

    assert resolution.is_unmatched
    assert resolution.matched == []
    assert store.count_matching(source_filter="nope_does_not_exist.md") == 0


def test_search_and_count_matching_resolve_identically(indexed_multi_source_store):
    """A divergence here makes the 'Showing N of M' header report a total the
    result set can never reach."""
    store = indexed_multi_source_store
    query_vector = np.random.default_rng(7).normal(size=_WideIdentityMapper.DIM).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)

    for value in ("group_1", "doc_4.md", "/abs/root/group_2/doc_2.md"):
        results = store.search(
            query="widgets", query_vector=query_vector, source_filter=value, limit=500
        )
        assert len(results) == store.count_matching(source_filter=value), (
            f"search/count disagree for source_filter={value!r}"
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
