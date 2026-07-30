"""Floor-mechanics integration tests: real tmpdir LanceDB, synthetic vectors, no MLX.

Vector geometry (dimension 4):
  axis 0 = beekeeping topic, axis 1 = code topic,
  axis 2 = off-corpus query axis (near-orthogonal to the corpus: only d3
  carries a 0.1 component on it), axis 3 = shop/store topic.
"""

import numpy as np
import pytest

from dbs_vector.core.models import Chunk
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import DocumentMapper
from dbs_vector.services.search import SearchService

DIM = 4

DOCS = [
    # (id, text, source, vector)
    ("d1", "beekeeping hive maintenance in spring", "bees.md", [1.0, 0.0, 0.0, 0.0]),
    (
        "d2",
        "def delete_by_source(x): removes all rows for a source",
        "store.py",
        [0.0, 1.0, 0.0, 0.0],
    ),
    ("d3", "the uv.lock lockfile pins python dependencies", "uv.md", [0.0, 0.9, 0.1, 0.0]),
    ("d4", "arrow record batches stream to lancedb", "arch.md", [0.7, 0.7, 0.0, 0.0]),
    ("d5", "pydantic models validate configuration", "config.md", [0.6, 0.8, 0.0, 0.0]),
    ("d6", "the store opens at nine", "shop.md", [0.0, 0.0, 0.0, 1.0]),
]

QUERY_VECTORS = {
    "beekeeping spring": [1.0, 0.0, 0.0, 0.0],
    "delete_by_source": [0.0, 0.0, 1.0, 0.0],  # off-corpus axis
    "narrowboat lock": [0.0, 0.0, 1.0, 0.0],  # off-corpus axis
    "quantum chromodynamics": [0.0, 0.0, 1.0, 0.0],  # off-corpus axis
    # cos(d6) ~= 0.30: makes d6 the best REJECTED candidate, proving the
    # FTS stem-hit happened while staying below the 0.5 floor.
    "stores": [0.0, 0.0, 0.95, 0.3],
}


class _FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(QUERY_VECTORS[query], dtype=np.float32)


def _make_store(tmp_path, with_fts: bool = True) -> LanceDBStore:
    store = LanceDBStore(
        db_path=str(tmp_path / "db"),
        table_name="floor_ci",
        vector_dimension=DIM,
        mapper=DocumentMapper(vector_dimension=DIM),
    )
    chunks = [Chunk(id=i, text=t, source=s, content_hash=f"hash_{i}") for i, t, s, _ in DOCS]
    vectors = np.asarray([v for *_, v in DOCS], dtype=np.float32)
    store.ingest_chunks(chunks, vectors, workflow="test")
    if with_fts:
        store.create_indices()
    return store


@pytest.fixture()
def store(tmp_path):
    return _make_store(tmp_path)


def _service(store, floor):
    return SearchService(_FakeEmbedder(), store, similarity_floor=floor)


def test_on_topic_query_carries_exact_similarity(store):
    resp = _service(store, floor=None).execute_query("beekeeping spring", limit=5)
    assert resp.floor is None
    by_id = {r.chunk.id: r for r in resp.results}
    assert by_id["d1"].similarity == pytest.approx(1.0, abs=1e-4)
    # d4 = [0.7, 0.7]: cos with [1, 0] = 0.7071
    assert by_id["d4"].similarity == pytest.approx(0.7071, abs=1e-3)
    # FTS index exists, so hybrid ran — fail fast if the environment degraded
    assert store._hybrid_ok is True


def test_floor_orthogonal_query_returns_empty_with_evidence(store):
    resp = _service(store, floor=0.5).execute_query("quantum chromodynamics", limit=5)
    assert resp.results == []
    assert resp.floor == 0.5
    # vector leg returns all 6 docs (flat search, fetch limit 15); FTS matches none
    assert resp.inspected == 6
    assert resp.best_rejected is not None
    assert resp.best_rejected.similarity < 0.5


def test_lexical_gate_rescues_verbatim_identifier(store):
    # Vector-orthogonal query whose token appears verbatim in d2's text: the
    # FTS leg returns it and the gate admits it despite similarity ~0.
    resp = _service(store, floor=0.5).execute_query("delete_by_source", limit=5)
    ids = [r.chunk.id for r in resp.results]
    assert ids == ["d2"]
    assert resp.results[0].retrieved_by in ("fts", "both")
    assert resp.results[0].similarity < 0.5


def test_all_terms_rule_rejects_partial_verbatim_match(store):
    # 'lock' is verbatim in d3 (FTS returns it) but 'narrowboat' is absent:
    # the all-terms rule rejects — the measured-stemming-noise defense.
    resp = _service(store, floor=0.5).execute_query("narrowboat lock", limit=5)
    assert resp.results == []
    assert resp.inspected > 0
    assert resp.best_rejected is not None


def test_stemming_overmatch_rejected_by_gate(store):
    # The measured false-positive class, end to end: FTS stemming makes the
    # query 'stores' retrieve d6 ('store' in text) — verified live on the
    # installed LanceDB/Tantivy — but the gate demands the token VERBATIM,
    # so the row is rejected. best_rejected's channel proves the stem hit
    # actually happened (a 'vector'-only channel here would mean the FTS
    # premise failed — investigate the FTS backend, don't loosen the assert).
    resp = _service(store, floor=0.5).execute_query("stores", limit=5)
    assert resp.results == []
    assert resp.best_rejected is not None
    assert resp.best_rejected.source == "shop.md"
    assert resp.best_rejected.retrieved_by == "both"
    assert resp.best_rejected.similarity < 0.5


def test_vector_only_fallback_floor_applies_no_lexical_rescue(tmp_path):
    # No FTS index: hybrid degrades to pure vector; retrieved_by is always
    # 'vector', so the lexical gate can never rescue a below-floor row.
    store = _make_store(tmp_path, with_fts=False)
    resp = _service(store, floor=0.5).execute_query("delete_by_source", limit=5)
    assert store._hybrid_ok is False
    assert resp.results == []  # verbatim token can't rescue: no FTS channel
    resp_unfloored = _service(store, floor=None).execute_query("delete_by_source", limit=5)
    assert resp_unfloored.results  # rows exist; the floor was what dropped them
    assert all(r.retrieved_by == "vector" for r in resp_unfloored.results)
    assert all(r.rrf_score is None for r in resp_unfloored.results)


def test_disable_similarity_floor_restores_baseline_pool(store):
    floored = _service(store, floor=0.5)
    baseline = floored.execute_query("beekeeping spring", limit=2, disable_similarity_floor=True)
    assert baseline.floor is None
    assert baseline.inspected == 2  # original pool: fetch limit == limit
    active = floored.execute_query("beekeeping spring", limit=2)
    assert active.inspected > 2  # oversampled pool (limit * 3 per leg)
