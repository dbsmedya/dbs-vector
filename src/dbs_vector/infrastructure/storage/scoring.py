"""Pure scoring helpers: exact cosine similarity + retrieval-channel provenance.

No LanceDB import, no I/O — unit-testable in isolation. Used by
LanceDBStore.search to annotate every returned row.
"""

import math
from typing import Any

import numpy as np
from loguru import logger

from dbs_vector.core.models import RetrievedBy


def cosine_similarity(query_vector: Any, row_vector: Any) -> float:
    """Exact cosine similarity in [-1, 1] between two vectors.

    Metric-independent and defined even for FTS-only rows (whose LanceDB
    `_distance` is null). Guards: either norm 0 -> 0.0; non-finite result
    (inf/nan inputs) -> 0.0 with a warning, because a NaN would silently
    fail every floor comparison and poison best_rejected selection.
    """
    q = np.asarray(query_vector, dtype=np.float64).ravel()
    v = np.asarray(row_vector, dtype=np.float64).ravel()
    q_norm = float(np.linalg.norm(q))
    v_norm = float(np.linalg.norm(v))
    if q_norm == 0.0 or v_norm == 0.0:
        return 0.0
    sim = float(np.dot(q, v) / (q_norm * v_norm))
    if not math.isfinite(sim):
        logger.warning("Non-finite cosine similarity computed; substituting 0.0")
        return 0.0
    return max(-1.0, min(1.0, sim))


def classify_retrieved_by(distance: float | None, fts_score: float | None) -> RetrievedBy:
    """Map the RRFReranker(return_score="all") null pattern to channel membership.

    `distance` is the row's `_distance` (vector leg), `fts_score` its `_score`
    (FTS/BM25 leg); null means that leg did not return the row.
    """
    if distance is not None and fts_score is not None:
        return "both"
    if distance is not None:
        return "vector"
    if fts_score is not None:
        return "fts"
    raise ValueError(
        "Hybrid result row carries neither _distance nor _score; "
        "cannot classify retrieval channel (programming error)."
    )
