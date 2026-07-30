"""Unit tests for pure scoring helpers: exact cosine + channel provenance."""

import math

import numpy as np
import pytest

from dbs_vector.infrastructure.storage.scoring import (
    classify_retrieved_by,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_unit_vectors_give_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_give_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_give_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_scale_invariant(self):
        assert cosine_similarity([2.0, 0.0], [0.5, 0.0]) == pytest.approx(1.0)

    def test_known_angle(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(math.sqrt(2) / 2)

    def test_zero_query_norm_gives_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_row_norm_gives_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_result_clamped_to_declared_range(self):
        v = np.asarray([0.1] * 768, dtype=np.float32)
        assert -1.0 <= cosine_similarity(v, v) <= 1.0

    def test_non_finite_input_gives_zero(self):
        assert cosine_similarity([np.inf, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_similarity([np.nan, 0.0], [1.0, 0.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_accepts_python_lists_and_numpy_arrays(self):
        q = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(q, [1.0, 2.0, 3.0]) == pytest.approx(1.0)


class TestClassifyRetrievedBy:
    def test_both_legs(self):
        assert classify_retrieved_by(0.12, 3.4) == "both"

    def test_vector_only(self):
        assert classify_retrieved_by(0.12, None) == "vector"

    def test_fts_only(self):
        assert classify_retrieved_by(None, 3.4) == "fts"

    def test_neither_leg_raises(self):
        with pytest.raises(ValueError, match="neither"):
            classify_retrieved_by(None, None)
