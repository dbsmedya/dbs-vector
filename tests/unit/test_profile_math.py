import pytest

from dbs_vector.core.model_registry import ModelContract
from dbs_vector.core.profile_math import (
    _PEAK_BUFFER_OVERHEAD,
    estimate_peak_buffer_bytes,
    recommend_profile,
)


class FakeProfile:
    def __init__(self, max_token_length: int, batch_size: int, chunk_max_chars: int = 0) -> None:
        self.max_token_length = max_token_length
        self.batch_size = batch_size
        self.chunk_max_chars = chunk_max_chars


GRANITE = ModelContract(
    model_name="ibm-granite/granite-embedding-311m-multilingual-r2",
    vector_dimension=768,
    model_max_token_length=32768,
    attention_mask_dtype=None,
    compute_dtype_bytes=2,
)


def test_estimate_calibration_point():
    """Calibration: batch=64, seq=16384, bf16 → raw 34.4 GB; with 3.0× factor ~103 GB."""
    profile = FakeProfile(max_token_length=16384, batch_size=64)
    bytes_estimated = estimate_peak_buffer_bytes(profile, GRANITE)
    raw_attention = 64 * (16384 ** 2) * 2
    expected = int(_PEAK_BUFFER_OVERHEAD * raw_attention)
    assert bytes_estimated == expected
    assert bytes_estimated > 100 * 1000 ** 3  # 100 GB decimal, matching the ~103 GB comment
    assert bytes_estimated < 110 * 1024 ** 3


def test_estimate_small_profile_fits_easily():
    profile = FakeProfile(max_token_length=2048, batch_size=64)
    bytes_estimated = estimate_peak_buffer_bytes(profile, GRANITE)
    assert bytes_estimated < 2 * 1024 ** 3


def test_recommend_preserves_target_seq_len():
    """The 16K/64 OOM case should be recommended as 16K/<smaller batch>."""
    result = recommend_profile(GRANITE, memory_budget_gb=22.0, target_seq_len=16384)
    assert result["max_token_length"] == 16384
    assert result["batch_size"] >= 1
    assert result["seq_len_reduced"] is False


def test_recommend_clamps_to_model_cap():
    result = recommend_profile(GRANITE, memory_budget_gb=22.0, target_seq_len=99999)
    assert result["max_token_length"] <= GRANITE.model_max_token_length


def test_recommend_atomic_chunks_for_sql():
    result = recommend_profile(
        GRANITE, memory_budget_gb=22.0, target_chunker="duckdb", target_seq_len=8192
    )
    assert result["chunk_max_chars"] == 0


def test_recommend_doc_chunks_use_char_heuristic():
    result = recommend_profile(
        GRANITE, memory_budget_gb=22.0, target_chunker="document", target_seq_len=8192
    )
    expected_chunk = int(8192 * 2.5 * 0.5)
    assert result["chunk_max_chars"] == expected_chunk


def test_recommend_seq_reduced_flag_when_halving():
    """A tiny budget that can't fit even batch=1 at target seq forces halving."""
    result = recommend_profile(GRANITE, memory_budget_gb=0.5, target_seq_len=32768)
    assert result["seq_len_reduced"] is True
    assert result["max_token_length"] < 32768


def test_recommend_raises_if_no_fit():
    """A pathologically tiny budget can't fit even seq=512."""
    with pytest.raises(ValueError, match="No profile fits"):
        recommend_profile(GRANITE, memory_budget_gb=0.0001, target_seq_len=512)


def test_recommend_defaults_target_to_model_cap():
    result = recommend_profile(GRANITE, memory_budget_gb=22.0)
    assert result["max_token_length"] >= 1
