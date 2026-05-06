import pytest
from pydantic import ValidationError

from dbs_vector.config import TuningProfile


def test_parses_valid_profile():
    profile = TuningProfile(max_token_length=8192, chunk_max_chars=4000, batch_size=8)
    assert profile.max_token_length == 8192
    assert profile.chunk_max_chars == 4000
    assert profile.batch_size == 8


def test_atomic_chunks_allowed():
    profile = TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=64)
    assert profile.chunk_max_chars == 0


def test_zero_max_token_length_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=0, chunk_max_chars=100, batch_size=8)


def test_negative_max_token_length_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=-1, chunk_max_chars=100, batch_size=8)


def test_zero_batch_size_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=2048, chunk_max_chars=100, batch_size=0)


def test_negative_chunk_max_rejected():
    with pytest.raises(ValidationError):
        TuningProfile(max_token_length=2048, chunk_max_chars=-1, batch_size=8)


def test_extra_fields_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        TuningProfile(
            max_token_length=2048,
            chunk_max_chars=100,
            batch_size=8,
            unknown_field="x",
        )
