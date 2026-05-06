from unittest.mock import patch

import pytest

from dbs_vector.infrastructure.hardware import (
    detect_memory_budget_gb,
    resolve_memory_budget_gb,
)


def test_detect_returns_gb_from_mlx():
    fake_info = {"max_buffer_length": 22 * 1024 ** 3}
    with patch("mlx.core.metal.device_info", return_value=fake_info):
        assert detect_memory_budget_gb() == pytest.approx(22.0, rel=1e-6)


def test_detect_returns_none_when_mlx_unavailable():
    with patch("mlx.core.metal.device_info", side_effect=ImportError("no metal")):
        assert detect_memory_budget_gb() is None


def test_detect_returns_none_when_key_missing():
    with patch("mlx.core.metal.device_info", return_value={}):
        assert detect_memory_budget_gb() is None


def test_resolve_configured_wins():
    with patch(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=22.0
    ):
        assert resolve_memory_budget_gb(8.0) == 8.0


def test_resolve_falls_back_to_detected():
    with patch(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=22.0
    ):
        assert resolve_memory_budget_gb(None) == 22.0


def test_resolve_raises_when_neither_available():
    with patch(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=None
    ):
        with pytest.raises(ValueError, match="Could not auto-detect"):
            resolve_memory_budget_gb(None)
