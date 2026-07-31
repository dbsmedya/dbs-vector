from unittest.mock import patch

import pytest

from dbs_vector.infrastructure.hardware import (
    configure_mlx_memory_limits,
    detect_memory_budget_gb,
    resolve_memory_budget_gb,
)


def test_detect_returns_gb_from_mlx():
    fake_info = {"max_buffer_length": 22 * 1024**3}
    with patch("mlx.core.device_info", return_value=fake_info):
        assert detect_memory_budget_gb() == pytest.approx(22.0, rel=1e-6)


def test_detect_returns_none_when_mlx_unavailable():
    with patch("mlx.core.device_info", side_effect=ImportError("no metal")):
        assert detect_memory_budget_gb() is None


def test_detect_returns_none_when_key_missing():
    with patch("mlx.core.device_info", return_value={}):
        assert detect_memory_budget_gb() is None


def test_resolve_configured_wins():
    with patch("dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=22.0):
        assert resolve_memory_budget_gb(8.0) == 8.0


def test_resolve_falls_back_to_detected():
    with patch("dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=22.0):
        assert resolve_memory_budget_gb(None) == 22.0


def test_resolve_raises_when_neither_available():
    with patch("dbs_vector.infrastructure.hardware.detect_memory_budget_gb", return_value=None):
        with pytest.raises(ValueError, match="Could not auto-detect"):
            resolve_memory_budget_gb(None)


def test_configure_explicit_limits(monkeypatch):
    monkeypatch.setattr("dbs_vector.infrastructure.hardware._applied_limits", None)
    with (
        patch("dbs_vector.infrastructure.hardware.resolve_memory_budget_gb") as resolve,
        patch("mlx.core.set_memory_limit") as set_memory,
        patch("mlx.core.set_cache_limit") as set_cache,
    ):
        result = configure_mlx_memory_limits(
            memory_budget_gb=22.0,
            memory_limit_gb=16.0,
            cache_limit_gb=2.0,
        )

    assert result == (16.0, 2.0)
    resolve.assert_not_called()
    set_memory.assert_called_once_with(16 * 1024**3)
    set_cache.assert_called_once_with(2 * 1024**3)


def test_configure_missing_limits_fall_back_to_memory_budget(monkeypatch):
    monkeypatch.setattr("dbs_vector.infrastructure.hardware._applied_limits", None)
    with (
        patch("mlx.core.set_memory_limit") as set_memory,
        patch("mlx.core.set_cache_limit") as set_cache,
    ):
        result = configure_mlx_memory_limits(
            memory_budget_gb=22.0,
            memory_limit_gb=None,
            cache_limit_gb=None,
        )

    assert result == (22.0, 22.0)
    set_memory.assert_called_once_with(22 * 1024**3)
    set_cache.assert_called_once_with(22 * 1024**3)


def test_configure_cache_defaults_to_explicit_memory_limit(monkeypatch):
    monkeypatch.setattr("dbs_vector.infrastructure.hardware._applied_limits", None)
    with (
        patch("dbs_vector.infrastructure.hardware.resolve_memory_budget_gb") as resolve,
        patch("mlx.core.set_memory_limit") as set_memory,
        patch("mlx.core.set_cache_limit") as set_cache,
    ):
        result = configure_mlx_memory_limits(
            memory_budget_gb=None,
            memory_limit_gb=8.0,
            cache_limit_gb=None,
        )

    assert result == (8.0, 8.0)
    resolve.assert_not_called()
    set_memory.assert_called_once_with(8 * 1024**3)
    set_cache.assert_called_once_with(8 * 1024**3)


def test_configure_zero_cache_limit_disables_mlx_cache(monkeypatch):
    monkeypatch.setattr("dbs_vector.infrastructure.hardware._applied_limits", None)
    with (
        patch("mlx.core.set_memory_limit"),
        patch("mlx.core.set_cache_limit") as set_cache,
    ):
        configure_mlx_memory_limits(
            memory_budget_gb=22.0,
            memory_limit_gb=8.0,
            cache_limit_gb=0.0,
        )

    set_cache.assert_called_once_with(0)


def test_configure_rejects_cache_above_total_limit(monkeypatch):
    monkeypatch.setattr("dbs_vector.infrastructure.hardware._applied_limits", None)
    with (
        patch("mlx.core.set_memory_limit") as set_memory,
        patch("mlx.core.set_cache_limit") as set_cache,
        pytest.raises(ValueError, match="mlx_cache_limit_gb"),
    ):
        configure_mlx_memory_limits(
            memory_budget_gb=22.0,
            memory_limit_gb=8.0,
            cache_limit_gb=9.0,
        )

    set_memory.assert_not_called()
    set_cache.assert_not_called()


def test_configure_same_limits_is_idempotent(monkeypatch):
    monkeypatch.setattr("dbs_vector.infrastructure.hardware._applied_limits", None)
    with (
        patch("mlx.core.set_memory_limit") as set_memory,
        patch("mlx.core.set_cache_limit") as set_cache,
    ):
        kwargs = {
            "memory_budget_gb": 22.0,
            "memory_limit_gb": 8.0,
            "cache_limit_gb": 1.0,
        }
        configure_mlx_memory_limits(**kwargs)
        configure_mlx_memory_limits(**kwargs)

    set_memory.assert_called_once()
    set_cache.assert_called_once()
