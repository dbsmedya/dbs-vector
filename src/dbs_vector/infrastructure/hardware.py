import threading

from loguru import logger

_GIB = 1024**3
_limits_lock = threading.Lock()
_applied_limits: tuple[int, int] | None = None


def detect_memory_budget_gb() -> float | None:
    """Try to read Metal's max buffer length. Return None if unavailable."""
    try:
        import mlx.core as mx

        info = mx.device_info()
        max_bytes = info.get("max_buffer_length")
        if isinstance(max_bytes, (int, float)) and max_bytes:
            return max_bytes / (1024**3)
    except Exception as e:  # noqa: BLE001 — any failure means "fall back to config"
        logger.debug("Metal device_info unavailable: {}", e)
    return None


def resolve_memory_budget_gb(configured: float | None) -> float:
    """Resolve final memory budget. Configured wins; else auto-detect; else raise."""
    if configured is not None:
        return configured
    detected = detect_memory_budget_gb()
    if detected is not None:
        logger.info("Auto-detected memory budget: {:.1f} GB", detected)
        return detected
    raise ValueError(
        "Could not auto-detect Metal memory budget. "
        "Set system.memory_budget_gb in config.yaml (e.g., 16.0)."
    )


def configure_mlx_memory_limits(
    *,
    memory_budget_gb: float | None,
    memory_limit_gb: float | None,
    cache_limit_gb: float | None,
) -> tuple[float, float]:
    """Apply process-wide MLX allocator limits and return them in GiB.

    ``memory_limit_gb`` is optional and falls back to ``memory_budget_gb``
    (including its Metal auto-detection). ``cache_limit_gb`` is optional and
    falls back to the resolved total memory limit. Repeated calls with the same
    byte values are no-ops because one process may build several engines that
    share the same MLX allocator.
    """
    global _applied_limits

    resolved_memory_gb = (
        memory_limit_gb
        if memory_limit_gb is not None
        else resolve_memory_budget_gb(memory_budget_gb)
    )
    resolved_cache_gb = cache_limit_gb if cache_limit_gb is not None else resolved_memory_gb
    if resolved_cache_gb > resolved_memory_gb:
        raise ValueError(
            "system.mlx_cache_limit_gb must be <= the resolved MLX memory limit "
            f"({resolved_cache_gb:g} GB > {resolved_memory_gb:g} GB)."
        )

    limits = (
        int(resolved_memory_gb * _GIB),
        int(resolved_cache_gb * _GIB),
    )
    with _limits_lock:
        if limits == _applied_limits:
            return resolved_memory_gb, resolved_cache_gb

        import mlx.core as mx

        mx.set_memory_limit(limits[0])
        mx.set_cache_limit(limits[1])
        _applied_limits = limits

    logger.info(
        "Configured MLX allocator limits: memory={:.1f} GB, cache={:.1f} GB",
        resolved_memory_gb,
        resolved_cache_gb,
    )
    return resolved_memory_gb, resolved_cache_gb
