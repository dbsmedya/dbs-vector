from loguru import logger


def detect_memory_budget_gb() -> float | None:
    """Try to read Metal's max buffer length. Return None if unavailable."""
    try:
        import mlx.core as mx

        info = mx.metal.device_info()
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
