"""Centralized Loguru configuration for dbs-vector."""

import os
import sys

# Suppress Hugging Face progress bars and telemetry globally
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from loguru import logger

# Remove the default stderr sink so we can reconfigure it
logger.remove()

# Default sink: stderr with INFO level, colored, concise format
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
)


def configure_logger(level: str = "INFO", serialize: bool = False) -> None:
    """Reconfigure the global logger (called from CLI or config).

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        serialize: If True, output JSON lines instead of human-readable text.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "{message}"
            if serialize
            else "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        ),
        serialize=serialize,
        colorize=not serialize,
    )
