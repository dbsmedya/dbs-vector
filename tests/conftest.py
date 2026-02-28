"""Pytest configuration and fixtures."""

import pytest
from loguru import logger


@pytest.fixture
def caplog(caplog):
    """Enable Loguru logging to be captured by pytest's caplog fixture."""
    import logging

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)
