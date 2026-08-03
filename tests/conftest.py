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


@pytest.fixture
def temp_model_registry():
    """Snapshot and restore ModelRegistry so a test may register synthetic models.

    Required: tests/unit/test_model_registry.py asserts the exact key list
    ['gemma-bf16', 'granite-r2'], so a leaked registration fails an unrelated
    test. ModelRegistry has no _reset_for_testing(), hence the manual snapshot.
    """
    from dbs_vector.core.model_registry import ModelRegistry

    snapshot = dict(ModelRegistry._models)
    yield ModelRegistry
    ModelRegistry._models.clear()
    ModelRegistry._models.update(snapshot)
