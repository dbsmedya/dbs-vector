from loguru import logger

from dbs_vector.cli import _build_dependencies
from dbs_vector.config import settings
from dbs_vector.services.search import SearchService

# Global service instances holding the initialized models and databases
_services: dict[str, SearchService] = {}


def initialize_services() -> dict[str, SearchService]:
    """Initialize configured search services and return the service map."""
    _services.clear()
    for engine_name in settings.engines.keys():
        logger.info("Loading engine: {}", engine_name)
        deps = _build_dependencies(engine_name)
        _services[engine_name] = SearchService(deps.embedder, deps.store)
    return _services
