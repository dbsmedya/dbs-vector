from loguru import logger

from dbs_vector.config import settings
from dbs_vector.services.bootstrap import build_search_service
from dbs_vector.services.search import SearchService

# Global service instances holding the initialized models and databases
_services: dict[str, SearchService] = {}


def initialize_services() -> dict[str, SearchService]:
    """Initialize configured search services and return the service map."""
    _services.clear()
    for engine_name in settings.engines.keys():
        logger.info("Loading engine: {}", engine_name)
        _services[engine_name] = build_search_service(engine_name)
    return _services
