from dbs_vector.services.search import SearchService

# Global service instances holding the initialized models and databases
_services: dict[str, SearchService] = {}
