from typing import Any

from loguru import logger

from dbs_vector.core.ports import IEmbedder, IVectorStore


class SearchService:
    """Orchestrates hybrid vector search and formats results."""

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def execute_query(
        self,
        query: str,
        source_filter: str | None = None,
        limit: int = 5,
        extra_filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Embeds the query and fetches top matches from the high-performance store."""
        logger.info("Executing query: {}", query)

        if extra_filters is None:
            extra_filters = {}

        # Step 1: Embed Query (Ensures correct shape)
        query_vector = self.embedder.embed_query(query)

        # Step 2: Rust-level Vector & FTS Search
        results = self.vector_store.search(
            query=query,
            query_vector=query_vector,
            source_filter=source_filter,
            limit=limit,
            **extra_filters,
        )
        return results

    def print_results(self, results: list[Any]) -> None:
        """Formats and prints the parsed search results."""
        if not results:
            logger.info("No results found")
            return

        logger.info("Top Results:")
        for res in results:
            dist_str = f"{res.distance:.4f}" if res.distance is not None else "N/A (FTS Match)"

            # Polymorphic printing
            if hasattr(res.chunk, "raw_query"):
                # SQL Result
                logger.info(
                    "[Score/Dist: {} | DB: {} | Calls: {} | Time: {}ms]",
                    dist_str,
                    res.chunk.source,
                    res.chunk.calls,
                    res.chunk.execution_time_ms,
                )
                snippet = res.chunk.raw_query[:100].replace("\n", " ")
            else:
                # Document Result
                logger.info(
                    "[Score/Dist: {} | Source: {} | Hash: {}]",
                    dist_str,
                    res.chunk.source,
                    res.chunk.content_hash,
                )
                snippet = res.chunk.text[:100].replace("\n", " ")

            logger.info('  --> "{}..."', snippet)
