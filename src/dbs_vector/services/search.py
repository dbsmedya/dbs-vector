import json
from typing import Any

from loguru import logger

from dbs_vector.core.ports import IEmbedder, IVectorStore

_RETRIEVED_BY_LABELS = {"both": "vector+fts", "vector": "vector-only", "fts": "fts-only"}


def retrieved_by_label(value: str) -> str:
    """Render channel membership for text surfaces (vector+fts / vector-only / fts-only)."""
    return _RETRIEVED_BY_LABELS.get(value, value)


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

    def count_matching(
        self,
        source_filter: str | None = None,
        extra_filters: dict[str, Any] | None = None,
    ) -> int:
        """Count rows that would survive the same prefilters as execute_query."""
        return self.vector_store.count_matching(
            source_filter=source_filter,
            **(extra_filters or {}),
        )

    def results_to_json(self, results: list[Any]) -> str:
        """Serialize search results to a JSON array string with full fidelity.

        Unlike print_results, nothing is truncated: every result carries its
        similarity, retrieved_by, rrf_score, source, full text, and all chunk
        metadata.
        """
        payload = [res.model_dump(mode="json") for res in results]
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def print_results(self, results: list[Any]) -> None:
        """Formats and prints the parsed search results."""
        if not results:
            logger.info("No results found")
            return

        logger.info("Top Results:")
        for res in results:
            sim_str = f"{res.similarity:.2f} ({retrieved_by_label(res.retrieved_by)})"
            if hasattr(res.chunk, "raw_query"):
                logger.info(
                    "[Similarity: {} | DB: {} | Calls: {} | Time: {}ms]",
                    sim_str,
                    res.chunk.source,
                    res.chunk.calls,
                    res.chunk.execution_time_ms,
                )
                snippet = res.chunk.raw_query[:100].replace("\n", " ")
            else:
                logger.info(
                    "[Similarity: {} | Source: {} | Hash: {}]",
                    sim_str,
                    res.chunk.source,
                    res.chunk.content_hash,
                )
                snippet = res.chunk.text[:100].replace("\n", " ")
            logger.info('  --> "{}..."', snippet)
