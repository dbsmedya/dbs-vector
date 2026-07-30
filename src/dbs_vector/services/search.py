import json
from typing import Any

from loguru import logger

from dbs_vector.core.models import RejectedCandidate, SearchResponse, SqlChunk
from dbs_vector.core.ports import IEmbedder, IVectorStore
from dbs_vector.services.admission import eligible_tokens, lexical_gate

_RETRIEVED_BY_LABELS = {"both": "vector+fts", "vector": "vector-only", "fts": "fts-only"}

# When a floor is active, fetch this multiple of `limit` so floor-filtering
# doesn't starve the requested limit. Enlarged per-leg pools change RRF
# fusion inputs — a deliberate, spec-stated behavior change (floor-active
# paths only; measured in the companion spec).
_FLOOR_OVERSAMPLE = 3

# LLM-callable surface guard: LanceDB treats a non-positive limit as "no
# limit" (unbounded fetch before app-side truncation), and floor mode
# multiplies the fetch by _FLOOR_OVERSAMPLE.
_MAX_LIMIT = 100


def retrieved_by_label(value: str) -> str:
    """Render channel membership for text surfaces (vector+fts / vector-only / fts-only)."""
    return _RETRIEVED_BY_LABELS.get(value, value)


def admission_phrase(floor: float) -> str:
    """The one-line admission rule, rendered identically on every surface."""
    return f"similarity >= {floor:g} or all query terms verbatim"


def format_admission_empty(query: str, response: SearchResponse) -> str:
    """Empty-because-admission-filtered message: low retrieval confidence for
    THIS attempt — never an assertion of corpus-level absence (only the
    inspected pool is known)."""
    floor = response.floor
    if floor is None:  # defensive; callers gate on floor being active
        return f"No results found for query: '{query}'"
    msg = (
        f"No inspected candidate passed admission ({admission_phrase(floor)}) "
        f"for '{query}'. Inspected {response.inspected} hybrid-ranked candidates"
    )
    best = response.best_rejected
    if best is not None:
        msg += (
            f"; best was similarity {best.similarity:.2f} "
            f"({best.source}, {retrieved_by_label(best.retrieved_by)})"
        )
    msg += (
        ". Retrieval confidence for this attempt is low; this does not "
        "establish corpus-level absence. Retry with different terms or a "
        "lower min_similarity if you expected a match."
    )
    return msg


class SearchService:
    """Orchestrates hybrid vector search and formats results."""

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        similarity_floor: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_floor = similarity_floor

    def execute_query(
        self,
        query: str,
        source_filter: str | None = None,
        limit: int = 5,
        extra_filters: dict[str, Any] | None = None,
        min_similarity: float | None = None,
        disable_similarity_floor: bool = False,
    ) -> SearchResponse:
        """Embed, fetch hybrid-ranked candidates, apply admission policy.

        Floor precedence: disable_similarity_floor (no floor AND the original
        candidate-pool size — the exact-baseline state; min_similarity=0.0 is
        NOT equivalent) > per-call min_similarity > engine similarity_floor >
        no floor. Input validation (limit and min_similarity ranges) is
        unconditional — it runs even when disable_similarity_floor=True, so
        garbage input fails loudly instead of being masked by the flag.
        """
        if not 1 <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be within [1, {_MAX_LIMIT}]; got {limit}")
        if min_similarity is not None and not (-1.0 <= min_similarity <= 1.0):
            raise ValueError(f"min_similarity must be within [-1, 1]; got {min_similarity}")
        logger.info("Executing query: {}", query)
        if extra_filters is None:
            extra_filters = {}

        if disable_similarity_floor:
            floor: float | None = None
        elif min_similarity is not None:
            floor = min_similarity
        else:
            floor = self.similarity_floor
        fetch_limit = limit if floor is None else limit * _FLOOR_OVERSAMPLE

        query_vector = self.embedder.embed_query(query)
        candidates = self.vector_store.search(
            query=query,
            query_vector=query_vector,
            source_filter=source_filter,
            limit=fetch_limit,
            **extra_filters,
        )
        inspected = len(candidates)
        if floor is None:
            return SearchResponse(
                results=candidates[:limit], floor=None, inspected=inspected, best_rejected=None
            )

        eligible = eligible_tokens(query)
        admitted: list[Any] = []
        rejected: list[Any] = []
        for cand in candidates:
            if cand.similarity >= floor or lexical_gate(
                eligible, cand.retrieved_by, cand.chunk.text
            ):
                admitted.append(cand)
            else:
                rejected.append(cand)
        best = max(rejected, key=lambda c: c.similarity, default=None)
        best_rejected = (
            RejectedCandidate(
                similarity=best.similarity,
                source=best.chunk.source,
                retrieved_by=best.retrieved_by,
            )
            if best is not None
            else None
        )
        return SearchResponse(
            results=admitted[:limit],
            floor=floor,
            inspected=inspected,
            best_rejected=best_rejected,
        )

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

    def results_to_json(self, response: SearchResponse) -> str:
        """Serialize a SearchResponse to a JSON envelope with full fidelity.

        Unlike print_results, nothing is truncated: every result carries its
        similarity, retrieved_by, rrf_score, source, full text, and all chunk
        metadata.
        """
        payload = {
            "floor": response.floor,
            "inspected": response.inspected,
            "best_rejected": (
                response.best_rejected.model_dump(mode="json")
                if response.best_rejected is not None
                else None
            ),
            "results": [res.model_dump(mode="json") for res in response.results],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def print_results(self, response: SearchResponse, query: str = "") -> None:
        """Formats and prints the parsed search results."""
        results = response.results
        if not results:
            if response.floor is not None and response.inspected > 0:
                logger.info("{}", format_admission_empty(query, response))
            else:
                logger.info("No results found")
            return

        logger.info("Top Results:")
        for res in results:
            sim_str = f"{res.similarity:.2f} ({retrieved_by_label(res.retrieved_by)})"
            if isinstance(res.chunk, SqlChunk):
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
