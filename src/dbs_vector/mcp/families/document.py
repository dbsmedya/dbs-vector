"""DocumentFamily — search engines whose results are document chunks."""

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar, cast

from mcp.types import CallToolResult, TextContent

from dbs_vector.core.models import ChunkDirection, ChunkPage, SearchResponse
from dbs_vector.mcp.families.base import (
    RESPONSE_BUDGET_BYTES,
    embeddings_phrase,
    floor_clause,
    render_with_budget,
)
from dbs_vector.services.search import (
    ACCEPTED_SOURCE_FORMS_DOCUMENT,
    SearchService,
    admission_phrase,
    envelope_payload,
    format_admission_empty,
    format_unmatched_source,
    retrieved_by_label,
)

if TYPE_CHECKING:
    from dbs_vector.config import EngineConfig


def _error_result(message: str) -> "SearchResponse":
    """isError result that still satisfies the declared envelope schema."""
    empty = SearchResponse(results=[], floor=None, inspected=0, best_rejected=None)
    return cast(
        SearchResponse,
        CallToolResult(
            content=[TextContent(type="text", text=message)],
            isError=True,
            structuredContent=envelope_payload(empty),
        ),
    )


class DocumentFamily:
    """SearchFamily implementation for document-style engines (markdown,
    prose, etc.)."""

    name: str = "document"
    # Document search declares structuredContent/outputSchema (spec decision
    # 7); SQL families stay text-only in v1 (absent means False via getattr).
    structured_search: ClassVar[bool] = True

    def run_search(
        self,
        service: SearchService,
        query: str,
        limit: int,
        source_filter: str | None,
        **family_kwargs: Any,
    ) -> SearchResponse:
        return service.execute_query(
            query,
            source_filter,
            limit,
            extra_filters={},
            min_similarity=family_kwargs.get("min_similarity"),
            disable_similarity_floor=bool(family_kwargs.get("disable_similarity_floor", False)),
        )

    def format_results(self, response: SearchResponse, query: str, total_matching: int = 0) -> str:
        results = response.results
        if not results:
            resolution = response.source_resolution
            if resolution is not None and resolution.is_unmatched:
                return format_unmatched_source(response, ACCEPTED_SOURCE_FORMS_DOCUMENT)
            if response.floor is not None and response.inspected > 0:
                return format_admission_empty(query, response)
            return f"No results found for query: '{query}'"

        suffix = (
            "" if response.floor is None else f", admission: {admission_phrase(response.floor)}"
        )
        header = f"Found {len(results)} results for '{query}' (hybrid-ranked{suffix}):\n"

        def _block(res: Any) -> str:
            chunk = res.chunk
            metadata = (
                f"--- Result (similarity {res.similarity:.2f}, "
                f"retrieved by: {retrieved_by_label(res.retrieved_by)}) ---\n"
                f"Source: {chunk.source}\n"
                f"Chunk cursor: {chunk.id}\n"
            )
            if chunk.parent_scope:
                metadata += f"Section: {chunk.parent_scope}\n"
            if chunk.line_range:
                metadata += f"Lines: {chunk.line_range}\n"
            return metadata + f"Content:\n{chunk.text}\n"

        return render_with_budget(
            header,
            (_block(res) for res in results),
            RESPONSE_BUDGET_BYTES,
            total=len(results),
        )

    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        emb = embeddings_phrase(engine.model)
        floor = floor_clause(engine)
        return (
            f"Hybrid semantic + full-text search over Markdown documentation "
            f"chunks ({emb}). Each result carries `similarity`: exact cosine "
            f"similarity in [-1, 1] between query and chunk embeddings — a "
            f"consistent geometric scale, NOT a calibrated probability of "
            f"relevance; comparisons are meaningful only within this engine/"
            f"configuration. Results are ordered by hybrid rank fusion, so "
            f"display order may disagree with similarity order. `retrieved_by` "
            f"reports only which retrieval channel(s) returned the row "
            f"(vector, fts, or both) — not evidence the match is correct. "
            f"{floor}`min_similarity` sets a per-call floor and "
            f"`disable_similarity_floor=true` disables admission filtering "
            f"entirely (exact unfloored baseline). `source_filter` narrows the "
            f"corpus and accepts a full stored path, a trailing fragment "
            f"('specs/api.md', 'api.md'), or a directory to scope to ('specs') "
            f"— but not globs or wildcards. If it resolves to nothing you get "
            f"a message saying so, which is NOT evidence about the corpus. "
            f"An empty response means no "
            f"inspected candidate passed admission — a low-confidence signal "
            f"for this attempt, NOT proof the corpus lacks relevant content."
            f" Each result includes a `Chunk cursor`; call `read_{engine_name.replace('-', '_')}` "
            f"with that cursor only when adjacent context is needed."
        )

    def format_chunk_page(self, page: ChunkPage) -> str:
        """Render an adjacent page while retaining every continuation signal."""
        if not page.chunks:
            edge = "start" if page.direction == "previous" else "end"
            return (
                f"Reached the {edge} of the document at chunk cursor "
                f"'{page.anchor_id}'; no {page.direction} chunk exists."
            )

        header = (
            f"Adjacent document chunks (direction: {page.direction}, "
            f"anchor: {page.anchor_id}, has_more: {str(page.has_more).lower()}, "
            f"continuation_cursor: {page.continuation_cursor or 'none'}, "
            f"boundary: {page.boundary or 'none'}):\n"
        )

        def _block(chunk: Any) -> str:
            metadata = f"--- Chunk ---\nSource: {chunk.source}\nChunk cursor: {chunk.id}\n"
            if chunk.parent_scope:
                metadata += f"Section: {chunk.parent_scope}\n"
            if chunk.line_range:
                metadata += f"Lines: {chunk.line_range}\n"
            return metadata + f"Content:\n{chunk.text}\n"

        return render_with_budget(
            header,
            (_block(chunk) for chunk in page.chunks),
            RESPONSE_BUDGET_BYTES,
            total=len(page.chunks),
        )

    def read_description(self, engine_name: str, engine: "EngineConfig") -> str:
        search_tool = f"search_{engine_name.replace('-', '_')}"
        return (
            f"Read previous or next chunks from the same document in engine "
            f"'{engine_name}'. Pass a `chunk_id` returned as `Chunk cursor` by "
            f"`{search_tool}`. This is an exact stored-text read: it performs no "
            f"embedding or semantic search. Use it only when a search result needs "
            f"adjacent context. `count` defaults to 1 and is limited to 3."
        )

    def make_read_handler(self, engine_name: str) -> Any:
        family = self

        async def handler(
            chunk_id: str,
            direction: ChunkDirection,
            count: int = 1,
        ) -> str:
            from dbs_vector.mcp.state import _services

            service = _services.get(engine_name)
            if service is None:
                return f"Error: search service '{engine_name}' is not initialized."
            try:
                page = await asyncio.to_thread(
                    service.read_adjacent_chunks,
                    chunk_id,
                    direction,
                    count,
                )
                return family.format_chunk_page(page)
            except Exception as e:
                return f"Chunk read failed: {e}"

        return handler

    def make_handler(self, engine_name: str, allow_raw_queries: bool = False) -> Any:
        # documents have no raw_query; the egress flag is accepted for
        # SearchFamily Protocol parity and intentionally ignored.
        family = self  # closure capture

        async def handler(
            query: str,
            limit: int = 5,
            source_filter: str | None = None,
            min_similarity: float | None = None,
            disable_similarity_floor: bool = False,
        ) -> SearchResponse:
            from dbs_vector.mcp.state import _services  # lazy import

            service = _services.get(engine_name)
            if service is None:
                return _error_result(f"Error: search service '{engine_name}' is not initialized.")
            if min_similarity is not None and not (-1.0 <= min_similarity <= 1.0):
                return _error_result(
                    f"min_similarity must be within [-1, 1]; got {min_similarity}."
                )
            try:
                response = await asyncio.to_thread(
                    family.run_search,
                    service,
                    query,
                    limit,
                    source_filter,
                    min_similarity=min_similarity,
                    disable_similarity_floor=disable_similarity_floor,
                )
                prose = family.format_results(response, query)
                return cast(
                    SearchResponse,  # annotation drives outputSchema; SDK accepts CallToolResult
                    CallToolResult(
                        content=[TextContent(type="text", text=prose)],
                        structuredContent=envelope_payload(response),
                    ),
                )
            except Exception as e:
                return _error_result(f"Search execution failed: {e}")

        return handler
