"""envelope_payload is THE serializer: results_to_json must be a thin dumps."""

import json

from dbs_vector.core.models import Chunk, SearchResponse, SearchResult
from dbs_vector.services.search import envelope_payload


def _response() -> SearchResponse:
    chunk = Chunk(
        id="c1",
        text="hello world",
        source="notes/a.md",
        content_hash="deadbeefdeadbeef",
        parent_scope="## Heading",
        line_range="3-9",
    )
    return SearchResponse(
        results=[SearchResult(chunk=chunk, similarity=0.72, retrieved_by="both", rrf_score=0.03)],
        floor=None,
        inspected=5,
        best_rejected=None,
    )


def test_envelope_payload_shape() -> None:
    payload = envelope_payload(_response())
    assert set(payload) == {"floor", "inspected", "best_rejected", "source_resolution", "results"}
    assert payload["inspected"] == 5
    assert payload["results"][0]["similarity"] == 0.72
    assert payload["results"][0]["chunk"]["source"] == "notes/a.md"


def test_results_to_json_is_thin_dumps_of_envelope_payload() -> None:
    # Golden: the CLI --json output and the envelope are one serializer.
    from dbs_vector.services.search import SearchService

    response = _response()
    text = SearchService.results_to_json(SearchService.__new__(SearchService), response)
    assert json.loads(text) == envelope_payload(response)
