"""No query text at INFO-and-above: the daemon's logfile must stay corpus-free."""

from datetime import UTC, datetime

from loguru import logger

from dbs_vector.core.models import SearchResponse, SqlChunk, SqlSearchResult
from dbs_vector.mcp.families.sql import SqlFamily


def _response():
    chunk = SqlChunk(
        id="q1",
        text="SELECT ?",
        raw_query="SELECT 1",
        source="db1",
        execution_time_ms=1.0,
        calls=1,
        content_hash="deadbeefdeadbeef",
        tables=["t"],
        latest_ts=datetime(2026, 8, 4, tzinfo=UTC),
    )
    return SearchResponse(
        results=[SqlSearchResult(chunk=chunk, similarity=0.8, retrieved_by="vector")],
        floor=None,
        inspected=1,
    )


def test_mismatch_warning_fires_without_query_text():
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="INFO")
    try:
        # len(results)=1 > total_matching=0 triggers the invariant warning.
        SqlFamily().format_results(_response(), "SECRET-NEEDLE", total_matching=0)
    finally:
        logger.remove(sink_id)
    assert any("count_matching and search disagreed" in r for r in records)
    assert not any("SECRET-NEEDLE" in r for r in records)
