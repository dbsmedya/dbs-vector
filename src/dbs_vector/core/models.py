import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel


class Chunk(BaseModel):
    """A semantic piece of data extracted from a document."""

    id: str
    text: str
    source: str
    content_hash: str
    # Phase 2 AST Additions (Ready for LibCST)
    node_type: str | None = None
    parent_scope: str | None = None
    line_range: str | None = None


class SqlChunk(BaseModel):
    """A single normalized SQL query parsed from a slow query log or stat statements."""

    id: str
    text: str  # The normalized query used for embedding
    raw_query: str
    source: str  # e.g., database name
    execution_time_ms: float
    calls: int
    content_hash: str
    tables: list[str] = []
    latest_ts: datetime
    user: str | None = None
    host: str | None = None
    rows_sent: int | None = None
    rows_examined: int | None = None
    lock_time_sec: float | None = None


def _coerce_optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _coerce_optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _coerce_latest_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(UTC)
    return datetime.now(UTC)


def sql_chunk_from_record(record: Mapping[str, Any]) -> SqlChunk:
    """Build a SqlChunk from a normalized record mapping."""
    text = str(record["text"])
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    tables = record.get("tables") or []

    return SqlChunk(
        id=str(record["id"]),
        text=text,
        raw_query=str(record.get("raw_query") or ""),
        source=str(record["source"]),
        execution_time_ms=float(record.get("execution_time_ms") or 0.0),
        calls=int(record.get("calls") or 1),
        content_hash=content_hash,
        tables=list(tables),
        latest_ts=_coerce_latest_ts(record.get("latest_ts")),
        user=str(record["user"]) if record.get("user") is not None else None,
        host=str(record["host"]) if record.get("host") is not None else None,
        rows_sent=_coerce_optional_int(record.get("rows_sent")),
        rows_examined=_coerce_optional_int(record.get("rows_examined")),
        lock_time_sec=_coerce_optional_float(record.get("lock_time_sec")),
    )


class Document(BaseModel):
    """A raw document before chunking."""

    filepath: str
    content: str
    content_hash: str


class SearchResult(BaseModel):
    """A matched chunk returned from the vector store."""

    chunk: Chunk
    score: float | None = None
    distance: float | None = None
    is_fts_match: bool = False


class SqlSearchResult(BaseModel):
    """A matched SQL chunk returned from the vector store."""

    chunk: SqlChunk
    score: float | None = None
    distance: float | None = None
    is_fts_match: bool = False
