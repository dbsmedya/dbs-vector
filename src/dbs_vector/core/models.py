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
