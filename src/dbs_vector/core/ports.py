from collections.abc import Callable, Iterator
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from dbs_vector.core.models import RetrievedBy
from dbs_vector.core.source_scope import SourceResolution


class IEmbedder(Protocol):
    """Protocol defining how an embedder should behave."""

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension size."""
        ...

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Converts a batch of text strings into a contiguous float32 NumPy array."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Converts a single query string into a flat float32 NumPy array."""
        ...

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens (incl. special tokens) for `text`."""
        ...


class IStoreMapper(Protocol):
    def __init__(self, vector_dimension: int) -> None: ...

    @property
    def schema(self) -> Any:
        """Returns the PyArrow schema for the table."""
        ...

    def to_record_batch(
        self, chunks: list[Any], vectors: NDArray[np.float32], workflow: str
    ) -> Any:
        """Converts domain chunks and vectors into a PyArrow RecordBatch."""
        ...

    def from_polars_row(
        self,
        row: dict[str, Any],
        *,
        similarity: float,
        retrieved_by: RetrievedBy,
        rrf_score: float | None,
    ) -> Any:
        """Convert one Polars result row into a domain search-result model.

        `similarity` is exact query-to-row cosine; `retrieved_by` is the
        retrieval channel that returned the row; `rrf_score` is the fused RRF
        value (None when no fusion ran — the pure-vector fallback path).
        """
        ...


class IVectorStore(Protocol):
    """Protocol defining how a high-performance vector store behaves."""

    def clear(self) -> None:
        """Completely drops the vector store table/index to start fresh."""
        ...

    def ingest_chunks(self, chunks: list[Any], vectors: NDArray[np.float32], workflow: str) -> None:
        """Appends chunks and their matching vectors into the store (Zero-Copy ideal)."""
        ...

    def compact(self) -> None:
        """Compacts fragmented datasets into optimal read files."""
        ...

    def create_indices(self) -> None:
        """Generates dynamic vector (IVF-PQ) and Full-Text Search indices."""
        ...

    def get_existing_hashes(self) -> set[str]:
        """Returns a set of all unique content hashes currently in the store."""
        ...

    def delete_by_source(self, source: str) -> None:
        """Delete every row whose `source` column equals `source`.

        Whole-file replace primitive for the watcher: no row count is
        returned (a pre-count scan per event is not worth it).
        """
        ...

    def refresh_fts(self) -> None:
        """Rebuild ONLY the full-text index. No vector index work."""
        ...

    def search(
        self,
        query: str,
        query_vector: NDArray[np.float32],
        source_filter: str | None = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> list[Any]:
        """Performs a hybrid search and returns mapped SearchResult models."""
        ...

    def count_matching(
        self,
        source_filter: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Count rows that survive the same prefilter set used by `search`."""
        ...

    def resolve_source_filter(self, value: str) -> SourceResolution:
        """Resolve a caller's `source_filter` against the stored sources.

        Implementations MUST use the same resolution in `search` and
        `count_matching`, or a "showing N of M" caller reports a total its
        result set can never reach.
        """
        ...

    def scan(self, columns: list[str] | None = None) -> Any:
        """Read ALL rows as a pyarrow.Table for in-process analytical SQL.

        `columns=None` projects every column EXCEPT the embedding `vector`
        and `workflow`. Implementations MUST call checkout_latest() first
        (like search()/count_matching()) so the read sees the latest version.
        No vector query, no row cap.
        """
        ...


class IContentFilter(Protocol):
    """Decides whether to skip a whole file or drop a single block."""

    name: str

    def should_skip_file(self, filepath: str, content: str) -> bool: ...

    def should_drop_block(self, text: str, info_string: str | None) -> bool: ...


class IChunker(Protocol):
    """Protocol defining how documents are split into logical boundaries."""

    @property
    def supported_extensions(self) -> list[str]:
        """Returns a list of file extensions this chunker can process (e.g., ['.md', '.txt'])."""
        ...

    def process(self, document: Any) -> Iterator[Any]:
        """Yields chunks from a raw document or parsed log."""
        ...


# (path, kind, is_directory, dest) — kind is one of
# "created" | "modified" | "deleted" | "moved"; `dest` is set for moves only.
WatchCallback = Callable[[str, str, bool, str | None], None]


class IWatchBackend(Protocol):
    """Minimal filesystem-watch port. One instance per watched engine."""

    def start(self, roots: list[str], on_event: WatchCallback) -> None:
        """Begin recursive observation of `roots`, calling `on_event` per event."""
        ...

    def stop(self) -> None:
        """Stop observing and release the backend's threads. Idempotent."""
        ...
