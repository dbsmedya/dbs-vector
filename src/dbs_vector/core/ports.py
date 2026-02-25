from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


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


class IStoreMapper(Protocol):
    def __init__(self, vector_dimension: int, **kwargs: Any) -> None: ...

    @property
    def schema(self) -> Any:
        """Returns the PyArrow schema for the table."""
        ...

    def to_record_batch(self, chunks: list[Any], vectors: NDArray[np.float32]) -> Any:
        """Converts domain chunks and vectors into a PyArrow RecordBatch."""
        ...

    def from_polars_row(self, row: dict[str, Any], score: float | None) -> Any:
        """Converts a Polars row back into a domain SearchResult."""
        ...


class IVectorStore(Protocol):
    """Protocol defining how a high-performance vector store behaves."""

    def clear(self) -> None:
        """Completely drops the vector store table/index to start fresh."""
        ...

    def ingest_chunks(self, chunks: list[Any], vectors: NDArray[np.float32]) -> None:
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


class IChunker(Protocol):
    """Protocol defining how documents are split into logical boundaries."""

    def process(self, document: Any) -> Iterator[Any]:
        """Yields chunks from a raw document or parsed log."""
        ...
