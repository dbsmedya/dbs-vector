from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from dbs_vector.core.models import Chunk, SearchResult, SqlChunk, SqlSearchResult


class DocumentMapper:
    """Mapper for mapping Document chunks to PyArrow structures and vice versa."""

    def __init__(self, vector_dimension: int) -> None:
        self.vector_dimension = vector_dimension
        self._schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.vector_dimension)),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field(
                    "workflow", pa.string()
                ),  # Identifies the exact model/task prefix space used
                pa.field("node_type", pa.string(), nullable=True),
                pa.field("parent_scope", pa.string(), nullable=True),
                pa.field("line_range", pa.string(), nullable=True),
            ]
        )

    @property
    def schema(self) -> Any:
        return self._schema

    def to_record_batch(
        self, chunks: list[Any], vectors: NDArray[np.float32], workflow: str
    ) -> Any:
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        sources = [c.source for c in chunks]
        hashes = [c.content_hash for c in chunks]
        workflows = [workflow for _ in chunks]
        node_types = [c.node_type for c in chunks]
        scopes = [c.parent_scope for c in chunks]
        lines = [c.line_range for c in chunks]

        return pa.RecordBatch.from_arrays(
            [
                pa.array(ids),
                pa.FixedSizeListArray.from_arrays(vectors.ravel(), list_size=self.vector_dimension),
                pa.array(texts),
                pa.array(sources),
                pa.array(hashes),
                pa.array(workflows),
                pa.array(node_types, type=pa.string()),
                pa.array(scopes, type=pa.string()),
                pa.array(lines, type=pa.string()),
            ],
            schema=self._schema,
        )

    def from_polars_row(self, row: dict[str, Any], score: float | None) -> Any:
        chunk = Chunk(
            id=row["id"],
            text=row["text"],
            source=row["source"],
            content_hash=row["content_hash"],
            node_type=row.get("node_type"),
            parent_scope=row.get("parent_scope"),
            line_range=row.get("line_range"),
        )
        # Assuming workflow might be needed in SearchResult in future, not added now for simplicity
        return SearchResult(chunk=chunk, score=score, distance=score, is_fts_match=(score is None))


class SqlMapper:
    """Mapper for mapping SQL chunks to PyArrow structures and vice versa."""

    def __init__(self, vector_dimension: int) -> None:
        self.vector_dimension = vector_dimension
        self._schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.vector_dimension)),
                pa.field("text", pa.string()),
                pa.field("raw_query", pa.string()),
                pa.field("source", pa.string()),
                pa.field("execution_time_ms", pa.float64()),
                pa.field("calls", pa.int64()),
                pa.field("content_hash", pa.string()),
                pa.field(
                    "workflow", pa.string()
                ),  # Identifies the exact model/task prefix space used
            ]
        )

    @property
    def schema(self) -> Any:
        return self._schema

    def to_record_batch(
        self, chunks: list[Any], vectors: NDArray[np.float32], workflow: str
    ) -> Any:
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        sources = [c.source for c in chunks]
        hashes = [c.content_hash for c in chunks]
        raw_queries = [c.raw_query for c in chunks]
        execution_times = [c.execution_time_ms for c in chunks]
        calls = [c.calls for c in chunks]
        workflows = [workflow for _ in chunks]

        return pa.RecordBatch.from_arrays(
            [
                pa.array(ids),
                pa.FixedSizeListArray.from_arrays(vectors.ravel(), list_size=self.vector_dimension),
                pa.array(texts),
                pa.array(raw_queries),
                pa.array(sources),
                pa.array(execution_times, type=pa.float64()),
                pa.array(calls, type=pa.int64()),
                pa.array(hashes),
                pa.array(workflows),
            ],
            schema=self._schema,
        )

    def from_polars_row(self, row: dict[str, Any], score: float | None) -> Any:
        chunk = SqlChunk(
            id=row["id"],
            text=row["text"],
            raw_query=row["raw_query"],
            source=row["source"],
            execution_time_ms=row["execution_time_ms"],
            calls=row["calls"],
            content_hash=row["content_hash"],
        )
        return SqlSearchResult(
            chunk=chunk, score=score, distance=score, is_fts_match=(score is None)
        )
