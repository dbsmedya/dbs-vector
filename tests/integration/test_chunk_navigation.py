"""Exact adjacent-chunk storage reads against a real LanceDB table."""

import numpy as np

from dbs_vector.core.models import Chunk
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import DocumentMapper


def test_get_chunks_by_ids_preserves_order_escapes_ids_and_omits_missing(tmp_path):
    source = str(tmp_path / "owner's-guide_chunk_notes.md")
    chunks = [
        Chunk(
            id=f"{source}_chunk_{index}",
            text=f"content {index}",
            source=source,
            content_hash=f"hash_{index}",
            parent_scope="Guide",
            line_range=f"{index + 1}-{index + 1}",
        )
        for index in range(4)
    ]
    store = LanceDBStore(
        db_path=str(tmp_path / "db"),
        table_name="documents",
        vector_dimension=3,
        mapper=DocumentMapper(vector_dimension=3),
    )
    store.ingest_chunks(chunks, np.zeros((4, 3), dtype=np.float32), workflow="docs")

    result = store.get_chunks_by_ids([chunks[2].id, "missing_chunk_99", chunks[0].id, chunks[2].id])

    assert [chunk.id for chunk in result] == [chunks[2].id, chunks[0].id, chunks[2].id]
    assert [chunk.text for chunk in result] == ["content 2", "content 0", "content 2"]
    assert all(isinstance(chunk, Chunk) for chunk in result)
