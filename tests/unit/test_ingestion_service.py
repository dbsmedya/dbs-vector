"""Unit tests for IngestionService."""

import hashlib
from unittest.mock import MagicMock

from dbs_vector.core.models import Chunk
from dbs_vector.services.ingestion import IngestionService


def test_ingest_skips_chunker_when_doc_hash_already_present(tmp_path):
    """If a markdown file's content_hash is already in the store, the
    chunker must not be called for that file at all."""

    # Build a 1-file directory
    md_file = tmp_path / "doc.md"
    md_file.write_text("# Hello\n\nBody paragraph.")

    chunker = MagicMock()
    chunker.supported_extensions = [".md"]

    embedder = MagicMock()
    store = MagicMock()

    # Pre-compute the file hash the service will compute
    content = md_file.read_text()
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # Pretend the store already contains that hash
    store.get_existing_hashes.return_value = {expected_hash}

    svc = IngestionService(chunker, embedder, store, workflow="md_search")
    svc.ingest_directory(str(tmp_path))

    # The chunker must never see this document
    chunker.process.assert_not_called()
    # And no vectors were embedded for it
    embedder.embed_batch.assert_not_called()
    # The hoist: get_existing_hashes must be called exactly once per ingest
    store.get_existing_hashes.assert_called_once()


def test_ingest_still_chunks_when_doc_hash_is_new(tmp_path):
    """If a file's hash is NOT in the store, the chunker IS called."""

    md_file = tmp_path / "doc.md"
    md_file.write_text("# New content\n\nNot seen before.")

    chunker = MagicMock()
    chunker.supported_extensions = [".md"]
    # Simulate chunker yielding one chunk
    fake_chunk = MagicMock()
    fake_chunk.content_hash = "new_file_hash"
    fake_chunk.text = "chunk text"
    chunker.process.return_value = iter([fake_chunk])

    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]

    store = MagicMock()
    store.get_existing_hashes.return_value = set()  # empty

    svc = IngestionService(chunker, embedder, store, workflow="md_search")
    svc.ingest_directory(str(tmp_path))

    chunker.process.assert_called_once()
    embedder.embed_batch.assert_called_once()
    store.ingest_chunks.assert_called_once()


def test_ingestion_service_accepts_batch_size_kwarg():
    chunker = MagicMock()
    embedder = MagicMock()
    store = MagicMock()
    svc = IngestionService(
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
        workflow="w",
        batch_size=8,
    )
    assert svc.batch_size == 8


def test_ingestion_service_uses_self_batch_size_in_batched():
    """_batched yields batches sized by self.batch_size."""
    svc = IngestionService(
        chunker=MagicMock(),
        embedder=MagicMock(),
        vector_store=MagicMock(),
        workflow="w",
        batch_size=3,
    )
    batches = list(svc._batched(iter(range(10)), svc.batch_size))
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_intra_run_dedup_stores_duplicate_hash_once(tmp_path):
    """Two files whose chunks share one content_hash in ONE run → stored once."""
    (tmp_path / "a.md").write_text("# A\n\nbody a")  # different contents ⇒
    (tmp_path / "b.md").write_text("# B\n\nbody b")  # different file hashes

    dup_hash = "deadbeefdeadbeef"
    chunker = MagicMock()
    chunker.supported_extensions = [".md"]
    chunker.process.side_effect = lambda doc: iter(
        [
            Chunk(
                id=f"{doc.filepath}_chunk_0",
                text="same body",
                source=doc.filepath,
                content_hash=dup_hash,
            )
        ]
    )

    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]
    store = MagicMock()
    store.get_existing_hashes.return_value = set()

    # 1 chunk/batch → each file is its own batch, exercising the CROSS-batch dedup path
    # (a larger batch would dedupe within one batch and pass for the wrong reason)
    svc = IngestionService(chunker, embedder, store, batch_size=1)
    svc.ingest_directory(str(tmp_path))

    stored = [
        c.content_hash for call in store.ingest_chunks.call_args_list for c in call.kwargs["chunks"]
    ]
    assert stored.count(dup_hash) == 1
