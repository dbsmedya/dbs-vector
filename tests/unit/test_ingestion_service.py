"""Unit tests for IngestionService."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


def test_ingestion_batches_by_self_batch_size(tmp_path):
    """Chunks are streamed to the store in batches sized by self.batch_size."""
    (tmp_path / "doc.md").write_text("# Doc\n\nbody")

    chunker = MagicMock()
    chunker.supported_extensions = [".md"]
    chunker.process.side_effect = lambda doc: iter(
        [
            Chunk(
                id=f"{doc.filepath}_chunk_{i}",
                text=f"chunk {i}",
                source=doc.filepath,
                content_hash=f"hash{i}",  # distinct ⇒ none deduped
            )
            for i in range(10)
        ]
    )

    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    store = MagicMock()
    store.get_existing_hashes.return_value = set()

    svc = IngestionService(chunker, embedder, store, batch_size=3)
    svc.ingest_directory(str(tmp_path))

    batch_sizes = [len(call.kwargs["chunks"]) for call in store.ingest_chunks.call_args_list]
    assert batch_sizes == [3, 3, 3, 1]


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

    # 1 chunk/batch → each file is its own batch, exercising the CROSS-batch
    # dedup path. The INTRA-batch path (both duplicates inside one batch) is
    # covered separately below — it requires check-and-add at filter time, not
    # just a post-ingest set update.
    svc = IngestionService(chunker, embedder, store, batch_size=1)
    svc.ingest_directory(str(tmp_path))

    stored = [
        c.content_hash for call in store.ingest_chunks.call_args_list for c in call.kwargs["chunks"]
    ]
    assert stored.count(dup_hash) == 1


def test_intra_batch_dedup_stores_duplicate_hash_once(tmp_path):
    """Two same-hash chunks landing in the SAME batch → stored once.

    This is the production-realistic case (batch_size=64 puts chunks from
    adjacent identical files into one batch). A dedup that only folds hashes
    into the set AFTER each batch is ingested misses it.
    """
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
    embedder.embed_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    store = MagicMock()
    store.get_existing_hashes.return_value = set()

    # Large batch → both files' chunks arrive in ONE batch.
    svc = IngestionService(chunker, embedder, store, batch_size=64)
    svc.ingest_directory(str(tmp_path))

    stored = [
        c.content_hash for call in store.ingest_chunks.call_args_list for c in call.kwargs["chunks"]
    ]
    assert stored.count(dup_hash) == 1


class TestPathFilterDiscovery:
    """With a PathFilter, discovery goes through the ONE filtering path."""

    def _service(self, roots, store=None, extensions=(".md", ".txt")):
        from dbs_vector.services.path_filter import PathFilter

        chunker = MagicMock()
        chunker.supported_extensions = list(extensions)
        chunker.process.return_value = iter([])
        store = store or MagicMock()
        store.get_existing_hashes.return_value = set()
        pf = PathFilter(
            roots=[str(r) for r in roots],
            extensions=list(extensions),
            ignore_patterns=[".#*", "*~", "*.tmp", ".DS_Store"],
        )
        svc = IngestionService(chunker, MagicMock(), store, "wf", path_filter=pf)
        return svc, chunker

    def test_multi_root_list_ingests_every_root_in_one_run(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "one.md").write_text("# one")
        (b / "two.md").write_text("# two")
        svc, chunker = self._service([a, b])

        svc.ingest_directory([str(a), str(b)], rebuild=True)

        ingested = {Path(c.args[0].filepath).name for c in chunker.process.call_args_list}
        assert ingested == {"one.md", "two.md"}
        # ONE run: one clear, one dedup snapshot, one index pass.
        svc.vector_store.clear.assert_called_once()
        svc.vector_store.get_existing_hashes.assert_called_once()
        svc.vector_store.create_indices.assert_called_once()

    def test_ignored_files_are_not_ingested(self, tmp_path):
        (tmp_path / "keep.md").write_text("# keep")
        (tmp_path / "scratch.tmp").write_text("junk")
        (tmp_path / ".#lock.md").write_text("lock")
        svc, chunker = self._service([tmp_path])

        svc.ingest_directory([str(tmp_path)])

        ingested = {Path(c.args[0].filepath).name for c in chunker.process.call_args_list}
        assert ingested == {"keep.md"}

    def test_sources_are_stored_absolute(self, tmp_path, monkeypatch):
        (tmp_path / "doc.md").write_text("# doc")
        svc, chunker = self._service([tmp_path])
        monkeypatch.chdir(tmp_path)

        svc.ingest_directory("doc.md")  # explicit relative path

        source = chunker.process.call_args.args[0].filepath
        assert Path(source).is_absolute()
        assert Path(source) == (tmp_path / "doc.md").resolve()

    def test_explicit_glob_still_applies_the_extension_gate(self, tmp_path):
        (tmp_path / "a.md").write_text("# a")
        (tmp_path / "b.pdf").write_bytes(b"%PDF")
        svc, chunker = self._service([tmp_path])

        svc.ingest_directory(f"{tmp_path}/*")

        ingested = {Path(c.args[0].filepath).name for c in chunker.process.call_args_list}
        assert ingested == {"a.md"}

    def test_multi_root_list_without_a_path_filter_is_a_programming_error(self):
        svc = IngestionService(MagicMock(), MagicMock(), MagicMock(), "wf")
        with pytest.raises(ValueError, match="requires a PathFilter"):
            svc.ingest_directory(["/a", "/b"])


class TestLegacyDiscoveryUnchanged:
    """Non-document engines (path_filter=None) keep byte-identical behaviour."""

    def test_glob_expansion_is_unfiltered_without_a_path_filter(self, tmp_path):
        (tmp_path / "queries.json").write_text("[]")
        chunker = MagicMock()
        chunker.supported_extensions = [".json"]
        chunker.process.return_value = iter([])
        store = MagicMock()
        store.get_existing_hashes.return_value = set()

        svc = IngestionService(chunker, MagicMock(), store, "wf")
        svc.ingest_directory(f"{tmp_path}/*.json")

        assert chunker.process.call_count == 1


class TestUpsertFile:
    """Whole-file replace, source-aware."""

    def _service(self, tmp_path, rows):
        """rows: list of (source, content_hash) tuples already in the store."""
        import pyarrow as pa

        from dbs_vector.services.path_filter import PathFilter

        chunker = MagicMock()
        chunker.supported_extensions = [".md"]
        chunker.process.return_value = iter([])
        store = MagicMock()
        store.scan.return_value = pa.table(
            {
                "source": [r[0] for r in rows],
                "content_hash": [r[1] for r in rows],
            }
        )
        pf = PathFilter(roots=[str(tmp_path)], extensions=[".md"], ignore_patterns=[])
        return IngestionService(chunker, MagicMock(), store, "wf", path_filter=pf), store, chunker

    def test_unchanged_file_is_a_no_op(self, tmp_path):
        doc = tmp_path / "a.md"
        doc.write_text("# A")
        file_hash = hashlib.sha256(b"# A").hexdigest()[:16]
        svc, store, chunker = self._service(tmp_path, [(str(doc), file_hash)])

        svc.upsert_file(str(doc))

        store.delete_by_source.assert_not_called()
        chunker.process.assert_not_called()

    def test_same_hash_under_a_different_source_does_not_skip_this_source(self, tmp_path):
        """The pair check closes the silent-poisoning hole: editing B.md into a
        duplicate of A.md must still DELETE B's stale rows."""
        doc = tmp_path / "b.md"
        doc.write_text("# A")
        file_hash = hashlib.sha256(b"# A").hexdigest()[:16]
        svc, store, chunker = self._service(
            tmp_path, [(str(tmp_path / "a.md"), file_hash), (str(doc), "staleoldhash")]
        )

        svc.upsert_file(str(doc))

        store.delete_by_source.assert_called_once_with(str(doc))
        # Global dedup preserved: content is indexed once, under a.md.
        chunker.process.assert_not_called()

    def test_changed_file_is_deleted_then_reingested(self, tmp_path):
        doc = tmp_path / "a.md"
        doc.write_text("# NEW")
        svc, store, chunker = self._service(tmp_path, [(str(doc), "oldhash")])

        svc.upsert_file(str(doc))

        store.delete_by_source.assert_called_once_with(str(doc))
        chunker.process.assert_called_once()

    def test_new_file_is_ingested_unconditionally(self, tmp_path):
        doc = tmp_path / "new.md"
        doc.write_text("# NEW")
        svc, store, chunker = self._service(tmp_path, [])

        svc.upsert_file(str(doc))

        chunker.process.assert_called_once()

    def test_unreadable_file_is_skipped_without_raising(self, tmp_path):
        doc = tmp_path / "bin.md"
        doc.write_bytes(b"\xff\xfe\x00binary")
        svc, store, chunker = self._service(tmp_path, [])

        svc.upsert_file(str(doc))

        chunker.process.assert_not_called()
        store.delete_by_source.assert_not_called()


class TestRemoveSource:
    def test_remove_source_delegates_to_the_store(self):
        store = MagicMock()
        svc = IngestionService(MagicMock(), MagicMock(), store, "wf")

        svc.remove_source("/vault/a.md")

        store.delete_by_source.assert_called_once_with("/vault/a.md")


class TestReconcile:
    def _service(self, roots, rows, extensions=(".md",)):
        import pyarrow as pa

        from dbs_vector.services.path_filter import PathFilter

        chunker = MagicMock()
        chunker.supported_extensions = list(extensions)
        chunker.process.return_value = iter([])
        store = MagicMock()
        store.scan.return_value = pa.table(
            {"source": [r[0] for r in rows], "content_hash": [r[1] for r in rows]}
        )
        pf = PathFilter(
            roots=[str(r) for r in roots],
            extensions=list(extensions),
            ignore_patterns=[".#*", "*~", "*.tmp", ".DS_Store"],
        )
        return IngestionService(chunker, MagicMock(), store, "wf", path_filter=pf), store, chunker

    def test_deleted_file_is_pruned(self, tmp_path):
        gone = str(tmp_path / "gone.md")
        svc, store, _ = self._service([tmp_path], [(gone, "h1")])

        pruned, upserted = svc.reconcile()

        assert (pruned, upserted) == (1, 0)
        store.delete_by_source.assert_called_once_with(gone)

    def test_newly_excluded_file_is_pruned(self, tmp_path):
        """The index tracks what the engine WOULD ingest today."""
        excluded = tmp_path / "notes.tmp"
        excluded.write_text("still on disk")
        svc, store, _ = self._service([tmp_path], [(str(excluded), "h1")])

        pruned, _ = svc.reconcile()

        assert pruned == 1
        store.delete_by_source.assert_called_once_with(str(excluded))

    def test_rows_outside_active_roots_are_never_pruned(self, tmp_path):
        outside = "/somewhere/else/old.md"
        svc, store, _ = self._service([tmp_path], [(outside, "h1")])

        pruned, _ = svc.reconcile()

        assert pruned == 0
        store.delete_by_source.assert_not_called()

    def test_renamed_while_down_is_pruned_and_reingested(self, tmp_path):
        """Pruned under the old source, must NOT be skipped as 'present'
        under the new one."""
        new = tmp_path / "new.md"
        new.write_text("# same content")
        file_hash = hashlib.sha256(b"# same content").hexdigest()[:16]
        old = str(tmp_path / "old.md")
        svc, store, chunker = self._service([tmp_path], [(old, file_hash)])

        pruned, upserted = svc.reconcile()

        assert (pruned, upserted) == (1, 1)
        store.delete_by_source.assert_any_call(old)
        chunker.process.assert_called_once()

    def test_missing_root_skips_the_whole_pass(self, tmp_path):
        svc, store, _ = self._service(
            [tmp_path / "unmounted"], [(str(tmp_path / "unmounted" / "a.md"), "h1")]
        )

        pruned, upserted = svc.reconcile()

        assert (pruned, upserted) == (0, 0)
        store.delete_by_source.assert_not_called()

    def test_reconcile_scans_the_store_once(self, tmp_path):
        for name in ("a.md", "b.md", "c.md"):
            (tmp_path / name).write_text(f"# {name}")
        svc, store, _ = self._service([tmp_path], [])

        svc.reconcile()

        assert store.scan.call_count == 1

    def test_reconcile_requires_a_path_filter(self):
        svc = IngestionService(MagicMock(), MagicMock(), MagicMock(), "wf")
        with pytest.raises(ValueError, match="requires a PathFilter"):
            svc.reconcile()
