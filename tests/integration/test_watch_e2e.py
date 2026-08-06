"""End-to-end watch behaviour: tmpdir LanceDB, real chunker/mapper, fake
embedder, SYNTHETIC events. No timers, no real observers."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import DocumentChunker
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import DocumentMapper
from dbs_vector.services.ingestion import IngestionService
from dbs_vector.services.path_filter import PathFilter
from dbs_vector.services.watcher import WatchedEngine, WatcherService

DIM = 8


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for MLXEmbedder."""

    dimension = DIM

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [[float((len(t) + i) % 5) for i in range(DIM)] for t in texts], dtype=np.float32
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def count_tokens(self, text: str) -> int:
        return len(text)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def build_stack(tmp_path: Path, vault: Path, use_gitignore: bool = False):
    mapper = DocumentMapper(vector_dimension=DIM)
    store = LanceDBStore(
        db_path=str(tmp_path / "db"),
        table_name="watch_test",
        vector_dimension=DIM,
        mapper=mapper,
    )
    chunker = DocumentChunker(max_chars=1000, target_tokens=200, max_tokens=400, length_fn=len)
    path_filter = PathFilter(
        roots=[str(vault)],
        extensions=chunker.supported_extensions,
        ignore_patterns=[".#*", "*~", "*.tmp", ".DS_Store"],
        use_gitignore=use_gitignore,
    )
    ingestion = IngestionService(
        chunker, FakeEmbedder(), store, "wf", batch_size=8, path_filter=path_filter
    )
    return ingestion, store, path_filter


_DOC = "# Guide\n\n## A\n\ntiny\n\n## B\n\n" + ("Body words. " * 40) + "\n"


def _rows(store):
    t = store.scan(columns=["id", "text"]).sort_by([("id", "ascending")])
    return list(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))


def test_reingest_after_boundary_change_writes_nothing_without_rebuild(tmp_path, monkeypatch):
    """The failure mode looks like success.

    `_chunk_content_hash` keys on file content plus position, never chunk
    content, so a boundary change leaves hashes identical and both ingest
    paths skip the file while logging success. This is why 1.7.0 requires
    `--rebuild --force` and why a plain re-ingest is not enough.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "doc.md").write_text(_DOC)
    service, store, _ = build_stack(tmp_path, vault)

    # "Old" chunker == fold disabled. Changing target_tokens would NOT simulate
    # the old chunker: a larger target makes a tiny section MORE fold-eligible,
    # so both passes would fold and the test would prove nothing.
    monkeypatch.setattr(DocumentChunker, "_fold", lambda self, sections: sections)
    service.ingest_directory([str(vault)])
    before = _rows(store)
    monkeypatch.undo()

    # Prove the two chunkers genuinely disagree on this file, else the skip
    # assertion below is vacuous.
    new_texts = [
        c.text
        for c in DocumentChunker(target_tokens=200, max_tokens=400, length_fn=len).process(
            Document(filepath=str(vault / "doc.md"), content=_DOC, content_hash="h")
        )
    ]
    assert new_texts != [t for _, t in before], "fixture must actually change boundaries"

    # Same unchanged file, new chunker, no --rebuild: still a no-op.
    service.ingest_directory([str(vault)])
    assert _rows(store) == before, (
        "re-ingest without --rebuild must be a no-op: identical file hash means "
        "the file is skipped even though boundaries changed"
    )


def test_upsert_path_has_the_same_skip_behaviour(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    target = vault / "doc.md"
    target.write_text(_DOC)
    service, store, _ = build_stack(tmp_path, vault)

    monkeypatch.setattr(DocumentChunker, "_fold", lambda self, sections: sections)
    service.upsert_file(str(target))
    before = _rows(store)
    monkeypatch.undo()

    service.upsert_file(str(target))
    assert _rows(store) == before


def sources(store: LanceDBStore) -> set[str]:
    table = store.scan(columns=["source"])
    return set(table.column("source").to_pylist())


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    return root


class TestUpsertLifecycle:
    def test_create_edit_delete_round_trip(self, tmp_path, vault):
        ingestion, store, _ = build_stack(tmp_path, vault)
        doc = vault / "a.md"

        doc.write_text("# Alpha\n\nFirst version of the note.")
        ingestion.upsert_file(str(doc))
        assert sources(store) == {str(doc)}
        first = store.scan(columns=["source", "content_hash"]).num_rows

        doc.write_text("# Alpha\n\nSecond version, completely different body text.")
        ingestion.upsert_file(str(doc))
        assert sources(store) == {str(doc)}
        # Whole-file replace: the old rows are gone, not accumulated.
        assert store.scan(columns=["source"]).num_rows == first

        ingestion.remove_source(str(doc))
        assert sources(store) == set()

    def test_edit_into_duplicate_leaves_no_stale_rows(self, tmp_path, vault):
        """The (source, bare_hash) PAIR check. A global-hash check would skip
        b.md forever and keep its stale rows."""
        ingestion, store, _ = build_stack(tmp_path, vault)
        a, b = vault / "a.md", vault / "b.md"
        a.write_text("# Shared\n\nIdentical body.")
        b.write_text("# Different\n\nOriginal body of B.")
        ingestion.upsert_file(str(a))
        ingestion.upsert_file(str(b))
        assert sources(store) == {str(a), str(b)}

        b.write_text("# Shared\n\nIdentical body.")  # b becomes a duplicate of a
        ingestion.upsert_file(str(b))

        # b's stale rows are DELETED; the content stays indexed once, under a.
        assert sources(store) == {str(a)}

    def test_rename_moves_the_rows(self, tmp_path, vault):
        ingestion, store, _ = build_stack(tmp_path, vault)
        old, new = vault / "old.md", vault / "new.md"
        old.write_text("# Note\n\nBody text here.")
        ingestion.upsert_file(str(old))

        old.rename(new)
        ingestion.remove_source(str(old))
        ingestion.upsert_file(str(new))

        assert sources(store) == {str(new)}


class TestReconcile:
    def test_startup_reconcile_is_a_full_delta_ingest(self, tmp_path, vault):
        ingestion, store, _ = build_stack(tmp_path, vault)
        (vault / "one.md").write_text("# One\n\nBody one.")
        (vault / "two.md").write_text("# Two\n\nBody two.")

        pruned, upserted = ingestion.reconcile()

        assert (pruned, upserted) == (0, 2)
        assert sources(store) == {str(vault / "one.md"), str(vault / "two.md")}

    def test_reconcile_is_idempotent(self, tmp_path, vault):
        ingestion, store, _ = build_stack(tmp_path, vault)
        (vault / "one.md").write_text("# One\n\nBody one.")
        ingestion.reconcile()

        assert ingestion.reconcile() == (0, 0)

    def test_reconcile_heals_a_deletion_that_happened_while_down(self, tmp_path, vault):
        ingestion, store, _ = build_stack(tmp_path, vault)
        gone = vault / "gone.md"
        gone.write_text("# Gone\n\nBody.")
        ingestion.reconcile()

        gone.unlink()
        pruned, _ = ingestion.reconcile()

        assert pruned == 1
        assert sources(store) == set()

    def test_reconcile_prunes_newly_gitignored_files(self, tmp_path, vault):
        ingestion, store, path_filter = build_stack(tmp_path, vault, use_gitignore=True)
        (vault / "build").mkdir()
        (vault / "build" / "gen.md").write_text("# Generated\n\nBody.")
        ingestion.reconcile()
        assert sources(store) == {str(vault / "build" / "gen.md")}

        (vault / ".gitignore").write_text("build/\n")
        path_filter._gitignore_cache.clear()  # simulate the restart the spec requires
        pruned, _ = ingestion.reconcile()

        assert pruned == 1
        assert sources(store) == set()


class TestWatcherIntegration:
    def test_synthetic_events_drive_real_store_writes(self, tmp_path, vault):
        ingestion, store, path_filter = build_stack(tmp_path, vault)
        clock = FakeClock()
        engine = WatchedEngine(
            name="md",
            ingestion=ingestion,
            path_filter=path_filter,
            store=store,
            debounce_seconds=3.0,
            backend=MagicMock(),  # start() is never called in this test
        )
        svc = WatcherService({"md": engine}, clock=clock)

        doc = vault / "a.md"
        doc.write_text("# Alpha\n\nBody.")
        svc.handle_event("md", str(doc), "created")
        clock.advance(3.0)
        svc.run_once()
        assert sources(store) == {str(doc)}

        doc.unlink()
        svc.handle_event("md", str(doc), "deleted")
        clock.advance(3.0)
        svc.run_once()
        assert sources(store) == set()

    def test_refresh_fts_runs_against_a_real_table(self, tmp_path, vault):
        ingestion, store, _ = build_stack(tmp_path, vault)
        (vault / "a.md").write_text("# Alpha\n\nSearchable body text.")
        ingestion.reconcile()

        store.refresh_fts()  # must not raise on a real, small table


class TestCliMultiRoot:
    def test_multi_root_ingest_covers_every_root_in_one_run(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "one.md").write_text("# One\n\nBody one.")
        (b / "two.md").write_text("# Two\n\nBody two.")

        mapper = DocumentMapper(vector_dimension=DIM)
        store = LanceDBStore(
            db_path=str(tmp_path / "db"),
            table_name="multi",
            vector_dimension=DIM,
            mapper=mapper,
        )
        chunker = DocumentChunker(max_chars=1000, target_tokens=200, max_tokens=400, length_fn=len)
        path_filter = PathFilter(
            roots=[str(a), str(b)],
            extensions=chunker.supported_extensions,
            ignore_patterns=[],
        )
        svc = IngestionService(
            chunker, FakeEmbedder(), store, "wf", batch_size=8, path_filter=path_filter
        )

        svc.ingest_directory([str(a), str(b)], rebuild=True)

        assert sources(store) == {str(a / "one.md"), str(b / "two.md")}
