import glob
import hashlib
import os
from collections.abc import Iterable, Iterator
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from dbs_vector.core.models import Document
from dbs_vector.core.ports import IChunker, IEmbedder, IVectorStore

if TYPE_CHECKING:
    from dbs_vector.services.path_filter import PathFilter


class IngestionService:
    """Orchestrates the chunking, embedding, and storage of documents."""

    def __init__(
        self,
        chunker: IChunker,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        workflow: str = "default",
        batch_size: int = 64,
        path_filter: "PathFilter | None" = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.workflow = workflow
        self.batch_size = batch_size
        # Document engines only. None => legacy discovery, untouched.
        self.path_filter = path_filter

    # ---- discovery -----------------------------------------------------

    def _discover_files(self, target: str | list[str]) -> list[Path]:
        """Resolve an ingestion target to concrete files.

        A `list` target is a configured-roots run: the PathFilter owns
        discovery (it also warns about and skips missing roots).
        A `str` target is an explicit CLI path: a directory walks through the
        run-scoped PathFilter, a glob/file expands as given and is then
        filter-gated. Without a PathFilter the legacy behaviour is verbatim.
        """
        pf = self.path_filter

        if isinstance(target, list):
            if pf is None:
                raise ValueError(
                    "Multi-root ingestion requires a PathFilter "
                    "(document engines only). Pass a single target instead."
                )
            return list(pf.iter_files())

        if pf is None:
            if os.path.isdir(target):
                files: list[Path] = []
                base_dir = Path(target)
                for ext in self.chunker.supported_extensions:
                    files.extend(base_dir.rglob(f"*{ext}"))
                return files
            return [Path(p) for p in glob.glob(target, recursive=True)]

        if os.path.isdir(target):
            # The run-scoped PathFilter is anchored at exactly this directory.
            return list(pf.iter_files())

        # Glob / single file: canonicalize, then gate. Explicit-path runs
        # resolve so root_for() matches the resolved anchor.
        gated: list[Path] = []
        for raw in glob.glob(target, recursive=True):
            candidate = Path(raw).resolve()
            if not candidate.is_file():
                continue
            if pf.is_ingestable(candidate):
                gated.append(candidate)
            else:
                logger.warning(
                    "Skipping {} — excluded by extension / ignore_patterns / gitignore",
                    candidate,
                )
        return gated

    def ingest_directory(self, target: str | list[str], rebuild: bool = False) -> None:
        """Reads documents, chunks them, and streams them to the Vector Store.

        `target` is either a list of configured roots (one run over all of
        them: one clear, one dedup snapshot, one index pass) or a single
        explicit path / glob / URL.
        """
        if rebuild:
            logger.warning("Rebuilding vector store (clearing existing data)")
            self.vector_store.clear()

        logger.info("Starting streaming ingestion for {}", target)
        logger.info("Checking for existing documents (deduplication enabled)")
        existing_hashes = self.vector_store.get_existing_hashes()

        def _chunk_generator() -> Iterator[Any]:
            # API mode: target is a URL — bypass file discovery entirely
            if isinstance(target, str) and target.startswith(("http://", "https://")):
                doc = Document(filepath=target, content="", content_hash="api-chunker")
                yield from self.chunker.process(doc)
                return

            for filepath in self._discover_files(target):
                if not filepath.is_file():
                    continue

                filepath_str = str(filepath)
                content = ""

                # Skip UTF-8 read for binary duckdb files
                if not filepath_str.endswith(".duckdb"):
                    try:
                        with open(filepath_str, encoding="utf-8") as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        logger.warning("Skipping non-UTF-8 file: {}", filepath_str)
                        continue

                # Calculate file hash for delta updates
                if content:
                    file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                else:
                    # For duckdb or empty files, use a hash of the filepath and modification time
                    stat = filepath.stat()
                    file_hash = hashlib.sha256(
                        f"{filepath_str}{stat.st_mtime}".encode()
                    ).hexdigest()[:16]

                # Short-circuit: DocumentChunker propagates this file_hash to every
                # chunk, so if the hash is already in the store the whole file is
                # already indexed. Safe no-op for per-chunk-hash chunkers (SQL /
                # DuckDB / API): their chunk hashes are per-record SHA-256s so a
                # file hash never matches.
                if file_hash in existing_hashes:
                    logger.debug("Skipping unchanged file {}", filepath_str)
                    continue

                doc = Document(
                    filepath=filepath_str,
                    content=content,
                    content_hash=file_hash,
                )
                yield from self.chunker.process(doc)

        total_chunks = 0
        skipped_chunks = 0
        for batch in batched(_chunk_generator(), self.batch_size):
            # Check-AND-ADD against the live hash set so a duplicate is skipped
            # wherever it appears: already in the store, in an earlier batch, or
            # earlier in THIS batch (two identical files routinely land in one
            # batch at the default batch_size=64 — a post-ingest set update
            # alone misses that case). get_existing_hashes() is a one-time
            # snapshot and never sees in-run inserts; this set is its live
            # extension.
            new_chunks = []
            for c in batch:
                if c.content_hash in existing_hashes:
                    continue
                existing_hashes.add(c.content_hash)
                new_chunks.append(c)

            if not new_chunks:
                skipped_chunks += len(batch)
                continue

            texts = [c.text for c in new_chunks]
            vectors = self.embedder.embed_batch(texts)

            self.vector_store.ingest_chunks(
                chunks=new_chunks, vectors=vectors, workflow=self.workflow
            )
            total_chunks += len(new_chunks)
            skipped_chunks += len(batch) - len(new_chunks)
            logger.info("Streamed {} new chunks (total: {})", len(new_chunks), total_chunks)

        if skipped_chunks > 0:
            logger.info("Skipped {} already-indexed chunks", skipped_chunks)

        logger.info("Creating explicit index strategies")
        self.vector_store.create_indices()

        logger.info("Running dataset compaction")
        self.vector_store.compact()
        logger.success("Ingestion complete")

    # ---- source-scoped operations (watcher / reconcile) ----------------

    def _store_hash_index(self) -> tuple[dict[str, set[str]], set[str]]:
        """(source -> its content hashes, all content hashes) from a FRESH scan.

        `scan()` guarantees checkout_latest(), so this always sees rows
        committed by another process.
        """
        table = self.vector_store.scan(columns=["source", "content_hash"])
        if table.num_rows == 0:
            return {}, set()
        sources = table.column("source").to_pylist()
        hashes = table.column("content_hash").to_pylist()
        by_source: dict[str, set[str]] = {}
        for source, content_hash in zip(sources, hashes, strict=True):
            by_source.setdefault(source, set()).add(content_hash)
        return by_source, set(hashes)

    def _embed_and_store(self, chunks: Iterable[Any]) -> int:
        """Embed + write every chunk, UNCONDITIONALLY.

        Never the snapshot-based dedup used by ingest_directory: the caller has
        already deleted this source's rows, and the snapshot would still show
        the file's own pre-delete hashes and skip everything.
        """
        total = 0
        for batch in batched(chunks, self.batch_size):
            items = list(batch)
            if not items:
                continue
            vectors = self.embedder.embed_batch([c.text for c in items])
            self.vector_store.ingest_chunks(
                chunks=items, vectors=vectors, workflow=self.workflow
            )
            total += len(items)
        return total

    def upsert_file(
        self,
        path: str,
        *,
        index: tuple[dict[str, set[str]], set[str]] | None = None,
    ) -> None:
        """Whole-file replace for one source, source-aware.

        `index` is the (by_source, all_hashes) pair from `_store_hash_index()`.
        Pass it from a reconcile loop to avoid one full scan per file; it is
        UPDATED IN PLACE so later files in the same pass see this write.
        """
        filepath = Path(path)
        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 file: {}", path)
            return
        except OSError as e:
            logger.warning("Cannot read {}: {}", path, e)
            return

        source = str(filepath)
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        by_source, all_hashes = self._store_hash_index() if index is None else index

        # "Already ingested" is the (source, bare_hash) PAIR — never the hash
        # alone. A global-hash check would let an edit-into-duplicate keep its
        # stale rows forever.
        if file_hash in by_source.get(source, set()):
            logger.debug("Unchanged, skipping {}", source)
            return

        self.vector_store.delete_by_source(source)
        # Drop this source's hashes from BOTH views. Global dedup means a bare
        # hash lives under exactly one source, so the subtraction is safe — and
        # without it a shared reconcile index would still claim the content is
        # "indexed elsewhere" after we just deleted it.
        all_hashes -= by_source.pop(source, set())

        if file_hash in all_hashes:
            # Existing global dedup: the content is indexed once, under
            # another source. Deleting that copy heals this at the next
            # reconcile.
            logger.info(
                "Content of {} is already indexed under another source; not re-adding",
                source,
            )
            return

        doc = Document(filepath=source, content=content, content_hash=file_hash)
        written = self._embed_and_store(self.chunker.process(doc))
        logger.info("Upserted {} ({} chunks)", source, written)

        # Only the BARE hash is tracked: it is the pair-check key, and chunks
        # 1..N carry `{file_hash}_{i}` which nothing ever looks up.
        all_hashes.add(file_hash)
        by_source[source] = {file_hash}

    def remove_source(self, path: str) -> None:
        """Delete every row for one source."""
        self.vector_store.delete_by_source(str(path))
        logger.info("Removed source {}", path)

    def reconcile(self) -> tuple[int, int]:
        """Full delta pass over the ACTIVE roots. Returns (pruned, upserted).

        Compares content hashes, so it detects FILE changes only. Any change to
        the engine's roots, model, prefixes, tuning profile, chunker behaviour
        or filters invalidates the corpus — run `ingest --rebuild --force`.
        """
        pf = self.path_filter
        if pf is None:
            raise ValueError("reconcile() requires a PathFilter (document engines only)")

        roots = pf.active_roots()
        if not roots:
            logger.warning("No active ingestion roots; skipping reconciliation")
            return (0, 0)

        # 1. Disk: walk active roots through the PathFilter.
        disk: dict[str, str] = {}
        for candidate in pf.iter_files():
            try:
                content = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                logger.warning("Skipping unreadable file {}: {}", candidate, e)
                continue
            disk[str(candidate)] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        # 2. Store.
        by_source, all_hashes = self._store_hash_index()

        # 3. Prune: rows under an ACTIVE root that disk no longer has — which
        #    includes files that became excluded (newly ignored/gitignored).
        #    Sources outside the active roots are unmanaged: never touched.
        pruned = 0
        for source in list(by_source):
            if source in disk or pf.root_for(source, roots) is None:
                continue
            self.vector_store.delete_by_source(source)
            # Subtract from all_hashes too, or step 4 would treat a
            # renamed-while-down file as "already indexed under another
            # source" and silently drop it.
            all_hashes -= by_source.pop(source, set())
            pruned += 1

        # 4. Ingest whatever the pruned store map no longer vouches for. The
        #    subtraction in step 3 handles renamed-while-down: pruned under the
        #    old source, so it must not be skipped as "present" under the new.
        index = (by_source, all_hashes)
        upserted = 0
        for source, file_hash in disk.items():
            if file_hash in by_source.get(source, set()):
                continue
            self.upsert_file(source, index=index)
            upserted += 1

        logger.info("Reconciliation complete: {} pruned, {} upserted", pruned, upserted)
        return (pruned, upserted)
