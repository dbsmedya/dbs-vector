import glob
import hashlib
import os
from collections.abc import Iterator
from itertools import batched
from pathlib import Path
from typing import Any

from loguru import logger

from dbs_vector.core.models import Document
from dbs_vector.core.ports import IChunker, IEmbedder, IVectorStore


class IngestionService:
    """Orchestrates the chunking, embedding, and storage of documents."""

    def __init__(
        self,
        chunker: IChunker,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        workflow: str = "default",
        batch_size: int = 64,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.workflow = workflow
        self.batch_size = batch_size

    def ingest_directory(self, target_path: str, rebuild: bool = False) -> None:
        """Reads documents, chunks them, and streams them to the Vector Store."""
        if rebuild:
            logger.warning("Rebuilding vector store (clearing existing data)")
            self.vector_store.clear()

        logger.info("Starting streaming ingestion for {}", target_path)
        logger.info("Checking for existing documents (deduplication enabled)")
        existing_hashes = self.vector_store.get_existing_hashes()

        def _chunk_generator() -> Iterator[Any]:
            # API mode: target_path is a URL — bypass file discovery entirely
            if target_path.startswith(("http://", "https://")):
                doc = Document(filepath=target_path, content="", content_hash="api-chunker")
                yield from self.chunker.process(doc)
                return

            if os.path.isdir(target_path):
                files: list[Path] = []
                base_dir = Path(target_path)
                for ext in self.chunker.supported_extensions:
                    files.extend(base_dir.rglob(f"*{ext}"))
            else:
                files = [Path(p) for p in glob.glob(target_path, recursive=True)]

            for filepath in files:
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
