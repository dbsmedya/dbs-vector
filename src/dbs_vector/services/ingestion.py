import glob
import hashlib
import os
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from dbs_vector.config import settings
from dbs_vector.core.models import Document
from dbs_vector.core.ports import IChunker, IEmbedder, IVectorStore


class IngestionService:
    """Orchestrates the chunking, embedding, and storage of documents."""

    def __init__(
        self,
        chunker: IChunker,
        embedder: IEmbedder,
        vector_store: IVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def _batched(self, iterable: Iterator[Any], n: int) -> Iterator[list[Any]]:
        """Yields successive n-sized chunks from an iterable (Python 3.12+ backport)."""
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch

    def ingest_directory(self, target_path: str, rebuild: bool = False) -> None:
        """Reads documents, chunks them, and streams them to the Vector Store."""
        if rebuild:
            print("\n[!] Rebuilding vector store (clearing existing data)...")
            self.vector_store.clear()

        print(f"\nStarting Streaming Ingestion for {target_path}...")

        def _chunk_generator() -> Iterator[Any]:
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

                with open(filepath_str, encoding="utf-8") as f:
                    content = f.read()

                # Calculate file hash for delta updates from the already loaded content
                file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

                doc = Document(
                    filepath=filepath_str,
                    content=content,
                    content_hash=file_hash,
                )
                yield from self.chunker.process(doc)

        print("\nChecking for existing documents (Deduplication enabled)...")
        existing_hashes = self.vector_store.get_existing_hashes()

        total_chunks = 0
        skipped_chunks = 0
        for batch in self._batched(_chunk_generator(), settings.batch_size):
            if not batch:
                continue

            # Filter out chunks whose content hash is already present in the store
            new_chunks = [c for c in batch if c.content_hash not in existing_hashes]

            if not new_chunks:
                skipped_chunks += len(batch)
                continue

            texts = [c.text for c in new_chunks]
            vectors = self.embedder.embed_batch(texts)

            self.vector_store.ingest_chunks(chunks=new_chunks, vectors=vectors)
            total_chunks += len(new_chunks)
            skipped_chunks += len(batch) - len(new_chunks)
            print(f" -> Streamed {len(new_chunks)} new chunks (Total: {total_chunks}).")

        if skipped_chunks > 0:
            print(f" -> Skipped {skipped_chunks} already-indexed chunks.")

        print("\nCreating Explicit Index Strategies...")
        self.vector_store.create_indices()

        print("Running Dataset Compaction...")
        self.vector_store.compact()
        print("\nIngestion Complete!")
