import hashlib
import json
from collections.abc import Iterator

from dbs_vector.core.models import Document, SqlChunk


class SqlChunker:
    """
    Parses JSON exports of slow query logs or pg_stat_statements.
    The normalized query string must be pre-provided in the JSON payload.
    Implements the IChunker protocol.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".json"]

    def process(self, document: Document) -> Iterator[SqlChunk]:
        """Parses the JSON content from a Document and yields SqlChunks."""
        try:
            records = json.loads(document.content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {document.filepath}: {e}")
            return

        if not isinstance(records, list):
            print(f"Warning: Expected a JSON array of query records in {document.filepath}")
            return

        for record in records:
            # Safely handle potential missing fields depending on the exact JSON schema
            raw = record.get("query") or ""
            normalized = record.get("normalized_query") or record.get("normalized") or raw
            query_id = (
                record.get("query_hash")
                or record.get("id")
                or hashlib.md5(raw.encode()).hexdigest()
            )
            database = record.get("database") or record.get("source") or "unknown"
            duration = float(record.get("duration") or record.get("execution_time_ms") or 0.0)
            calls = int(record.get("calls") or 1)

            if not normalized.strip():
                continue

            content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

            yield SqlChunk(
                id=str(query_id),
                text=normalized,
                raw_query=raw,
                source=database,
                execution_time_ms=duration,
                calls=calls,
                content_hash=content_hash,
            )
