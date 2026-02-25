import hashlib
import json
from collections.abc import Iterator

from dbs_vector.core.models import SqlChunk


class SqlChunker:
    """
    Parses JSON exports of slow query logs or pg_stat_statements.
    The normalized query string must be pre-provided in the JSON payload.
    """

    def parse_query_log(self, filepath: str) -> Iterator[SqlChunk]:
        """Reads a JSON file containing query records and yields SqlChunks."""
        with open(filepath, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array of query records in {filepath}")

        for record in records:
            # Safely handle potential missing fields depending on the exact JSON schema
            raw = record.get("query", "")
            normalized = record.get("normalized_query") or record.get("normalized") or raw
            query_id = (
                record.get("query_hash")
                or record.get("id")
                or hashlib.md5(raw.encode()).hexdigest()
            )
            database = record.get("database") or record.get("source") or "unknown"
            duration = float(record.get("duration", 0.0) or record.get("execution_time_ms", 0.0))
            calls = int(record.get("calls", 1))

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
