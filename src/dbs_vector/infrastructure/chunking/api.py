import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from dbs_vector.core.models import Document, SqlChunk


class ApiChunker:
    """Fetches pre-processed SQL slow queries from a remote HTTP API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        page_size: int = 200,
        since_days: int = 15,
        timeout_sec: int = 30,
        min_execution_ms: float = 0.0,
        database: str | None = None,
        custom_query: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._page_size = page_size
        self._since_days = since_days
        self._timeout_sec = timeout_sec
        self._min_execution_ms = min_execution_ms
        self._database = database or ""
        self._custom_query = custom_query

    @property
    def supported_extensions(self) -> list[str]:
        # Routed via URL, not file discovery
        return []

    def process(self, document: Document) -> Iterator[SqlChunk]:
        """Fetch SqlChunks from the remote API."""
        try:
            import httpx
        except ImportError:
            logger.error(
                "httpx package is not installed. Please install with 'uv pip install dbs-vector[api]' or 'uv pip install httpx'."
            )
            return

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept-Encoding": "gzip",
        }

        try:
            with httpx.Client(timeout=self._timeout_sec) as client:
                if self._custom_query:
                    yield from self._fetch_custom_query(client, headers)
                else:
                    yield from self._fetch_paginated(client, headers)
        except httpx.ConnectError as e:
            logger.error(f"Connection error fetching from {self._base_url}: {e}")
            return
        except httpx.TimeoutException as e:
            logger.error(f"Timeout fetching from {self._base_url}: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error fetching from {self._base_url}: {e}")
            return

    def _fetch_paginated(self, client: Any, headers: dict) -> Iterator[SqlChunk]:
        from datetime import timedelta

        url = f"{self._base_url}/sql/queries"
        since_dt = datetime.now(UTC) - timedelta(days=self._since_days)
        since_str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        params: dict = {
            "limit": self._page_size,
            "since": since_str,
            "min_execution_ms": self._min_execution_ms,
        }
        if self._database:
            params["database"] = self._database

        cursor: str | None = None

        while True:
            if cursor is not None:
                params["cursor"] = cursor

            response = client.get(url, headers=headers, params=params)
            if response.status_code >= 400:
                logger.error(f"HTTP {response.status_code} from {url}")
                return

            data = response.json()
            for record in data.get("data", []):
                chunk = self._to_sql_chunk_safe(record)
                if chunk is not None:
                    yield chunk

            if not data.get("has_more", False):
                break

            cursor = data.get("next_cursor")

    def _fetch_custom_query(self, client: Any, headers: dict) -> Iterator[SqlChunk]:
        url = f"{self._base_url}/sql/execute"
        body: dict = {
            "query": self._custom_query,
            "timeout_ms": self._timeout_sec * 1000,
        }
        if self._database:
            body["database"] = self._database

        response = client.post(url, headers=headers, json=body)
        if response.status_code >= 400:
            logger.error(f"HTTP {response.status_code} from {url}")
            return

        data = response.json()
        columns: list[str] = data.get("columns", [])
        rows: list[list] = data.get("rows", [])

        for row in rows:
            record = dict(zip(columns, row, strict=False))
            chunk = self._to_sql_chunk_safe(record)
            if chunk is not None:
                yield chunk

    def _to_sql_chunk_safe(self, record: dict) -> SqlChunk | None:
        if record.get("id") is None or record.get("text") is None or record.get("source") is None:
            logger.debug(
                f"Skipping row missing required fields (id/text/source): {record.get('id')}"
            )
            return None
        return self._to_sql_chunk(record)

    def _to_sql_chunk(self, record: dict) -> SqlChunk:
        content_hash = hashlib.sha256(record["text"].encode()).hexdigest()[:16]

        latest_ts_raw = record.get("latest_ts")
        if latest_ts_raw:
            try:
                latest_ts = datetime.fromisoformat(latest_ts_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                latest_ts = datetime.now(UTC)
        else:
            latest_ts = datetime.now(UTC)

        return SqlChunk(
            id=str(record["id"]),
            text=record["text"],
            raw_query=record.get("raw_query", ""),
            source=record["source"],
            execution_time_ms=float(record.get("execution_time_ms", 0.0)),
            calls=int(record.get("calls", 1)),
            content_hash=content_hash,
            tables=list(record.get("tables") or []),
            latest_ts=latest_ts,
            user=record.get("user"),
            host=record.get("host"),
            rows_sent=int(record["rows_sent"]) if record.get("rows_sent") is not None else None,
            rows_examined=int(record["rows_examined"])
            if record.get("rows_examined") is not None
            else None,
            lock_time_sec=float(record["lock_time_sec"])
            if record.get("lock_time_sec") is not None
            else None,
        )
