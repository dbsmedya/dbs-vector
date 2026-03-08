from collections.abc import Iterator

from loguru import logger

from dbs_vector.core.models import Document, SqlChunk, sql_chunk_from_record


class DuckDBChunker:
    """Reads pre-processed SQL slow queries from DuckDB files."""

    def __init__(self, query: str | None = None, batch_id: str | None = None) -> None:
        self._query = query
        self._batch_id = batch_id

        self._default_query = """
            SELECT 
                fingerprint_id as id,
                arg_max(sanitized_sql, ts) as text,
                arg_max(sample_sql, ts) as raw_query,
                arg_max(db, ts) as source,
                SUM(query_time_sec) * 1000 as execution_time_ms,
                COUNT(*) as calls,
                arg_max("tables", ts) as tables,
                MAX(ts) as latest_ts,
                arg_max("user", ts) as user,
                arg_max(host, ts) as host,
                arg_max(rows_sent, ts) as rows_sent,
                arg_max(rows_examined, ts) as rows_examined,
                arg_max(lock_time_sec, ts) as lock_time_sec
            FROM slow_logs
            WHERE ts > current_date - INTERVAL '15 days'
            GROUP BY fingerprint_id
            ORDER BY execution_time_ms DESC
            LIMIT 500
        """

    @property
    def supported_extensions(self) -> list[str]:
        return [".duckdb"]

    def process(self, document: Document) -> Iterator[SqlChunk]:
        """Process a DuckDB file and yield SqlChunks."""
        try:
            import duckdb
        except ImportError:
            logger.error(
                "duckdb package is not installed. Please install with 'uv pip install dbs-vector[sql]' or 'uv pip install duckdb'."
            )
            return

        conn = None
        try:
            # DuckDB connection may fail if file is locked or invalid
            conn = duckdb.connect(document.filepath, read_only=True)

            query = self._query if self._query else self._default_query

            # Simple handling of batch_id. If batch_id is set and default query is used,
            # we need to append it. PRD says: "If provided, appends WHERE batch_id = ? filter to the default query. Ignored if query is set."
            if not self._query and self._batch_id:
                # The default query uses a WHERE clause, so we add AND batch_id = ?
                query = query.replace(
                    "WHERE ts > current_date - INTERVAL '15 days'",
                    f"WHERE ts > current_date - INTERVAL '15 days' AND batch_id = '{self._batch_id}'",
                )

            try:
                result = conn.execute(query).fetchall()
                columns = [desc[0] for desc in conn.description]
            except Exception as e:
                logger.error(f"SQL query syntax error or execution failed: {e}\\nQuery: {query}")
                return

            # Identify expected columns
            expected_columns = {"id", "text", "raw_query", "source", "execution_time_ms", "calls"}
            missing_cols = expected_columns - set(columns)
            if missing_cols:
                logger.warning(f"Missing expected columns in query result: {missing_cols}")
                # We do not return here, we let the row-level checks handle missing required fields.

            for row in result:
                row_dict = dict(zip(columns, row, strict=False))

                # PRD: NULL values in required fields (normalized(text), database(source)): skip the row with a debug-level log
                if row_dict.get("text") is None or row_dict.get("source") is None:
                    logger.debug(
                        f"Skipping row with NULL in required fields (text or source): {row_dict.get('id')}"
                    )
                    continue

                yield sql_chunk_from_record(
                    {
                        "id": row_dict["id"],
                        "text": row_dict["text"],
                        "raw_query": row_dict.get("raw_query", ""),
                        "source": row_dict["source"],
                        "execution_time_ms": row_dict.get("execution_time_ms"),
                        "calls": row_dict.get("calls"),
                        "tables": row_dict.get("tables"),
                        "latest_ts": row_dict.get("latest_ts"),
                        "user": row_dict.get("user"),
                        "host": row_dict.get("host"),
                        "rows_sent": row_dict.get("rows_sent"),
                        "rows_examined": row_dict.get("rows_examined"),
                        "lock_time_sec": row_dict.get("lock_time_sec"),
                    }
                )

        except Exception as e:
            logger.error(
                f"Connection failure or unexpected error processing {document.filepath}: {e}"
            )
            return
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Failed to close DuckDB connection: {e}")
