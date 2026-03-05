import hashlib
from collections.abc import Iterator
from typing import Any

from loguru import logger

from dbs_vector.core.models import Document, SqlChunk


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
            logger.error("duckdb package is not installed. Please install with 'uv pip install dbs-vector[sql]' or 'uv pip install duckdb'.")
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
                query = query.replace("WHERE ts > current_date - INTERVAL '15 days'", 
                                      f"WHERE ts > current_date - INTERVAL '15 days' AND batch_id = '{self._batch_id}'")
            
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
                row_dict = dict(zip(columns, row))
                
                # PRD: NULL values in required fields (normalized(text), database(source)): skip the row with a debug-level log
                if row_dict.get("text") is None or row_dict.get("source") is None:
                    logger.debug(f"Skipping row with NULL in required fields (text or source): {row_dict.get('id')}")
                    continue
                
                # Derived hash
                text = str(row_dict["text"])
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                
                # calls defaults to 1 when NULL
                calls = row_dict.get("calls")
                if calls is None:
                    calls = 1
                    
                # Safe fallback for optional fields
                tables = row_dict.get("tables", [])
                if tables is None:
                    tables = []
                    
                yield SqlChunk(
                    id=str(row_dict["id"]),
                    text=text,
                    raw_query=str(row_dict.get("raw_query", "")),
                    source=str(row_dict["source"]),
                    execution_time_ms=float(row_dict.get("execution_time_ms") or 0.0),
                    calls=int(calls),
                    content_hash=content_hash,
                    tables=list(tables),
                    latest_ts=row_dict["latest_ts"],
                    user=str(row_dict["user"]) if row_dict.get("user") is not None else None,
                    host=str(row_dict["host"]) if row_dict.get("host") is not None else None,
                    rows_sent=int(row_dict["rows_sent"]) if row_dict.get("rows_sent") is not None else None,
                    rows_examined=int(row_dict["rows_examined"]) if row_dict.get("rows_examined") is not None else None,
                    lock_time_sec=float(row_dict["lock_time_sec"]) if row_dict.get("lock_time_sec") is not None else None,
                )
                
        except Exception as e:
            logger.error(f"Connection failure or unexpected error processing {document.filepath}: {e}")
            return
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Failed to close DuckDB connection: {e}")
