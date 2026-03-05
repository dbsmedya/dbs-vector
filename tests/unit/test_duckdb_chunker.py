import pytest
import duckdb
from datetime import datetime

from dbs_vector.infrastructure.chunking.duckdb import DuckDBChunker
from dbs_vector.core.models import Document

@pytest.fixture
def temp_duckdb_file(tmp_path):
    filepath = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(filepath))
    
    conn.execute("""
        CREATE TABLE slow_logs(
            fingerprint_id VARCHAR, 
            sanitized_sql VARCHAR, 
            sample_sql VARCHAR, 
            db VARCHAR, 
            query_time_sec DOUBLE,
            ts TIMESTAMP,
            "tables" VARCHAR[],
            "user" VARCHAR,
            host VARCHAR,
            rows_sent BIGINT,
            rows_examined BIGINT,
            lock_time_sec DOUBLE
        );
    """)
    
    # Insert some test data
    conn.execute("""
        INSERT INTO slow_logs VALUES 
        ('fp1', 'SELECT * FROM users', 'SELECT * FROM users WHERE id=1', 'db1', 1.5, current_date, ['users'], 'admin', 'localhost', 10, 100, 0.01),
        ('fp1', 'SELECT * FROM users', 'SELECT * FROM users WHERE id=2', 'db1', 2.0, current_date, ['users'], 'admin', 'localhost', 10, 100, 0.01),
        ('fp2', 'SELECT * FROM orders', 'SELECT * FROM orders', 'db1', 0.5, current_date, ['orders'], NULL, NULL, 1, 1, 0.001)
    """)
    conn.close()
    return str(filepath)

def test_duckdb_chunker_default_query(temp_duckdb_file):
    chunker = DuckDBChunker()
    doc = Document(filepath=temp_duckdb_file, content="", content_hash="")
    
    chunks = list(chunker.process(doc))
    
    assert len(chunks) == 2
    
    # Sort by execution_time_ms DESC (fp1 has 3.5s total, fp2 has 0.5s)
    assert chunks[0].id == "fp1"
    assert chunks[0].calls == 2
    assert chunks[0].execution_time_ms == 3500.0
    assert chunks[0].tables == ["users"]
    assert chunks[0].user == "admin"
    assert chunks[0].host == "localhost"
    assert chunks[0].rows_sent == 10
    assert chunks[0].rows_examined == 100
    assert chunks[0].lock_time_sec == 0.01
    
    assert chunks[1].id == "fp2"
    assert chunks[1].calls == 1
    assert chunks[1].execution_time_ms == 500.0
    assert chunks[1].user is None
    assert chunks[1].host is None
    assert chunks[1].rows_sent == 1
    assert chunks[1].rows_examined == 1
    assert chunks[1].lock_time_sec == 0.001

def test_duckdb_chunker_custom_query(temp_duckdb_file):
    query = 'SELECT fingerprint_id as id, sanitized_sql as text, db as source, current_date as latest_ts FROM slow_logs LIMIT 1'
    chunker = DuckDBChunker(query=query)
    doc = Document(filepath=temp_duckdb_file, content="", content_hash="")
    
    chunks = list(chunker.process(doc))
    assert len(chunks) == 1
    assert chunks[0].id == "fp1"
