"""Unit tests for LanceDBStore."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore


@pytest.fixture
def mock_mapper():
    """Create a mock mapper with schema."""
    mapper = MagicMock()
    # Create a simple arrow schema
    import pyarrow as pa

    mapper.schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 3)),
        ]
    )
    return mapper


@pytest.fixture
def lancedb_store(mock_mapper, tmp_path):
    """Create a LanceDBStore with mocked lancedb connection.
    
    Returns a tuple of (store, mock_db, mock_table, mock_lancedb) for test verification.
    """
    db_path = str(tmp_path / "test.db")

    with patch("dbs_vector.infrastructure.storage.lancedb_engine.lancedb") as mock_lancedb:
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.create_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        store = LanceDBStore(
            db_path=db_path,
            table_name="test_table",
            vector_dimension=3,
            mapper=mock_mapper,
            nprobes=10,
        )
        yield store, mock_db, mock_table, mock_lancedb


class TestLanceDBStoreInit:
    """Tests for LanceDBStore initialization."""

    def test_init_connects_to_db(self, mock_mapper, tmp_path):
        """Test that initialization connects to LanceDB."""
        db_path = str(tmp_path / "test.db")

        with patch("dbs_vector.infrastructure.storage.lancedb_engine.lancedb") as mock_lancedb:
            mock_db = MagicMock()
            mock_lancedb.connect.return_value = mock_db

            store = LanceDBStore(
                db_path=db_path,
                table_name="test_table",
                vector_dimension=3,
                mapper=mock_mapper,
            )

            mock_lancedb.connect.assert_called_once_with(db_path)
            assert store.db_path == db_path
            assert store.table_name == "test_table"
            assert store.vector_dimension == 3
            assert store.nprobes == 20  # default value

    def test_init_creates_table(self, mock_mapper, tmp_path):
        """Test that initialization creates the table with schema."""
        db_path = str(tmp_path / "test.db")

        with patch("dbs_vector.infrastructure.storage.lancedb_engine.lancedb") as mock_lancedb:
            mock_db = MagicMock()
            mock_lancedb.connect.return_value = mock_db

            LanceDBStore(
                db_path=db_path,
                table_name="my_vault",
                vector_dimension=3,
                mapper=mock_mapper,
            )

            mock_db.create_table.assert_called_once_with(
                "my_vault",
                schema=mock_mapper.schema,
                exist_ok=True,
            )

    def test_init_custom_nprobes(self, mock_mapper, tmp_path):
        """Test initialization with custom nprobes value."""
        db_path = str(tmp_path / "test.db")

        with patch("dbs_vector.infrastructure.storage.lancedb_engine.lancedb"):
            store = LanceDBStore(
                db_path=db_path,
                table_name="test_table",
                vector_dimension=3,
                mapper=mock_mapper,
                nprobes=50,
            )

            assert store.nprobes == 50


class TestClear:
    """Tests for the clear method."""

    def test_clear_drops_and_recreates_table(self, lancedb_store):
        """Test that clear drops the table and recreates it."""
        store, mock_db, _, _ = lancedb_store
        store.clear()

        mock_db.drop_table.assert_called_once_with(
            "test_table",
            ignore_missing=True,
        )
        mock_db.create_table.assert_called_with(
            "test_table",
            schema=store.mapper.schema,
        )


class TestIngestChunks:
    """Tests for the ingest_chunks method."""

    def test_ingest_chunks_empty_list(self, lancedb_store):
        """Test ingesting empty chunks list does nothing."""
        store, _, mock_table, _ = lancedb_store
        store.ingest_chunks([], np.array([], dtype=np.float32).reshape(0, 3))

        store.mapper.to_record_batch.assert_not_called()
        mock_table.add.assert_not_called()

    def test_ingest_chunks_calls_mapper_and_adds(self, lancedb_store):
        """Test ingesting chunks converts to batch and adds to table."""
        store, _, mock_table, _ = lancedb_store
        chunks = [{"id": "chunk_0"}, {"id": "chunk_1"}]
        vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

        mock_batch = MagicMock()
        store.mapper.to_record_batch.return_value = mock_batch

        store.ingest_chunks(chunks, vectors)

        store.mapper.to_record_batch.assert_called_once_with(chunks, vectors)
        mock_table.add.assert_called_once_with(mock_batch)


class TestCompact:
    """Tests for the compact method."""

    def test_compact_calls_optimize(self, lancedb_store):
        """Test that compact calls table.optimize()."""
        store, _, mock_table, _ = lancedb_store
        store.compact()

        mock_table.optimize.assert_called_once()


class TestCreateIndices:
    """Tests for the create_indices method."""

    def test_create_indices_with_few_rows(self, lancedb_store):
        """Test index creation with fewer than 256 rows skips vector index."""
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 100

        store.create_indices()

        # Should not create vector index for small tables
        mock_table.create_index.assert_not_called()

    def test_create_indices_vector_index_large_table(self, lancedb_store):
        """Test vector index creation for large tables."""
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 10000

        store.create_indices()

        mock_table.create_index.assert_called_once()
        call_kwargs = mock_table.create_index.call_args.kwargs
        assert call_kwargs["metric"] == "cosine"
        assert call_kwargs["vector_column_name"] == "vector"
        assert call_kwargs["index_type"] == "IVF_PQ"

    def test_create_indices_fts_index(self, lancedb_store):
        """Test FTS index creation."""
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 100

        store.create_indices()

        # FTS index should be created even for small tables
        mock_table.create_fts_index.assert_called_once_with(
            "text",
            replace=True,
        )

    def test_create_indices_fts_failure_handled(self, lancedb_store, capsys):
        """Test that FTS index failure is handled gracefully."""
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 100
        mock_table.create_fts_index.side_effect = Exception(
            "tantivy not installed"
        )

        # Should not raise
        store.create_indices()

        captured = capsys.readouterr()
        assert "FTS Indexing failed" in captured.out

    def test_vector_index_partitions_scaling(self, lancedb_store):
        """Test that vector index partitions scale with table size."""
        # sqrt(10000) = 100, but capped at 256
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 10000

        store.create_indices()

        call_kwargs = mock_table.create_index.call_args.kwargs
        assert call_kwargs["num_partitions"] == 100

    def test_vector_index_partitions_capped(self, lancedb_store):
        """Test that vector index partitions are capped at 256."""
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 100000

        store.create_indices()

        call_kwargs = mock_table.create_index.call_args.kwargs
        assert call_kwargs["num_partitions"] == 256  # capped


class TestGetExistingHashes:
    """Tests for the get_existing_hashes method."""

    def test_get_hashes_empty_table(self, lancedb_store):
        """Test getting hashes from empty table returns empty set."""
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 0

        result = store.get_existing_hashes()

        assert result == set()
        mock_table.to_polars.assert_not_called()

    def test_get_hashes_returns_unique_set(self, lancedb_store):
        """Test getting hashes returns set of unique hashes."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 10

        # Mock polars DataFrame
        mock_df = pl.DataFrame(
            {
                "content_hash": ["hash1", "hash2", "hash1", "hash3"]  # hash1 is duplicated
            }
        )
        mock_table.to_polars.return_value = mock_df

        result = store.get_existing_hashes()

        assert result == {"hash1", "hash2", "hash3"}
        mock_table.to_polars.assert_called_once_with(
            columns=["content_hash"]
        )


class TestSearch:
    """Tests for the search method."""

    def test_search_basic_hybrid(self, lancedb_store):
        """Test basic hybrid search."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Mock the search chain
        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search

        mock_results = pl.DataFrame(
            {
                "id": ["chunk_0"],
                "text": ["content"],
                "source": ["file.md"],
                "content_hash": ["hash1"],
                "_distance": [0.9],
            }
        )
        mock_search.to_polars.return_value = mock_results
        mock_table.search.return_value = mock_search

        # Setup mapper to return a result
        expected_result = MagicMock()
        store.mapper.from_polars_row.return_value = expected_result

        results = store.search(
            query="test query",
            query_vector=query_vector,
            limit=5,
        )

        mock_table.search.assert_called_once_with(query_type="hybrid")
        mock_search.vector.assert_called_once_with(query_vector)
        mock_search.text.assert_called_once_with("test query")
        mock_search.nprobes.assert_called_once_with(10)
        mock_search.limit.assert_called_once_with(5)
        assert results == [expected_result]

    def test_search_with_source_filter(self, lancedb_store):
        """Test search with source filter."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.where.return_value = mock_search

        mock_results = pl.DataFrame({"id": [], "text": [], "source": [], "content_hash": [], "_distance": []})
        mock_search.to_polars.return_value = mock_results
        mock_table.search.return_value = mock_search

        store.search(
            query="test",
            query_vector=query_vector,
            source_filter="docs/specific.md",
        )

        # Verify where clause was applied with escaped filter
        mock_search.where.assert_called_once_with(
            "source = 'docs/specific.md'",
            prefilter=True,
        )

    def test_search_source_filter_sql_injection_protection(self, lancedb_store):
        """Test that source filter escapes single quotes."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.where.return_value = mock_search

        mock_results = pl.DataFrame({"id": [], "text": [], "source": [], "content_hash": [], "_distance": []})
        mock_search.to_polars.return_value = mock_results
        mock_table.search.return_value = mock_search

        # Attempt SQL injection
        store.search(
            query="test",
            query_vector=query_vector,
            source_filter="file' OR '1'='1",
        )

        # Verify quotes are escaped
        mock_search.where.assert_called_once_with(
            "source = 'file'' OR ''1''=''1'",
            prefilter=True,
        )

    def test_search_with_min_time_filter(self, lancedb_store):
        """Test search with min_time filter for SQL queries."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.where.return_value = mock_search

        mock_results = pl.DataFrame({"id": [], "text": [], "source": [], "content_hash": [], "_distance": []})
        mock_search.to_polars.return_value = mock_results
        mock_table.search.return_value = mock_search

        store.search(
            query="slow query",
            query_vector=query_vector,
            min_time=100.0,
        )

        # Verify both filters are applied
        assert mock_search.where.call_count == 1
        mock_search.where.assert_called_with(
            "execution_time_ms >= 100.0",
            prefilter=True,
        )

    def test_search_falls_back_to_vector_on_hybrid_failure(self, lancedb_store):
        """Test that search falls back to pure vector search if hybrid fails."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # First call (hybrid) fails, second call (vector) succeeds
        mock_hybrid_search = MagicMock()
        mock_hybrid_search.vector.side_effect = Exception("Hybrid not available")

        mock_vector_search = MagicMock()
        mock_vector_search.metric.return_value = mock_vector_search
        mock_vector_search.nprobes.return_value = mock_vector_search
        mock_vector_search.limit.return_value = mock_vector_search

        mock_results = pl.DataFrame({"id": [], "text": [], "source": [], "content_hash": [], "_distance": []})
        mock_vector_search.to_polars.return_value = mock_results

        mock_table.search.side_effect = [
            mock_hybrid_search,  # First call for hybrid
            mock_vector_search,  # Second call for fallback
        ]

        store.search(query="test", query_vector=query_vector)

        # Verify fallback was used
        assert mock_table.search.call_count == 2
        mock_vector_search.metric.assert_called_once_with("cosine")

    def test_search_handles_null_distance(self, lancedb_store):
        """Test that search handles null _distance values (FTS matches)."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search

        # Result with null distance (FTS match)
        mock_results = pl.DataFrame(
            {
                "id": ["chunk_0", "chunk_1"],
                "text": ["content1", "content2"],
                "source": ["file1.md", "file2.md"],
                "content_hash": ["hash1", "hash2"],
                "_distance": [0.9, None],  # None for FTS match
            }
        )
        mock_search.to_polars.return_value = mock_results
        mock_table.search.return_value = mock_search

        store.mapper.from_polars_row = MagicMock(
            side_effect=lambda row, score: (row["id"], score)
        )

        results = store.search(query="test", query_vector=query_vector)

        # Verify both results are returned with correct scores
        assert results == [("chunk_0", 0.9), ("chunk_1", None)]
