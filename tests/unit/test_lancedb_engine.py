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

    def test_clear_resets_hybrid_cache(self, lancedb_store):
        """clear() must reset _hybrid_ok to None so hybrid search is retried
        after the table (and its FTS index) has been dropped and recreated."""
        store, _, _, _ = lancedb_store
        store._hybrid_ok = False  # simulate a prior failed hybrid attempt
        store.clear()
        assert store._hybrid_ok is None


class TestIngestChunks:
    """Tests for the ingest_chunks method."""

    def test_ingest_chunks_empty_list(self, lancedb_store):
        """Test ingesting empty chunks list does nothing."""
        store, _, mock_table, _ = lancedb_store
        store.ingest_chunks([], np.array([], dtype=np.float32).reshape(0, 3), workflow="default")

        store.mapper.to_record_batch.assert_not_called()
        mock_table.add.assert_not_called()

    def test_ingest_chunks_calls_mapper_and_adds(self, lancedb_store):
        """Test ingesting chunks converts to batch and adds to table."""
        store, _, mock_table, _ = lancedb_store
        chunks = [{"id": "chunk_0"}, {"id": "chunk_1"}]
        vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

        mock_batch = MagicMock()
        store.mapper.to_record_batch.return_value = mock_batch

        store.ingest_chunks(chunks, vectors, workflow="default")

        store.mapper.to_record_batch.assert_called_once_with(chunks, vectors, "default")
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

    def test_create_indices_fts_failure_handled(self, lancedb_store, caplog):
        """Test that FTS index failure is handled gracefully."""
        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 100
        mock_table.create_fts_index.side_effect = Exception("tantivy not installed")

        # Should not raise
        store.create_indices()

        assert "FTS indexing failed" in caplog.text

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
        mock_table.search.assert_not_called()

    def test_get_hashes_returns_unique_set(self, lancedb_store):
        """Test getting hashes returns set of unique hashes."""
        import pyarrow as pa

        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 10

        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.to_arrow.return_value = pa.table(
            {"content_hash": ["hash1", "hash2", "hash1", "hash3"]}
        )
        mock_table.search.return_value = mock_query

        result = store.get_existing_hashes()

        assert result == {"hash1", "hash2", "hash3"}
        mock_query.select.assert_called_once_with(["content_hash"])

    def test_get_hashes_uses_search_select_projection(self, lancedb_store):
        """Regression guard: get_existing_hashes must go through
        table.search().select([...]).to_arrow(), NOT table.to_lance()
        (which requires pylance and breaks at runtime)."""
        import pyarrow as pa

        store, _, mock_table, _ = lancedb_store
        mock_table.__len__.return_value = 3

        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.to_arrow.return_value = pa.table({"content_hash": ["h1", "h2", "h3"]})
        mock_table.search.return_value = mock_query

        result = store.get_existing_hashes()

        assert result == {"h1", "h2", "h3"}
        mock_table.search.assert_called_once_with()
        mock_query.select.assert_called_once_with(["content_hash"])
        mock_query.to_arrow.assert_called_once()
        mock_table.to_lance.assert_not_called()


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

        mock_results = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
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

        mock_results = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
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

        mock_results = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
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
        mock_hybrid_search.vector.side_effect = ValueError("Hybrid not available")

        mock_vector_search = MagicMock()
        mock_vector_search.metric.return_value = mock_vector_search
        mock_vector_search.nprobes.return_value = mock_vector_search
        mock_vector_search.limit.return_value = mock_vector_search

        mock_results = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
        mock_vector_search.to_polars.return_value = mock_results

        mock_table.search.side_effect = [
            mock_hybrid_search,  # First call for hybrid
            mock_vector_search,  # Second call for fallback
        ]

        store.search(query="test", query_vector=query_vector)

        # Verify fallback was used
        assert mock_table.search.call_count == 2
        mock_vector_search.metric.assert_called_once_with("cosine")

    def test_search_reads_relevance_score_for_hybrid(self, lancedb_store):
        """Hybrid search rows carry _relevance_score, not _distance.

        The mapper must receive the relevance score as `score` (not None),
        and is_fts_match must be False for genuine hybrid hits.
        """
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search

        # Hybrid result: _relevance_score column, no _distance column
        mock_results = pl.DataFrame(
            {
                "id": ["chunk_0"],
                "text": ["content"],
                "source": ["file.md"],
                "content_hash": ["hash1"],
                "_relevance_score": [0.82],
            }
        )
        mock_search.to_polars.return_value = mock_results
        mock_table.search.return_value = mock_search

        store.mapper.from_polars_row = MagicMock(
            side_effect=lambda row, score, distance: (row["id"], score, distance)
        )

        results = store.search(query="test", query_vector=query_vector)

        assert results == [("chunk_0", 0.82, None)]

    def test_search_fallback_preserves_filters(self, lancedb_store):
        """When hybrid to_polars() fails and falls back to vector search,
        the source_filter and min_time constraints must still be applied.
        """
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # First (hybrid) search object — .to_polars() blows up
        hybrid_op = MagicMock(name="hybrid_op")
        hybrid_op.vector.return_value = hybrid_op
        hybrid_op.text.return_value = hybrid_op
        hybrid_op.nprobes.return_value = hybrid_op
        hybrid_op.limit.return_value = hybrid_op
        hybrid_op.where.return_value = hybrid_op
        hybrid_op.to_polars.side_effect = RuntimeError("FTS index missing")

        # Second (vector fallback) search object — must also receive .where() calls
        fallback_op = MagicMock(name="fallback_op")
        fallback_op.metric.return_value = fallback_op
        fallback_op.nprobes.return_value = fallback_op
        fallback_op.limit.return_value = fallback_op
        fallback_op.where.return_value = fallback_op
        fallback_op.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )

        # First call → hybrid_op, second call (fallback) → fallback_op
        mock_table.search.side_effect = [hybrid_op, fallback_op]

        store.search(
            query="test",
            query_vector=query_vector,
            source_filter="some/file.md",
            min_time=100.0,
        )

        # The fallback vector op MUST have received both filter pushdowns
        where_calls = [c.args[0] for c in fallback_op.where.call_args_list]
        assert any("source = 'some/file.md'" in w for w in where_calls), (
            f"source filter not applied on fallback; got where calls: {where_calls}"
        )
        assert any("execution_time_ms >= 100.0" in w for w in where_calls), (
            f"min_time filter not applied on fallback; got where calls: {where_calls}"
        )

    def test_search_handles_fts_only_match(self, lancedb_store):
        """Rows with neither _relevance_score nor _distance are FTS-only matches."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search

        mock_results = pl.DataFrame(
            {
                "id": ["chunk_0", "chunk_1"],
                "text": ["content1", "content2"],
                "source": ["file1.md", "file2.md"],
                "content_hash": ["hash1", "hash2"],
                "_distance": [0.9, None],
            }
        )
        mock_search.to_polars.return_value = mock_results
        mock_table.search.return_value = mock_search

        store.mapper.from_polars_row = MagicMock(
            side_effect=lambda row, score, distance: (row["id"], score, distance)
        )

        results = store.search(query="test", query_vector=query_vector)
        assert results == [("chunk_0", None, 0.9), ("chunk_1", None, None)]

    def test_search_min_lock_time_filter_emits_where_clause(self, lancedb_store):
        """min_lock_time kwarg must add a `lock_time_sec >= X` prefilter."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.where.return_value = mock_search
        mock_search.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
        mock_table.search.return_value = mock_search

        store.search(query="q", query_vector=query_vector, min_lock_time=10.0)

        where_calls = [c.args[0] for c in mock_search.where.call_args_list]
        assert any("lock_time_sec >= 10.0" in w for w in where_calls), (
            f"min_lock_time filter not emitted; got where calls: {where_calls}"
        )

    def test_search_table_filter_emits_id_prefilter_and_bypasses_index(self, lancedb_store):
        """table_filter resolves exact matching ids and bypasses the IVF index."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        store._matching_table_ids = MagicMock(return_value=["fp1"])  # type: ignore[method-assign]
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.bypass_vector_index.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.where.return_value = mock_search
        mock_search.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
        mock_table.search.return_value = mock_search

        store.search(query="q", query_vector=query_vector, table_filter="tx_process")

        where_calls = [c.args[0] for c in mock_search.where.call_args_list]
        assert any("id IN ('fp1')" in w for w in where_calls), (
            f"table_filter id prefilter not emitted; got where calls: {where_calls}"
        )
        # When table_filter is set, the IVF index must be bypassed.
        mock_search.bypass_vector_index.assert_called()
        mock_search.nprobes.assert_not_called()

    def test_search_no_table_filter_uses_nprobes(self, lancedb_store):
        """Without table_filter, the default IVF approximate path with
        nprobes() is used (faster, well-suited for non-selective queries)."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.bypass_vector_index.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
        mock_table.search.return_value = mock_search

        store.search(query="q", query_vector=query_vector)

        mock_search.nprobes.assert_called()
        mock_search.bypass_vector_index.assert_not_called()

    def test_search_refreshes_table_to_latest_version(self, lancedb_store):
        """Each search must call table.checkout_latest() so a long-lived
        SearchService picks up rows committed by another process (e.g.,
        an out-of-band `dbs-vector ingest` while MCP is running)."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
        mock_table.search.return_value = mock_search

        # Two calls — checkout_latest must fire each time, not just once.
        store.search(query="first", query_vector=query_vector)
        store.search(query="second", query_vector=query_vector)

        assert mock_table.checkout_latest.call_count == 2

    def test_search_filters_applied_before_bypass_index(self, lancedb_store):
        """Filters must be added before bypass_vector_index() when table_filter is set."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        store._matching_table_ids = MagicMock(return_value=["fp1"])  # type: ignore[method-assign]
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        call_log: list[str] = []
        mock_search = MagicMock()

        def _log(method_name):
            def _f(*args, **kwargs):
                call_log.append(method_name)
                return mock_search

            return _f

        mock_search.vector.side_effect = _log("vector")
        mock_search.text.side_effect = _log("text")
        mock_search.where.side_effect = _log("where")
        mock_search.bypass_vector_index.side_effect = _log("bypass_vector_index")
        mock_search.nprobes.side_effect = _log("nprobes")
        mock_search.limit.side_effect = _log("limit")
        mock_search.metric.side_effect = _log("metric")
        mock_search.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )
        mock_table.search.return_value = mock_search

        store.search(
            query="q",
            query_vector=query_vector,
            min_lock_time=1.0,
            table_filter="magentoorders",
        )

        last_where = max(i for i, method in enumerate(call_log) if method == "where")
        bypass_idx = call_log.index("bypass_vector_index")
        assert last_where < bypass_idx, (
            f"Filter ordering bug: bypass_vector_index() called at index "
            f"{bypass_idx}, final where() at index {last_where}. "
            f"Call order: {call_log}"
        )

    def test_search_dedupes_results_by_id(self, lancedb_store):
        """Hybrid search may return the same id from vector and FTS branches."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.bypass_vector_index.return_value = mock_search
        mock_search.where.return_value = mock_search
        mock_search.to_polars.return_value = pl.DataFrame(
            {
                "id": ["dup_0", "dup_0", "uniq_1"],
                "text": ["a", "a", "b"],
                "source": ["s", "s", "s"],
                "content_hash": ["h", "h", "h2"],
                "_relevance_score": [0.9, 0.8, 0.7],
            }
        )
        mock_table.search.return_value = mock_search

        store.mapper.from_polars_row = MagicMock(side_effect=lambda row, score, distance: row["id"])

        results = store.search(query="q", query_vector=query_vector)

        assert results == ["dup_0", "uniq_1"]
        assert store.mapper.from_polars_row.call_count == 2

    def test_hybrid_failure_cached_and_reset_by_create_indices(self, lancedb_store, monkeypatch):
        """After hybrid fails once, subsequent searches skip the hybrid attempt
        (cache hit). create_indices resets the cache so hybrid is retried."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Vector-path chain succeeds; hybrid path raises deterministically.
        mock_search = MagicMock()
        mock_search.metric.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )

        def search_side_effect(*args, **kwargs):
            if kwargs.get("query_type") == "hybrid":
                raise ValueError("fts index not found")
            return mock_search

        mock_table.search.side_effect = search_side_effect

        warnings: list[str] = []
        monkeypatch.setattr(
            "dbs_vector.infrastructure.storage.lancedb_engine.logger.warning",
            lambda msg, *a, **k: warnings.append(str(msg)),
        )

        store.search("q", qv, limit=3)
        store.search("q", qv, limit=3)

        hybrid_attempts = [
            c for c in mock_table.search.call_args_list if c.kwargs.get("query_type") == "hybrid"
        ]
        assert len(hybrid_attempts) == 1, (
            f"Expected 1 hybrid attempt (2nd call should be cached), "
            f"got {len(hybrid_attempts)}. All calls: {mock_table.search.call_args_list}"
        )
        assert sum("Hybrid search unavailable" in w for w in warnings) == 1, (
            f"Expected exactly 1 warning, got: {warnings}"
        )
        assert store._hybrid_ok is False

        store.create_indices()  # FTS rebuilt → cache reset
        assert store._hybrid_ok is None

    def test_hybrid_cache_resets_when_table_version_advances(self, lancedb_store):
        """A version advance seen via checkout_latest() (e.g. ANOTHER process
        ingesting and creating the FTS index) must allow a hybrid retry —
        otherwise a long-lived MCP server stays vector-only until restart."""
        import polars as pl

        store, _, mock_table, _ = lancedb_store
        qv = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_search = MagicMock()
        mock_search.vector.return_value = mock_search
        mock_search.text.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.nprobes.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_polars.return_value = pl.DataFrame(
            {"id": [], "text": [], "source": [], "content_hash": [], "_distance": []}
        )

        fts_exists = {"v": False}

        def search_side_effect(*args, **kwargs):
            if kwargs.get("query_type") == "hybrid" and not fts_exists["v"]:
                raise ValueError("fts index not found")
            return mock_search

        mock_table.search.side_effect = search_side_effect

        mock_table.version = 1
        store.search("q", qv, limit=3)  # hybrid fails → cached False
        store.search("q", qv, limit=3)  # same version → cache hit, no retry

        # Another process ingests and builds the FTS index → version advances.
        mock_table.version = 2
        fts_exists["v"] = True
        store.search("q", qv, limit=3)  # version change → cache reset → retry

        hybrid_attempts = [
            c for c in mock_table.search.call_args_list if c.kwargs.get("query_type") == "hybrid"
        ]
        assert len(hybrid_attempts) == 2, (
            f"Expected hybrid retried after version bump, calls: {mock_table.search.call_args_list}"
        )
        assert store._hybrid_ok is True
