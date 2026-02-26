"""Unit tests for DocumentMapper and SqlMapper."""
import numpy as np
import pyarrow as pa
import pytest

from dbs_vector.core.models import Chunk, SearchResult, SqlChunk
from dbs_vector.infrastructure.storage.mappers import DocumentMapper, SqlMapper


class TestDocumentMapper:
    """Tests for DocumentMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a DocumentMapper with 3-dimensional vectors for testing."""
        return DocumentMapper(vector_dimension=3)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                id="chunk_0",
                text="First chunk content",
                source="docs/file1.md",
                content_hash="hash1",
                node_type="paragraph",
                parent_scope="# Section 1",
                line_range="1-10",
            ),
            Chunk(
                id="chunk_1",
                text="Second chunk content",
                source="docs/file1.md",
                content_hash="hash2",
                node_type="code",
                parent_scope="## Subsection",
                line_range="20-30",
            ),
        ]

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors matching the chunks."""
        return np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=np.float32,
        )

    def test_schema_structure(self, mapper):
        """Test that the schema has the correct structure."""
        schema = mapper.schema

        assert isinstance(schema, pa.Schema)
        field_names = [field.name for field in schema]

        assert "id" in field_names
        assert "vector" in field_names
        assert "text" in field_names
        assert "source" in field_names
        assert "content_hash" in field_names
        assert "node_type" in field_names
        assert "parent_scope" in field_names
        assert "line_range" in field_names

    def test_vector_field_type(self, mapper):
        """Test that vector field has correct type with fixed size list."""
        vector_field = mapper.schema.field("vector")

        assert pa.types.is_fixed_size_list(vector_field.type)
        assert vector_field.type.list_size == 3
        assert pa.types.is_float32(vector_field.type.value_type)

    def test_to_record_batch(self, mapper, sample_chunks, sample_vectors):
        """Test converting chunks and vectors to a RecordBatch."""
        batch = mapper.to_record_batch(sample_chunks, sample_vectors)

        assert isinstance(batch, pa.RecordBatch)
        assert batch.num_rows == 2
        assert batch.num_columns == 8

        # Check that data is correctly placed
        assert batch.column("id").to_pylist() == ["chunk_0", "chunk_1"]
        assert batch.column("text").to_pylist() == [
            "First chunk content",
            "Second chunk content",
        ]
        assert batch.column("source").to_pylist() == ["docs/file1.md", "docs/file1.md"]
        assert batch.column("content_hash").to_pylist() == ["hash1", "hash2"]
        assert batch.column("node_type").to_pylist() == ["paragraph", "code"]

    def test_to_record_batch_with_none_values(self, mapper):
        """Test RecordBatch creation with None values in nullable fields."""
        chunks = [
            Chunk(
                id="chunk_0",
                text="Content",
                source="file.md",
                content_hash="hash1",
                node_type=None,
                parent_scope=None,
                line_range=None,
            )
        ]
        vectors = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        batch = mapper.to_record_batch(chunks, vectors)

        assert batch.num_rows == 1
        # None values should be handled gracefully
        assert batch.column("node_type")[0].as_py() is None
        assert batch.column("parent_scope")[0].as_py() is None

    def test_from_polars_row(self, mapper):
        """Test converting a polars row dict to SearchResult."""
        row = {
            "id": "chunk_42",
            "text": "Sample content",
            "source": "docs/readme.md",
            "content_hash": "abc123",
            "node_type": "paragraph",
            "parent_scope": "# Header",
            "line_range": "5-15",
        }

        result = mapper.from_polars_row(row, score=0.95)

        assert isinstance(result, SearchResult)
        assert result.chunk.id == "chunk_42"
        assert result.chunk.text == "Sample content"
        assert result.chunk.source == "docs/readme.md"
        assert result.chunk.content_hash == "abc123"
        assert result.score == 0.95
        assert result.distance == 0.95
        assert result.is_fts_match is False

    def test_from_polars_row_with_none_score(self, mapper):
        """Test converting row with None score (FTS match)."""
        row = {
            "id": "chunk_0",
            "text": "Content",
            "source": "file.md",
            "content_hash": "hash1",
            "node_type": None,
            "parent_scope": None,
            "line_range": None,
        }

        result = mapper.from_polars_row(row, score=None)

        assert result.score is None
        assert result.distance is None
        assert result.is_fts_match is True

    def test_from_polars_row_missing_optional_fields(self, mapper):
        """Test converting row without optional fields in dict."""
        row = {
            "id": "chunk_0",
            "text": "Content",
            "source": "file.md",
            "content_hash": "hash1",
            # node_type, parent_scope, line_range missing
        }

        result = mapper.from_polars_row(row, score=0.8)

        assert result.chunk.node_type is None
        assert result.chunk.parent_scope is None
        assert result.chunk.line_range is None


class TestSqlMapper:
    """Tests for SqlMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a SqlMapper with 4-dimensional vectors for testing."""
        return SqlMapper(vector_dimension=4)

    @pytest.fixture
    def sample_sql_chunks(self):
        """Create sample SQL chunks for testing."""
        return [
            SqlChunk(
                id="sql_0",
                text="SELECT * FROM users",
                raw_query="SELECT * FROM users WHERE active = 1",
                source="db_production",
                execution_time_ms=150.5,
                calls=42,
                content_hash="sql_hash1",
            ),
            SqlChunk(
                id="sql_1",
                text="UPDATE orders SET status = 'shipped'",
                raw_query="UPDATE orders SET status = 'shipped' WHERE id = 123",
                source="db_production",
                execution_time_ms=89.0,
                calls=15,
                content_hash="sql_hash2",
            ),
        ]

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors matching the SQL chunks."""
        return np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            dtype=np.float32,
        )

    def test_schema_structure(self, mapper):
        """Test that the SQL schema has the correct structure."""
        schema = mapper.schema

        field_names = [field.name for field in schema]

        assert "id" in field_names
        assert "vector" in field_names
        assert "text" in field_names
        assert "raw_query" in field_names
        assert "source" in field_names
        assert "execution_time_ms" in field_names
        assert "calls" in field_names
        assert "content_hash" in field_names

    def test_vector_dimension(self, mapper):
        """Test that vector field has correct dimension."""
        vector_field = mapper.schema.field("vector")
        assert vector_field.type.list_size == 4

    def test_numeric_field_types(self, mapper):
        """Test that numeric fields have correct types."""
        exec_time_field = mapper.schema.field("execution_time_ms")
        calls_field = mapper.schema.field("calls")

        assert pa.types.is_float64(exec_time_field.type)
        assert pa.types.is_int64(calls_field.type)

    def test_to_record_batch(self, mapper, sample_sql_chunks, sample_vectors):
        """Test converting SQL chunks and vectors to a RecordBatch."""
        batch = mapper.to_record_batch(sample_sql_chunks, sample_vectors)

        assert isinstance(batch, pa.RecordBatch)
        assert batch.num_rows == 2
        assert batch.num_columns == 8

        assert batch.column("id").to_pylist() == ["sql_0", "sql_1"]
        assert batch.column("text").to_pylist() == [
            "SELECT * FROM users",
            "UPDATE orders SET status = 'shipped'",
        ]
        assert batch.column("raw_query").to_pylist() == [
            "SELECT * FROM users WHERE active = 1",
            "UPDATE orders SET status = 'shipped' WHERE id = 123",
        ]
        assert batch.column("source").to_pylist() == ["db_production", "db_production"]
        assert batch.column("execution_time_ms").to_pylist() == [150.5, 89.0]
        assert batch.column("calls").to_pylist() == [42, 15]

    def test_from_polars_row(self, mapper):
        """Test converting a polars row dict to SqlSearchResult."""
        from dbs_vector.core.models import SqlSearchResult

        row = {
            "id": "sql_42",
            "text": "SELECT COUNT(*) FROM orders",
            "raw_query": "SELECT COUNT(*) FROM orders WHERE status = 'pending'",
            "source": "db_analytics",
            "execution_time_ms": 245.7,
            "calls": 1000,
            "content_hash": "hash_abc",
        }

        result = mapper.from_polars_row(row, score=0.88)

        assert isinstance(result, SqlSearchResult)
        assert result.chunk.id == "sql_42"
        assert result.chunk.text == "SELECT COUNT(*) FROM orders"
        assert result.chunk.raw_query == "SELECT COUNT(*) FROM orders WHERE status = 'pending'"
        assert result.chunk.source == "db_analytics"
        assert result.chunk.execution_time_ms == 245.7
        assert result.chunk.calls == 1000
        assert result.score == 0.88
        assert result.is_fts_match is False

    def test_from_polars_row_fts_match(self, mapper):
        """Test SqlSearchResult with None score (FTS match)."""
        row = {
            "id": "sql_0",
            "text": "INSERT INTO logs",
            "raw_query": "INSERT INTO logs VALUES (...)",
            "source": "db_logs",
            "execution_time_ms": 10.0,
            "calls": 5000,
            "content_hash": "hash_xyz",
        }

        result = mapper.from_polars_row(row, score=None)

        assert result.score is None
        assert result.is_fts_match is True


class TestMapperEdgeCases:
    """Edge case tests for both mappers."""

    def test_document_mapper_empty_chunks(self):
        """Test DocumentMapper with empty chunks list."""
        mapper = DocumentMapper(vector_dimension=3)
        empty_vectors = np.array([], dtype=np.float32).reshape(0, 3)

        batch = mapper.to_record_batch([], empty_vectors)

        assert batch.num_rows == 0
        assert batch.num_columns == 8

    def test_sql_mapper_empty_chunks(self):
        """Test SqlMapper with empty chunks list."""
        mapper = SqlMapper(vector_dimension=4)
        empty_vectors = np.array([], dtype=np.float32).reshape(0, 4)

        batch = mapper.to_record_batch([], empty_vectors)

        assert batch.num_rows == 0
        assert batch.num_columns == 8

    def test_document_mapper_single_chunk(self):
        """Test DocumentMapper with single chunk."""
        mapper = DocumentMapper(vector_dimension=2)
        chunks = [
            Chunk(
                id="single",
                text="Only chunk",
                source="file.md",
                content_hash="hash",
            )
        ]
        vectors = np.array([[0.5, 0.5]], dtype=np.float32)

        batch = mapper.to_record_batch(chunks, vectors)

        assert batch.num_rows == 1
        assert batch.column("id").to_pylist() == ["single"]
