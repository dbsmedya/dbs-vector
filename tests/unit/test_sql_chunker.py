"""Unit tests for SqlChunker."""

import json

import pytest

from dbs_vector.core.models import Document, SqlChunk
from dbs_vector.infrastructure.chunking.sql import SqlChunker


@pytest.fixture
def chunker():
    """Create a SqlChunker instance."""
    return SqlChunker()


class TestSupportedExtensions:
    """Tests for supported_extensions property."""

    def test_supported_extensions(self, chunker):
        """Test that only .json is supported."""
        assert chunker.supported_extensions == [".json"]


class TestProcessValidRecords:
    """Tests for processing valid SQL query records."""

    def test_single_query_record(self, chunker):
        """Test processing a single query record."""
        records = [
            {
                "query": "SELECT * FROM users WHERE id = 1",
                "normalized_query": "SELECT * FROM users WHERE id = ?",
                "query_hash": "abc123",
                "database": "production",
                "duration": 150.5,
                "calls": 42,
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash1")

        chunks = list(chunker.process(doc))

        assert len(chunks) == 1
        assert isinstance(chunks[0], SqlChunk)
        assert chunks[0].id == "abc123"
        assert chunks[0].text == "SELECT * FROM users WHERE id = ?"
        assert chunks[0].raw_query == "SELECT * FROM users WHERE id = 1"
        assert chunks[0].source == "production"
        assert chunks[0].execution_time_ms == 150.5
        assert chunks[0].calls == 42
        assert len(chunks[0].content_hash) == 16  # SHA256 truncated to 16 chars

    def test_multiple_query_records(self, chunker):
        """Test processing multiple query records."""
        records = [
            {
                "query": "SELECT 1",
                "normalized_query": "SELECT ?",
                "database": "db1",
            },
            {
                "query": "SELECT 2",
                "normalized_query": "SELECT ?",
                "database": "db2",
            },
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash2")

        chunks = list(chunker.process(doc))

        assert len(chunks) == 2
        assert chunks[0].source == "db1"
        assert chunks[1].source == "db2"

    def test_normalized_query_fallback(self, chunker):
        """Test fallback to 'normalized' field if 'normalized_query' not present."""
        records = [
            {
                "query": "SELECT * FROM orders",
                "normalized": "SELECT * FROM orders",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash3")

        chunks = list(chunker.process(doc))

        assert len(chunks) == 1
        assert chunks[0].text == "SELECT * FROM orders"

    def test_normalized_fallback_to_raw(self, chunker):
        """Test fallback to raw query if no normalized version present."""
        records = [
            {
                "query": "SELECT * FROM items WHERE id = 123",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash4")

        chunks = list(chunker.process(doc))

        assert len(chunks) == 1
        assert chunks[0].text == "SELECT * FROM items WHERE id = 123"
        assert chunks[0].raw_query == "SELECT * FROM items WHERE id = 123"

    def test_query_id_fallback_to_id_field(self, chunker):
        """Test fallback to 'id' field if 'query_hash' not present."""
        records = [
            {
                "query": "SELECT 1",
                "id": "query_123",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash5")

        chunks = list(chunker.process(doc))

        assert chunks[0].id == "query_123"

    def test_query_id_fallback_to_md5(self, chunker):
        """Test fallback to MD5 hash of raw query if no ID field present."""
        raw_query = "SELECT * FROM users"
        records = [
            {
                "query": raw_query,
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash6")

        chunks = list(chunker.process(doc))

        import hashlib

        expected_hash = hashlib.md5(raw_query.encode()).hexdigest()
        assert chunks[0].id == expected_hash

    def test_database_fallback_to_source(self, chunker):
        """Test fallback to 'source' field if 'database' not present."""
        records = [
            {
                "query": "SELECT 1",
                "source": "analytics_db",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash7")

        chunks = list(chunker.process(doc))

        assert chunks[0].source == "analytics_db"

    def test_database_fallback_to_unknown(self, chunker):
        """Test fallback to 'unknown' if no database/source field present."""
        records = [
            {
                "query": "SELECT 1",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash8")

        chunks = list(chunker.process(doc))

        assert chunks[0].source == "unknown"

    def test_duration_fallback_to_execution_time_ms(self, chunker):
        """Test fallback to 'execution_time_ms' field if 'duration' not present."""
        records = [
            {
                "query": "SELECT 1",
                "execution_time_ms": 250.75,
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash9")

        chunks = list(chunker.process(doc))

        assert chunks[0].execution_time_ms == 250.75

    def test_duration_default_to_zero(self, chunker):
        """Test default duration of 0.0 if no time field present."""
        records = [
            {
                "query": "SELECT 1",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash10")

        chunks = list(chunker.process(doc))

        assert chunks[0].execution_time_ms == 0.0

    def test_calls_default_to_one(self, chunker):
        """Test default calls value of 1 if not present."""
        records = [
            {
                "query": "SELECT 1",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash11")

        chunks = list(chunker.process(doc))

        assert chunks[0].calls == 1

    def test_content_hash_consistency(self, chunker):
        """Test that same normalized query produces same content hash."""
        records = [
            {
                "query": "SELECT * FROM users WHERE id = 1",
                "normalized_query": "SELECT * FROM users WHERE id = ?",
            },
            {
                "query": "SELECT * FROM users WHERE id = 2",
                "normalized_query": "SELECT * FROM users WHERE id = ?",
            },
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash12")

        chunks = list(chunker.process(doc))

        # Both have same normalized query, so same hash
        assert chunks[0].content_hash == chunks[1].content_hash


class TestProcessInvalidRecords:
    """Tests for handling invalid/malformed records."""

    def test_empty_normalized_query_skipped(self, chunker):
        """Test that records with empty normalized query are skipped."""
        records = [
            {
                "query": "   ",  # whitespace only
                "normalized_query": "   ",
            },
            {
                "query": "SELECT 1",
                "normalized_query": "SELECT 1",
            },
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash13")

        chunks = list(chunker.process(doc))

        assert len(chunks) == 1
        assert chunks[0].text == "SELECT 1"

    def test_null_query_and_normalized_handled(self, chunker):
        """Test that records with null query and normalized_query are handled gracefully.

        Regression test: Previously, if query was None, raw.encode() would fail
        with AttributeError. The fix uses `or ""` to ensure raw is always a string.
        """
        records = [
            {
                "query": None,  # Explicit null in JSON
                "normalized_query": None,
            },
            {
                "query": "SELECT 1",
                "normalized_query": "SELECT 1",
            },
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash14")

        # Should not raise AttributeError
        chunks = list(chunker.process(doc))

        # First record skipped (empty after null handling), second processed
        assert len(chunks) == 1
        assert chunks[0].text == "SELECT 1"


class TestProcessEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_json_array(self, chunker):
        """Test processing empty JSON array."""
        doc = Document(filepath="empty.json", content="[]", content_hash="hash15")

        chunks = list(chunker.process(doc))

        assert chunks == []

    def test_empty_document(self, chunker):
        """Test processing empty document content."""
        doc = Document(filepath="empty.json", content="", content_hash="hash16")

        chunks = list(chunker.process(doc))

        assert chunks == []

    def test_invalid_json(self, chunker, caplog):
        """Test handling of invalid JSON content."""
        doc = Document(filepath="invalid.json", content="not valid json", content_hash="hash17")

        chunks = list(chunker.process(doc))

        assert chunks == []
        assert "Error decoding JSON" in caplog.text

    def test_json_object_not_array(self, chunker, caplog):
        """Test handling of JSON object instead of array."""
        doc = Document(
            filepath="object.json", content='{"query": "SELECT 1"}', content_hash="hash18"
        )

        chunks = list(chunker.process(doc))

        assert chunks == []
        assert "Expected a JSON array" in caplog.text

    def test_null_duration_treated_as_zero(self, chunker):
        """Test that null duration is treated as 0.0."""
        records = [
            {
                "query": "SELECT 1",
                "duration": None,
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash19")

        chunks = list(chunker.process(doc))

        assert chunks[0].execution_time_ms == 0.0

    def test_string_duration_converted(self, chunker):
        """Test that string duration values are converted to float."""
        records = [
            {
                "query": "SELECT 1",
                "duration": "150.5",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash20")

        chunks = list(chunker.process(doc))

        assert chunks[0].execution_time_ms == 150.5

    def test_string_calls_converted(self, chunker):
        """Test that string calls values are converted to int."""
        records = [
            {
                "query": "SELECT 1",
                "calls": "100",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash21")

        chunks = list(chunker.process(doc))

        assert chunks[0].calls == 100

    def test_complex_query_with_special_chars(self, chunker):
        """Test processing complex query with special characters."""
        complex_query = """
            INSERT INTO users (name, email)
            VALUES ('John O''Connor', 'john@example.com')
            ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name
            RETURNING id, name, email
        """
        records = [
            {
                "query": complex_query,
                "normalized_query": complex_query,
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash22")

        chunks = list(chunker.process(doc))

        assert len(chunks) == 1
        assert chunks[0].text == complex_query

    def test_unicode_query(self, chunker):
        """Test processing query with unicode characters."""
        records = [
            {
                "query": "SELECT '日本語' FROM users WHERE name = 'José'",
                "normalized_query": "SELECT ? FROM users WHERE name = ?",
            }
        ]
        doc = Document(filepath="queries.json", content=json.dumps(records), content_hash="hash23")

        chunks = list(chunker.process(doc))

        assert len(chunks) == 1
        assert "日本語" in chunks[0].raw_query
        assert "José" in chunks[0].raw_query
