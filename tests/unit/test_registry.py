"""Unit tests for ComponentRegistry."""
import pytest

from dbs_vector.core.registry import ComponentRegistry
from dbs_vector.infrastructure.chunking.document import DocumentChunker
from dbs_vector.infrastructure.chunking.sql import SqlChunker
from dbs_vector.infrastructure.storage.mappers import DocumentMapper, SqlMapper


class TestGetMapper:
    """Tests for the get_mapper method."""

    def test_get_document_mapper(self):
        """Test retrieving DocumentMapper class."""
        mapper_class = ComponentRegistry.get_mapper("document")
        assert mapper_class is DocumentMapper

    def test_get_sql_mapper(self):
        """Test retrieving SqlMapper class."""
        mapper_class = ComponentRegistry.get_mapper("sql")
        assert mapper_class is SqlMapper

    def test_get_unknown_mapper_raises(self):
        """Test that unknown mapper type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mapper type: 'unknown'"):
            ComponentRegistry.get_mapper("unknown")


class TestGetChunker:
    """Tests for the get_chunker method."""

    def test_get_document_chunker(self):
        """Test retrieving DocumentChunker class."""
        chunker_class = ComponentRegistry.get_chunker("document")
        assert chunker_class is DocumentChunker

    def test_get_sql_chunker(self):
        """Test retrieving SqlChunker class."""
        chunker_class = ComponentRegistry.get_chunker("sql")
        assert chunker_class is SqlChunker

    def test_get_unknown_chunker_raises(self):
        """Test that unknown chunker type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunker type: 'unknown'"):
            ComponentRegistry.get_chunker("unknown")


class TestRegistryUsage:
    """Tests for typical registry usage patterns."""

    def test_instantiate_components_via_registry(self):
        """Test typical pattern: get classes from registry and instantiate."""
        # Get components for document engine
        mapper_class = ComponentRegistry.get_mapper("document")
        chunker_class = ComponentRegistry.get_chunker("document")

        # Verify they can be instantiated
        mapper = mapper_class(vector_dimension=384)
        chunker = chunker_class(max_chars=1000)

        assert isinstance(mapper, DocumentMapper)
        assert isinstance(chunker, DocumentChunker)
        assert mapper.vector_dimension == 384

    def test_sql_components_via_registry(self):
        """Test SQL components work via registry."""
        mapper_class = ComponentRegistry.get_mapper("sql")
        chunker_class = ComponentRegistry.get_chunker("sql")

        mapper = mapper_class(vector_dimension=768)
        chunker = chunker_class()  # SQL chunker takes no args

        assert isinstance(mapper, SqlMapper)
        assert isinstance(chunker, SqlChunker)
