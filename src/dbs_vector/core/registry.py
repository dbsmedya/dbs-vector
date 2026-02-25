from typing import Any

from dbs_vector.core.ports import IStoreMapper
from dbs_vector.infrastructure.chunking.document import DocumentChunker
from dbs_vector.infrastructure.chunking.sql import SqlChunker
from dbs_vector.infrastructure.storage.mappers import DocumentMapper, SqlMapper


class ComponentRegistry:
    """Registry pattern to dynamically map string names to class implementations."""

    _mappers: dict[str, type[IStoreMapper]] = {
        "document": DocumentMapper,
        "sql": SqlMapper,
    }

    _chunkers: dict[str, Any] = {
        "document": DocumentChunker,
        "sql": SqlChunker,
    }

    @classmethod
    def get_mapper(cls, name: str) -> type[IStoreMapper]:
        if name not in cls._mappers:
            raise ValueError(f"Unknown mapper type: '{name}'")
        return cls._mappers[name]

    @classmethod
    def get_chunker(cls, name: str) -> Any:
        if name not in cls._chunkers:
            raise ValueError(f"Unknown chunker type: '{name}'")
        return cls._chunkers[name]
