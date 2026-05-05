"""Dependency-injection factory for engines.

Extracted from cli.py so that api/state.py does not have to import the
CLI layer (which pulls in typer + Hugging Face env-var side effects).
"""

import os
from typing import Any, NamedTuple

from dbs_vector.config import settings
from dbs_vector.core.registry import ComponentRegistry
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore


class EngineDeps(NamedTuple):
    """Resolved per-engine runtime dependencies."""

    embedder: Any
    store: Any
    chunker: Any
    workflow: str


def build_dependencies(
    engine_name: str,
    query_override: str | None = None,
    url_override: str | None = None,
) -> EngineDeps:
    """Resolve the chunker / mapper / embedder / store stack for an engine."""
    if engine_name not in settings.engines:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Check {os.environ.get('DBS_CONFIG_FILE', 'config.yaml')}."
        )

    config = settings.engines[engine_name]

    embedder = MLXEmbedder(
        model_name=config.model_name,
        max_token_length=config.max_token_length,
        dimension=config.vector_dimension,
        passage_prefix=config.passage_prefix,
        query_prefix=config.query_prefix,
        attention_mask_dtype=config.attention_mask_dtype,
    )

    MapperClass = ComponentRegistry.get_mapper(config.mapper_type)
    ChunkerClass = ComponentRegistry.get_chunker(config.chunker_type)

    mapper = MapperClass(vector_dimension=config.vector_dimension)
    chunker = ChunkerClass(
        **config.chunker_kwargs(query_override=query_override, url_override=url_override)
    )

    store = LanceDBStore(
        db_path=settings.db_path,
        table_name=config.table_name,
        vector_dimension=config.vector_dimension,
        mapper=mapper,
        nprobes=settings.nprobes,
    )

    return EngineDeps(embedder=embedder, store=store, chunker=chunker, workflow=config.workflow)
