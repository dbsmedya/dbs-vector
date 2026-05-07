"""Dependency-injection factory for engines."""

import os
from typing import Any, NamedTuple

from dbs_vector.config import settings
from dbs_vector.core.model_registry import ModelRegistry
from dbs_vector.core.registry import ComponentRegistry
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore


class EngineDeps(NamedTuple):
    """Resolved per-engine runtime dependencies."""

    embedder: Any
    store: Any
    chunker: Any
    workflow: str
    batch_size: int


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

    engine = settings.engines[engine_name]
    contract = ModelRegistry.get(engine.model)

    if engine.tuning_profile not in settings.profiles:
        raise ValueError(
            f"Engine '{engine_name}' references unknown tuning profile "
            f"'{engine.tuning_profile}'. Known: {sorted(settings.profiles)}"
        )
    profile = settings.profiles[engine.tuning_profile]

    embedder = MLXEmbedder(
        model_name=contract.model_name,
        max_token_length=profile.max_token_length,
        dimension=contract.vector_dimension,
        passage_prefix=engine.passage_prefix,
        query_prefix=engine.query_prefix,
        attention_mask_dtype=contract.attention_mask_dtype,
    )

    MapperClass = ComponentRegistry.get_mapper(engine.mapper_type)
    ChunkerClass = ComponentRegistry.get_chunker(engine.chunker_type)

    mapper = MapperClass(vector_dimension=contract.vector_dimension)
    chunker = ChunkerClass(
        **engine.chunker_kwargs(
            chunk_max_chars=profile.chunk_max_chars,
            query_override=query_override,
            url_override=url_override,
        )
    )

    store = LanceDBStore(
        db_path=settings.db_path,
        table_name=engine.table_name,
        vector_dimension=contract.vector_dimension,
        mapper=mapper,
        nprobes=settings.nprobes,
    )

    return EngineDeps(
        embedder=embedder,
        store=store,
        chunker=chunker,
        workflow=engine.workflow,
        batch_size=profile.batch_size,
    )
