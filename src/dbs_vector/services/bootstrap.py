"""Dependency-injection factory for engines."""

import os
from typing import Any, NamedTuple

from dbs_vector.config import settings
from dbs_vector.core.model_registry import ModelRegistry
from dbs_vector.core.registry import ComponentRegistry
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder
from dbs_vector.infrastructure.hardware import configure_mlx_memory_limits
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.services.search import SearchService


class EngineDeps(NamedTuple):
    """Resolved per-engine runtime dependencies."""

    embedder: Any
    store: Any
    chunker: Any
    workflow: str
    batch_size: int
    # PathFilter for document engines; None elsewhere (those code paths are
    # deliberately untouched in v1). Last field with a default so existing
    # keyword construction keeps working.
    path_filter: Any = None


def build_dependencies(
    engine_name: str,
    query_override: str | None = None,
    url_override: str | None = None,
    roots_override: list[str] | None = None,
) -> EngineDeps:
    """Resolve the chunker / mapper / embedder / store / path-filter stack.

    `roots_override` replaces the engine's configured `paths:` for this build
    only — the CLI uses it to anchor an explicit-path run.
    """
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

    # MLX allocator limits are process-wide. Apply them before loading model
    # weights; the helper is idempotent across engines sharing this process.
    configure_mlx_memory_limits(
        memory_budget_gb=settings.memory_budget_gb,
        memory_limit_gb=settings.mlx_memory_limit_gb,
        cache_limit_gb=settings.mlx_cache_limit_gb,
    )

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

    if engine.chunker_type == "document":
        # deferred import: avoids a circular import (config.py uses the same pattern)
        from dbs_vector.infrastructure.chunking.filters import FilterRegistry
        from dbs_vector.services.path_filter import PathFilter

        # Token budgets are passed straight through: validation (config Rule for
        # document engines) guarantees both are > 0, so NO `or <default>` here —
        # a 0 would be a validation bug, not a silent fallback. `max_chars`
        # keeps its `or 1000` because it only feeds the rarely-used .txt path
        # and md profiles legitimately set chunk_max_chars: 0.
        chunker = ChunkerClass(
            max_chars=profile.chunk_max_chars or 1000,
            target_tokens=profile.chunk_target_tokens,
            max_tokens=profile.chunk_max_tokens,
            length_fn=embedder.count_tokens,
            filters=FilterRegistry.resolve(engine.exclusion_filters),
        )
        path_filter: Any = PathFilter(
            roots=roots_override if roots_override is not None else engine.paths,
            extensions=chunker.supported_extensions,
            ignore_patterns=engine.ignore_patterns,
            use_gitignore="gitignore" in engine.exclusion_filters,
        )
    else:
        chunker = ChunkerClass(
            **engine.chunker_kwargs(
                query_override=query_override,
                url_override=url_override,
            )
        )
        path_filter = None

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
        path_filter=path_filter,
    )


def build_search_service(engine_name: str, deps: EngineDeps | None = None) -> SearchService:
    """SearchService with the engine's similarity_floor injected.

    The single construction path for every search surface (MCP, CLI,
    dbs-web) — a hand-wired SearchService is exactly where a floor would
    drift. Pass prebuilt `deps` to reuse cached embedder/store handles.
    """
    if engine_name not in settings.engines:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Check {os.environ.get('DBS_CONFIG_FILE', 'config.yaml')}."
        )
    if deps is None:
        deps = build_dependencies(engine_name)
    return SearchService(
        deps.embedder,
        deps.store,
        similarity_floor=settings.engines[engine_name].similarity_floor,
    )


def build_watcher_service() -> Any:
    """Compose a WatcherService for every engine with `watch.enabled`.

    Returns None when nothing is watched. Deliberately creates a SECOND
    LanceDBStore handle per engine (the writer) alongside SearchService's
    (the readers) — that separation is what makes MVCC / checkout_latest()
    reads safe while the watcher writes. Keep it.
    """
    from dbs_vector.infrastructure.watch.watchdog_backend import WatchdogBackend
    from dbs_vector.services.ingestion import IngestionService
    from dbs_vector.services.watcher import WatchedEngine, WatcherService

    watched_names = [name for name, cfg in settings.engines.items() if cfg.watch.enabled]
    if not watched_names:
        return None

    engines: dict[str, Any] = {}
    for name in watched_names:
        deps = build_dependencies(name)
        engines[name] = WatchedEngine(
            name=name,
            ingestion=IngestionService(
                deps.chunker,
                deps.embedder,
                deps.store,
                deps.workflow,
                batch_size=deps.batch_size,
                path_filter=deps.path_filter,
            ),
            path_filter=deps.path_filter,
            store=deps.store,
            debounce_seconds=settings.engines[name].watch.debounce_seconds,
            backend=WatchdogBackend(owner=name),
        )
    return WatcherService(engines)


def build_store(engine_name: str) -> LanceDBStore:
    """Resolve ONLY the vector store for an engine — no embedder, no chunker.

    For read paths (browse) that never embed. Avoids MLX model load and the
    cost/failure modes of constructing a chunker. The mapper (hence the table
    schema) and vector_dimension come from the engine's model contract.
    """
    if engine_name not in settings.engines:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Check {os.environ.get('DBS_CONFIG_FILE', 'config.yaml')}."
        )
    engine = settings.engines[engine_name]
    contract = ModelRegistry.get(engine.model)
    MapperClass = ComponentRegistry.get_mapper(engine.mapper_type)
    mapper = MapperClass(vector_dimension=contract.vector_dimension)
    return LanceDBStore(
        db_path=settings.db_path,
        table_name=engine.table_name,
        vector_dimension=contract.vector_dimension,
        mapper=mapper,
        nprobes=settings.nprobes,
    )
