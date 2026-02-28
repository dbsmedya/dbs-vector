import os
from typing import Annotated, Any, NamedTuple

import typer

from dbs_vector.config import settings
from dbs_vector.core.registry import ComponentRegistry
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.services.ingestion import IngestionService
from dbs_vector.services.search import SearchService

app = typer.Typer(
    help="dbs-vector: Local Arrow-Native Codebase Search Engine",
    no_args_is_help=True,
)


class EngineDeps(NamedTuple):
    """Container for resolved engine dependencies."""

    embedder: Any
    store: Any
    chunker: Any
    workflow: str


def version_callback(value: bool) -> None:
    if value:
        from dbs_vector import __version__

        typer.echo(f"dbs-vector version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    config_file: Annotated[
        str, typer.Option("--config-file", "-c", help="Path to config.yaml file.")
    ] = "config.yaml",
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show the version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """dbs-vector: Configurable Arrow-Native Search Engine."""
    import os

    from dbs_vector.config import load_settings, settings

    # Export to environment so uvicorn subprocesses (in API mode) inherit it
    os.environ["DBS_CONFIG_FILE"] = config_file

    # Dynamically update the current process global settings singleton
    new_settings = load_settings(config_file)
    settings.db_path = new_settings.db_path
    settings.batch_size = new_settings.batch_size
    settings.nprobes = new_settings.nprobes
    settings.engines = new_settings.engines


def _build_dependencies(engine_name: str) -> EngineDeps:
    """Dependency Injection Factory driven by config.yaml configuration."""
    if engine_name not in settings.engines:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. Check {os.environ.get('DBS_CONFIG_FILE', 'config.yaml')}."
        )

    config = settings.engines[engine_name]

    # Initialize Embedder
    embedder = MLXEmbedder(
        model_name=config.model_name,
        max_token_length=config.max_token_length,
        dimension=config.vector_dimension,
        passage_prefix=config.passage_prefix,
        query_prefix=config.query_prefix,
    )

    # Resolve components via Registry
    MapperClass = ComponentRegistry.get_mapper(config.mapper_type)
    ChunkerClass = ComponentRegistry.get_chunker(config.chunker_type)

    mapper = MapperClass(vector_dimension=config.vector_dimension)

    # Optional arguments based on chunker type
    chunker_kwargs = {}
    if config.chunk_max_chars > 0:
        chunker_kwargs["max_chars"] = config.chunk_max_chars

    chunker = ChunkerClass(**chunker_kwargs)

    try:
        store = LanceDBStore(
            db_path=settings.db_path,
            table_name=config.table_name,
            vector_dimension=config.vector_dimension,
            mapper=mapper,
            nprobes=settings.nprobes,
        )
    except ValueError as e:
        if "Schema mismatch" in str(e):
            typer.echo(f"\n[!] Database Error: {e}", err=True)
            raise typer.Exit(code=1) from e
        raise

    return EngineDeps(embedder=embedder, store=store, chunker=chunker, workflow=config.workflow)


@app.command()
def ingest(
    path: Annotated[
        str, typer.Argument(help="Directory path, glob pattern, or JSON file to ingest.")
    ],
    engine_name: Annotated[
        str, typer.Option("--type", "-t", help="The type of data to ingest (md, sql, etc).")
    ] = "md",
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild", "-r", help="Drop the existing vector store and recreate it from scratch."
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Bypass confirmation prompt when rebuilding."),
    ] = False,
) -> None:
    """Ingests documents or SQL query logs into the Arrow-native vector store."""
    if engine_name not in settings.engines:
        typer.echo(
            f"Error: Unknown engine type '{engine_name}'. Available: {list(settings.engines.keys())}"
        )
        raise typer.Exit(code=1)

    if rebuild and not force:
        typer.confirm(
            f"Are you sure you want to completely rebuild the '{engine_name}' vector store? This will erase all existing data.",
            abort=True,
        )

    deps = _build_dependencies(engine_name)
    service = IngestionService(deps.chunker, deps.embedder, deps.store, deps.workflow)
    service.ingest_directory(path, rebuild=rebuild)


@app.command()
def search(
    query: Annotated[
        str, typer.Argument(help="The text or SQL to search for within the indexed data.")
    ],
    engine_name: Annotated[
        str, typer.Option("--type", "-t", help="The type of data to search (md, sql, etc).")
    ] = "md",
    filter_source: Annotated[
        str | None,
        typer.Option("--source", "-s", help="Filter results to a specific file or database."),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Maximum number of search results to return.")
    ] = 5,
    # SQL specific filters
    min_time: Annotated[
        float | None, typer.Option("--min-time", help="(SQL Only) Minimum execution time in ms.")
    ] = None,
) -> None:
    """Searches the vector store using hybrid retrieval (Vector + Full-Text)."""
    if engine_name not in settings.engines:
        typer.echo(
            f"Error: Unknown engine type '{engine_name}'. Available: {list(settings.engines.keys())}"
        )
        raise typer.Exit(code=1)

    deps = _build_dependencies(engine_name)
    service = SearchService(deps.embedder, deps.store)

    extra_filters = {}
    if min_time is not None and engine_name == "sql":
        extra_filters["min_time"] = min_time

    results = service.execute_query(
        query, source_filter=filter_source, limit=limit, extra_filters=extra_filters
    )
    service.print_results(results)


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", "-h", help="Host to bind the API server to.")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to bind the API server to.")
    ] = 8000,
    reload: Annotated[
        bool, typer.Option("--reload", help="Enable auto-reload for development.")
    ] = False,
) -> None:
    """Starts the asynchronous FastAPI search server."""
    import uvicorn

    print(f"Starting dbs-vector API server at http://{host}:{port}...")
    uvicorn.run("dbs_vector.api.main:app", host=host, port=port, reload=reload)


@app.command()
def mcp(
    config_file: Annotated[
        str, typer.Option("--config-file", "-c", help="Path to config.yaml file.")
    ] = "config.yaml",
) -> None:
    """Starts the FastMCP standard input/output (stdio) server for integrations."""
    import os
    import sys

    from dbs_vector.api.mcp_server import mcp as mcp_server
    from dbs_vector.api.state import _services

    # Export to environment so the MCP subprocess inherits it
    os.environ["DBS_CONFIG_FILE"] = config_file

    print("[MCP Startup] Initializing MLX Embedders and LanceDB connections...", file=sys.stderr)
    try:
        for engine_name in settings.engines.keys():
            print(f"  -> Loading Engine ({engine_name})...", file=sys.stderr)
            deps = _build_dependencies(engine_name)
            _services[engine_name] = SearchService(deps.embedder, deps.store)
    except Exception as e:
        print(f"[MCP Startup] Failed to initialize search services: {e}", file=sys.stderr)
        raise

    mcp_server.run()


if __name__ == "__main__":
    app()
