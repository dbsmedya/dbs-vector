import os
from typing import Annotated

import typer
from loguru import logger

from dbs_vector.config import settings
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.logger import configure_logger
from dbs_vector.services.bootstrap import EngineDeps, build_dependencies, build_store
from dbs_vector.services.browse import BrowseService, result_to_json, result_to_table
from dbs_vector.services.ingestion import IngestionService
from dbs_vector.services.search import SearchService

app = typer.Typer(
    help="dbs-vector: Local Arrow-Native Codebase Search Engine",
    no_args_is_help=True,
    rich_markup_mode=None,
)


def version_callback(value: bool) -> None:
    if value:
        from dbs_vector import __version__

        typer.echo(f"dbs-vector version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
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
    # Skip config loading when just showing help or version (no subcommand invoked)
    if ctx.invoked_subcommand is None:
        return

    import os

    from dbs_vector.config import _populate_singleton_from, load_settings, settings

    # Export to environment so any spawned subprocesses (e.g., MCP stdio
    # transport invoked via uv run) inherit the same config.
    os.environ["DBS_CONFIG_FILE"] = config_file

    # Load AND validate the config; copy fields onto the singleton.
    new_settings = load_settings(config_file, validate=True)
    _populate_singleton_from(new_settings)

    # Configure logger based on settings
    configure_logger(level=settings.log_level, serialize=settings.log_serialize)


def _build_dependencies(
    engine_name: str,
    query_override: str | None = None,
    url_override: str | None = None,
) -> EngineDeps:
    """CLI-facing wrapper: converts schema-mismatch errors to typer exits."""
    try:
        return build_dependencies(
            engine_name,
            query_override=query_override,
            url_override=url_override,
        )
    except ValueError as e:
        if "Schema mismatch" in str(e):
            typer.echo(f"\n[!] Database Error: {e}", err=True)
            raise typer.Exit(code=1) from e
        raise


def _build_store(engine_name: str) -> LanceDBStore:
    """CLI-facing store-only builder: converts schema-mismatch to a typer exit."""
    try:
        return build_store(engine_name)
    except ValueError as e:
        if "Schema mismatch" in str(e):
            typer.echo(f"\n[!] Database Error: {e}", err=True)
            raise typer.Exit(code=1) from e
        raise


@app.command()
def ingest(
    path: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Directory path, glob pattern, JSON file, or URL to ingest. "
                "Optional for api-chunker engines: omit to use api_base_url from config.yaml."
            ),
        ),
    ] = None,
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
    query: Annotated[
        str | None,
        typer.Option("--query", "-q", help="Custom SQL query for DuckDB extraction."),
    ] = None,
) -> None:
    """Ingests documents or SQL query logs into the Arrow-native vector store."""
    if engine_name not in settings.engines:
        typer.echo(
            f"Error: Unknown engine type '{engine_name}'. Available: {list(settings.engines.keys())}"
        )
        raise typer.Exit(code=1)

    engine = settings.engines[engine_name]

    # Resolve the path argument. For api-chunker engines, fall back to
    # api_base_url from config.yaml when no path was supplied.
    if path is None:
        if engine.chunker_type == "api":
            if not engine.api_base_url:
                typer.echo(
                    f"Error: engine '{engine_name}' has no api_base_url configured "
                    f"and no URL was passed on the command line.",
                    err=True,
                )
                raise typer.Exit(code=1)
            path = engine.api_base_url
        else:
            typer.echo(
                f"Error: 'path' argument is required for engine type '{engine_name}' "
                f"(chunker_type='{engine.chunker_type}'). Only api-chunker engines may "
                f"omit it (the URL falls back to api_base_url from config.yaml).",
                err=True,
            )
            raise typer.Exit(code=1)

    if rebuild and not force:
        typer.confirm(
            f"Are you sure you want to completely rebuild the '{engine_name}' vector store? This will erase all existing data.",
            abort=True,
        )

    url_override = path if path.startswith(("http://", "https://")) else None
    deps = _build_dependencies(engine_name, query_override=query, url_override=url_override)
    service = IngestionService(
        deps.chunker, deps.embedder, deps.store, deps.workflow, batch_size=deps.batch_size
    )
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
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit full results (score, source, full text, metadata) as JSON to stdout.",
        ),
    ] = False,
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
    if min_time is not None and settings.engines[engine_name].resolved_family == "sql":
        extra_filters["min_time"] = min_time

    results = service.execute_query(
        query, source_filter=filter_source, limit=limit, extra_filters=extra_filters
    )
    if json_output:
        typer.echo(service.results_to_json(results))
    else:
        service.print_results(results)


@app.command()
def browse(
    sql: Annotated[str, typer.Option("--sql", help="A read-only SELECT (polars SQL dialect).")],
    engine_name: Annotated[
        str, typer.Option("--type", "-t", help="SQL engine to browse (sql, sql-api, ...).")
    ] = "sql-api",
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit rows as JSON instead of a table.")
    ] = False,
) -> None:
    """Analytical SQL over a SQL engine's table (no embedder, no ranking).

    Frames: `t` (one row per fingerprint), `t_by_table` (exploded on `tables`),
    and the engine name with dashes->underscores. Quote "user" (SQL keyword).
    Unbounded - use LIMIT in your SQL; `SELECT * FROM t` is a full export.
    """
    if engine_name not in settings.engines:
        typer.echo(
            f"Error: Unknown engine type '{engine_name}'. Available: "
            f"{list(settings.engines.keys())}"
        )
        raise typer.Exit(code=1)
    if settings.engines[engine_name].resolved_family != "sql":
        sql_engines = [n for n, e in settings.engines.items() if e.resolved_family == "sql"]
        typer.echo(
            f"Error: browse is only available for SQL engines. "
            f"'{engine_name}' is not one. Available SQL engines: {sql_engines}"
        )
        raise typer.Exit(code=1)

    store = _build_store(engine_name)
    frame_alias = engine_name.replace("-", "_")
    service = BrowseService(store, frame_alias)
    try:
        result = service.run_sql(sql)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(result_to_json(result))
    else:
        typer.echo(result_to_table(result))


@app.command()
def mcp(
    config_file: Annotated[
        str | None,
        typer.Option(
            "--config-file",
            "-c",
            help="Override the global --config-file for this subcommand.",
        ),
    ] = None,
    allow_raw_queries: Annotated[
        bool,
        typer.Option(
            "--allow-raw-queries",
            help="Expose the verbatim raw_query column (literal PII values) to "
            "browse MCP tools. Default off - enable only for a trusted local model.",
        ),
    ] = False,
) -> None:
    """Starts the FastMCP standard input/output (stdio) server for integrations."""
    from dbs_vector.config import _populate_singleton_from, load_settings
    from dbs_vector.mcp.server import start_stdio_server

    # If the subcommand was given a config file (e.g., `dbs-vector mcp -c X`),
    # re-load and re-populate the singleton; otherwise rely on what the global
    # callback already loaded. Also re-export DBS_CONFIG_FILE so spawned
    # subprocesses see the same path.
    if config_file is not None:
        os.environ["DBS_CONFIG_FILE"] = config_file
        new_settings = load_settings(config_file, validate=True)
        _populate_singleton_from(new_settings)

    logger.info("Initializing MLX Embedders and LanceDB connections")
    try:
        start_stdio_server(allow_raw_queries=allow_raw_queries)
    except Exception as e:
        logger.error("Failed to initialize search services: {}", e)
        raise


if __name__ == "__main__":
    app()
