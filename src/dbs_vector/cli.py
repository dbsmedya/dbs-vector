import os
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from dbs_vector.config import settings
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.logger import configure_logger
from dbs_vector.services.bootstrap import (
    EngineDeps,
    build_dependencies,
    build_search_service,
    build_store,
)
from dbs_vector.services.browse import BrowseService, result_to_json, result_to_table
from dbs_vector.services.ingestion import IngestionService
from dbs_vector.services.path_filter import anchor_for
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

    # `init` GENERATES the config, so it must not require a loadable one.
    # An absent config.yaml is already survivable (no engines -> the validator
    # returns early), but a malformed one raises here - which is exactly when
    # a user reaches for init.
    if ctx.invoked_subcommand == "init":
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
    roots_override: list[str] | None = None,
) -> EngineDeps:
    """CLI-facing wrapper: converts schema-mismatch errors to typer exits."""
    try:
        return build_dependencies(
            engine_name,
            query_override=query_override,
            url_override=url_override,
            roots_override=roots_override,
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


def _build_search_service(engine_name: str) -> SearchService:
    """CLI-facing search-service builder: converts schema-mismatch to a typer exit."""
    try:
        return build_search_service(engine_name)
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

    # Resolve the ingestion target. Three shapes:
    #   - api engines: fall back to api_base_url when no path is given
    #   - document engines: fall back to the engine's configured `paths:` roots
    #     (one run over all of them)
    #   - everything else: a path is required
    target: str | list[str]
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
            target = path
        elif engine.chunker_type == "document":
            if not engine.paths:
                typer.echo(
                    f"Error: engine '{engine_name}' has no `paths:` configured, so "
                    f"there is nothing to ingest. Add absolute directory roots under "
                    f"engines.{engine_name}.paths in config.yaml, or pass a path "
                    f"explicitly.",
                    err=True,
                )
                raise typer.Exit(code=1)
            target = list(engine.paths)
        else:
            typer.echo(
                f"Error: 'path' argument is required for engine type '{engine_name}' "
                f"(chunker_type='{engine.chunker_type}'). Only api-chunker engines may "
                f"omit it (the URL falls back to api_base_url from config.yaml), and "
                f"document engines with `paths:` configured.",
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        target = path

    if rebuild and not force:
        typer.confirm(
            f"Are you sure you want to completely rebuild the '{engine_name}' vector store? This will erase all existing data.",
            abort=True,
        )

    # An explicit path is its own filtering anchor (directory -> itself; file ->
    # parent; glob -> longest non-glob prefix; URL -> none). Filtering RULES
    # still apply: extension gate, ignore_patterns, gitignore, canonical sources.
    roots_override: list[str] | None = None
    if engine.chunker_type == "document" and path is not None:
        anchor = anchor_for(path)
        roots_override = [anchor] if anchor else []
        if engine.watch.enabled and anchor is not None:
            anchor_path = Path(anchor)
            inside = any(
                anchor_path == Path(root) or Path(root) in anchor_path.parents
                for root in engine.paths
            )
            if not inside:
                typer.echo(
                    f"Note: '{path}' is outside the configured paths for engine "
                    f"'{engine_name}'. These rows will not be watched or reconciled "
                    f"and stay until the next --rebuild."
                )

    url_override = path if path is not None and path.startswith(("http://", "https://")) else None
    deps = _build_dependencies(
        engine_name,
        query_override=query,
        url_override=url_override,
        roots_override=roots_override,
    )
    service = IngestionService(
        deps.chunker,
        deps.embedder,
        deps.store,
        deps.workflow,
        batch_size=deps.batch_size,
        path_filter=deps.path_filter,
    )
    service.ingest_directory(target, rebuild=rebuild)


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
        typer.Option(
            "--source",
            "-s",
            help=(
                "Restrict to part of the corpus: a full stored path, a trailing "
                "fragment (specs/api.md, api.md), or a directory to scope to "
                "(specs). SQL engines take a database name. Not a glob."
            ),
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Maximum number of search results to return.")
    ] = 5,
    # SQL specific filters
    min_time: Annotated[
        float | None, typer.Option("--min-time", help="(SQL Only) Minimum execution time in ms.")
    ] = None,
    min_similarity: Annotated[
        float | None,
        typer.Option(
            "--min-similarity",
            help="Admission floor: only return results with cosine similarity >= this "
            "value (or all query terms verbatim). Overrides the engine's configured floor.",
        ),
    ] = None,
    no_similarity_floor: Annotated[
        bool,
        typer.Option(
            "--no-similarity-floor",
            help="Disable admission filtering entirely (exact unfloored baseline: "
            "no floor AND the original candidate-pool size).",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the full envelope (floor, inspected, best_rejected, results with "
            "similarity/retrieved_by/rrf_score) as JSON to stdout.",
        ),
    ] = False,
) -> None:
    """Searches the vector store using hybrid retrieval (Vector + Full-Text)."""
    if engine_name not in settings.engines:
        typer.echo(
            f"Error: Unknown engine type '{engine_name}'. Available: {list(settings.engines.keys())}"
        )
        raise typer.Exit(code=1)

    service = _build_search_service(engine_name)

    extra_filters = {}
    if min_time is not None and settings.engines[engine_name].resolved_family == "sql":
        extra_filters["min_time"] = min_time

    try:
        response = service.execute_query(
            query,
            source_filter=filter_source,
            limit=limit,
            extra_filters=extra_filters,
            min_similarity=min_similarity,
            disable_similarity_floor=no_similarity_floor,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(service.results_to_json(response))
    else:
        service.print_results(response, query)


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
            "browse and search MCP tools (search honours it via include_raw). "
            "Default off - enable only for a trusted local model.",
        ),
    ] = False,
    http_mode: Annotated[
        bool,
        typer.Option(
            "--http",
            help="Serve MCP over streamable HTTP using the config's server: "
            "block and per-engine tokens, instead of stdio. See "
            "docs/README_MCP.md.",
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

    if http_mode and allow_raw_queries:
        typer.echo(
            "Error: --allow-raw-queries is the stdio knob. Over --http, raw-query "
            "egress is declared per client: send the header "
            "X-DBS-Allow-Raw-Queries: true from the client's .mcp.json. "
            "See docs/README_MCP.md.",
            err=True,
        )
        raise typer.Exit(code=2)

    logger.info("Initializing MLX Embedders and LanceDB connections")
    try:
        if http_mode:
            from dbs_vector.mcp.server import start_http_server

            start_http_server()
        else:
            start_stdio_server(allow_raw_queries=allow_raw_queries)
    except Exception as e:
        logger.error("Failed to initialize search services: {}", e)
        raise


class TyperPromptIO:
    """PromptIO backed by typer. The only implementation that touches a TTY."""

    def echo(self, message: str) -> None:
        typer.echo(message)

    def ask_text(self, prompt: str, default: str = "") -> str:
        return str(typer.prompt(prompt, default=default, show_default=True))

    def ask_choice(self, prompt: str, options: list[tuple[str, str]], default: str) -> str:
        typer.echo(f"\n{prompt}:")
        width = max((len(key) for key, _ in options), default=0)
        for key, label in options:
            typer.echo(f"  {key:<{width}}    {label}")
        valid = [key for key, _ in options]
        while True:
            choice = str(typer.prompt("  choice", default=default, show_default=True)).strip()
            if choice in valid:
                return choice
            typer.echo(f"  Please choose one of: {', '.join(valid)}")

    def ask_multi(self, prompt: str, options: list[str], default: list[str]) -> list[str]:
        typer.echo(f"\n{prompt} (comma-separated; available: {', '.join(options)}):")
        while True:
            raw = str(typer.prompt("  values", default=",".join(default), show_default=True))
            chosen = [v.strip() for v in raw.split(",") if v.strip()]
            unknown = [v for v in chosen if v not in options]
            if not unknown:
                return chosen
            typer.echo(f"  Unknown: {', '.join(unknown)}. Available: {', '.join(options)}")

    def ask_bool(self, prompt: str, default: bool) -> bool:
        return bool(typer.confirm(prompt, default=default))


@app.command()
def init() -> None:
    """Interactively generate a config.yaml and a Claude-format .mcp.json."""
    from pathlib import Path as _Path

    from dbs_vector.services.initializer import run_init

    io = TyperPromptIO()
    io.echo("dbs-vector init - generating a single-engine configuration.\n")
    try:
        result = run_init(io, cwd=_Path.cwd())
    except ValueError as e:
        typer.echo(f"\n[!] {e}", err=True)
        raise typer.Exit(code=1) from e

    io.echo("")
    for note in result.notes:
        io.echo(f"  note: {note}")
    io.echo(f"\nWrote {result.config_path}")
    if result.config_backup:
        io.echo(f"  previous version saved to {result.config_backup}")
    io.echo(f"Wrote {result.mcp_path}")
    if result.mcp_backup:
        io.echo(f"  previous version saved to {result.mcp_backup}")
    # `ingest` has NO --all flag, and its engine default is `md` via --type.
    # An arbitrary generated engine name must be passed explicitly, or the
    # command fails with "Unknown engine type". The path argument is omitted
    # deliberately: a document engine with no path ingests its configured
    # `paths:` roots.
    # `uv run` requires uv, which a `pip install`/installed user may not have.
    prefix = "uv run dbs-vector" if result.used_checkout else "dbs-vector"
    io.echo(
        f"\nNext: {prefix} --config-file {result.config_path} "
        f"ingest --type {result.engine_name}\n"
        f"Then restart your MCP client to pick up {result.mcp_path}."
    )


if __name__ == "__main__":
    app()
