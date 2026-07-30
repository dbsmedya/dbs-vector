"""FastMCP server instance + stdio entry point.

Tool registration is dynamic: per-engine search tools come from
register_search_tools(); browse tools come from register_browse_tools();
the list_engines tool comes from register_discovery_tool(). All three run
inside start_stdio_server() before the mcp.run() loop begins.
"""

from mcp.server.fastmcp import FastMCP

from dbs_vector.mcp.discovery import register_discovery_tool
from dbs_vector.mcp.dynamic_tools import (
    register_browse_tools,
    register_search_tools,
    register_triage_tools,
)
from dbs_vector.mcp.state import initialize_services
from dbs_vector.services.bootstrap import build_watcher_service

mcp = FastMCP("dbs-vector")


def start_stdio_server(allow_raw_queries: bool = False) -> None:
    """Initialize services, register all tools, start watchers, and run stdio MCP.

    Watchers start AFTER initialize_services() so the embedding models are
    already in `_MODEL_CACHE` — a watched engine's stack reuses the resident
    model instead of loading a second copy. Startup reconciliation runs on the
    watcher's worker thread, so MCP answers requests immediately. Teardown sits
    in a `finally` so a crashing tool loop still stops the observer threads.

    `allow_raw_queries` is the sole explicit argument — the server-level
    raw-egress gate from the CLI's --allow-raw-queries flag. Everything else
    comes from dbs_vector.config.settings, already populated by the CLI
    callback's _populate_singleton_from(...) call before this runs.
    """
    initialize_services()
    register_search_tools(mcp, allow_raw_queries=allow_raw_queries)
    register_browse_tools(mcp, allow_raw_queries=allow_raw_queries)
    register_triage_tools(mcp, allow_raw_queries=allow_raw_queries)
    register_discovery_tool(mcp)

    watcher = build_watcher_service()
    try:
        # start() is INSIDE the try: if it fails partway through, the finally
        # still tears down whichever backends did start. stop() is safe on a
        # never-started watcher.
        if watcher is not None:
            watcher.start()
        mcp.run()
    finally:
        if watcher is not None:
            watcher.stop()
