"""FastMCP server instance + stdio entry point.

Tool registration is dynamic: per-engine search tools come from
register_search_tools(); browse tools come from register_browse_tools();
the list_engines tool comes from register_discovery_tool(). All three run
inside start_stdio_server() before the mcp.run() loop begins.
"""

from mcp.server.fastmcp import FastMCP

from dbs_vector.mcp.discovery import register_discovery_tool
from dbs_vector.mcp.dynamic_tools import register_browse_tools, register_search_tools
from dbs_vector.mcp.state import initialize_services

mcp = FastMCP("dbs-vector")


def start_stdio_server(allow_raw_queries: bool = False) -> None:
    """Initialize services, register all tools, and run stdio MCP.

    Takes no arguments — dbs_vector.config.settings is already populated by
    the CLI callback's _populate_singleton_from(...) call before this runs.
    initialize_services(), register_search_tools(mcp), register_browse_tools(mcp),
    and register_discovery_tool(mcp) all read from the singleton too, so
    settings ownership is consistent across the lifecycle.
    """
    initialize_services()
    register_search_tools(mcp)
    register_browse_tools(mcp, allow_raw_queries=allow_raw_queries)
    register_discovery_tool(mcp)
    mcp.run()
