"""FastMCP server instance + stdio/HTTP entry points.

Tool registration is dynamic: per-engine search and adjacent-read tools come
from register_search_tools()/register_read_tools(); browse tools come from
register_browse_tools(); the list_engines tool comes from
register_discovery_tool(). All registrars run inside start_stdio_server()
before the mcp.run() loop begins.

start_http_server() reuses the SAME registrars, once per token scope (Task
4's HttpPlan), so a `dbs-vector mcp --http` sub-app never differs in
registration LOGIC from the stdio path — only in which engines it sees.
"""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from dbs_vector.mcp.discovery import register_discovery_tool
from dbs_vector.mcp.dynamic_tools import (
    register_browse_tools,
    register_read_tools,
    register_search_tools,
    register_triage_tools,
)
from dbs_vector.mcp.state import initialize_services
from dbs_vector.services.bootstrap import build_watcher_service

if TYPE_CHECKING:
    from dbs_vector.mcp.http_config import HttpPlan

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
    register_read_tools(mcp)
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


def build_http_app(plan: "HttpPlan") -> Any:
    """Compose the HTTP ASGI app: one FastMCP sub-app per token scope.

    Does NOT initialize services — start_http_server owns that, mirroring
    the stdio split, and it keeps this function cheap for tests.
    """
    from collections.abc import AsyncIterator
    from contextlib import AsyncExitStack, asynccontextmanager

    from starlette.applications import Starlette
    from starlette.routing import Mount

    from dbs_vector.mcp.auth import TokenRouterMiddleware

    subs: list[tuple[str, str, FastMCP]] = []
    for i, scope in enumerate(plan.scopes):
        # host/port matter: FastMCP auto-enables Host/Origin validation
        # (DNS-rebinding defense) only when its host is loopback, and it must
        # match what we actually bind. json_response=True: plain JSON bodies
        # instead of SSE frames — simpler for every client we serve.
        sub = FastMCP(
            f"dbs-vector-{i}",
            host=plan.bind,
            port=plan.port,
            json_response=True,
        )
        names = set(scope.engines)
        # Baked allow_raw_queries stays False over HTTP: the per-request
        # header is the only way to raw egress (mcp/policy.py).
        register_search_tools(sub, allow_raw_queries=False, engine_names=names)
        register_read_tools(sub, engine_names=names)
        register_browse_tools(sub, allow_raw_queries=False, engine_names=names)
        register_triage_tools(sub, allow_raw_queries=False, engine_names=names)
        register_discovery_tool(sub, engine_names=names)
        subs.append((scope.token, f"/s{i}", sub))

    mounts = [Mount(prefix, app=sub.streamable_http_app()) for _, prefix, sub in subs]

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        # Mounted sub-apps' own lifespans never run under Starlette Mount;
        # every session manager must be entered here.
        async with AsyncExitStack() as stack:
            for _, _, sub in subs:
                await stack.enter_async_context(sub.session_manager.run())
            yield

    inner = Starlette(routes=mounts, lifespan=lifespan)
    return TokenRouterMiddleware(inner, [(token, prefix) for token, prefix, _ in subs])


def start_http_server() -> None:
    """Initialize services, start watchers, serve MCP over streamable HTTP.

    Same bootstrap order as start_stdio_server; uvicorn replaces mcp.run().
    TLS comes from the plan (non-loopback binds are refused without it).
    """
    import uvicorn

    from dbs_vector.config import settings
    from dbs_vector.mcp.http_config import build_http_plan

    plan = build_http_plan(settings)  # fail fast, before loading any model
    initialize_services()
    app = build_http_app(plan)

    watcher = build_watcher_service()
    try:
        if watcher is not None:
            watcher.start()
        tls_kwargs: dict[str, Any] = (
            {"ssl_certfile": plan.tls[0], "ssl_keyfile": plan.tls[1]} if plan.tls else {}
        )
        uvicorn.run(app, host=plan.bind, port=plan.port, **tls_kwargs)
    finally:
        if watcher is not None:
            watcher.stop()
