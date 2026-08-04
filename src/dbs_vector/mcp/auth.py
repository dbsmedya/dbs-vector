"""Bearer-token routing for `dbs-vector mcp --http`.

One ASGI layer in front of the per-scope sub-apps: constant-time token
compare, bare 401 (no engine names — a failed auth must not learn what it
is locked out of), and rejection of browser-Origin requests (DNS-rebinding
defense; no MCP client we serve sends an Origin header). Scoping itself is
structural — each token routes to a sub-app that only ever contained its
own engines — so there is no response filtering here to get wrong.
"""

import secrets

from loguru import logger
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class TokenRouterMiddleware:
    def __init__(self, app: ASGIApp, routes: list[tuple[str, str]]) -> None:
        """routes: (resolved_token, internal_mount_prefix) pairs."""
        self._app = app
        self._routes = routes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if headers.get("origin") is not None:
            await PlainTextResponse("forbidden", status_code=403)(scope, receive, send)
            return
        if scope["path"] != "/mcp":
            await PlainTextResponse("not found", status_code=404)(scope, receive, send)
            return
        auth = headers.get("authorization", "")
        presented = auth[7:] if auth.startswith("Bearer ") else ""
        matched: str | None = None
        for token, prefix in self._routes:
            # Compare EVERY route (no early break): uniform timing.
            if secrets.compare_digest(token, presented):
                matched = prefix
        if matched is None:
            logger.info("HTTP MCP auth failure from {}", scope.get("client"))
            await PlainTextResponse("unauthorized", status_code=401)(scope, receive, send)
            return
        routed = dict(scope)
        routed["path"] = f"{matched}/mcp"
        routed["raw_path"] = routed["path"].encode()
        await self._app(routed, receive, send)
