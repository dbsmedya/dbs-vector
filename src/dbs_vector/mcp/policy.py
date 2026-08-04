"""Per-request raw-query egress policy.

stdio: the baked --allow-raw-queries CLI flag decides (there is no HTTP
side to a stdio request). HTTP: the client's X-DBS-Allow-Raw-Queries
header decides, per request — the server does not police egress policy;
the token is the access boundary, the header is declared intent
(docs/README_MCP.md). Fail-closed: absent header, absent request context,
or anything unparseable → no raw queries.
"""

from typing import Any

RAW_QUERIES_HEADER = "x-dbs-allow-raw-queries"


def raw_queries_effective(baked: bool) -> bool:
    request = _current_http_request()
    if request is None:
        return baked
    value = request.headers.get(RAW_QUERIES_HEADER, "")
    return value.strip().lower() == "true"


def _current_http_request() -> Any:
    """The live HTTP request, or None on stdio / outside a request."""
    try:
        from mcp.server.lowlevel.server import request_ctx

        ctx = request_ctx.get()
    except LookupError:
        return None
    request = getattr(ctx, "request", None)
    return request if request is not None and hasattr(request, "headers") else None
