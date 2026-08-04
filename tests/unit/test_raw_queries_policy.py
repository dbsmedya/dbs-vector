"""Client-knob raw queries: header on HTTP, baked flag on stdio."""

from types import SimpleNamespace

from mcp.server.lowlevel.server import request_ctx

from dbs_vector.mcp.policy import RAW_QUERIES_HEADER, raw_queries_effective


def _with_request_headers(headers: dict[str, str]):
    fake_request = SimpleNamespace(headers=headers)
    fake_ctx = SimpleNamespace(request=fake_request)
    return request_ctx.set(fake_ctx)


def test_no_request_context_falls_back_to_baked() -> None:
    # stdio: request_ctx not set (or carries no HTTP request).
    assert raw_queries_effective(True) is True
    assert raw_queries_effective(False) is False


def test_stdio_style_context_without_request_uses_baked() -> None:
    token = request_ctx.set(SimpleNamespace(request=None))
    try:
        assert raw_queries_effective(True) is True
    finally:
        request_ctx.reset(token)


def test_http_header_true_enables() -> None:
    token = _with_request_headers({RAW_QUERIES_HEADER: "true"})
    try:
        assert raw_queries_effective(False) is True
    finally:
        request_ctx.reset(token)


def test_http_header_absent_or_false_disables() -> None:
    for headers in ({}, {RAW_QUERIES_HEADER: "false"}, {RAW_QUERIES_HEADER: "1"}):
        token = _with_request_headers(headers)
        try:
            # Fail-closed even against a baked True: on HTTP the header decides.
            assert raw_queries_effective(True) is False
        finally:
            request_ctx.reset(token)
