"""Pure planning for `dbs-vector mcp --http`.

Token resolution, scope grouping, and the bind/TLS gate — everything the
spec calls "server-mode validation". Pure: no I/O beyond os.environ, so it
is fully unit-testable and stdio never imports it.
"""

import ipaddress
import os
import re
from dataclasses import dataclass

from dbs_vector.config import Settings

# `${VAR}` full-value expansion only — an embedded reference is not a token.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}\Z")
MIN_TOKEN_CHARS = 32


@dataclass(frozen=True)
class TokenScope:
    token: str
    engines: tuple[str, ...]


@dataclass(frozen=True)
class HttpPlan:
    scopes: tuple[TokenScope, ...]
    bind: str
    port: int
    tls: tuple[str, str] | None  # (cert_path, key_path)


def _is_loopback(bind: str) -> bool:
    if bind == "localhost":
        return True
    try:
        return ipaddress.ip_address(bind).is_loopback
    except ValueError:
        return False


def resolve_token(raw: str, engine_name: str) -> str:
    """Resolve `${VAR}` and enforce minimum entropy. HTTP-startup only."""
    match = _ENV_REF.match(raw)
    if match:
        value = os.environ.get(match.group(1))
        if value is None or not value.strip():
            raise ValueError(
                f"Engine '{engine_name}': token references ${{{match.group(1)}}} "
                f"but the environment variable is unset or blank."
            )
        raw = value
    token = raw.strip()
    if not token.isascii():
        raise ValueError(
            f"Engine '{engine_name}': resolved token contains non-ASCII "
            f"characters, which HTTP Authorization headers cannot carry "
            f"reliably. Generate one with e.g. `openssl rand -hex 32`."
        )
    if len(token) < MIN_TOKEN_CHARS:
        raise ValueError(
            f"Engine '{engine_name}': resolved token is shorter than "
            f"{MIN_TOKEN_CHARS} characters. Generate one with e.g. "
            f"`openssl rand -hex 32`."
        )
    return token


def build_http_plan(settings: Settings) -> HttpPlan:
    server = settings.server

    tls: tuple[str, str] | None = None
    if server.tls_cert or server.tls_key:
        if not (server.tls_cert and server.tls_key):
            raise ValueError("server.tls_cert and server.tls_key must be set together.")
        for label, path in (("tls_cert", server.tls_cert), ("tls_key", server.tls_key)):
            # Actually read a byte: existence is not readability, and uvicorn's
            # SSL error at bind time is far less actionable than this one.
            try:
                with open(path, "rb") as fh:
                    fh.read(1)
            except OSError as e:
                raise ValueError(f"server.{label} '{path}' is not readable: {e}") from e
        tls = (server.tls_cert, server.tls_key)

    if not _is_loopback(server.bind) and tls is None:
        raise ValueError(
            f"server.bind '{server.bind}' is not loopback: refusing to serve "
            f"plain HTTP off-host. Set server.tls_cert and server.tls_key "
            f"(self-signed is fine), or bind 127.0.0.1."
        )

    groups: dict[str, list[str]] = {}
    for name, engine in settings.engines.items():
        if engine.token is None:
            continue  # fail-closed: an untokened engine is never served over HTTP
        groups.setdefault(resolve_token(engine.token, name), []).append(name)
    if not groups:
        raise ValueError(
            "No engine has a token: nothing would be served over --http. "
            "Add `token:` to at least one engine (stdio needs no tokens)."
        )

    scopes = tuple(
        sorted(
            (TokenScope(token=t, engines=tuple(sorted(names))) for t, names in groups.items()),
            key=lambda s: s.engines,
        )
    )
    return HttpPlan(scopes=scopes, bind=server.bind, port=server.port, tls=tls)
