"""build_http_plan: pure planning for mcp --http."""

import pytest

from dbs_vector.config import EngineConfig, ServerConfig, Settings, TuningProfile
from dbs_vector.mcp.http_config import build_http_plan

TOKEN_A = "a" * 32
TOKEN_B = "b" * 32


def _engine(table: str, token: str | None) -> EngineConfig:
    return EngineConfig(
        description="t",
        model="gemma-bf16",
        mapper_type="document",
        chunker_type="document",
        table_name=table,
        workflow="w",
        tuning_profile="p",
        token=token,
    )


def _settings(**kwargs) -> Settings:
    s = Settings(_env_file=None)
    s.profiles = {
        "p": TuningProfile(
            max_token_length=2048,
            chunk_max_chars=1000,
            batch_size=8,
            chunk_target_tokens=256,
            chunk_max_tokens=512,
        )
    }
    s.engines = kwargs.pop("engines")
    s.server = kwargs.pop("server", ServerConfig())
    return s


def test_groups_by_token_and_excludes_untokened() -> None:
    s = _settings(
        engines={
            "a-md": _engine("ta", TOKEN_A),
            "b-md": _engine("tb", TOKEN_A),  # same token = same scope
            "c-md": _engine("tc", TOKEN_B),
            "hidden-md": _engine("th", None),  # fail-closed: never over HTTP
        }
    )
    plan = build_http_plan(s)
    scopes = {scope.engines: scope.token for scope in plan.scopes}
    assert scopes == {("a-md", "b-md"): TOKEN_A, ("c-md",): TOKEN_B}


def test_env_expansion(monkeypatch) -> None:
    monkeypatch.setenv("DBS_T", "e" * 40)
    s = _settings(engines={"a-md": _engine("ta", "${DBS_T}")})
    assert build_http_plan(s).scopes[0].token == "e" * 40


def test_unset_env_var_is_an_error(monkeypatch) -> None:
    monkeypatch.delenv("DBS_MISSING", raising=False)
    s = _settings(engines={"a-md": _engine("ta", "${DBS_MISSING}")})
    with pytest.raises(ValueError, match="DBS_MISSING"):
        build_http_plan(s)


def test_short_token_is_an_error() -> None:
    s = _settings(engines={"a-md": _engine("ta", "short")})
    with pytest.raises(ValueError, match="32"):
        build_http_plan(s)


def test_non_ascii_token_is_an_error() -> None:
    # A non-ASCII token cannot travel in an Authorization header reliably —
    # refuse at startup instead of serving eternal 401s.
    s = _settings(engines={"a-md": _engine("ta", "é" * 32)})
    with pytest.raises(ValueError, match="ASCII"):
        build_http_plan(s)


def test_zero_tokened_engines_is_an_error() -> None:
    s = _settings(engines={"a-md": _engine("ta", None)})
    with pytest.raises(ValueError, match="[Nn]o engine has a token"):
        build_http_plan(s)


def test_nonloopback_without_tls_refused() -> None:
    s = _settings(
        engines={"a-md": _engine("ta", TOKEN_A)},
        server=ServerConfig(bind="192.168.1.10"),
    )
    with pytest.raises(ValueError, match="loopback"):
        build_http_plan(s)


def test_nonloopback_with_tls_ok(tmp_path) -> None:
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    s = _settings(
        engines={"a-md": _engine("ta", TOKEN_A)},
        server=ServerConfig(bind="0.0.0.0", tls_cert=str(cert), tls_key=str(key)),
    )
    assert build_http_plan(s).tls == (str(cert), str(key))


def test_unreadable_tls_file_refused(tmp_path) -> None:
    import os

    if os.geteuid() == 0:
        pytest.skip("root reads anything; permission test is meaningless")
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    key.chmod(0o000)
    s = _settings(
        engines={"a-md": _engine("ta", TOKEN_A)},
        server=ServerConfig(bind="127.0.0.1", tls_cert=str(cert), tls_key=str(key)),
    )
    with pytest.raises(ValueError, match="not readable"):
        build_http_plan(s)


def test_half_tls_pair_refused(tmp_path) -> None:
    cert = tmp_path / "c.pem"
    cert.write_text("x")
    s = _settings(
        engines={"a-md": _engine("ta", TOKEN_A)},
        server=ServerConfig(bind="127.0.0.1", tls_cert=str(cert)),
    )
    with pytest.raises(ValueError, match="together"):
        build_http_plan(s)
