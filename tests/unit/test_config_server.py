"""server:/token: parse from YAML; both are inert data until --http uses them."""

from pathlib import Path

from dbs_vector.config import load_settings

_YAML = """
system:
  db_path: "{db}"
  memory_budget_gb: 8.0  # hermetic: keep validate=True off the MLX auto-detect
server:
  bind: "127.0.0.1"
  port: 9999
profiles:
  p: {{max_token_length: 2048, chunk_max_chars: 1000, batch_size: 8,
       chunk_target_tokens: 256, chunk_max_tokens: 512}}
engines:
  alpha-md:
    description: "t"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "t_alpha"
    workflow: "w"
    tuning_profile: "p"
    token: "${{DBS_TEST_TOKEN}}"
"""


def _write(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_YAML.format(db=tmp_path / "db"))
    return cfg


def test_server_block_and_token_parse(tmp_path: Path) -> None:
    s = load_settings(str(_write(tmp_path)))
    assert s.server.bind == "127.0.0.1"
    assert s.server.port == 9999
    assert s.server.tls_cert is None
    assert s.engines["alpha-md"].token == "${DBS_TEST_TOKEN}"  # raw, unresolved


def test_server_block_defaults_when_absent(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("system:\n  db_path: '%s'\n" % (tmp_path / "db"))
    s = load_settings(str(cfg))
    assert s.server.bind == "127.0.0.1"
    assert s.server.port == 8765


def test_token_is_inert_for_stdio_validation(tmp_path: Path) -> None:
    # validate=True must NOT resolve tokens: an unset env var is fine on stdio.
    load_settings(str(_write(tmp_path)), validate=True)
