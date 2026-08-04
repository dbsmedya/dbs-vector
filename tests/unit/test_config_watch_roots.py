"""Rule 12: one watcher per directory per process, refused at load."""

import pytest

from dbs_vector.config import load_settings

_BASE = """
system:
  db_path: "{db}"
  memory_budget_gb: 8.0  # hermetic: keep validate=True off the MLX auto-detect
profiles:
  p: {{max_token_length: 2048, chunk_max_chars: 1000, batch_size: 8,
       chunk_target_tokens: 256, chunk_max_tokens: 512}}
engines:
  one-md:
    description: "t"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "t_one"
    workflow: "w"
    tuning_profile: "p"
    paths: ["{root_one}"]
    watch: {{enabled: true}}
  two-md:
    description: "t"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "t_two"
    workflow: "w"
    tuning_profile: "p"
    paths: ["{root_two}"]
    watch: {{enabled: true}}
"""


def _cfg(tmp_path, root_one, root_two):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_BASE.format(db=tmp_path / "db", root_one=root_one, root_two=root_two))
    return str(cfg)


def test_same_root_refused(tmp_path):
    shared = tmp_path / "docs"
    shared.mkdir()
    with pytest.raises(ValueError, match="one-md.*two-md|two-md.*one-md"):
        load_settings(_cfg(tmp_path, shared, shared), validate=True)


def test_nested_roots_allowed(tmp_path):
    outer = tmp_path / "docs"
    inner = outer / "sub"
    inner.mkdir(parents=True)
    load_settings(_cfg(tmp_path, outer, inner), validate=True)  # must not raise


def test_unwatched_shared_root_allowed(tmp_path):
    # Only watch-enabled engines participate in the exclusivity rule. Flip
    # two-md's watch off in the YAML by string replace.
    shared = tmp_path / "docs"
    shared.mkdir()
    cfg_path = _cfg(tmp_path, shared, shared)
    text = open(cfg_path).read()
    open(cfg_path, "w").write(text.replace("watch: {enabled: true}", "watch: {enabled: false}", 1))
    load_settings(cfg_path, validate=True)  # must not raise
