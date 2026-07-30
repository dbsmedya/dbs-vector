# tests/unit/test_watch_config.py
import textwrap

import pytest
from pydantic import ValidationError

from dbs_vector.config import EngineConfig, load_settings

_BASE_ENGINE = {
    "description": "test engine",
    "model": "gemma-bf16",
    "mapper_type": "document",
    "chunker_type": "document",
    "table_name": "t",
    "workflow": "w",
    "tuning_profile": "p",
}


def _write(tmp_path, engines_yaml: str) -> str:
    # Build the file without wrapping the full document in dedent after
    # interpolating engines — that would leave top-level keys over-indented.
    p = tmp_path / "config.yaml"
    engines_block = textwrap.indent(textwrap.dedent(engines_yaml).strip("\n") + "\n", "  ")
    p.write_text(
        "system:\n"
        "  memory_budget_gb: 22.0\n"
        "profiles:\n"
        "  p: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 16,\n"
        "       chunk_target_tokens: 512, chunk_max_tokens: 1024}\n"
        "engines:\n"
        f"{engines_block}"
    )
    return str(p)


class TestEngineDefaults:
    def test_paths_defaults_to_empty(self):
        assert EngineConfig(**_BASE_ENGINE).paths == []

    def test_ignore_patterns_default_set(self):
        assert EngineConfig(**_BASE_ENGINE).ignore_patterns == [
            ".#*",
            "*~",
            "*.tmp",
            ".DS_Store",
        ]

    def test_watch_defaults_to_disabled(self):
        watch = EngineConfig(**_BASE_ENGINE).watch
        assert watch.enabled is False
        assert watch.debounce_seconds == 3.0

    def test_setting_ignore_patterns_replaces_the_defaults(self):
        engine = EngineConfig(**_BASE_ENGINE, ignore_patterns=["*.bak"])
        assert engine.ignore_patterns == ["*.bak"]


class TestPathsValidation:
    def test_relative_path_is_rejected(self):
        with pytest.raises(ValidationError, match="must be an absolute directory path"):
            EngineConfig(**_BASE_ENGINE, paths=["vault"])

    def test_absolute_path_is_normalized(self, tmp_path):
        messy = f"{tmp_path}/vault/../vault/./notes"
        engine = EngineConfig(**_BASE_ENGINE, paths=[messy])
        assert engine.paths == [str((tmp_path / "vault" / "notes").resolve())]

    def test_missing_root_is_accepted_at_load_time(self, tmp_path):
        # Existence is checked at USE time — an unmounted vault must not
        # break `dbs-vector search`.
        engine = EngineConfig(**_BASE_ENGINE, paths=[str(tmp_path / "not-there")])
        assert engine.paths == [str(tmp_path / "not-there")]


class TestDebounceValidation:
    def test_negative_debounce_is_rejected(self):
        with pytest.raises(ValidationError):
            EngineConfig(**_BASE_ENGINE, watch={"enabled": True, "debounce_seconds": -1})

    def test_zero_debounce_is_allowed(self):
        engine = EngineConfig(**_BASE_ENGINE, watch={"debounce_seconds": 0})
        assert engine.watch.debounce_seconds == 0


class TestWatchRules:
    def test_watch_requires_non_empty_paths(self, tmp_path):
        cfg = _write(
            tmp_path,
            """
            md:
              description: d
              model: gemma-bf16
              mapper_type: document
              chunker_type: document
              table_name: t
              workflow: w
              tuning_profile: p
              watch: {enabled: true}
            """,
        )
        with pytest.raises(ValueError, match="requires a non-empty `paths:`"):
            load_settings(cfg, validate=True)

    def test_watch_requires_document_chunker(self, tmp_path):
        cfg = _write(
            tmp_path,
            f"""
            sqlish:
              description: d
              model: gemma-bf16
              mapper_type: sql
              chunker_type: sql
              table_name: t
              workflow: w
              tuning_profile: p
              paths: ["{tmp_path}"]
              watch: {{enabled: true}}
            """,
        )
        with pytest.raises(ValueError, match='chunker_type: "document"'):
            load_settings(cfg, validate=True)

    def test_watched_engine_may_not_share_a_table_name(self, tmp_path):
        cfg = _write(
            tmp_path,
            f"""
            md:
              description: d
              model: gemma-bf16
              mapper_type: document
              chunker_type: document
              table_name: shared
              workflow: w
              tuning_profile: p
              paths: ["{tmp_path}"]
              watch: {{enabled: true}}
            md2:
              description: d
              model: gemma-bf16
              mapper_type: document
              chunker_type: document
              table_name: shared
              workflow: w
              tuning_profile: p
            """,
        )
        with pytest.raises(ValueError, match="shares table_name"):
            load_settings(cfg, validate=True)

    def test_valid_watched_engine_loads(self, tmp_path):
        cfg = _write(
            tmp_path,
            f"""
            md:
              description: d
              model: gemma-bf16
              mapper_type: document
              chunker_type: document
              table_name: t
              workflow: w
              tuning_profile: p
              paths: ["{tmp_path}"]
              exclusion_filters: [gitignore]
              watch: {{enabled: true, debounce_seconds: 0}}
            """,
        )
        loaded = load_settings(cfg, validate=True)
        assert loaded.engines["md"].watch.enabled is True
        assert loaded.engines["md"].paths == [str(tmp_path)]
