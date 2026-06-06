# tests/unit/test_config_validation.py
import textwrap

import pytest
import yaml

from dbs_vector.config import load_settings


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


@pytest.fixture
def write_config(tmp_path):
    """Fixture returning a helper that writes a minimal valid document-engine
    config YAML and returns the path string.

    profile_overrides: dict merged into the default profile knobs.
    engine_overrides: dict merged into the default engine fields.
    """

    def _helper(
        profile_overrides: dict | None = None,
        engine_overrides: dict | None = None,
    ) -> str:
        profile: dict = {
            "max_token_length": 2048,
            "chunk_max_chars": 1000,
            "batch_size": 16,
            "chunk_target_tokens": 512,
            "chunk_max_tokens": 1024,
        }
        if profile_overrides:
            profile.update(profile_overrides)

        engine: dict = {
            "description": "test engine",
            "model": "gemma-bf16",
            "mapper_type": "document",
            "chunker_type": "document",
            "table_name": "t",
            "workflow": "w",
            "tuning_profile": "test-profile",
        }
        if engine_overrides:
            engine.update(engine_overrides)

        data = {
            "system": {"memory_budget_gb": 22.0},
            "profiles": {"test-profile": profile},
            "engines": {"md": engine},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data))
        return str(p)

    return _helper


GENERAL_PROFILES = """\
profiles:
  gemma-md:           {max_token_length: 2048,  chunk_max_chars: 1000, batch_size: 64, chunk_target_tokens: 512,  chunk_max_tokens: 1024}
  gemma-too-big:      {max_token_length: 99999, chunk_max_chars: 1000, batch_size: 64, chunk_target_tokens: 512,  chunk_max_tokens: 1024}
  granite-oom:        {max_token_length: 16384, chunk_max_chars: 1000, batch_size: 64, chunk_target_tokens: 512,  chunk_max_tokens: 1024}
  granite-md-large:   {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8,  chunk_target_tokens: 1024, chunk_max_tokens: 2048}
"""


def test_unknown_model_raises(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + textwrap.dedent("""
        engines:
          md:
            description: "x"
            model: "nonexistent-model"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """),
    )
    with pytest.raises(ValueError, match="unknown model contract 'nonexistent-model'"):
        load_settings(yaml_path, validate=True)


def test_unknown_profile_raises(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + textwrap.dedent("""
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "nonexistent-profile"
        """),
    )
    with pytest.raises(ValueError, match="unknown tuning profile 'nonexistent-profile'"):
        load_settings(yaml_path, validate=True)


def test_profile_exceeds_model_cap_raises(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + textwrap.dedent("""
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-too-big"
        """),
    )
    with pytest.raises(ValueError, match="requires 99999 tokens.*cap 2048"):
        load_settings(yaml_path, validate=True)


def test_profile_oom_raises_with_recommendation(tmp_path):
    """The user's calibration crash: 16K seq × batch 64 on granite, 22 GB cap."""
    yaml_path = _write_yaml(
        tmp_path,
        textwrap.dedent("""\
        system:
          memory_budget_gb: 22.0
        """)
        + GENERAL_PROFILES
        + textwrap.dedent("""
        engines:
          md-granite:
            description: "x"
            model: "granite-r2"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "granite-oom"
        """),
    )
    with pytest.raises(ValueError) as exc_info:
        load_settings(yaml_path, validate=True)
    msg = str(exc_info.value)
    assert "granite-oom" in msg
    assert "md-granite" in msg
    assert "conservative estimate" in msg
    assert "raw attention buffer" in msg
    assert "41 GB" in msg  # observed OOM data point from calibration note
    assert "16384" in msg  # recommendation preserves seq len


def test_unknown_model_fires_before_memory_check(tmp_path, monkeypatch):
    """Validation ordering: unknown model must fail BEFORE memory budget resolution."""
    # Force memory detection to fail; we should never reach it.
    monkeypatch.setattr(
        "dbs_vector.infrastructure.hardware.detect_memory_budget_gb",
        lambda: None,
    )
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + textwrap.dedent("""
        engines:
          md:
            description: "x"
            model: "nonexistent-model"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-md"
        """),
    )
    with pytest.raises(ValueError, match="unknown model contract"):
        load_settings(yaml_path, validate=True)
    # Specifically NOT a memory-budget error
    with pytest.raises(ValueError) as exc_info:
        load_settings(yaml_path, validate=True)
    assert "Could not auto-detect" not in str(exc_info.value)


def test_validate_false_skips_checks(tmp_path):
    """Default validate=False does not run the chain; broken profile is loaded."""
    yaml_path = _write_yaml(
        tmp_path,
        GENERAL_PROFILES
        + textwrap.dedent("""
        engines:
          md:
            description: "x"
            model: "gemma-bf16"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "gemma-too-big"
        """),
    )
    s = load_settings(yaml_path)  # default validate=False
    assert "md" in s.engines


def test_validate_empty_engines_is_noop(tmp_path):
    yaml_path = _write_yaml(tmp_path, 'system:\n  db_path: "/tmp"\n')
    s = load_settings(yaml_path, validate=True)
    assert s.engines == {}


def test_valid_config_passes_validation(tmp_path):
    yaml_path = _write_yaml(
        tmp_path,
        textwrap.dedent("""\
        system:
          memory_budget_gb: 22.0
        """)
        + GENERAL_PROFILES
        + textwrap.dedent("""
        engines:
          md-granite:
            description: "x"
            model: "granite-r2"
            mapper_type: "document"
            chunker_type: "document"
            table_name: "t"
            workflow: "w"
            tuning_profile: "granite-md-large"
        """),
    )
    s = load_settings(yaml_path, validate=True)
    assert "md-granite" in s.engines


def test_resolved_family_falls_back_to_mapper_type():
    """When EngineConfig.family is None, resolved_family equals mapper_type."""
    from dbs_vector.config import EngineConfig

    engine = EngineConfig(
        description="t",
        model="gemma-bf16",
        mapper_type="document",
        chunker_type="document",
        table_name="t",
        workflow="t",
        tuning_profile="p",
    )
    assert engine.family is None
    assert engine.resolved_family == "document"


def test_resolved_family_uses_explicit_family_when_set():
    """When EngineConfig.family is set, resolved_family uses it."""
    from dbs_vector.config import EngineConfig

    engine = EngineConfig(
        description="t",
        model="gemma-bf16",
        mapper_type="custom-mapper",
        chunker_type="document",
        table_name="t",
        workflow="t",
        tuning_profile="p",
        family="document",
    )
    assert engine.resolved_family == "document"


def test_validate_rejects_illegal_engine_name():
    """Engine names must match ^[a-z0-9][a-z0-9_-]*$ (Rule 6)."""
    import tempfile
    from pathlib import Path

    from dbs_vector.config import load_settings

    config_yaml = """
profiles:
  p: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}

engines:
  Bad-Name:
    description: "x"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "t"
    workflow: "w"
    tuning_profile: "p"
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(config_yaml)
        with pytest.raises(ValueError, match="Engine name 'Bad-Name'"):
            load_settings(str(path), validate=True)


def test_validate_rejects_unknown_family():
    """An engine whose resolved_family is not in FamilyKeyRegistry raises (Rule 7)."""
    import tempfile
    from pathlib import Path

    from dbs_vector.config import load_settings

    config_yaml = """
profiles:
  p: {max_token_length: 2048, chunk_max_chars: 1000, batch_size: 64}

engines:
  oddball:
    description: "x"
    model: "gemma-bf16"
    mapper_type: "nope-not-a-family"
    chunker_type: "document"
    table_name: "t"
    workflow: "w"
    tuning_profile: "p"
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(config_yaml)
        with pytest.raises(ValueError, match="unknown family"):
            load_settings(str(path), validate=True)


def test_document_engine_requires_nonzero_token_budgets(write_config):
    cfg = write_config(
        profile_overrides={
            "chunk_target_tokens": 0,
            "chunk_max_tokens": 1024,
        }
    )
    with pytest.raises(ValueError, match="requires .* chunk_target_tokens"):
        load_settings(cfg, validate=True)


def test_chunk_max_tokens_below_target_is_rejected(write_config):
    cfg = write_config(
        profile_overrides={
            "chunk_target_tokens": 1024,
            "chunk_max_tokens": 512,
        }
    )
    with pytest.raises(ValueError, match="chunk_max_tokens"):
        load_settings(cfg, validate=True)


def test_chunk_max_tokens_above_model_cap_is_rejected(write_config):
    cfg = write_config(
        profile_overrides={
            "max_token_length": 2048,
            "chunk_target_tokens": 512,
            "chunk_max_tokens": 4096,
        }
    )
    with pytest.raises(ValueError, match="chunk_max_tokens"):
        load_settings(cfg, validate=True)


def test_unknown_exclusion_filter_is_rejected(write_config):
    cfg = write_config(engine_overrides={"exclusion_filters": ["bogus"]})
    with pytest.raises(ValueError, match="Unknown exclusion filter"):
        load_settings(cfg, validate=True)


def test_document_engine_rejects_zero_max_tokens(write_config):
    cfg = write_config(
        profile_overrides={
            "chunk_target_tokens": 512,
            "chunk_max_tokens": 0,
        }
    )
    with pytest.raises(ValueError, match="requires .* chunk_target_tokens"):
        load_settings(cfg, validate=True)


def test_non_document_engine_exempt_from_token_budget_rule(write_config):
    cfg = write_config(
        profile_overrides={"chunk_target_tokens": 0, "chunk_max_tokens": 0},
        engine_overrides={"chunker_type": "duckdb", "mapper_type": "sql"},
    )
    load_settings(cfg, validate=True)  # must not raise
