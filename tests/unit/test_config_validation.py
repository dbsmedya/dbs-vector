# tests/unit/test_config_validation.py
import textwrap

import pytest

from dbs_vector.config import load_settings


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


GENERAL_PROFILES = """\
profiles:
  gemma-md:           {max_token_length: 2048,  chunk_max_chars: 1000, batch_size: 64}
  gemma-too-big:      {max_token_length: 99999, chunk_max_chars: 1000, batch_size: 64}
  granite-oom:        {max_token_length: 16384, chunk_max_chars: 1000, batch_size: 64}
  granite-md-large:   {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
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
