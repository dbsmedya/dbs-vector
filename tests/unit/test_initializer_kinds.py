import pytest

from dbs_vector.core.model_registry import ModelRegistry
from dbs_vector.services.initializer.answers import KindAnswers
from dbs_vector.services.initializer.kinds import (
    DEFAULT_IGNORE_PATTERNS,
    DocumentKind,
    EngineKindRegistry,
)

BUDGET = 21.0
PATH_PROMPT = "Directory to index (blank when done)"


def test_document_kind_is_registered():
    assert EngineKindRegistry.keys() == ["document"]
    assert EngineKindRegistry.get("document").chunker_type == "document"
    assert EngineKindRegistry.get("document").mapper_type == "document"
    assert EngineKindRegistry.get("document").supports_watch is True


def test_get_unknown_kind_raises():
    with pytest.raises(KeyError, match="Unknown engine kind 'duckdb'"):
        EngineKindRegistry.get("duckdb")


def test_build_profile_delegates_to_document_derivation():
    contract = ModelRegistry.get("granite-r2")
    profile = DocumentKind().build_profile(contract, "medium", BUDGET)
    assert profile["chunk_target_tokens"] == 768
    assert profile["chunk_max_tokens"] == 1536


def test_ask_collects_multiple_paths_until_blank(scripted_io):
    """The list is consumed one answer per call; exhaustion yields the blank
    default that ends the loop."""
    io = scripted_io({PATH_PROMPT: ["/tmp/a", "/tmp/b"]})
    answers = DocumentKind().ask(io)
    assert answers.paths == ["/tmp/a", "/tmp/b"]


def test_ask_rejects_a_relative_path(scripted_io):
    io = scripted_io({PATH_PROMPT: ["relative/dir"]})
    with pytest.raises(ValueError, match="must be an absolute"):
        DocumentKind().ask(io)


def test_ask_appends_extra_ignore_patterns_to_the_defaults(scripted_io):
    """Setting ignore_patterns REPLACES the defaults in config.py, so the
    wizard must always emit the full list."""
    io = scripted_io(
        {
            PATH_PROMPT: ["/tmp/a"],
            "Additional ignore patterns (comma-separated)": ".ayder/archived/*",
        }
    )
    answers = DocumentKind().ask(io)
    assert answers.ignore_patterns == [*DEFAULT_IGNORE_PATTERNS, ".ayder/archived/*"]


def test_ask_defaults_ignore_patterns_to_exactly_the_config_defaults(scripted_io):
    io = scripted_io({PATH_PROMPT: ["/tmp/a"]})
    answers = DocumentKind().ask(io)
    assert answers.ignore_patterns == [".#*", "*~", "*.tmp", ".DS_Store"]


def test_ask_defaults_exclusion_filters_to_the_shipped_pair(scripted_io):
    io = scripted_io({PATH_PROMPT: ["/tmp/a"]})
    answers = DocumentKind().ask(io)
    assert answers.exclusion_filters == ["excalidraw", "compressed_json"]


def test_ask_does_not_prompt_for_debounce_when_watch_is_off(scripted_io):
    io = scripted_io({PATH_PROMPT: ["/tmp/a"]})
    DocumentKind().ask(io)
    assert "Debounce seconds" not in io.asked


def test_ask_prompts_for_debounce_when_watch_is_on(scripted_io):
    io = scripted_io({PATH_PROMPT: ["/tmp/a"], "Watch these paths for changes?": True})
    answers = DocumentKind().ask(io)
    assert answers.watch_enabled is True
    assert answers.watch_debounce_seconds == 3.0


def test_ask_refuses_watch_without_paths(scripted_io):
    """config.py rule 10: watch.enabled requires a non-empty paths list."""
    io = scripted_io({"Watch these paths for changes?": True})
    with pytest.raises(ValueError, match="at least one directory"):
        DocumentKind().ask(io)


def test_build_engine_block_emits_watch_only_when_enabled():
    kind = KindAnswers(paths=["/tmp/a"], ignore_patterns=[".#*"], exclusion_filters=[])
    block = DocumentKind().build_engine_block({"description": "d"}, kind)
    assert "watch" not in block

    kind_watched = KindAnswers(
        paths=["/tmp/a"],
        ignore_patterns=[".#*"],
        exclusion_filters=[],
        watch_enabled=True,
        watch_debounce_seconds=2.5,
    )
    block_watched = DocumentKind().build_engine_block({"description": "d"}, kind_watched)
    assert block_watched["watch"] == {"enabled": True, "debounce_seconds": 2.5}


def test_build_engine_block_preserves_base_keys():
    kind = KindAnswers(paths=["/tmp/a"], ignore_patterns=[], exclusion_filters=[])
    block = DocumentKind().build_engine_block({"description": "d", "model": "granite-r2"}, kind)
    assert block["description"] == "d"
    assert block["model"] == "granite-r2"
    assert block["paths"] == ["/tmp/a"]
