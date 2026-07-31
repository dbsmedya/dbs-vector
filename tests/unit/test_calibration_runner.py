import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "calibrate_similarity_floor.py"
SPEC = importlib.util.spec_from_file_location("calibrate_similarity_floor", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _query_set(name="dev"):
    return SimpleNamespace(
        name=name,
        corpus="test",
        queries=[],
        relevant=[],
        absent=[],
        hard_negatives=[],
        off_domain=[],
    )


def _store():
    table = MagicMock()
    table.num_rows = 1
    table.column.return_value.to_pylist.return_value = ["/repo/source.md"]
    store = MagicMock()
    store.scan.return_value = table
    return store


@pytest.fixture()
def fake_engine(monkeypatch):
    engine = SimpleNamespace(table_name="table", paths=[])
    monkeypatch.setattr(RUNNER.settings, "engines", {"fake": engine})
    return engine


def test_preflight_never_builds_dependencies_or_searches(monkeypatch, fake_engine):
    store = _store()
    monkeypatch.setattr(RUNNER, "load_query_set", lambda path: _query_set())
    monkeypatch.setattr(RUNNER, "source_resolution_errors", lambda query_set, sources: [])
    monkeypatch.setattr(RUNNER, "build_store", lambda engine: store)
    build_dependencies = MagicMock()
    monkeypatch.setattr(RUNNER, "build_dependencies", build_dependencies)

    assert RUNNER.main(["--engine", "fake", "--set", "unused.json", "--preflight-only"]) == 0
    build_dependencies.assert_not_called()
    store.search.assert_not_called()


def test_development_mode_rejects_floor_before_store_access(
    monkeypatch,
    fake_engine,
):
    monkeypatch.setattr(RUNNER, "load_query_set", lambda path: _query_set())
    build_store = MagicMock()
    monkeypatch.setattr(RUNNER, "build_store", build_store)
    assert RUNNER.main(["--engine", "fake", "--set", "unused.json", "--floor", "0.5"]) == 2
    build_store.assert_not_called()


def test_evaluation_requires_floor_and_choice_before_store_access(
    monkeypatch,
    fake_engine,
):
    monkeypatch.setattr(RUNNER, "load_query_set", lambda path: _query_set("eval"))
    build_store = MagicMock()
    monkeypatch.setattr(RUNNER, "build_store", build_store)
    assert RUNNER.main(["--engine", "fake", "--set", "unused.json"]) == 2
    build_store.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--limit", "0"],
        ["--limit", "101"],
        ["--floor", "-1.1"],
        ["--floor", "1.1"],
    ],
)
def test_numeric_ranges_are_rejected_before_loading_set(
    monkeypatch,
    fake_engine,
    arguments,
):
    load_query_set = MagicMock()
    monkeypatch.setattr(RUNNER, "load_query_set", load_query_set)
    assert RUNNER.main(["--engine", "fake", "--set", "unused.json", *arguments]) == 2
    load_query_set.assert_not_called()


def test_existing_output_is_never_overwritten(
    tmp_path,
    monkeypatch,
    fake_engine,
):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    load_query_set = MagicMock()
    monkeypatch.setattr(RUNNER, "load_query_set", load_query_set)
    assert (
        RUNNER.main(
            [
                "--engine",
                "fake",
                "--set",
                "unused.json",
                "--out",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_text(encoding="utf-8") == "keep"
    load_query_set.assert_not_called()


def test_unsealed_eval_stops_before_dependencies_or_spend_marker(
    tmp_path,
    monkeypatch,
    fake_engine,
):
    choice = tmp_path / "choice.json"
    choice.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(RUNNER, "load_query_set", lambda path: _query_set("eval"))
    monkeypatch.setattr(RUNNER, "source_resolution_errors", lambda query_set, sources: [])
    monkeypatch.setattr(RUNNER, "build_store", lambda engine: _store())
    monkeypatch.setattr(
        RUNNER,
        "_require_committed_inputs",
        MagicMock(side_effect=ValueError("dirty")),
    )
    build_dependencies = MagicMock()
    consume = MagicMock()
    monkeypatch.setattr(RUNNER, "build_dependencies", build_dependencies)
    monkeypatch.setattr(RUNNER, "_consume_eval_once", consume)

    assert (
        RUNNER.main(
            [
                "--engine",
                "fake",
                "--set",
                "unused.json",
                "--floor",
                "0.5",
                "--choice-record",
                str(choice),
            ]
        )
        == 2
    )
    build_dependencies.assert_not_called()
    consume.assert_not_called()


def test_eval_spend_marker_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "REPO_ROOT", tmp_path)
    RUNNER._consume_eval_once("md", "abc123", 0.5)
    with pytest.raises(ValueError, match="already spent"):
        RUNNER._consume_eval_once("md", "abc123", 0.5)
