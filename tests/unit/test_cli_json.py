"""Unit tests for the CLI search --json output flag."""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from dbs_vector.core.models import Chunk, SearchResult


@pytest.fixture
def runner():
    return CliRunner()


def _settings_with(engines: dict):
    s = MagicMock()
    s.engines = {
        name: MagicMock(mapper_type=mapper_type, resolved_family=mapper_type)
        for name, mapper_type in engines.items()
    }
    return s


def _doc_result():
    return SearchResult(
        chunk=Chunk(
            id="chunk_0",
            text="Full document content here.",
            source="docs/a.md",
            content_hash="hash_a",
        ),
        score=0.9,
        distance=0.42,
        is_fts_match=False,
    )


def test_json_flag_emits_json_to_stdout_and_skips_pretty_print(runner):
    from dbs_vector.cli import app

    deps = MagicMock()
    deps.embedder = MagicMock()
    deps.store = MagicMock()

    with (
        patch("dbs_vector.cli.settings", _settings_with({"md": "document"})),
        patch("dbs_vector.cli._build_dependencies", return_value=deps),
        patch("dbs_vector.cli.SearchService") as MockService,
    ):
        service = MockService.return_value
        service.execute_query.return_value = [_doc_result()]
        service.results_to_json.return_value = json.dumps([_doc_result().model_dump(mode="json")])

        result = runner.invoke(app, ["search", "q", "--type", "md", "--json"])

        assert result.exit_code == 0, result.output
        # JSON went to stdout and parses back to the full result payload.
        payload = json.loads(result.stdout)
        assert payload[0]["chunk"]["source"] == "docs/a.md"
        assert payload[0]["chunk"]["text"] == "Full document content here."
        assert payload[0]["score"] == 0.9
        # Human-readable printer is bypassed in JSON mode.
        service.results_to_json.assert_called_once()
        service.print_results.assert_not_called()


def test_default_uses_pretty_print_not_json(runner):
    from dbs_vector.cli import app

    deps = MagicMock()
    deps.embedder = MagicMock()
    deps.store = MagicMock()

    with (
        patch("dbs_vector.cli.settings", _settings_with({"md": "document"})),
        patch("dbs_vector.cli._build_dependencies", return_value=deps),
        patch("dbs_vector.cli.SearchService") as MockService,
    ):
        service = MockService.return_value
        service.execute_query.return_value = [_doc_result()]

        result = runner.invoke(app, ["search", "q", "--type", "md"])

        assert result.exit_code == 0, result.output
        service.print_results.assert_called_once()
        service.results_to_json.assert_not_called()
