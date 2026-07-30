"""Unit tests for the CLI search --json output flag."""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from dbs_vector.core.models import Chunk, SearchResponse, SearchResult


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
        similarity=0.9,
        retrieved_by="both",
        rrf_score=0.42,
    )


def test_json_flag_emits_json_to_stdout_and_skips_pretty_print(runner):
    from dbs_vector.cli import app

    service = MagicMock()
    service.execute_query.return_value = SearchResponse(results=[_doc_result()], inspected=1)
    service.results_to_json.return_value = json.dumps(
        {
            "floor": None,
            "inspected": 1,
            "best_rejected": None,
            "results": [_doc_result().model_dump(mode="json")],
        }
    )

    with (
        patch("dbs_vector.cli.settings", _settings_with({"md": "document"})),
        patch("dbs_vector.cli.build_search_service", return_value=service),
    ):
        result = runner.invoke(app, ["search", "q", "--type", "md", "--json"])

        assert result.exit_code == 0, result.output
        # JSON went to stdout and parses back to the full envelope payload.
        payload = json.loads(result.stdout)
        assert payload["results"][0]["chunk"]["source"] == "docs/a.md"
        assert payload["results"][0]["chunk"]["text"] == "Full document content here."
        assert payload["results"][0]["similarity"] == 0.9
        # Human-readable printer is bypassed in JSON mode.
        service.results_to_json.assert_called_once()
        service.print_results.assert_not_called()


def test_default_uses_pretty_print_not_json(runner):
    from dbs_vector.cli import app

    service = MagicMock()
    service.execute_query.return_value = SearchResponse(results=[_doc_result()], inspected=1)

    with (
        patch("dbs_vector.cli.settings", _settings_with({"md": "document"})),
        patch("dbs_vector.cli.build_search_service", return_value=service),
    ):
        result = runner.invoke(app, ["search", "q", "--type", "md"])

        assert result.exit_code == 0, result.output
        service.print_results.assert_called_once()
        service.results_to_json.assert_not_called()
