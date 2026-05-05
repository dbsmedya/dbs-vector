"""Unit tests for the CLI --min-time predicate widening."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def _settings_with(engines: dict):
    """Build a settings mock whose .engines dict keys+mapper_type match the input."""
    s = MagicMock()
    s.engines = {name: MagicMock(mapper_type=mapper_type) for name, mapper_type in engines.items()}
    return s


@pytest.mark.parametrize(
    "engine_name, mapper_type",
    [
        ("sql", "sql"),
        ("sql-api", "sql"),
        ("sql-granite", "sql"),
        ("sql-api-granite", "sql"),
    ],
)
def test_min_time_forwarded_for_sql_family_engines(runner, engine_name, mapper_type):
    from dbs_vector.cli import app

    deps = MagicMock()
    deps.embedder = MagicMock()
    deps.store = MagicMock()

    with (
        patch("dbs_vector.cli.settings", _settings_with({engine_name: mapper_type})),
        patch("dbs_vector.cli._build_dependencies", return_value=deps),
        patch("dbs_vector.cli.SearchService") as MockService,
    ):
        MockService.return_value.execute_query.return_value = []
        result = runner.invoke(app, ["search", "q", "--type", engine_name, "--min-time", "100"])
        assert result.exit_code == 0, result.output
        _, kwargs = MockService.return_value.execute_query.call_args
        assert kwargs["extra_filters"] == {"min_time": 100.0}


def test_min_time_ignored_for_document_engines(runner):
    from dbs_vector.cli import app

    deps = MagicMock()
    deps.embedder = MagicMock()
    deps.store = MagicMock()

    with (
        patch("dbs_vector.cli.settings", _settings_with({"md": "document"})),
        patch("dbs_vector.cli._build_dependencies", return_value=deps),
        patch("dbs_vector.cli.SearchService") as MockService,
    ):
        MockService.return_value.execute_query.return_value = []
        result = runner.invoke(app, ["search", "q", "--type", "md", "--min-time", "100"])
        assert result.exit_code == 0, result.output
        _, kwargs = MockService.return_value.execute_query.call_args
        assert kwargs["extra_filters"] == {}
