"""The generated config must actually work, not merely parse.

The end-to-end test downloads a real embedding model on first run. Opt in:

    RUN_SLOW_TESTS=1 uv run pytest tests/integration/test_init_e2e.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dbs_vector.config import load_settings
from dbs_vector.services.initializer import run_init

BUDGET = 21.0


def slow(fn):
    """Marker + env gate, matching tests/integration/test_granite_engines.py.

    `poe test` runs plain `pytest tests/ -v` with no -m filter, so the marker
    alone would not keep a model download out of `poe check`.
    """
    fn = pytest.mark.slow(fn)
    fn = pytest.mark.skipif(
        os.getenv("RUN_SLOW_TESTS") != "1",
        reason="Downloads a real embedding model; set RUN_SLOW_TESTS=1 to enable.",
    )(fn)
    return fn


def _answers(tmp_path: Path) -> dict:
    return {
        "Engine name": "docs",
        "Embedding model": "granite-r2",
        "Chunk granularity": "medium",
        "Directory to index (blank when done)": [str(tmp_path / "notes")],
        "Where should LanceDB store its tables?": str(tmp_path / "lancedb"),
        "Where is dbs-vector installed?": str(tmp_path / "repo"),
        "Write config to": str(tmp_path / "config.yaml"),
        "Write MCP config to": str(tmp_path / ".mcp.json"),
    }


def _prepare(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "guide.md").write_text(
        "# Vector search\n\nLanceDB stores Arrow record batches on disk.\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    # `version` is required for `uv run` itself to parse this as a project
    # (is_dbs_vector_checkout only needs `name`); the Part D smoke test is the
    # one test in this file that actually invokes `uv run` as a subprocess.
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "dbs-vector"\nversion = "0.0.0"\n', encoding="utf-8"
    )


def test_generated_config_loads_through_the_real_loader(tmp_path, scripted_io):
    _prepare(tmp_path)
    result = run_init(scripted_io(_answers(tmp_path)), cwd=tmp_path, memory_budget_gb=BUDGET)

    settings = load_settings(str(result.config_path), validate=True)

    assert "docs" in settings.engines
    assert settings.engines["docs"].tuning_profile in settings.profiles


@slow
def test_generated_config_ingests_and_searches(tmp_path, monkeypatch, scripted_io):
    _prepare(tmp_path)
    result = run_init(scripted_io(_answers(tmp_path)), cwd=tmp_path, memory_budget_gb=BUDGET)

    from dbs_vector import config as config_module
    from dbs_vector.services import bootstrap as bootstrap_module
    from dbs_vector.services.bootstrap import build_dependencies, build_search_service
    from dbs_vector.services.ingestion import IngestionService

    # Point BOTH singletons at the generated config, and let monkeypatch undo
    # it afterwards so no other test inherits this engine set.
    generated = load_settings(str(result.config_path), validate=True)
    monkeypatch.setattr(config_module, "settings", generated)
    monkeypatch.setattr(bootstrap_module, "settings", generated)

    deps = build_dependencies("docs")
    ingestion = IngestionService(
        chunker=deps.chunker,
        embedder=deps.embedder,
        vector_store=deps.store,
        workflow=deps.workflow,
        batch_size=deps.batch_size,
        path_filter=deps.path_filter,
    )
    # A list target means "the engine's configured roots", discovered through
    # the PathFilter.
    ingestion.ingest_directory(generated.engines["docs"].paths)

    response = build_search_service("docs", deps=deps).execute_query(
        "arrow record batches", limit=3
    )
    assert response.results, "generated engine returned no hits for an obvious query"


@slow
def test_the_generated_mcp_command_starts_a_server(tmp_path, scripted_io):
    """The generated launch line must actually run. Sends one JSON-RPC
    initialize frame and expects a response."""
    import json as _json
    import subprocess

    _prepare(tmp_path)
    result = run_init(scripted_io(_answers(tmp_path)), cwd=tmp_path, memory_budget_gb=BUDGET)
    entry = _json.loads(result.mcp_path.read_text(encoding="utf-8"))["mcpServers"]["dbs-vector"]

    frame = _json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "0"},
            },
        }
    )
    proc = subprocess.run(
        [entry["command"], *entry["args"]],
        input=frame + "\n",
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert '"result"' in proc.stdout, f"server did not respond: {proc.stderr[-2000:]}"
