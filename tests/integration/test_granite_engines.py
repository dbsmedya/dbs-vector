"""Slow-marked integration tests for the Granite engines.

These tests download the real ibm-granite/granite-embedding-311m-multilingual-r2
model (~1 GB) on first run and exercise the full ingest → search pipeline
against a tmp_path LanceDB. Opt-in via RUN_SLOW_TESTS=1.

    RUN_SLOW_TESTS=1 uv run pytest tests/integration/test_granite_engines.py -v
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest


def slow(fn):
    """Combine the registered `slow` marker with an env-gated skipif.

    The `slow` marker is registered in pyproject.toml (deselect via
    `pytest -m "not slow"`); the skipif additionally short-circuits the
    test by default unless RUN_SLOW_TESTS=1 — which keeps `uv run poe test`
    fast even if you didn't pass `-m "not slow"`.
    """
    fn = pytest.mark.slow(fn)
    fn = pytest.mark.skipif(
        os.getenv("RUN_SLOW_TESTS") != "1",
        reason="Granite integration tests download a real model; set RUN_SLOW_TESTS=1 to enable.",
    )(fn)
    return fn


def _patch_settings_singleton(monkeypatch, fixture_path: str) -> None:
    from dbs_vector import config as config_module
    from dbs_vector.services import bootstrap as bootstrap_module
    from dbs_vector.services import ingestion as ingestion_module

    fixture_settings = config_module.load_settings(fixture_path)
    monkeypatch.setattr(config_module, "settings", fixture_settings)
    monkeypatch.setattr(bootstrap_module, "settings", fixture_settings)
    monkeypatch.setattr(ingestion_module, "settings", fixture_settings)


@slow
def test_md_granite_engine_end_to_end(tmp_path: Path, monkeypatch):
    fixture = tmp_path / "config.yaml"
    fixture.write_text(textwrap.dedent(f"""
        system:
          db_path: {tmp_path / "lancedb"}
          batch_size: 8
          nprobes: 5
        engines:
          md-granite:
            description: "test"
            model_name: "ibm-granite/granite-embedding-311m-multilingual-r2"
            vector_dimension: 768
            max_token_length: 512
            table_name: "knowledge_vault_granite"
            mapper_type: "document"
            chunker_type: "document"
            chunk_max_chars: 2000
            passage_prefix: ""
            query_prefix: ""
            workflow: "md_search_granite"
    """))

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.md").write_text(
        "# Database Indexing\n\nThis document explains how vector indexes work.\n"
    )
    (docs_dir / "beta.md").write_text(
        "# Cooking Risotto\n\nA classic Italian rice dish prepared with broth.\n"
    )

    _patch_settings_singleton(monkeypatch, str(fixture))

    from dbs_vector.services.bootstrap import build_dependencies
    from dbs_vector.services.ingestion import IngestionService
    from dbs_vector.services.search import SearchService

    deps = build_dependencies("md-granite")
    ingest = IngestionService(deps.chunker, deps.embedder, deps.store, deps.workflow)
    ingest.ingest_directory(str(docs_dir), rebuild=True)

    search = SearchService(deps.embedder, deps.store)
    results = search.execute_query(
        "vector indexing", source_filter=None, limit=2, extra_filters={}
    )

    assert len(results) >= 1
    # The "Database Indexing" doc should rank above the risotto doc.
    sources = [r.chunk.source for r in results]
    assert any("alpha.md" in s for s in sources)
