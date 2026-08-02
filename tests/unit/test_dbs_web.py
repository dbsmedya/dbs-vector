"""Tests for the stdlib web UI's document chunk navigation adapter."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dbs_vector.core.models import Chunk, ChunkPage, SearchResult


@pytest.fixture(scope="module")
def web_module():
    path = Path(__file__).parents[2] / "scripts" / "dbs-web.py"
    spec = importlib.util.spec_from_file_location("dbs_web_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chunk(index: int) -> Chunk:
    return Chunk(
        id=f"docs/guide.md_chunk_{index}",
        text=f"content {index}",
        source="docs/guide.md",
        content_hash=f"h{index}",
        parent_scope="Guide",
        line_range=f"{index + 1}-{index + 1}",
    )


def test_document_search_result_exposes_navigation_cursor(web_module):
    result = SearchResult(
        chunk=_chunk(0),
        similarity=0.9,
        retrieved_by="vector",
    )

    payload = web_module._serialize(result)

    assert payload["chunk_id"] == "docs/guide.md_chunk_0"
    assert payload["can_navigate"] is True
    assert payload["prev_disabled"] is True
    assert payload["next_disabled"] is False
    assert ["Chunk cursor", "docs/guide.md_chunk_0"] in payload["header"]


def test_handle_chunk_uses_exact_service_method_and_marks_end(web_module, monkeypatch):
    web_module.STATE.settings = SimpleNamespace(
        engines={"md": SimpleNamespace(resolved_family="document")}
    )
    deps = object()
    monkeypatch.setattr(web_module, "_get_deps", lambda engine: deps)

    service = MagicMock()
    service.read_adjacent_chunks.return_value = ChunkPage(
        anchor_id="docs/guide.md_chunk_1",
        direction="next",
        chunks=[_chunk(2)],
        boundary="end",
    )
    monkeypatch.setattr(
        "dbs_vector.services.bootstrap.build_search_service",
        lambda engine, deps: service,
    )

    payload = web_module._handle_chunk(
        {"engine": "md", "chunk_id": "docs/guide.md_chunk_1", "direction": "next"}
    )

    service.read_adjacent_chunks.assert_called_once_with("docs/guide.md_chunk_1", "next", count=1)
    assert payload["boundary"] == "end"
    assert payload["chunk"]["chunk_id"] == "docs/guide.md_chunk_2"
    assert payload["chunk"]["prev_disabled"] is False
    assert payload["chunk"]["next_disabled"] is True


def test_handle_chunk_rejects_non_document_engine(web_module):
    web_module.STATE.settings = SimpleNamespace(
        engines={"sql": SimpleNamespace(resolved_family="sql")}
    )

    with pytest.raises(ValueError, match="does not support"):
        web_module._handle_chunk({"engine": "sql", "chunk_id": "sql_1", "direction": "next"})


def test_page_contains_navigation_buttons_and_endpoint(web_module):
    assert 'id="prevChunk"' in web_module.PAGE
    assert 'id="nextChunk"' in web_module.PAGE
    assert 'postJSON("/api/chunk"' in web_module.PAGE
