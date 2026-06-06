import os

import pytest

from dbs_vector.config import settings
from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import DocumentChunker
from dbs_vector.infrastructure.chunking.filters import FilterRegistry
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.services.ingestion import IngestionService
from dbs_vector.services.search import SearchService

# Skip conditions for this E2E test:
# - CI environment (set via CI=true env var)
# - Missing docs/ directory
# - Missing "md" engine in settings
# - Not running on Apple Silicon (MLX requirement)
skip_in_ci = pytest.mark.skipif(
    os.environ.get("CI") == "true" or not os.path.exists("docs/") or "md" not in settings.engines,
    reason="E2E test requires local docs/, mlx engine, and Apple Silicon - not suitable for CI",
)


@skip_in_ci
@pytest.mark.slow
@pytest.mark.e2e
def test_ingestion_and_search_integration(tmp_path):
    """
    End-to-End integration test using the markdown files in the docs/ directory.
    Uses a temporary directory for the LanceDB store to isolate the test.
    """
    # 1. Setup Isolated Dependencies
    test_db_path = str(tmp_path / "test_lancedb")

    from dbs_vector.infrastructure.storage.mappers import DocumentMapper

    md_config = settings.engines["md"]

    embedder = MLXEmbedder(
        model_name=md_config.model_name,
        max_token_length=md_config.max_token_length,
        dimension=md_config.vector_dimension,
    )

    mapper = DocumentMapper(vector_dimension=embedder.dimension)

    store = LanceDBStore(
        db_path=test_db_path,
        table_name="test_vault",
        vector_dimension=embedder.dimension,
        mapper=mapper,
    )

    chunker = DocumentChunker(max_chars=md_config.chunk_max_chars)

    # 2. Ingestion Phase (Using the relative docs/ path!)
    ingestion_service = IngestionService(chunker, embedder, store)

    # Assert docs exist
    assert os.path.exists("docs/"), "The docs/ directory must exist for this test."

    # Ingest the markdown files we just moved to docs/
    ingestion_service.ingest_directory("docs/*.md")

    # 3. Search Phase
    search_service = SearchService(embedder, store)

    # Query for something we know is in MLX_LANCEDB_POLARS_via_ApacheArrow.md
    results = search_service.execute_query(query="Unified Memory Architecture", limit=3)

    # 4. Assertions
    assert len(results) > 0, "Search should return results."

    # Verify the schema parsing worked
    first_result = results[0]
    assert first_result.chunk.source.startswith("docs/"), (
        "Source metadata should reflect the docs/ directory."
    )
    assert first_result.chunk.id is not None
    assert first_result.chunk.content_hash is not None
    assert isinstance(first_result.is_fts_match, bool)


def test_section_chunking_no_noise_no_truncation(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "doc.md").write_text(
        "# Guide\n\n## Setup\n\nInstall and configure the service properly.\n\n"
        "## Big\n\n```python\n" + "\n".join(f"a{i}=1" for i in range(300)) + "\n```\n"
    )
    (vault / "draw.excalidraw.md").write_text("# d\n\n```compressed-json\nBLOB\n```\n")

    chunker = DocumentChunker(
        target_tokens=120, max_tokens=240,
        filters=FilterRegistry.resolve(["excalidraw", "compressed_json"]),
    )
    chunks = [
        c
        for f in vault.glob("*.md")
        for c in chunker.process(
            Document(filepath=str(f), content=f.read_text(), content_hash="h")
        )
    ]

    assert chunks, "expected chunks"
    assert all(c.source.endswith("doc.md") for c in chunks)  # excalidraw skipped
    assert all(len(c.text.strip()) >= 16 for c in chunks)    # no sub-16-char noise
    assert all("BLOB" not in c.text for c in chunks)          # compressed-json dropped
    assert all(c.parent_scope for c in chunks)                # heading context present
    assert all(len(c.text) <= 240 for c in chunks)  # size invariant: no chunk exceeds max_tokens
