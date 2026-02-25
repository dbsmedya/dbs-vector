import os

from dbs_vector.config import settings
from dbs_vector.infrastructure.chunking.document import DocumentChunker
from dbs_vector.infrastructure.embeddings.mlx_engine import MLXEmbedder
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.services.ingestion import IngestionService
from dbs_vector.services.search import SearchService


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
