from dbs_vector.config import EngineConfig
from dbs_vector.mcp.families.document import DocumentFamily
from dbs_vector.mcp.families.sql import SqlFamily


def _engine(**over) -> EngineConfig:
    base = dict(
        description="x",
        model="gemma-bf16",
        mapper_type="sql",
        chunker_type="api",
        table_name="query_vault",
        workflow="sql_clustering",
        tuning_profile="gemma-sql-atomic",
    )
    base.update(over)
    return EngineConfig(**base)


def test_sql_search_description_keeps_filter_docs_and_source_phrase():
    d = SqlFamily().search_description("sql-api", _engine(chunker_type="api", model="gemma-bf16"))
    assert "min_time" in d and "min_lock_time" in d and "table_filter" in d
    assert "API" in d  # source phrase from chunker_type="api"
    assert "Gemma" in d  # embeddings phrase from model
    assert "Showing" in d  # N-of-M note
    assert "browse_sql_api" in d  # sibling-tool pointer


def test_sql_search_description_duckdb_granite_phrases():
    d = SqlFamily().search_description(
        "sql-granite", _engine(chunker_type="duckdb", model="granite-r2")
    )
    assert "DuckDB" in d
    assert "Granite" in d


def test_document_search_description_similarity_clause():
    d = DocumentFamily().search_description(
        "md",
        _engine(
            mapper_type="document",
            chunker_type="document",
            model="gemma-bf16",
            tuning_profile="gemma-md",
        ),
    )
    assert "similarity" in d.lower()


def test_browse_description_off_omits_raw_query():
    d = SqlFamily().browse_description(
        "sql-api", _engine(chunker_type="api"), allow_raw_queries=False
    )
    assert "raw_query" not in d
    assert "group_by" in d and "order_by" in d
    assert "execution_time_ms" in d


def test_browse_description_on_includes_raw_query():
    d = SqlFamily().browse_description(
        "sql-api", _engine(chunker_type="api"), allow_raw_queries=True
    )
    assert "raw_query" in d


def test_browse_description_mentions_derived_columns():
    engine = EngineConfig(
        description="short summary",
        model="gemma-bf16",
        mapper_type="sql",
        chunker_type="api",
        table_name="query_vault",
        workflow="sql_clustering",
        tuning_profile="gemma-sql-atomic",
    )
    desc = SqlFamily().browse_description("sql-api", engine, allow_raw_queries=False)
    assert "impact_score" in desc
    assert "selectivity" in desc
