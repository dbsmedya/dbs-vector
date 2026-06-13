from dbs_vector.core.naming import normalize_tool_name


def test_default_verb_is_search():
    assert normalize_tool_name("sql-api") == "search_sql_api"


def test_browse_verb():
    assert normalize_tool_name("sql-api", verb="browse") == "browse_sql_api"
    assert normalize_tool_name("sql", verb="browse") == "browse_sql"
