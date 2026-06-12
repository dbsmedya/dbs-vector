from dbs_vector.core.naming import ENGINE_NAME_PATTERN, normalize_tool_name


def test_engine_name_pattern_accepts_valid_names():
    assert ENGINE_NAME_PATTERN.match("md-granite")
    assert ENGINE_NAME_PATTERN.match("sql_api_granite")
    assert ENGINE_NAME_PATTERN.match("9x")


def test_engine_name_pattern_rejects_invalid_names():
    assert not ENGINE_NAME_PATTERN.match("-leading-dash")
    assert not ENGINE_NAME_PATTERN.match("Upper")
    assert not ENGINE_NAME_PATTERN.match("has space")


def test_normalize_tool_name_dashes_to_underscores():
    assert normalize_tool_name("md-granite") == "search_md_granite"
    assert normalize_tool_name("sql") == "search_sql"
