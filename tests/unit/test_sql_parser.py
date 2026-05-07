"""Unit tests for sql_parser — synthetic SQL inputs covering the
predicate extraction shapes the slow-query-investigator skill relies on."""

from dbs_vector.services.sql_parser import (
    extract_predicate_signature,
    sanitize_placeholders,
)

# ---------- sanitize_placeholders ----------


def test_sanitize_collapses_question_plus():
    """Slow-log `?+` (variadic param list) becomes a single placeholder."""
    assert sanitize_placeholders("WHERE id IN (?+)") == "WHERE id IN (0)"


def test_sanitize_collapses_question_lists():
    """Comma-separated `?,?,?` collapses to a single placeholder."""
    assert sanitize_placeholders("VALUES (?, ?, ?)") == "VALUES (0)"


def test_sanitize_replaces_bare_question():
    """Bare `?` becomes `0` so sqlglot reads it as an integer literal."""
    assert sanitize_placeholders("WHERE x = ?") == "WHERE x = 0"


def test_sanitize_idempotent_on_clean_sql():
    """Already-substituted SQL passes through unchanged."""
    assert sanitize_placeholders("SELECT 1 FROM t") == "SELECT 1 FROM t"


# ---------- sqlglot path: WHERE eq predicates ----------


def test_select_where_eq_simple():
    sig = extract_predicate_signature("SELECT * FROM t WHERE customer_id = ?")
    assert sig["parser"] == "sqlglot"
    assert sig["eq_cols"] == ["customer_id"]
    assert sig["range_cols"] == []
    assert sig["is_write"] is False
    assert sig["kind"] == "SELECT"


def test_select_where_in_list():
    sig = extract_predicate_signature("SELECT * FROM t WHERE id IN (?+)")
    assert sig["parser"] == "sqlglot"
    assert sig["eq_cols"] == ["id"]


def test_select_where_compound_eq():
    sig = extract_predicate_signature("SELECT * FROM t WHERE a = ? AND b = ? AND c = ?")
    assert set(sig["eq_cols"]) == {"a", "b", "c"}


# ---------- sqlglot path: range predicates ----------


def test_select_where_range():
    sig = extract_predicate_signature("SELECT * FROM t WHERE created_at >= ? AND created_at < ?")
    assert sig["parser"] == "sqlglot"
    assert sig["range_cols"] == ["created_at"]
    assert sig["eq_cols"] == []


def test_select_where_between():
    sig = extract_predicate_signature("SELECT * FROM t WHERE x BETWEEN ? AND ?")
    assert "x" in sig["range_cols"]


def test_select_eq_and_range_together():
    sig = extract_predicate_signature("SELECT * FROM t WHERE status = ? AND created_at >= ?")
    assert sig["eq_cols"] == ["status"]
    assert sig["range_cols"] == ["created_at"]


# ---------- sqlglot path: JOIN keys ----------


def test_join_keys_extracted():
    sig = extract_predicate_signature(
        "SELECT 1 FROM tx_process p INNER JOIN tx_message m "
        "ON p.process_id = m.process_id WHERE p.customer_id = ?"
    )
    assert "process_id" in sig["join_keys"]
    assert "customer_id" in sig["eq_cols"]


# ---------- sqlglot path: ORDER BY ----------


def test_order_by_extracted_with_direction():
    sig = extract_predicate_signature(
        "SELECT * FROM t WHERE x = ? ORDER BY created_at DESC, id ASC"
    )
    assert sig["order_by"] == [("created_at", "DESC"), ("id", "ASC")]


def test_order_by_default_asc():
    sig = extract_predicate_signature("SELECT * FROM t ORDER BY x")
    assert sig["order_by"] == [("x", "ASC")]


# ---------- write statements ----------


def test_insert_pure_write_no_where():
    sig = extract_predicate_signature("INSERT INTO t (a, b) VALUES (?, ?)")
    assert sig["is_write"] is True
    assert sig["kind"] == "INSERT"
    assert sig["eq_cols"] == []
    assert sig["range_cols"] == []


def test_update_with_where_eq():
    sig = extract_predicate_signature("UPDATE t SET status = ? WHERE process_id = ?")
    assert sig["parser"] == "sqlglot"
    assert sig["is_write"] is True
    assert sig["kind"] == "UPDATE"
    assert sig["eq_cols"] == ["process_id"]


def test_delete_with_where():
    sig = extract_predicate_signature("DELETE FROM t WHERE id = ?")
    assert sig["is_write"] is True
    assert sig["kind"] == "DELETE"
    assert sig["eq_cols"] == ["id"]


# ---------- regex fallback path ----------


def test_on_duplicate_key_with_nested_if_falls_back_to_regex():
    """ON DUPLICATE KEY UPDATE with nested IF(values(col)=?,?,?) fails sqlglot
    in some forms. Regex fallback must still extract the WHERE eq column."""
    sql = """
    INSERT IGNORE INTO dt_report(customer_id, day, count)
    SELECT customer_id, ?, ? FROM tx_process p
    INNER JOIN tx_message m ON p.process_id = m.process_id
    WHERE p.process_id IN (?+)
    ON DUPLICATE KEY UPDATE
      count = IF(values(count) > ?, count + values(count), count),
      total = IF(values(success) = ?, ceil(total / success), total)
    """
    sig = extract_predicate_signature(sql)
    # Either parser path is acceptable; the assertion is that we extracted
    # process_id from the WHERE clause regardless.
    assert "process_id" in sig["eq_cols"], (
        f"failed to extract process_id; got eq_cols={sig['eq_cols']} via parser={sig['parser']}"
    )
    assert sig["is_write"] is True
    assert sig["kind"] == "INSERT"


def test_regex_fallback_skips_sql_keywords():
    """The regex fallback's predicate scanner must not treat AND/OR/IS/NULL
    as column names."""
    sql = (
        "INSERT IGNORE INTO foo(x,y) SELECT 1,2 FROM t "
        "WHERE a = ? AND b = ? AND c IS NULL "
        "ON DUPLICATE KEY UPDATE x = if(values(x) > ?, x + values(x), x)"
    )
    sig = extract_predicate_signature(sql)
    # `and`, `or`, `is`, `null` must NOT show up as eq_cols.
    for kw in ("and", "or", "is", "null"):
        assert kw not in sig["eq_cols"]


# ---------- robustness ----------


def test_unparseable_returns_safe_dict():
    """Pathological input still returns a well-formed signature dict."""
    sig = extract_predicate_signature("not even close to SQL")
    assert isinstance(sig, dict)
    assert "eq_cols" in sig
    assert "parser" in sig


def test_empty_string():
    sig = extract_predicate_signature("")
    assert sig["eq_cols"] == []
    assert sig["kind"] == "?"
