"""Integration test for the slow-query-investigator skill's predicate
extractor.

Exercises `dbs_vector.services.sql_parser.extract_predicate_signature`
against the user's actual ingested slow-log corpus (LanceDB), reports
the parse-success rate, and asserts a minimum threshold so parser
regressions surface at CI time before they can hide bad index
recommendations.

Skipped by default — requires a populated LanceDB store at
`./lancedb_dbs_vector` with at least one of the SQL-family vaults
ingested. Set `DBS_PARSER_INTEGRATION=1` to run locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dbs_vector.services.sql_parser import extract_predicate_signature

# Threshold for the parse-success rate. The slow-log-investigator skill's
# index recommendations rely on extracting WHERE/JOIN/ORDER columns; if
# fewer than this fraction of queries yield a usable signature, the
# 80%-coverage analysis becomes unreliable.
MIN_PARSE_RATE = 0.95
TOP_N = 100

DB_PATH = "./lancedb_dbs_vector"
SQL_TABLES = ("query_vault", "query_vault_granite", "query_vault_granite_api")


def _runnable() -> bool:
    """Whether this test can find a populated SQL vault."""
    if os.environ.get("DBS_PARSER_INTEGRATION") != "1":
        return False
    if not Path(DB_PATH).exists():
        return False
    try:
        import lancedb
    except ImportError:
        return False
    db = lancedb.connect(DB_PATH)
    tables = set(db.list_tables().tables)
    return any(name in tables for name in SQL_TABLES)


pytestmark = pytest.mark.skipif(
    not _runnable(),
    reason=(
        "Requires populated LanceDB SQL vault at ./lancedb_dbs_vector. "
        "Set DBS_PARSER_INTEGRATION=1 and ingest at least one sql/sql-api "
        "engine to run."
    ),
)


def _load_top_queries(n: int = TOP_N) -> list[dict]:
    """Pull the top-n queries by `calls` from whichever SQL vault exists."""
    import lancedb

    db = lancedb.connect(DB_PATH)
    table_names = set(db.list_tables().tables)
    available = [name for name in SQL_TABLES if name in table_names]
    table_name = available[0]
    table = db.open_table(table_name)
    df = table.to_pandas()

    has_calls = "calls" in df.columns
    sort_col = "calls" if has_calls else "execution_time_ms"
    df_sorted = df.sort_values(by=sort_col, ascending=False).head(n)

    return [
        {
            "id": row["id"],
            "calls": int(row["calls"]) if has_calls and row["calls"] is not None else 0,
            "raw_query": row.get("raw_query") or row.get("text") or "",
        }
        for _, row in df_sorted.iterrows()
    ]


@pytest.mark.slow
@pytest.mark.e2e
def test_parser_meets_min_success_rate_on_real_corpus():
    """Top-100 slow-log queries — at least MIN_PARSE_RATE must yield a
    usable signature (eq_cols, range_cols, join_keys, or order_by)."""
    queries = _load_top_queries(TOP_N)
    assert len(queries) > 0, "expected at least one ingested query"

    total = len(queries)
    sqlglot_ok = 0
    regex_ok = 0
    yielded_predicates = 0

    for q in queries:
        sig = extract_predicate_signature(q["raw_query"])
        if sig["parser"] == "sqlglot":
            sqlglot_ok += 1
        else:
            regex_ok += 1
        if sig["eq_cols"] or sig["range_cols"] or sig["join_keys"] or sig["order_by"]:
            yielded_predicates += 1
        elif sig["kind"] == "INSERT" and not sig["eq_cols"]:
            # Pure INSERT … VALUES queries legitimately have no WHERE.
            # Count them as "successfully parsed" — we know they don't
            # need a predicate signature.
            yielded_predicates += 1

    parse_rate = yielded_predicates / total
    print(
        f"\nCorpus parse stats over top-{total}:"
        f"\n  sqlglot path: {sqlglot_ok} ({sqlglot_ok / total:.0%})"
        f"\n  regex path:   {regex_ok} ({regex_ok / total:.0%})"
        f"\n  Yielded usable signature: {yielded_predicates} ({parse_rate:.0%})"
    )

    assert parse_rate >= MIN_PARSE_RATE, (
        f"parse-success rate {parse_rate:.0%} below required {MIN_PARSE_RATE:.0%}. "
        f"Inspect the {total - yielded_predicates} failing queries."
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_parser_returns_consistent_shape_for_all_queries():
    """Every signature dict must have the same keys regardless of parser
    path. This is a contract check — downstream skill steps assume the
    schema is stable."""
    queries = _load_top_queries(TOP_N)
    expected_keys = {
        "eq_cols",
        "range_cols",
        "join_keys",
        "order_by",
        "is_write",
        "kind",
        "parser",
    }
    for q in queries:
        sig = extract_predicate_signature(q["raw_query"])
        missing = expected_keys - set(sig)
        assert not missing, f"signature missing keys {missing} for query id={q['id']}"
        assert isinstance(sig["eq_cols"], list)
        assert isinstance(sig["range_cols"], list)
        assert isinstance(sig["join_keys"], list)
        assert isinstance(sig["order_by"], list)
        assert isinstance(sig["is_write"], bool)
        assert isinstance(sig["kind"], str)
        assert sig["parser"] in {"sqlglot", "regex"}


@pytest.mark.slow
@pytest.mark.e2e
def test_parser_correctly_classifies_writes_vs_reads():
    """The is_write flag must align with leading keyword. Useful sanity
    check that the kind detection isn't broken by sanitization."""
    queries = _load_top_queries(TOP_N)
    misclassified: list[tuple[str, str]] = []

    for q in queries:
        sig = extract_predicate_signature(q["raw_query"])
        head = q["raw_query"].lstrip()[:10].upper()
        is_write_kw = head.startswith(("INSERT", "UPDATE", "DELETE"))
        if is_write_kw != sig["is_write"]:
            misclassified.append((q["id"], head))

    assert not misclassified, (
        f"{len(misclassified)} queries misclassified as write/read. First few: {misclassified[:3]}"
    )
