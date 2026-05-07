"""SQL slow-log query → predicate signature extractor.

Used by the slow-query-investigator skill (Phase 2, Step 5) to extract
WHERE / JOIN / ORDER BY columns from query text in the slow-log corpus.

Two layers:
  1. Sanitize slow-log placeholder syntax (`?+`, `?,?,?`, bare `?`) so
     sqlglot's MySQL parser can handle the input.
  2. Try sqlglot first (handles dialect quirks); fall back to regex for
     queries sqlglot can't parse — most commonly INSERT … ON DUPLICATE
     KEY UPDATE forms with nested IF / VALUES expressions.

The sanitizer + fallback are baked in so callers (skill executions,
tests) never re-discover them.
"""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp

# Placeholder normalization patterns.
_RE_PLACEHOLDER_PLUS = re.compile(r"\?\+")
_RE_PLACEHOLDER_LIST = re.compile(r"\?\s*(?:,\s*\?\s*)+")

# Statement-kind detection (used by both paths).
_RE_KIND_INSERT = re.compile(r"^\s*INSERT\b", re.IGNORECASE)
_RE_KIND_UPDATE = re.compile(r"^\s*UPDATE\b", re.IGNORECASE)
_RE_KIND_DELETE = re.compile(r"^\s*DELETE\b", re.IGNORECASE)
_RE_KIND_SELECT = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

# Regex-fallback predicate extraction.
_RE_WHERE_BLOCK = re.compile(
    r"\bWHERE\b\s+(.+?)"
    r"(?:\bON\s+DUPLICATE\b|\bORDER\s+BY\b|\bGROUP\s+BY\b|\bLIMIT\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_EQ_PRED = re.compile(r"\b([a-z_][a-z0-9_]*)\s*(?:=|\bIN\s*\()", re.IGNORECASE)
_RE_RANGE_PRED = re.compile(
    r"\b([a-z_][a-z0-9_]*)\s*(?:<=|>=|<|>|\bBETWEEN\b)",
    re.IGNORECASE,
)
_SQL_KEYWORDS = {
    "and",
    "or",
    "not",
    "is",
    "null",
    "true",
    "false",
    "in",
    "between",
    "like",
    "regexp",
    "exists",
    "case",
    "when",
    "then",
    "else",
    "end",
    "values",
    "interval",
    "if",
}


def sanitize_placeholders(sql: str) -> str:
    """Convert slow-log placeholder syntax to literals sqlglot accepts.

    Transformations (in order):
      `?+`          → `?`     (slow-log "list of N params" marker)
      `?, ?, ?`     → `?`     (collapse comma-separated placeholder lists)
      `?`           → `0`     (final pass — bare placeholders become integer literals)
    """
    s = _RE_PLACEHOLDER_PLUS.sub("?", sql)
    s = _RE_PLACEHOLDER_LIST.sub("?", s)
    return s.replace("?", "0")


def _stmt_kind(sql: str) -> tuple[str, bool]:
    """Detect kind via leading keyword. Returns (kind, is_write)."""
    if _RE_KIND_INSERT.match(sql):
        return "INSERT", True
    if _RE_KIND_UPDATE.match(sql):
        return "UPDATE", True
    if _RE_KIND_DELETE.match(sql):
        return "DELETE", True
    if _RE_KIND_SELECT.match(sql):
        return "SELECT", False
    return "?", False


def _col_name(node: exp.Expression | None) -> str | None:
    return node.name.lower() if isinstance(node, exp.Column) else None


def _extract_via_sqlglot(sql: str) -> dict[str, Any] | None:
    """Best-effort sqlglot extraction. Returns None on parse failure."""
    try:
        ast = sqlglot.parse_one(sql, dialect="mysql")
    except Exception:
        return None

    sig: dict[str, Any] = {
        "eq_cols": [],
        "range_cols": [],
        "join_keys": [],
        "order_by": [],
        "is_write": False,
        "kind": ast.key.upper(),
    }
    sig["is_write"] = sig["kind"] in {"INSERT", "UPDATE", "DELETE"}

    # sqlglot's stubs type find_all() as accepting a single class, but the
    # runtime accepts a tuple of classes too — split into multiple loops to
    # keep mypy happy without losing readability.
    for where in ast.find_all(exp.Where):
        for eq_cls in (exp.EQ, exp.NEQ):
            for n in where.find_all(eq_cls):
                cn = _col_name(n.left) or _col_name(n.right)  # type: ignore[arg-type]
                if cn and cn not in sig["eq_cols"]:
                    sig["eq_cols"].append(cn)
        for in_node in where.find_all(exp.In):
            cn = _col_name(in_node.this)
            if cn and cn not in sig["eq_cols"]:
                sig["eq_cols"].append(cn)
        for range_cls in (exp.GT, exp.GTE, exp.LT, exp.LTE):
            for n in where.find_all(range_cls):
                cn = _col_name(n.left) or _col_name(n.right)  # type: ignore[arg-type]
                if cn and cn not in sig["range_cols"]:
                    sig["range_cols"].append(cn)
        for between in where.find_all(exp.Between):
            cn = _col_name(between.this)
            if cn and cn not in sig["range_cols"]:
                sig["range_cols"].append(cn)

    for join in ast.find_all(exp.Join):
        for cond in join.find_all(exp.EQ):
            for side in (cond.left, cond.right):
                cn = _col_name(side)  # type: ignore[arg-type]
                if cn and cn not in sig["join_keys"]:
                    sig["join_keys"].append(cn)

    for ob in ast.find_all(exp.Order):
        for o in ob.expressions:
            if isinstance(o, exp.Ordered) and isinstance(o.this, exp.Column):
                sig["order_by"].append(
                    (
                        o.this.name.lower(),
                        "DESC" if o.args.get("desc") else "ASC",
                    )
                )
    return sig


def _extract_via_regex(sql: str) -> dict[str, Any]:
    """Fallback when sqlglot can't parse. Catches the common WHERE shapes.

    Targeted at INSERT … ON DUPLICATE KEY UPDATE with nested IF/VALUES,
    which is structurally pathological for parsers but easy to scan with
    a regex if we anchor on `WHERE … (ON DUPLICATE | ORDER BY | EOF)`.
    """
    kind, is_write = _stmt_kind(sql)
    sig: dict[str, Any] = {
        "eq_cols": [],
        "range_cols": [],
        "join_keys": [],
        "order_by": [],
        "is_write": is_write,
        "kind": kind,
    }

    where_match = _RE_WHERE_BLOCK.search(sql)
    if not where_match:
        return sig
    where_clause = where_match.group(1)

    for m in _RE_EQ_PRED.finditer(where_clause):
        cn = m.group(1).lower()
        if cn not in _SQL_KEYWORDS and cn not in sig["eq_cols"]:
            sig["eq_cols"].append(cn)
    for m in _RE_RANGE_PRED.finditer(where_clause):
        cn = m.group(1).lower()
        if cn not in _SQL_KEYWORDS and cn not in sig["range_cols"]:
            sig["range_cols"].append(cn)

    return sig


def extract_predicate_signature(sql: str) -> dict[str, Any]:
    """Extract WHERE / JOIN / ORDER BY signature from a SQL query string.

    Sanitizes slow-log placeholders, attempts sqlglot first, falls back to
    regex for queries sqlglot can't parse (commonly INSERT … ON DUPLICATE
    KEY UPDATE with nested IF expressions).

    Returns a dict with these keys:
      eq_cols    list[str]            equality predicates (= and IN)
      range_cols list[str]            range predicates (>, >=, <, <=, BETWEEN)
      join_keys  list[str]            JOIN ON columns
      order_by   list[tuple[str,str]] (column, direction) pairs
      is_write   bool                 True for INSERT/UPDATE/DELETE
      kind       str                  statement kind
      parser     str                  "sqlglot" or "regex" — which path produced this
                                      (lets callers gauge confidence)
    """
    sanitized = sanitize_placeholders(sql)
    sig = _extract_via_sqlglot(sanitized)
    if sig is not None:
        sig["parser"] = "sqlglot"
        return sig
    sig = _extract_via_regex(sanitized)
    sig["parser"] = "regex"
    return sig
