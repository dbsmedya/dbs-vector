#!/usr/bin/env python3
"""
dbs-vector Remote API Compatibility Checker  (contract v1)
===========================================================
Verifies that a remote server correctly implements the HTTP contract that
ApiChunker depends on.  Run this manually before wiring up a new backend.

Usage
-----
  # Positional args
  python scripts/check_remote_api.py \\
      --base-url http://localhost:9000/api/v1 \\
      --api-key  sk-dev

  # Via environment variables
  DBS_API_BASE_URL=http://localhost:9000/api/v1 \\
  DBS_API_KEY=sk-dev \\
      python scripts/check_remote_api.py

Exit codes
----------
  0  — all checks passed (or only skipped)
  1  — one or more checks failed
  2  — argument / setup error
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required.  Install: uv pip install 'dbs-vector[api]'")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW = "\033[93m"
_CYAN  = "\033[96m"


def _g(s: str) -> str: return f"{_GREEN}{s}{_RESET}"
def _r(s: str) -> str: return f"{_RED}{s}{_RESET}"
def _y(s: str) -> str: return f"{_YELLOW}{s}{_RESET}"
def _b(s: str) -> str: return f"{_BOLD}{s}{_RESET}"
def _c(s: str) -> str: return f"{_CYAN}{s}{_RESET}"


# ---------------------------------------------------------------------------
# Minimal test-result tracking (zero pytest dependency)
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class _Result:
    name:    str
    status:  str
    detail:  str = ""


@dataclass
class Suite:
    results: list[_Result] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Assertion primitives
    # ------------------------------------------------------------------

    def ok(self, name: str, condition: bool, detail: str = "") -> bool:
        """Record a PASS/FAIL.  Returns True on pass so callers can gate
        follow-up checks."""
        status = PASS if condition else FAIL
        self.results.append(_Result(name, status, detail))
        sym   = _g("✓") if condition else _r("✗")
        label = _g(PASS) if condition else _r(FAIL)
        line  = f"  {sym} [{label}] {name}"
        if detail and not condition:
            line += f"\n         {_r(detail)}"
        print(line)
        return condition

    def skip(self, name: str, reason: str = "") -> None:
        self.results.append(_Result(name, SKIP, reason))
        tail = f" — {reason}" if reason else ""
        print(f"  - [{_y('SKIP')}] {name}{tail}")

    def section(self, title: str) -> None:
        print(f"\n{_b(_c(f'── {title}'))}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> int:
        passed  = sum(1 for r in self.results if r.status == PASS)
        failed  = sum(1 for r in self.results if r.status == FAIL)
        skipped = sum(1 for r in self.results if r.status == SKIP)
        total   = passed + failed

        print(f"\n{'─' * 56}")
        print(_b("Results"))
        print(
            f"  {_g(f'{passed}/{total} passed')}   "
            f"{_r(f'{failed} failed')}   "
            f"{_y(f'{skipped} skipped')}"
        )

        if failed:
            print(_b(_r("\nFailed checks:")))
            for r in self.results:
                if r.status == FAIL:
                    tail = f": {r.detail}" if r.detail else ""
                    print(f"  • {r.name}{tail}")

        return 1 if failed else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(value: object) -> bool:
    """Return True if *value* is a parseable ISO 8601 timestamp."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _json_or_none(resp: httpx.Response) -> dict | None:
    try:
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main check runner
# ---------------------------------------------------------------------------

def run_checks(base_url: str, api_key: str, timeout: int) -> int:
    s = Suite()
    base_url = base_url.rstrip("/")

    auth_headers = {
        "Authorization":   f"Bearer {api_key}",
        "Accept-Encoding": "gzip",
    }

    with httpx.Client(timeout=timeout) as client:

        # ================================================================
        # 1. Health — GET /health
        # ================================================================
        s.section("1. Health  GET /health")

        health: dict = {}
        try:
            r = client.get(f"{base_url}/health")
        except httpx.ConnectError as exc:
            s.ok("health_returns_200", False, f"Connection refused: {exc}")
            # Nothing else makes sense without connectivity.
            return s.summary()

        if s.ok("health_returns_200", r.status_code == 200, f"got {r.status_code}"):
            health = _json_or_none(r) or {}
            s.ok("health_status_is_ok",
                 health.get("status") == "ok",
                 f"status={health.get('status')!r}")
            s.ok("health_has_version_string",
                 isinstance(health.get("version"), str) and bool(health["version"]),
                 f"version={health.get('version')!r}")
            s.ok("health_has_databases_list",
                 isinstance(health.get("databases"), list),
                 f"databases type={type(health.get('databases')).__name__}")
            s.ok("health_no_auth_required",
                 r.status_code != 401,
                 "/health should not require authentication")
        else:
            for name in ["health_status_is_ok", "health_has_version_string",
                         "health_has_databases_list", "health_no_auth_required"]:
                s.skip(name, "health_returns_200 failed")

        # ================================================================
        # 2. Authentication
        # ================================================================
        s.section("2. Authentication")

        # 2a — no Authorization header
        r = client.get(f"{base_url}/sql/queries")
        if s.ok("auth_no_token_returns_401", r.status_code == 401, f"got {r.status_code}"):
            body = _json_or_none(r) or {}
            s.ok("auth_no_token_error_is_invalid_token",
                 body.get("error") == "invalid_token",
                 f"error={body.get('error')!r}")
        else:
            s.skip("auth_no_token_error_is_invalid_token", "status was not 401")

        # 2b — wrong bearer token
        r = client.get(f"{base_url}/sql/queries",
                       headers={"Authorization": "Bearer bad-token-xyz"})
        s.ok("auth_wrong_token_returns_401", r.status_code == 401, f"got {r.status_code}")

        # ================================================================
        # 3. GET /sql/queries — response structure
        # ================================================================
        s.section("3. GET /sql/queries — structure")

        r = client.get(f"{base_url}/sql/queries", headers=auth_headers)
        if not s.ok("queries_returns_200", r.status_code == 200, f"got {r.status_code}"):
            for name in [
                "queries_content_type_json", "queries_x_request_id_header",
                "queries_has_data_array", "queries_has_has_more_bool",
                "queries_has_total_count_numeric",
                "queries_record_has_required_fields", "queries_required_fields_non_null",
                "queries_tables_is_list_never_null", "queries_latest_ts_iso8601",
                "queries_text_not_empty_string", "queries_id_non_empty_string",
                "queries_execution_time_ms_numeric", "queries_calls_is_int",
            ]:
                s.skip(name, "queries_returns_200 failed")
        else:
            ct = r.headers.get("content-type", "")
            s.ok("queries_content_type_json", "application/json" in ct, f"got {ct!r}")
            s.ok("queries_x_request_id_header",
                 "x-request-id" in r.headers, "header absent")

            body = _json_or_none(r) or {}
            s.ok("queries_has_data_array",
                 isinstance(body.get("data"), list),
                 f"data type={type(body.get('data')).__name__}")
            s.ok("queries_has_has_more_bool",
                 isinstance(body.get("has_more"), bool),
                 f"has_more={body.get('has_more')!r}")
            s.ok("queries_has_total_count_numeric",
                 isinstance(body.get("total_count"), (int, float)),
                 f"total_count={body.get('total_count')!r}")

            records: list[dict] = body.get("data", [])
            if records:
                rec = records[0]
                required = ["id", "text", "source", "execution_time_ms",
                            "calls", "tables", "latest_ts"]
                missing = [f for f in required if f not in rec]
                s.ok("queries_record_has_required_fields", not missing,
                     f"missing: {missing}")
                s.ok("queries_required_fields_non_null",
                     all(rec.get(f) is not None for f in ["id", "text", "source"]),
                     "id / text / source must not be null")
                s.ok("queries_tables_is_list_never_null",
                     isinstance(rec.get("tables"), list),
                     f"tables={rec.get('tables')!r}")
                s.ok("queries_latest_ts_iso8601",
                     _parse_ts(rec.get("latest_ts")),
                     f"latest_ts={rec.get('latest_ts')!r}")
                s.ok("queries_text_not_empty_string",
                     isinstance(rec.get("text"), str) and bool(rec["text"].strip()),
                     "text is blank")
                s.ok("queries_id_non_empty_string",
                     isinstance(rec.get("id"), str) and bool(rec["id"]),
                     f"id={rec.get('id')!r}")
                s.ok("queries_execution_time_ms_numeric",
                     isinstance(rec.get("execution_time_ms"), (int, float)),
                     f"execution_time_ms={rec.get('execution_time_ms')!r}")
                s.ok("queries_calls_is_int",
                     isinstance(rec.get("calls"), int),
                     f"calls={rec.get('calls')!r}")
            else:
                for name in [
                    "queries_record_has_required_fields", "queries_required_fields_non_null",
                    "queries_tables_is_list_never_null", "queries_latest_ts_iso8601",
                    "queries_text_not_empty_string", "queries_id_non_empty_string",
                    "queries_execution_time_ms_numeric", "queries_calls_is_int",
                ]:
                    s.skip(name, "no records returned (empty dataset)")

        # ================================================================
        # 4. GET /sql/queries — pagination
        # ================================================================
        s.section("4. GET /sql/queries — pagination")

        r = client.get(f"{base_url}/sql/queries", headers=auth_headers,
                       params={"limit": 1})
        if r.status_code != 200:
            for name in [
                "queries_limit_param_respected",
                "queries_next_cursor_present_when_has_more",
                "queries_page2_returns_200",
                "queries_no_duplicate_ids_across_pages",
            ]:
                s.skip(name, f"request failed: {r.status_code}")
        else:
            page1 = _json_or_none(r) or {}
            data1: list[dict] = page1.get("data", [])
            s.ok("queries_limit_param_respected",
                 len(data1) <= 1, f"got {len(data1)} records with limit=1")

            if page1.get("has_more"):
                cursor = page1.get("next_cursor")
                s.ok("queries_next_cursor_present_when_has_more",
                     bool(cursor), f"next_cursor={cursor!r}")

                if cursor:
                    r2 = client.get(f"{base_url}/sql/queries", headers=auth_headers,
                                    params={"limit": 1, "cursor": cursor})
                    if s.ok("queries_page2_returns_200",
                            r2.status_code == 200, f"got {r2.status_code}"):
                        page2 = _json_or_none(r2) or {}
                        ids1 = {rec["id"] for rec in data1 if "id" in rec}
                        ids2 = {rec["id"] for rec in page2.get("data", []) if "id" in rec}
                        overlap = ids1 & ids2
                        s.ok("queries_no_duplicate_ids_across_pages",
                             not overlap, f"duplicate ids: {overlap}")
                    else:
                        s.skip("queries_no_duplicate_ids_across_pages", "page 2 failed")
                else:
                    s.skip("queries_page2_returns_200", "next_cursor absent")
                    s.skip("queries_no_duplicate_ids_across_pages", "next_cursor absent")
            else:
                s.skip("queries_next_cursor_present_when_has_more",
                       "has_more=false — single page dataset")
                s.skip("queries_page2_returns_200",
                       "has_more=false — single page dataset")
                s.skip("queries_no_duplicate_ids_across_pages",
                       "has_more=false — single page dataset")

        # ================================================================
        # 5. GET /sql/queries — filters
        # ================================================================
        s.section("5. GET /sql/queries — filters")

        # 5a — min_execution_ms: use a very large value to expect zero records
        r = client.get(f"{base_url}/sql/queries", headers=auth_headers,
                       params={"min_execution_ms": 999_999_999.0, "limit": 10})
        if r.status_code == 200:
            over_threshold = [
                rec for rec in (_json_or_none(r) or {}).get("data", [])
                if rec.get("execution_time_ms", 0) < 999_999_999.0
            ]
            s.ok("queries_min_execution_ms_filter_honored",
                 not over_threshold,
                 f"{len(over_threshold)} records below the threshold")
        else:
            s.skip("queries_min_execution_ms_filter_honored",
                   f"request failed: {r.status_code}")

        # 5b — since=future date → must return empty data
        r = client.get(f"{base_url}/sql/queries", headers=auth_headers,
                       params={"since": "2099-01-01T00:00:00Z", "limit": 10})
        if r.status_code == 200:
            future_data = (_json_or_none(r) or {}).get("data", [])
            s.ok("queries_since_future_returns_empty",
                 len(future_data) == 0,
                 f"got {len(future_data)} records with since=2099")
        else:
            s.skip("queries_since_future_returns_empty",
                   f"request failed: {r.status_code}")

        # 5c — database filter: if health reported databases, pick the first
        available_dbs: list[str] = health.get("databases", [])
        if available_dbs:
            db = available_dbs[0]
            r = client.get(f"{base_url}/sql/queries", headers=auth_headers,
                           params={"database": db, "limit": 20})
            if r.status_code == 200:
                db_records = (_json_or_none(r) or {}).get("data", [])
                wrong = [rec for rec in db_records if rec.get("source") != db]
                s.ok("queries_database_filter_returns_matching_source",
                     not wrong,
                     f"{len(wrong)} records with source != {db!r}")
            else:
                s.skip("queries_database_filter_returns_matching_source",
                       f"request failed: {r.status_code}")
        else:
            s.skip("queries_database_filter_returns_matching_source",
                   "no databases listed in /health")

        # ================================================================
        # 6. GET /sql/databases
        # ================================================================
        s.section("6. GET /sql/databases")

        r = client.get(f"{base_url}/sql/databases", headers=auth_headers)
        if s.ok("databases_returns_200", r.status_code == 200, f"got {r.status_code}"):
            db_body = _json_or_none(r) or {}
            if s.ok("databases_has_databases_list",
                    isinstance(db_body.get("databases"), list),
                    f"type={type(db_body.get('databases')).__name__}"):
                if available_dbs:
                    api_set    = set(db_body["databases"])
                    health_set = set(available_dbs)
                    s.ok("databases_consistent_with_health",
                         api_set == health_set,
                         f"/health={sorted(health_set)}  /sql/databases={sorted(api_set)}")
                else:
                    s.skip("databases_consistent_with_health",
                           "/health returned no databases to compare")
            else:
                s.skip("databases_consistent_with_health", "databases list missing")
        else:
            s.skip("databases_has_databases_list", "databases_returns_200 failed")
            s.skip("databases_consistent_with_health", "databases_returns_200 failed")

        # ================================================================
        # 7. POST /sql/execute — happy path
        # ================================================================
        s.section("7. POST /sql/execute — happy path")

        safe_select = (
            "SELECT 1 AS id, 'normalized sql' AS text, 'db' AS source, "
            "0.0 AS execution_time_ms, 1 AS calls"
        )
        r = client.post(
            f"{base_url}/sql/execute",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={"query": safe_select, "timeout_ms": 5000},
        )
        if s.ok("execute_select_returns_200",
                r.status_code == 200, f"got {r.status_code}"):
            ex = _json_or_none(r) or {}
            s.ok("execute_has_columns",
                 isinstance(ex.get("columns"), list) and len(ex["columns"]) > 0,
                 f"columns={ex.get('columns')!r}")
            s.ok("execute_has_rows",
                 isinstance(ex.get("rows"), list),
                 f"rows type={type(ex.get('rows')).__name__}")
            s.ok("execute_has_row_count",
                 isinstance(ex.get("row_count"), int),
                 f"row_count={ex.get('row_count')!r}")
            s.ok("execute_has_truncated_bool",
                 isinstance(ex.get("truncated"), bool),
                 f"truncated={ex.get('truncated')!r}")
            s.ok("execute_has_execution_time_ms",
                 isinstance(ex.get("execution_time_ms"), (int, float)),
                 f"execution_time_ms={ex.get('execution_time_ms')!r}")
            rows = ex.get("rows", [])
            cols = ex.get("columns", [])
            s.ok("execute_row_count_matches_len_rows",
                 ex.get("row_count") == len(rows),
                 f"row_count={ex.get('row_count')} vs len(rows)={len(rows)}")
            if rows and cols:
                bad = [i for i, row in enumerate(rows) if len(row) != len(cols)]
                s.ok("execute_rows_parallel_to_columns",
                     not bad, f"row indices with wrong arity: {bad[:5]}")
            else:
                s.skip("execute_rows_parallel_to_columns", "no rows to verify")
        else:
            for name in [
                "execute_has_columns", "execute_has_rows", "execute_has_row_count",
                "execute_has_truncated_bool", "execute_has_execution_time_ms",
                "execute_row_count_matches_len_rows", "execute_rows_parallel_to_columns",
            ]:
                s.skip(name, "execute_select_returns_200 failed")

        # ================================================================
        # 8. POST /sql/execute — safety enforcement
        # ================================================================
        s.section("8. POST /sql/execute — safety enforcement")

        unsafe_stmts = [
            ("INSERT INTO x VALUES (1)", "insert"),
            ("UPDATE x SET a = 1",       "update"),
            ("DELETE FROM x",            "delete"),
            ("DROP TABLE x",             "drop"),
        ]
        for stmt, key in unsafe_stmts:
            r = client.post(
                f"{base_url}/sql/execute",
                headers={**auth_headers, "Content-Type": "application/json"},
                json={"query": stmt, "timeout_ms": 5000},
            )
            if s.ok(f"execute_rejects_{key}_with_422",
                    r.status_code == 422,
                    f"got {r.status_code} for {stmt!r}"):
                body = _json_or_none(r) or {}
                s.ok(f"execute_{key}_error_code_unsafe_query",
                     body.get("error") == "unsafe_query",
                     f"error={body.get('error')!r}")
            else:
                s.skip(f"execute_{key}_error_code_unsafe_query", "status was not 422")

        # Multi-statement (SQLi probe)
        r = client.post(
            f"{base_url}/sql/execute",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={"query": "SELECT 1; DROP TABLE slow_logs", "timeout_ms": 5000},
        )
        s.ok("execute_multi_statement_rejected",
             r.status_code in (400, 422),
             f"got {r.status_code}")

        # ================================================================
        # 9. Error envelope structure
        # ================================================================
        s.section("9. Error envelope")

        # Re-use the unauthenticated 401 we can reliably reproduce
        r = client.get(f"{base_url}/sql/queries")
        body = _json_or_none(r) or {}
        s.ok("error_envelope_has_error_field",
             "error" in body, f"keys: {sorted(body)}")
        s.ok("error_envelope_has_message_field",
             "message" in body, f"keys: {sorted(body)}")

    return s.summary()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="check_remote_api.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DBS_API_BASE_URL", ""),
        metavar="URL",
        help=(
            "Base URL of the remote API, e.g. http://localhost:9000/api/v1. "
            "Reads DBS_API_BASE_URL env var when not given."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DBS_API_KEY", ""),
        metavar="TOKEN",
        help=(
            "Bearer token for authentication. "
            "Reads DBS_API_KEY env var when not given."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="SEC",
        help="HTTP request timeout in seconds (default: 30).",
    )

    args = parser.parse_args()

    if not args.base_url:
        parser.error("--base-url is required (or set DBS_API_BASE_URL)")
    if not args.api_key:
        parser.error("--api-key is required (or set DBS_API_KEY)")

    masked = args.api_key[:6] + "…" if len(args.api_key) > 6 else args.api_key
    print(_b("\ndbs-vector Remote API — Compatibility Check"))
    print(f"  Base URL : {_c(args.base_url)}")
    print(f"  API key  : {_c(masked)}")
    print(f"  Timeout  : {args.timeout}s")

    sys.exit(run_checks(args.base_url, args.api_key, args.timeout))


if __name__ == "__main__":
    main()
