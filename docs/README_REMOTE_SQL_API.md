# Remote SQL API Ingestion for dbs-vector

This document covers the `ApiChunker` — a new ingestion source that pulls
pre-aggregated SQL slow-query records from a remote HTTP server instead of
reading local files or DuckDB databases.

---

## Overview

dbs-vector already supports two local ingestion sources for SQL slow-query
data:

| Source | Chunker | Input |
|--------|---------|-------|
| JSON slow-query log | `SqlChunker` | `.json` file |
| DuckDB database | `DuckDBChunker` | `.duckdb` file |

`ApiChunker` adds a third path: fetch pre-aggregated query records from any
networked slow-log backend over HTTP.  Everything downstream — deduplication,
embedding, LanceDB storage, search — is unchanged.

```
Remote HTTP API ──► ApiChunker.process() ──► Iterator[SqlChunk]
                                                     │
                                           IngestionService
                                                     │
                              dedup (SHA-256 content_hash) ──► embed ──► LanceDB
```

The server is responsible for aggregating raw log lines by query fingerprint
before responding.  dbs-vector consumes one row per unique normalized query,
exactly as `DuckDBChunker` does.

---

## Requirements

The HTTP client (`httpx`) is an optional dependency grouped under the `api`
extra:

```bash
# Install with the api extra
uv sync --extra api

# Or add it directly
uv add "httpx>=0.27.0"
```

The `sql` extra (DuckDB) is independent — install both if you use both sources:

```bash
uv sync --extra api --extra sql
```

---

## Configuration

Add an engine block to `config.yaml` with `chunker_type: "api"`.  All
`api_*` fields are optional and default to sensible values, so the only
required additions are `api_base_url` and `api_key`.

```yaml
engines:
  sql-api:
    description: "Remote slow query log via HTTP API"
    model_name: "mlx-community/embeddinggemma-300m-bf16"
    vector_dimension: 768
    max_token_length: 2048
    table_name: "query_vault"
    mapper_type: "sql"
    chunker_type: "api"
    chunk_max_chars: 0
    passage_prefix: "task: clustering | query: "
    query_prefix: "task: clustering | query: "
    workflow: "sql_clustering"

    # --- ApiChunker-specific fields ---
    api_base_url: "https://slow-log-api.internal/api/v1"
    api_key: "sk-..."          # set via DBS_API_KEY env var in production
    api_page_size: 200         # records per GET request (max 1000)
    api_since_days: 15         # lower bound on latest_ts  (default: 15)
    api_timeout_sec: 30        # HTTP request timeout in seconds
    api_min_execution_ms: 0    # filter: skip queries below this threshold
    api_database: ""           # leave empty to fetch all databases
```

### Field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_base_url` | str | `""` | Base URL including version prefix, e.g. `https://host/api/v1` |
| `api_key` | str | `""` | Static bearer token sent as `Authorization: Bearer <token>` |
| `api_page_size` | int | `200` | Records requested per page (`limit` query param) |
| `api_since_days` | int | `15` | Fetch queries whose `latest_ts` is within the last N days |
| `api_timeout_sec` | int | `30` | Per-request HTTP timeout in seconds |
| `api_min_execution_ms` | float | `0.0` | Exclude queries below this cumulative execution time |
| `api_database` | str | `""` | Restrict fetch to one database/schema name; empty = all |

---

## CLI Usage

### Standard ingestion — paginated GET

Pass the base URL as the path argument and name the engine with `--type`:

```bash
uv run dbs-vector ingest "https://slow-log-api.internal/api/v1" --type sql-api
```

The chunker issues `GET /sql/queries` pages until `has_more` is `false`,
yielding one `SqlChunk` per record.

### Custom query — POST execution

Use `--query` to send a SQL string to the server's `POST /sql/execute`
endpoint.  The server runs it against its own data source and returns a
tabular response:

```bash
uv run dbs-vector ingest "https://slow-log-api.internal/api/v1" --type sql-api \
  --query "SELECT fingerprint_id AS id,
                  sanitized_sql  AS text,
                  db             AS source,
                  SUM(query_time_sec) * 1000 AS execution_time_ms,
                  COUNT(*)       AS calls
           FROM slow_logs
           WHERE db = 'analytics'
           GROUP BY fingerprint_id, sanitized_sql, db
           ORDER BY execution_time_ms DESC
           LIMIT 100"
```

The query must be a single `SELECT` statement.  The server enforces
read-only access and will reject anything else with `422 unsafe_query`.

### Rebuild

```bash
# Drop and re-ingest from scratch
uv run dbs-vector ingest "https://slow-log-api.internal/api/v1" \
    --type sql-api --rebuild --force
```

---

## How It Works

### URL routing in IngestionService

`IngestionService.ingest_directory()` normally walks the filesystem looking
for files matching `chunker.supported_extensions`.  `ApiChunker` returns an
empty list for `supported_extensions`, so the file walk would find nothing.

To handle this cleanly, a guard at the top of `_chunk_generator()` intercepts
any path that starts with `http://` or `https://` and bypasses file discovery
entirely:

```python
# services/ingestion.py  ─  _chunk_generator()
if target_path.startswith(("http://", "https://")):
    doc = Document(filepath=target_path, content="", content_hash="api-chunker")
    yield from self.chunker.process(doc)
    return
```

The `Document.filepath` field carries the base URL into `ApiChunker.process()`.

### Paginated GET mode (default)

`ApiChunker._fetch_paginated()` issues sequential `GET /sql/queries` requests
until the server signals the last page:

```
GET {base_url}/sql/queries
  ?limit=200
  &since=2026-02-18T00:00:00Z   ← now - api_since_days
  &min_execution_ms=0.0
  &database=prod                ← only if api_database is set
  &cursor=<token>               ← absent on first request, from next_cursor on subsequent
```

The loop:

```
page 1:  GET /sql/queries?limit=200&since=...
           → data=[...200 records...], has_more=true,  next_cursor="eyJ..."
page 2:  GET /sql/queries?limit=200&since=...&cursor=eyJ...
           → data=[...200 records...], has_more=true,  next_cursor="eyK..."
page N:  GET /sql/queries?limit=200&since=...&cursor=eyK...
           → data=[...50 records...],  has_more=false
```

Each record is mapped to a `SqlChunk` and yielded immediately — the page is
not buffered in memory before being handed to the batching pipeline.

### Custom query POST mode

When `--query` is provided, `ApiChunker._fetch_custom_query()` sends a single
`POST /sql/execute`:

```json
{
  "query":      "<the SQL string>",
  "database":   "analytics",
  "timeout_ms": 30000
}
```

The server returns a tabular response:

```json
{
  "columns": ["id", "text", "source", "execution_time_ms", "calls"],
  "rows":    [["fp1", "SELECT ...", "prod", 4230.5, 88], ...],
  "row_count":          42,
  "truncated":          false,
  "execution_time_ms":  120
}
```

`ApiChunker` zips `columns` + each row into a dict and calls the same
`_to_sql_chunk()` mapper used by the paginated path.

### Record → SqlChunk mapping

Every record (from either mode) is converted by `_to_sql_chunk()`:

| API field | SqlChunk field | Notes |
|-----------|----------------|-------|
| `id` | `id` | cast to `str` |
| `text` | `text` | normalized SQL; used for embedding |
| `raw_query` | `raw_query` | defaults to `""` if absent |
| `source` | `source` | database / schema name |
| `execution_time_ms` | `execution_time_ms` | cumulative float; defaults to `0.0` |
| `calls` | `calls` | defaults to `1` |
| *(derived)* | `content_hash` | `sha256(text)[:16]` — computed locally |
| `tables` | `tables` | defaults to `[]`; `null` treated as `[]` |
| `latest_ts` | `latest_ts` | ISO 8601 string parsed to `datetime`; falls back to `now(UTC)` |
| `user` | `user` | optional; `None` if absent |
| `host` | `host` | optional; `None` if absent |
| `rows_sent` | `rows_sent` | optional int; `None` if absent |
| `rows_examined` | `rows_examined` | optional int; `None` if absent |
| `lock_time_sec` | `lock_time_sec` | optional float; `None` if absent |

`content_hash` is **never** returned by the server — dbs-vector derives it
locally, consistent with how `DuckDBChunker` and `SqlChunker` work.  The
existing SHA-256-based deduplication in `IngestionService` therefore works
identically for all three sources.

### Error handling

| Condition | Behaviour |
|-----------|-----------|
| `httpx` not installed | `logger.error` + early return (zero chunks) |
| HTTP 4xx / 5xx | `logger.error(status, url)` + return |
| Connection refused / timeout | `logger.error` + return |
| Row missing `id`, `text`, or `source` | `logger.debug` + `continue` (row skipped) |
| `latest_ts` absent or malformed | falls back to `datetime.now(UTC)` |

The chunker never raises — broken API responses result in zero or partial
output rather than a crash.

---

## Server Contract

The full API specification is in [`claude_api_contract.md`](../claude_api_contract.md).
Key points for a server implementor:

### Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | none | Connectivity probe; returns `{"status":"ok","version":"...","databases":[...]}` |
| `GET` | `/sql/queries` | Bearer | Primary ingestion — paginated aggregated queries |
| `POST` | `/sql/execute` | Bearer | Optional custom SELECT |
| `GET` | `/sql/databases` | Bearer | List available database names |

### Authentication

Every endpoint except `/health` requires:

```
Authorization: Bearer <token>
```

A missing or invalid token must return `401` with body:

```json
{"error": "invalid_token", "message": "...", "request_id": "..."}
```

### Aggregation contract

`GET /sql/queries` must return **one row per unique normalized query
fingerprint** — the server aggregates raw log lines before responding.
dbs-vector never receives duplicate raw rows.

The semantic equivalent of the DuckDB default query that `DuckDBChunker` runs:

```sql
SELECT
    fingerprint_id               AS id,
    arg_max(sanitized_sql, ts)   AS text,
    arg_max(sample_sql, ts)      AS raw_query,
    arg_max(db, ts)              AS source,
    SUM(query_time_sec) * 1000   AS execution_time_ms,
    COUNT(*)                     AS calls,
    arg_max(tables, ts)          AS tables,
    MAX(ts)                      AS latest_ts,
    arg_max(user, ts)            AS user,
    arg_max(host, ts)            AS host,
    arg_max(rows_sent, ts)       AS rows_sent,
    arg_max(rows_examined, ts)   AS rows_examined,
    arg_max(lock_time_sec, ts)   AS lock_time_sec
FROM slow_logs
GROUP BY fingerprint_id
```

### Pagination

Use **keyset pagination** on `(execution_time_ms DESC, id ASC)`.
`next_cursor` is a base64-encoded JSON of `{"execution_time_ms": <float>, "id": "<str>"}`.
Never use SQL `OFFSET` — it degrades with large datasets.

### Safety rules for `POST /sql/execute`

The server must:

1. Reject any non-`SELECT` statement with `422 unsafe_query`
2. Reject multi-statement inputs (`;` followed by a second statement)
3. Execute with a read-only DB connection or role
4. Enforce a hard `timeout_ms` cap regardless of what the client sends

---

## Unit Tests

The unit test suite lives in `tests/unit/test_api_chunker.py`.  All tests
use `unittest.mock.patch` — no real network connection is needed.

```bash
uv run pytest tests/unit/test_api_chunker.py -v
```

| Test | What it verifies |
|------|-----------------|
| `test_paginated_single_page` | Single page, `has_more=false` → correct `SqlChunk` fields |
| `test_paginated_two_pages` | Two pages — both consumed; `cursor` forwarded on page 2 |
| `test_paginated_empty_response` | `data=[]` → yields nothing, no error |
| `test_custom_query_success` | POST mode, columns+rows → `SqlChunk` |
| `test_custom_query_missing_required_fields` | Row missing `text` → skipped, no crash |
| `test_http_error_401` | HTTP 401 → logs error, yields nothing |
| `test_connection_error` | `httpx.ConnectError` → logs error, yields nothing |
| `test_missing_httpx_import` | `ImportError` on httpx → logs error, yields nothing |
| `test_nullable_fields_none` | Optional fields absent → `SqlChunk` fields are `None` |
| `test_database_param_sent` | `database` configured → appears in GET request params |

---

## Compatibility Checker

`scripts/check_remote_api.py` is a standalone script that verifies a remote
server correctly implements the contract before you wire it up to dbs-vector.
It has no pytest dependency and is not collected by the test suite.

### Running it

```bash
# With explicit flags
python scripts/check_remote_api.py \
    --base-url http://localhost:9000/api/v1 \
    --api-key  sk-dev

# Via environment variables
DBS_API_BASE_URL=http://localhost:9000/api/v1 \
DBS_API_KEY=sk-dev \
    python scripts/check_remote_api.py

# Longer timeout for slow servers
python scripts/check_remote_api.py \
    --base-url https://api.internal/api/v1 \
    --api-key  sk-prod \
    --timeout  60
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed (or only skipped) |
| `1` | One or more checks failed |
| `2` | Argument / setup error |

### What it checks

The script runs 46 checks across 9 sections:

| Section | Checks |
|---------|--------|
| **1. Health** | `/health` returns 200; body has `status=ok`, `version`, `databases`; no auth required |
| **2. Authentication** | No token → 401; wrong token → 401; error body has `error=invalid_token` |
| **3. `/sql/queries` structure** | Status 200; `Content-Type: application/json`; `X-Request-Id` header; `data` / `has_more` / `total_count` fields; each record has all required fields, correct types, non-null required values; `tables` is always a list; `latest_ts` is ISO 8601 |
| **4. Pagination** | `limit=1` respected; `next_cursor` present when `has_more=true`; page 2 fetches correctly; no duplicate IDs across pages |
| **5. Filters** | `min_execution_ms` honored; `since=2099` returns empty; `database` filter returns only matching `source` |
| **6. `/sql/databases`** | Returns 200; `databases` list present; consistent with `/health` |
| **7. `/sql/execute` happy path** | Valid `SELECT` returns 200; `columns` / `rows` / `row_count` / `truncated` / `execution_time_ms` present; `row_count == len(rows)`; row arity matches columns |
| **8. Safety enforcement** | `INSERT`, `UPDATE`, `DELETE`, `DROP` each return 422 with `error=unsafe_query`; multi-statement input rejected |
| **9. Error envelope** | 4xx responses have `error` and `message` fields |

Checks that cannot run because data is unavailable (e.g. empty dataset,
server returned no databases) are automatically **skipped** with a reason
rather than marked as failed.

### Sample output

```
dbs-vector Remote API — Compatibility Check
  Base URL : http://localhost:9000/api/v1
  API key  : sk-dev…
  Timeout  : 30s

── 1. Health  GET /health
  ✓ [PASS] health_returns_200
  ✓ [PASS] health_status_is_ok
  ✓ [PASS] health_has_version_string
  ✓ [PASS] health_has_databases_list
  ✓ [PASS] health_no_auth_required

── 2. Authentication
  ✓ [PASS] auth_no_token_returns_401
  ✓ [PASS] auth_no_token_error_is_invalid_token
  ✓ [PASS] auth_wrong_token_returns_401

── 3. GET /sql/queries — structure
  ✓ [PASS] queries_returns_200
  ✓ [PASS] queries_content_type_json
  ✗ [FAIL] queries_x_request_id_header
         X-Request-Id header missing
  ✓ [PASS] queries_has_data_array
  ...

────────────────────────────────────────────────────────
Results
  41/43 passed   1 failed   2 skipped

Failed checks:
  • queries_x_request_id_header: X-Request-Id header missing
```

---

## Searching After Ingestion

Once records are ingested, searching works identically to the DuckDB-backed
`sql` engine — the underlying LanceDB table and `SqlMapper` are shared:

```bash
# Natural language search
uv run dbs-vector search "expensive joins on the orders table" --type sql-api

# Filter by database
uv run dbs-vector search "user lookup queries" --type sql-api --source production

# Filter by minimum execution time
uv run dbs-vector search "slow aggregates" --type sql-api --min-time 5000
```

The engine can also be exposed via the MCP server for use with AI assistants:

```bash
uv run dbs-vector mcp
```

See [`README_MCP.md`](README_MCP.md) for MCP integration details and
[`README_duckdb.md`](README_duckdb.md) for AI-assisted slow-query analysis
prompt examples that apply equally to API-ingested data.

---

## Comparison: ApiChunker vs DuckDBChunker

| Aspect | `DuckDBChunker` | `ApiChunker` |
|--------|-----------------|--------------|
| Input | Local `.duckdb` file | Remote HTTP endpoint |
| Aggregation | Runs SQL query locally | Server-side, pre-aggregated |
| File access | Direct read | None |
| Transport | — | HTTP/HTTPS + gzip |
| Pagination | Single query, `LIMIT 500` | Keyset cursor, configurable page size |
| Custom extraction | `--query` flag runs SQL locally | `--query` flag sends SQL to `POST /sql/execute` |
| Extra dependency | `duckdb` (`[sql]` extra) | `httpx` (`[api]` extra) |
| Deduplication | SHA-256 content hash | SHA-256 content hash (identical) |
| Registry key | `"duckdb"` | `"api"` |

Both chunkers yield `SqlChunk` objects and flow into the same
`IngestionService` → `SqlMapper` → `LanceDB` pipeline.

---

## Files Added or Modified

| File | Change |
|------|--------|
| `src/dbs_vector/infrastructure/chunking/api.py` | New — `ApiChunker` implementation |
| `src/dbs_vector/core/registry.py` | Added `"api": ApiChunker` |
| `src/dbs_vector/config.py` | Added 7 optional `api_*` fields to `EngineConfig` |
| `src/dbs_vector/services/ingestion.py` | URL bypass at top of `_chunk_generator()` |
| `src/dbs_vector/cli.py` | `elif chunker_type == "api"` branch in `_build_dependencies` |
| `pyproject.toml` | `api = ["httpx>=0.27.0"]` optional extra; mypy ignore for httpx |
| `tests/unit/test_api_chunker.py` | New — 10 unit tests (all mock-based) |
| `scripts/check_remote_api.py` | New — standalone compatibility checker (46 checks) |
| `claude_api_contract.md` | Contract spec for remote server implementors |
