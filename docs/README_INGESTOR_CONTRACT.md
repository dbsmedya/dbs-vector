# dbs-vector Remote SQL API — Contract v1.1

> **Purpose**: This document specifies the HTTP API that a remote server must implement so that
> dbs-vector's `ApiChunker` can pull, ingest, and optionally query SQL slow-log data over the
> network. It is written from the perspective of what dbs-vector _needs_, so the implementor of
> the remote endpoint can build exactly what is consumed — no more, no less.

> **v1.1 (2026-06-13):** adds the **required** `tables_original` field to `/sql/queries`
> records — original-case, schema-qualified table names so a fingerprint can be replayed
> against a case-sensitive MySQL server (`lower_case_table_names=0`). Additive (the URL stays
> `/api/v1`). See §1, §5.2, §11. Enforced by `dbs-vector/scripts/check_remote_api.py`.

---

## 1. Background & Integration Model

### How dbs-vector ingests data today

```
File / DuckDB ──► IChunker.process(Document) ──► Iterator[SqlChunk]
                                                        │
                                              IngestionService
                                                        │
                                    dedup (SHA-256 content_hash) ──► embed ──► LanceDB
```

The `ApiChunker` replaces the file/DuckDB source. It fetches pages of pre-aggregated query
records from the remote server and yields `SqlChunk` objects in exactly the same way.
Everything downstream (dedup, embedding, LanceDB storage) is unchanged.

### SqlChunk — the canonical domain object

Every record returned by the API must map to this Pydantic model
(`src/dbs_vector/core/models.py`):

| Field              | Type             | Nullable | Description                                             |
|--------------------|------------------|----------|---------------------------------------------------------|
| `id`               | `str`            | no       | Stable fingerprint/hash of the normalized query         |
| `text`             | `str`            | no       | **Normalized / sanitized SQL** — used for embedding     |
| `raw_query`        | `str`            | no       | One concrete sample of the raw SQL with real values     |
| `source`           | `str`            | no       | Database / schema name                                  |
| `execution_time_ms`| `float`          | no       | **Cumulative** total execution time across all `calls`  |
| `calls`            | `int`            | no       | Total number of times this fingerprint was observed     |
| `tables`           | `list[str]`      | no       | **Normalized** (lowercased, schema-qualified) table names (can be `[]`) |
| `tables_original`  | `list[str]`      | no       | **Original-case**, schema-qualified table names — same set as `tables`, for replay (v1.1; can be `[]`) |
| `latest_ts`        | `datetime`       | no       | Timestamp of the most recent execution                  |
| `user`             | `str \| null`    | yes      | DB user that ran the query                              |
| `host`             | `str \| null`    | yes      | Client host / IP                                        |
| `rows_sent`        | `int \| null`    | yes      | Rows returned to the client (from most recent call)     |
| `rows_examined`    | `int \| null`    | yes      | Rows scanned by the engine (from most recent call)      |
| `lock_time_sec`    | `float \| null`  | yes      | Lock wait time in seconds (from most recent call)       |

`content_hash` is **NOT** returned by the API — dbs-vector derives it locally:
```python
content_hash = hashlib.sha256(record["text"].encode()).hexdigest()[:16]
```

---

## 2. Base URL & Versioning

```
https://<host>/api/v1
```

- All paths below are relative to this base.
- Version is in the URL path. Breaking changes bump the version.
- The `ApiChunker` config field `api_base_url` points to the base URL including `/api/v1`.

---

## 3. Authentication

```
Authorization: Bearer <token>
```

- All endpoints require this header (except `/health`).
- Token is a static bearer token configured in `config.yaml` → `api_key`.
- The server must return `401 Unauthorized` with `{"error": "invalid_token"}` on failure.
- No OAuth / refresh flow is needed for the initial version.

---

## 4. Common Conventions

### Request headers

| Header            | Value                    | Required |
|-------------------|--------------------------|----------|
| `Authorization`   | `Bearer <token>`         | yes      |
| `Accept-Encoding` | `gzip`                   | recommended |
| `Content-Type`    | `application/json`       | POST only |

### Response headers

| Header             | Value                                       |
|--------------------|---------------------------------------------|
| `Content-Type`     | `application/json; charset=utf-8`           |
| `Content-Encoding` | `gzip` (when client sent `Accept-Encoding`) |
| `X-Request-Id`     | UUID for tracing                            |

### Error envelope

All 4xx / 5xx responses use:

```json
{
  "error": "machine_readable_code",
  "message": "Human readable description.",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Common error codes:

| HTTP | `error`              | Meaning                                              |
|------|----------------------|------------------------------------------------------|
| 400  | `invalid_param`      | Bad query parameter or request body field            |
| 401  | `invalid_token`      | Missing or invalid bearer token                      |
| 403  | `forbidden`          | Token valid but lacks permission for this resource   |
| 408  | `query_timeout`      | Custom SQL exceeded `timeout_ms`                     |
| 422  | `unsafe_query`       | Non-SELECT or DDL detected in custom SQL             |
| 429  | `rate_limited`       | Too many requests; include `Retry-After` header      |
| 500  | `internal_error`     | Server fault                                         |
| 503  | `db_unavailable`     | Upstream DB unreachable                              |

---

## 5. Endpoints

### 5.1 Health Check

```
GET /health
```

No authentication required. Used by dbs-vector at startup to verify connectivity.

**Response 200**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "databases": ["production", "analytics", "reporting"]
}
```

`databases` is the list of source names available for filtering — mirrors the `source` field in
SqlChunk. dbs-vector uses this only for informational logging.

---

### 5.2 List Aggregated Queries  ← primary ingestion endpoint

```
GET /sql/queries
```

Returns slow-query records **pre-aggregated by query fingerprint** (one row per unique normalized
query). The server is responsible for the aggregation — dbs-vector does not do it.

This is the direct API equivalent of the DuckDB default query:
```sql
SELECT
    fingerprint_id          AS id,
    arg_max(sanitized_sql)  AS text,
    arg_max(sample_sql)     AS raw_query,
    arg_max(db)             AS source,
    SUM(query_time_sec)*1000 AS execution_time_ms,
    COUNT(*)                AS calls,
    arg_max(tables)          AS tables,
    arg_max(tables_original) AS tables_original,
    MAX(ts)                  AS latest_ts,
    arg_max(user)           AS user,
    arg_max(host)           AS host,
    arg_max(rows_sent)      AS rows_sent,
    arg_max(rows_examined)  AS rows_examined,
    arg_max(lock_time_sec)  AS lock_time_sec
FROM slow_logs
GROUP BY fingerprint_id
```

#### Query parameters

| Parameter        | Type     | Default      | Description                                                    |
|------------------|----------|--------------|----------------------------------------------------------------|
| `limit`          | int      | `200`        | Records per page. Max `1000`.                                  |
| `cursor`         | str      | —            | Opaque pagination token from previous response `next_cursor`.  |
| `since`          | ISO8601  | `now - 15d`  | Lower bound on `latest_ts` (inclusive).                        |
| `until`          | ISO8601  | `now`        | Upper bound on `latest_ts` (inclusive).                        |
| `min_execution_ms` | float  | `0`          | Filter: cumulative `execution_time_ms >= value`.               |
| `database`       | str      | —            | Filter to a single source/database name.                       |
| `known_hashes`   | str      | —            | Comma-separated list of `content_hash` values dbs-vector already has. Server omits matching records. **Delta sync optimization.** |

**Performance note on `known_hashes`**: dbs-vector will send up to `batch_size` (default 64)
hashes per request when doing incremental syncs. The server should index its
`sha256(text)[:16]` column to make this filter cheap. If the dataset grows large, dbs-vector
will send hashes across multiple requests.

#### Response 200

```json
{
  "data": [
    {
      "id": "a3f8c2d1e4b09712",
      "text": "SELECT id, email FROM users WHERE status = ?",
      "raw_query": "SELECT id, email FROM users WHERE status = 'active'",
      "source": "production",
      "execution_time_ms": 84230.5,
      "calls": 14872,
      "tables": ["users"],
      "tables_original": ["Users"],
      "latest_ts": "2026-03-04T22:15:30Z",
      "user": "app_ro",
      "host": "10.0.1.22",
      "rows_sent": 1,
      "rows_examined": 48300,
      "lock_time_sec": 0.0
    }
  ],
  "next_cursor": "eyJleGVjdXRpb25fdGltZV9tcyI6ODQyMzAuNSwiaWQiOiJhM2Y4YzJkMWU0YjA5NzEyIn0=",
  "has_more": true,
  "total_count": 4821
}
```

| Field          | Description                                                               |
|----------------|---------------------------------------------------------------------------|
| `data`         | Array of query records; may be empty on last page                         |
| `next_cursor`  | Opaque base64 token for next page; absent or `null` when `has_more=false` |
| `has_more`     | Whether more pages exist                                                   |
| `total_count`  | Total matching records (for progress reporting); may be approximate        |

#### `tables` vs `tables_original` (v1.1)

`tables` is **normalized** (lowercased, schema-qualified) — the matching/dedup key dbs-vector
filters on. `tables_original` carries the **original case** of the same tables (e.g.
`Orders`, not `orders`) so a fingerprint can be replayed against a case-sensitive
MySQL server. The two describe the **same set** of tables; treat them as **set-equivalent, not
positionally 1:1** — both are independent `arg_max` aggregates, and during a backfill window
historical rows may carry a null `tables_original`. Recover the pairing by lowercasing an
original-case name (it equals the corresponding normalized entry).

#### Cursor design (server implementation guide)

Use **keyset pagination** on `(execution_time_ms DESC, id ASC)` — never OFFSET, which degrades
with large datasets:

```sql
-- First page
WHERE latest_ts >= :since AND latest_ts <= :until
  AND execution_time_ms >= :min_execution_ms
ORDER BY execution_time_ms DESC, id ASC
LIMIT :limit

-- Subsequent pages (cursor decoded to {execution_time_ms, id})
WHERE latest_ts >= :since AND latest_ts <= :until
  AND execution_time_ms >= :min_execution_ms
  AND (execution_time_ms < :cursor_ms OR (execution_time_ms = :cursor_ms AND id > :cursor_id))
ORDER BY execution_time_ms DESC, id ASC
LIMIT :limit
```

The cursor is a base64-encoded JSON of `{"execution_time_ms": <float>, "id": "<str>"}`.

---

### 5.3 Execute Custom SQL  ← optional, read-only

```
POST /sql/execute
```

Executes a caller-supplied SELECT against the server's underlying slow-log data source. This is
the API equivalent of the DuckDB `--query` CLI flag. The remote server **must** enforce
read-only access — no INSERT, UPDATE, DELETE, DROP, or SET statements.

#### Request body

```json
{
  "query": "SELECT fingerprint_id AS id, sanitized_sql AS text, db AS source, SUM(query_time_sec)*1000 AS execution_time_ms, COUNT(*) AS calls FROM slow_logs WHERE db = 'analytics' GROUP BY fingerprint_id, sanitized_sql, db ORDER BY execution_time_ms DESC LIMIT 100",
  "database": "analytics",
  "timeout_ms": 10000
}
```

| Field        | Type   | Required | Description                                              |
|--------------|--------|----------|----------------------------------------------------------|
| `query`      | str    | yes      | The SQL to execute. Must be a single SELECT statement.   |
| `database`   | str    | no       | Target database/schema if the server manages multiple.   |
| `timeout_ms` | int    | no       | Query timeout. Server default applies if omitted. Max server-enforced cap applies. |

#### Safety rules (server-side enforcement)

The server must:
1. Parse the SQL and reject if the top-level statement is not `SELECT` → `422 unsafe_query`
2. Strip or reject any semicolons followed by additional statements (SQLi prevention)
3. Execute with a read-only DB connection or role (e.g., `SET TRANSACTION READ ONLY`)
4. Enforce a hard `timeout_ms` cap (recommended: 30 000 ms) regardless of what the client sends
5. Never expose system tables, credentials, or internal schemas

#### Response 200 — tabular format

```json
{
  "columns": ["id", "text", "source", "execution_time_ms", "calls"],
  "rows": [
    ["a3f8c2d1e4b09712", "SELECT id FROM users WHERE ?", "analytics", 4230.5, 88],
    ["b7e1a9f3c2d04851", "INSERT INTO ...", "analytics", 1100.0, 12]
  ],
  "row_count": 2,
  "truncated": false,
  "execution_time_ms": 45
}
```

| Field              | Description                                                         |
|--------------------|---------------------------------------------------------------------|
| `columns`          | Ordered list of column names                                        |
| `rows`             | Ordered list of value arrays, parallel to `columns`                 |
| `row_count`        | Number of rows returned (after truncation if applied)               |
| `truncated`        | `true` if the server applied a row cap (e.g. server-side LIMIT)     |
| `execution_time_ms`| Wall time the server spent executing the query                      |

**Note for dbs-vector ApiChunker**: when using `POST /sql/execute` as an ingestion source, the
custom query **must** return columns matching the SqlChunk schema (at minimum: `id`, `text`,
`source`, `execution_time_ms`, `calls`). The ApiChunker will skip rows missing required fields
and log a warning, matching the behaviour of `DuckDBChunker`.

---

### 5.4 List Available Databases

```
GET /sql/databases
```

Returns the database/schema names available on the server. Useful for validating the `database`
filter before ingestion.

**Response 200**
```json
{
  "databases": ["production", "analytics", "reporting", "staging"]
}
```

---

## 6. Pagination Flow — dbs-vector Perspective

```
ApiChunker.process(Document(filepath=api_base_url))
    │
    ├─► GET /sql/queries?limit=200&since=2026-02-18T00:00:00Z&known_hashes=<64 hashes>
    │       → yields SqlChunk for each record in data[]
    │
    ├─► GET /sql/queries?limit=200&cursor=<next_cursor>&known_hashes=<64 hashes>
    │       → yields SqlChunk for each record in data[]
    │
    └─► ... until has_more=false
```

dbs-vector reads pages in sequence (not parallel) because the downstream embedder
(`MLXEmbedder`) is already GPU-bound. Parallel page fetching would only queue up RAM.

---

## 7. Performance & Efficiency Design Decisions

### 7.1 Server-side aggregation
The server aggregates by fingerprint **before** responding. dbs-vector never receives duplicate
raw log rows. This reduces payload size by 10–100× compared to streaming raw log lines.

### 7.2 Server-side delta filtering via `known_hashes`
When `known_hashes` is supplied, the server excludes already-indexed records. This means
incremental ingestion only transfers new queries — no bandwidth wasted on already-vectorized
data. The client (dbs-vector) sends up to `batch_size` hashes per request.

Alternatively (if `known_hashes` is not implemented server-side), dbs-vector's existing
content-hash deduplication in `IngestionService` still prevents re-ingestion — it just uses
more bandwidth.

### 7.3 Gzip compression
Normalized SQL text compresses very well (repetitive tokens). A 200-record page typically
compresses to < 20 KB with gzip. Always enable it.

### 7.4 Page size recommendation
- Default `limit=200` balances latency and throughput.
- The dbs-vector `batch_size` (default 64) controls embedding batch size, not fetch page size.
  The ApiChunker will internally buffer the API page and yield one `SqlChunk` at a time into
  the `_batched()` pipeline of `IngestionService`.
- Large `limit` values (500–1000) are safe as long as server query time stays under 2s.

### 7.5 Ordering by `execution_time_ms DESC`
The most expensive queries are ingested first. If ingestion is interrupted, the most impactful
queries are already in LanceDB and searchable. This also makes keyset pagination stable.

### 7.6 `total_count` for progress reporting
Return an approximate count (e.g., `COUNT(DISTINCT fingerprint_id)`) so dbs-vector can log
progress. Exact counts are not required.

---

## 8. config.yaml Extension for ApiChunker

```yaml
engines:
  sql-api:
    description: "Remote slow query log via HTTP API"
    model_name: "mlx-community/gemma-2-2b-it-4bit"
    vector_dimension: 2048
    max_token_length: 512
    table_name: "query_vault"
    mapper_type: "sql"
    chunker_type: "api"           # new registry key
    chunk_max_chars: 0
    query_prefix: "query: "
    passage_prefix: "passage: "
    workflow: "gemma2-sql-v1"

    # ApiChunker-specific fields
    api_base_url: "https://slow-log-api.internal/api/v1"
    api_key: "sk-..."             # injected from env: DBS_API_KEY
    api_page_size: 200
    api_since_days: 15            # equivalent to DuckDB default INTERVAL '15 days'
    api_timeout_sec: 30
    api_min_execution_ms: 0       # maps to min_execution_ms query param
    api_database: ""              # leave empty to fetch all databases
```

---

## 9. ApiChunker — Expected dbs-vector Implementation Sketch

This section describes what dbs-vector needs to implement (not the remote server):

```python
class ApiChunker:
    """Fetches pre-aggregated SQL slow-query records from a remote HTTP API."""

    supported_extensions = [".api"]  # virtual — path arg is the base URL

    def __init__(
        self,
        base_url: str,
        api_key: str,
        page_size: int = 200,
        since_days: int = 15,
        timeout_sec: int = 30,
        min_execution_ms: float = 0,
        database: str | None = None,
        custom_query: str | None = None,  # triggers POST /sql/execute
    ) -> None: ...

    def process(self, document: Document) -> Iterator[SqlChunk]:
        # document.filepath == base_url (passed through by IngestionService)
        # document.content is empty for API chunker
        if self.custom_query:
            yield from self._execute_custom(self.custom_query)
        else:
            yield from self._paginate()

    def _paginate(self) -> Iterator[SqlChunk]:
        cursor = None
        while True:
            page = self._get_page(cursor)
            for record in page["data"]:
                yield self._to_sql_chunk(record)
            if not page["has_more"]:
                break
            cursor = page["next_cursor"]

    def _to_sql_chunk(self, record: dict) -> SqlChunk:
        text = record["text"]
        return SqlChunk(
            id=record["id"],
            text=text,
            raw_query=record.get("raw_query", ""),
            source=record["source"],
            execution_time_ms=float(record.get("execution_time_ms", 0.0)),
            calls=int(record.get("calls", 1)),
            content_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            tables=record.get("tables", []),
            latest_ts=datetime.fromisoformat(record["latest_ts"]),
            user=record.get("user"),
            host=record.get("host"),
            rows_sent=record.get("rows_sent"),
            rows_examined=record.get("rows_examined"),
            lock_time_sec=record.get("lock_time_sec"),
        )
```

The CLI invocation:
```bash
# Standard ingestion (paginated GET /sql/queries)
uv run dbs-vector ingest "https://api.internal/api/v1" --type sql-api

# Custom query (POST /sql/execute)
uv run dbs-vector ingest "https://api.internal/api/v1" --type sql-api \
  --query "SELECT fingerprint_id AS id, sanitized_sql AS text, ..."
```

---

## 10. OpenAPI Summary

```yaml
openapi: "3.1.0"
info:
  title: dbs-vector Remote SQL API
  version: "1.0.0"

paths:
  /health:
    get:
      summary: Health check
      security: []

  /sql/queries:
    get:
      summary: List aggregated slow queries (paginated)
      parameters:
        - name: limit
          in: query
          schema: { type: integer, default: 200, maximum: 1000 }
        - name: cursor
          in: query
          schema: { type: string }
        - name: since
          in: query
          schema: { type: string, format: date-time }
        - name: until
          in: query
          schema: { type: string, format: date-time }
        - name: min_execution_ms
          in: query
          schema: { type: number, default: 0 }
        - name: database
          in: query
          schema: { type: string }
        - name: known_hashes
          in: query
          schema: { type: string, description: "Comma-separated SHA-256[:16] hashes" }

  /sql/execute:
    post:
      summary: Execute a custom read-only SQL query
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [query]
              properties:
                query:
                  type: string
                database:
                  type: string
                timeout_ms:
                  type: integer
                  default: 10000

  /sql/databases:
    get:
      summary: List available database names
```

---

## 11. Implementation Checklist for Remote Server

- [ ] `GET /health` returns `{"status": "ok", "databases": [...]}`
- [ ] `GET /sql/queries` aggregates by fingerprint server-side
- [ ] Keyset pagination on `(execution_time_ms DESC, id ASC)` — no OFFSET
- [ ] `known_hashes` filter excludes already-indexed records
- [ ] Gzip encoding supported
- [ ] `POST /sql/execute` rejects non-SELECT statements with `422`
- [ ] `POST /sql/execute` runs with a read-only DB connection/role
- [ ] `POST /sql/execute` enforces `timeout_ms` with a hard server cap
- [ ] All endpoints require `Authorization: Bearer` except `/health`
- [ ] `latest_ts` is always ISO 8601 UTC (`2026-03-04T22:15:30Z`)
- [ ] `tables` is always a JSON array (never `null`, may be `[]`) — normalized (lowercased, schema-qualified)
- [ ] `tables_original` (v1.1) is always a JSON array (never `null`, may be `[]`) — original-case, schema-qualified, same set as `tables`
- [ ] `id` is stable across runs for the same normalized query fingerprint
- [ ] `text` contains the **normalized** SQL (parameters replaced with `?` or `$1`)
- [ ] `execution_time_ms` is the **cumulative sum**, not per-call average
