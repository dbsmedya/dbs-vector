# DuckDB Ingestion for dbs-vector

This document covers the details of using DuckDB as a fast, columnar data source for the `sql` engine in `dbs-vector`.

## Overview

The `DuckDBChunker` reads pre-processed SQL slow query data from `.duckdb` database files directly into the dbs-vector ingestion pipeline. It serves as an alternative to flat JSON files, offering massive space savings and extremely fast analytical queries.

DuckDB files are highly optimized for this type of workload because of their columnar storage and dictionary encoding of repeated SQL fragments.

## Requirements

The DuckDB integration is an optional extra. To use it, you must install the `sql` dependencies:

```bash
uv pip install "dbs-vector[sql]"
# Or directly via uv add
uv add "duckdb>=1.2.0"
```

## How It Works

When `dbs-vector` encounters a file with the `.duckdb` extension during an `ingest` command, it will skip UTF-8 decoding and instead use the DuckDB Python library to connect to the file in `read_only` mode. 

By default, the chunker executes an internal query designed to aggregate identical queries (by `fingerprint_id`), rolling up metadata and computing total impact time over a recent 15-day window.

### Default Aggregation Strategy

The default aggregation query prioritizes queries by total time impact (duration multiplied by call frequency) over the last 15 days. The exact logic translates to:

1. **Group by:** `fingerprint_id`
2. **Sort by:** `execution_time_ms DESC`
3. **Limit:** Top 500 queries
4. **Aggregations:**
   * `text` (sanitized query): Most recent value (`arg_max(..., ts)`)
   * `source` (database name): Most recent value
   * `raw_query`: Most recent sample
   * `execution_time_ms`: Sum of `query_time_sec * 1000`
   * `calls`: Total rows for that fingerprint
   * `tables`, `user`, `host`: Most recent values
   * `rows_sent`, `rows_examined`, `lock_time_sec`: Most recent values

Here is the exact SQL executed by default:

```sql
SELECT 
    fingerprint_id as id,
    arg_max(sanitized_sql, ts) as text,
    arg_max(sample_sql, ts) as raw_query,
    arg_max(db, ts) as source,
    SUM(query_time_sec) * 1000 as execution_time_ms,
    COUNT(*) as calls,
    arg_max("tables", ts) as tables,
    MAX(ts) as latest_ts,
    arg_max("user", ts) as user,
    arg_max(host, ts) as host,
    arg_max(rows_sent, ts) as rows_sent,
    arg_max(rows_examined, ts) as rows_examined,
    arg_max(lock_time_sec, ts) as lock_time_sec
FROM slow_logs
WHERE ts > current_date - INTERVAL '15 days'
GROUP BY fingerprint_id
ORDER BY execution_time_ms DESC
LIMIT 500
```

## Configuration

You can override the default query by editing your `config.yaml` to specify a custom `duckdb_query` for the engine:

```yaml
engines:
  sql:
    description: "SQL Slow Query Log Engine"
    model_name: "mlx-community/embeddinggemma-300m-bf16"
    vector_dimension: 768
    max_token_length: 2048
    table_name: "query_vault"
    mapper_type: "sql"
    chunker_type: "duckdb"  # Important: Must be set to duckdb
    chunk_max_chars: 0
    # Custom Query Override:
    duckdb_query: >
      SELECT 
        fingerprint_id as id,
        arg_max(sanitized_sql, ts) as text,
        arg_max(sample_sql, ts) as raw_query,
        arg_max(db, ts) as source,
        SUM(query_time_sec) * 1000 as execution_time_ms,
        COUNT(*) as calls
      FROM slow_logs
      GROUP BY fingerprint_id
      ORDER BY calls DESC
      LIMIT 100
```

### Required Fields for Custom Queries
If you provide a custom query, your result set **must** include the following columns:
* `id` (string): Unique identifier for the chunk.
* `text` (string): The text that will be embedded and searched against.
* `source` (string): Used for filtering results.
* `raw_query` (string): Usually a sample of the actual query executed.
* `execution_time_ms` (float): Total or average execution time in milliseconds.
* `calls` (int): Number of times the query was executed.

**Optional Fields:** `tables` (list[str]), `latest_ts` (datetime), `user` (str), `host` (str), `rows_sent` (int), `rows_examined` (int), `lock_time_sec` (float).

## Ad-hoc CLI Extraction

For ad-hoc analysis or testing a new query without modifying `config.yaml`, you can pass a query directly via the `--query` (or `-q`) flag during ingestion. This takes precedence over any query defined in `config.yaml`.

```bash
uv run dbs-vector ingest ./prd/oracle_test.duckdb --type sql --rebuild -q "SELECT fingerprint_id as id, arg_max(sanitized_sql, ts) as text, arg_max(db, ts) as source, COUNT(*) as calls, SUM(query_time_sec)*1000 as execution_time_ms, arg_max(sample_sql, ts) as raw_query FROM slow_logs GROUP BY fingerprint_id ORDER BY calls DESC LIMIT 10"
```

## Expected Schema

The chunker is designed to work with an upstream Go preprocessor that writes to the following DuckDB schema:

```sql
CREATE TABLE slow_logs(
    id BIGINT DEFAULT(nextval('seq_slow_log_id')) PRIMARY KEY, 
    file_hash VARCHAR, 
    fingerprint_id VARCHAR NOT NULL, 
    sanitized_sql VARCHAR NOT NULL, 
    sample_sql VARCHAR NOT NULL, 
    "user" VARCHAR, 
    host VARCHAR, 
    db VARCHAR, 
    query_time_sec DOUBLE, 
    lock_time_sec DOUBLE, 
    rows_sent UBIGINT, 
    rows_examined UBIGINT, 
    ts TIMESTAMP, 
    created_at TIMESTAMP DEFAULT(CURRENT_TIMESTAMP), 
    "tables" VARCHAR[]
);
```

## Examples

Below are various examples of how you can ingest and query your DuckDB SQL logs:

### Ingestion

1. **Standard Ingestion:**
   ```bash
   uv run dbs-vector ingest ./prd/oracle_test.duckdb --type sql
   ```

2. **Rebuild the Entire Vector Store from a DuckDB file:**
   ```bash
   uv run dbs-vector ingest ./prd/oracle_test.duckdb --type sql --rebuild -f
   ```

3. **Ingest with a Custom Query (Ad-hoc Analysis):**
   ```bash
   uv run dbs-vector ingest ./prd/oracle_test.duckdb --type sql -q "SELECT fingerprint_id as id, arg_max(sanitized_sql, ts) as text, MAX(db) as source, COUNT(*) as calls FROM slow_logs GROUP BY fingerprint_id LIMIT 50"
   ```

### Searching (SQL Clustering)

4. **Basic Natural Language Search against SQL Logs:**
   ```bash
   uv run dbs-vector search "magento orders with picking type" --type sql
   ```

5. **Find Queries by Table Name:**
   ```bash
   uv run dbs-vector search "queries updating the magentoorders table" --type sql
   ```

6. **Filter by Execution Time:**
   ```bash
   uv run dbs-vector search "select from deliverycompany" --type sql --min-time 5000.0
   ```

7. **Limit the Number of Results Returned:**
   ```bash
   uv run dbs-vector search "warehouse stock sync" --type sql --limit 10
   ```

8. **Filter by Database Source (if your custom query populated 'source'):**
   ```bash
   uv run dbs-vector search "update branch info" --type sql --source "db_production"
   ```

9. **Hybrid Search (Matching specific column patterns):**
   ```bash
   uv run dbs-vector search "SELECT id, status FROM MagentoOrders" --type sql
   ```

10. **Broad Aggregation Searches:**
    ```bash
    uv run dbs-vector search "count aggregate queries on large tables" --type sql
    ```

## AI Optimization Workflows (MCP Server)

When `dbs-vector` is running as an **MCP (Model Context Protocol) Server**, you can use AI assistants (like Claude Desktop or Cursor) to autonomously fetch and analyze slow queries. 

Because the vector store clusters conceptually similar queries and includes execution telemetry (time, rows examined, locks), the LLM can spot architectural anti-patterns. 

Here are **5 prompt examples** you can give your AI assistant to kick off an optimization investigation:

### 1. Identify Missing Indexes
> "Use the search_sql_logs tool to find the most frequent SELECT queries hitting the `MagentoOrders` table. Analyze the ratio of `rows_examined` vs `rows_sent` in the results. If a query examines thousands of rows but only returns a few, suggest the exact `CREATE INDEX` statement needed to optimize it."

### 2. N+1 Query Detection
> "Search the SQL logs for simple `SELECT * FROM CustomerAddress WHERE id = ?` or similar exact-match lookups. Look at the `Calls` count. If you see an extremely high call count with a very low average execution time, this might be an N+1 query issue in the ORM. Explain how to refactor the application code to use a bulk `WHERE id IN (...)` fetch instead."

### 3. Subquery & Join Optimization
> "Search the SQL logs for 'complex joins and lateral subqueries'. Look for queries that have a high `execution_time_ms`. Review the `raw_query` sample and suggest a refactoring strategy. Could the subqueries be replaced with `LEFT JOIN`s, or should we introduce a materialized view for this data?"

### 4. Lock Contention Analysis
> "Use the SQL search tool to look for `UPDATE` or `INSERT` statements on the `DeliveryCompany` table. Check if any of these returned chunks have a high `lock_time_sec`. If so, analyze the transaction pattern and suggest ways to reduce the lock footprint (e.g., granular locking, batching updates, or moving heavy reads to a read-replica)."

### 5. Architectural Refactoring (Clustering)
> "Search the SQL logs for 'aggregate count and sum queries on orders'. Group the returned queries by their conceptual intent. Are we repeatedly calculating the same metrics across different endpoints? Suggest a Redis caching strategy or a background cron job pattern to pre-calculate these values rather than running them dynamically on every page load."