# SQL Vector Engine

The SQL engine in `dbs-vector` is specifically designed for analyzing and clustering database queries. It enables finding "semantically similar" slow queries, allowing developers to identify patterns in database performance bottlenecks.

## Overview
Unlike standard prose search, the SQL engine uses specialized code-native models (like `bigcode/starencoder`) to understand SQL structure. It relies on a separate schema in LanceDB that includes execution metrics like duration and call counts.

---

## Ingestion Format (JSON)
The SQL engine expects a JSON file containing an array of query records. This is typically exported from `pg_stat_statements` or slow query logs.

### Required JSON Fields:
| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The original, raw SQL query. |
| `normalized_query` | `string` | The structural version of the query (literals stripped). Used for embedding. |
| `database` | `string` | The name of the source database. |
| `duration` | `float` | Execution time in milliseconds. |
| `calls` | `integer` | Number of times the query was executed. |

### Example Input File (`queries.json`):
```json
[
  {
    "query": "SELECT * FROM users WHERE id = 42",
    "normalized_query": "SELECT * FROM users WHERE id = ?",
    "query_hash": "abc123hash",
    "database": "production_db",
    "duration": 1250.5,
    "calls": 500
  },
  {
    "query": "SELECT * FROM users WHERE id = 101",
    "normalized_query": "SELECT * FROM users WHERE id = ?",
    "query_hash": "abc123hash",
    "database": "production_db",
    "duration": 1100.2,
    "calls": 30
  }
]
```

---

## Command Line Usage

### 1. Ingesting SQL Logs
To index your JSON log file into the `query_vault` table:
```bash
uv run dbs-vector ingest "path/to/slow_queries.json" --type sql
```

### 2. Searching for Similar Slow Queries
You can search using a raw SQL string to find clusters of similar slow queries:
```bash
uv run dbs-vector search "SELECT * FROM users" --type sql --min-time 1000
```
*   `--min-time`: Filter results to only include queries that took longer than the specified milliseconds.

---

## Why use Vector Search for SQL?
Traditional structural analysis often relies on exact hash matching of normalized queries. While useful, it misses queries that are **logically identical but structurally different** (e.g., a `JOIN` written in a different order, or different aliasing). 

Vector similarity captures the **semantic intent** of the SQL, grouping together queries that touch the same tables and indices even if the syntax varies slightly.
