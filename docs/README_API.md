# dbs-vector API Documentation

This document outlines the API specifications and usage for the `dbs-vector` search engine.

## Overview
The `dbs-vector` API provides a high-performance, asynchronous REST interface for hybrid (Vector + Full-Text) codebase search. It is built using **FastAPI** and runs on a local Uvicorn ASGI server.

To minimize latency and maximize performance, the application leverages Apple's Unified Memory Architecture (UMA). The MLX embedding model and LanceDB connection are initialized *once* during the application's startup lifecycle, remaining resident in memory across requests.

## Starting the Server
You can launch the API server using the Typer CLI:

```bash
uv run dbs-vector serve
```

### Options
*   `--host` / `-h`: The host interface to bind to (Default: `127.0.0.1`).
*   `--port` / `-p`: The port to bind to (Default: `8000`).
*   `--reload`: Enable auto-reload for local development.

By default, the server will start at `http://127.0.0.1:8000`. 
FastAPI automatically generates interactive Swagger UI documentation, accessible at `http://127.0.0.1:8000/docs`.

---

## Endpoints

### 1. Health Check
Checks the initialization status of the API and the MLX model.

*   **URL:** `/health`
*   **Method:** `GET`

**Success Response (200 OK):**
```json
{
  "status": "healthy",
  "model": "intfloat/e5-small-v2"
}
```

**Initializing/Failed Response (200 OK):**
```json
{
  "status": "initializing or failed"
}
```

### 2. Search Codebase
Executes a hybrid search query against the local LanceDB vector store.

*   **URL:** `/search`
*   **Method:** `POST`
*   **Content-Type:** `application/json`

**Request Body (JSON):**
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | **Yes** | The semantic text query to search for. |
| `limit` | `integer` | No | Maximum number of results to return (Default: 5). |
| `source_filter` | `string` | No | Optional exact-match path/file to filter the search results. |

**Example Request:**
```json
{
  "query": "How does Unified Memory mapping work?",
  "limit": 3,
  "source_filter": "docs/MLX_LANCEDB_POLARS_via_ApacheArrow.md"
}
```

**Success Response (200 OK):**
Returns the original query and an array of `SearchResult` objects.

```json
{
  "query": "How does Unified Memory mapping work?",
  "results": [
    {
      "chunk": {
        "id": "MLX_LANCEDB_POLARS_via_ApacheArrow.md_chunk_5",
        "text": "Unified Memory mapping (Forces MLX Lazy Evaluation)...",
        "source": "docs/MLX_LANCEDB_POLARS_via_ApacheArrow.md",
        "content_hash": "a1b2c3d4e5f6g7h8",
        "node_type": null,
        "parent_scope": null,
        "line_range": null
      },
      "score": 0.8523,
      "distance": 0.1477,
      "is_fts_match": false
    }
  ]
}
```

**Error Responses:**
*   **503 Service Unavailable:** The search service (MLX or LanceDB) failed to initialize or is not ready.
*   **500 Internal Server Error:** An unexpected error occurred during search execution.

## Concurrency and Threading
Because the MLX embedder and LanceDB search operations are blocking (synchronous), the API utilizes `asyncio.to_thread`. This allows the asynchronous FastAPI event loop to remain unblocked and responsive to concurrent incoming HTTP requests (like `/health` checks) while the heavy GPU inference is offloaded to a thread pool.
