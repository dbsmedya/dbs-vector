# scripts/

Helper scripts for validating and maintaining dbs-vector.

---

## calibrate_similarity_floor.py

Measures a corpus- and engine-specific `similarity_floor` with the production
embedder, hybrid store, lexical gate, and candidate-pool geometry. Development
mode prints an exact sweep and a suggestion for human review:

```bash
uv run python scripts/calibrate_similarity_floor.py \
  --engine md-granite \
  --set scripts/calibration/documents/dev.json
```

Locked evaluation mode scores one committed human choice:

```bash
uv run python scripts/calibrate_similarity_floor.py \
  --engine md-granite \
  --set scripts/calibration/<corpus-name>/eval.json \
  --floor <chosen-value> \
  --choice-record scripts/calibration/choices/<choice>.json
```

An evaluation set is consumed per engine before its first search. A failed or
interrupted evaluation must not be rerun; author and commit a fresh set. Exit
code `0` means a development run or passing evaluation, `1` means evaluation
acceptance failed, and `2` means a setup or sealing error. Query sets, choice
records, and reports are deployment-local evidence about one corpus, so this
repository ships none of them. For the complete workflow — authoring sets,
sealing, spending an evaluation once, and recording the outcome — see
[`docs/README_CALIBRATE_CORPUS.md`](../docs/README_CALIBRATE_CORPUS.md).

Use `--preflight-only` to validate the corpus and expected-source labels
without loading an embedder or executing retrieval.

## measure_metric_delta.py

Isolates the raw vector-metric change from L2 to cosine by bypassing the ANN
index for both orderings. It reports embedding-norm percentiles, top-k
membership/order changes, top-1 changes, shared-member Kendall disagreement,
and the live IVF-index state:

```bash
uv run python scripts/measure_metric_delta.py \
  --engine md-granite \
  --set scripts/calibration/documents/dev.json
```

This probe measures vector geometry only; it deliberately excludes hybrid
fusion and admission behavior.

---

## check_remote_api.py

Verifies that a remote backend correctly implements the HTTP contract that
`ApiChunker` depends on. Run this before wiring a new server into `config.yaml`.

### What it checks

| Section | Endpoint | Checks |
|---------|----------|--------|
| 1. Health | `GET /health` | 200, `status=ok`, version string, databases list |
| 2. Auth | `GET /sql/queries` (no/wrong token) | 401 + `error=invalid_token` |
| 3. Structure | `GET /sql/queries` | JSON shape, required fields, types |
| 4. Pagination | `GET /sql/queries?limit=1` | cursor-based paging, no duplicate IDs |
| 5. Filters | `GET /sql/queries` | `min_execution_ms`, `since`, `database` params |
| 6. Databases | `GET /sql/databases` | list present, consistent with `/health` |
| 7. Execute (happy) | `POST /sql/execute` | columns/rows shape, row_count, truncated flag |
| 8. Execute (safety) | `POST /sql/execute` | INSERT/UPDATE/DELETE/DROP rejected with 422 |
| 9. Error envelope | `GET /sql/queries` (no token) | `error` + `message` fields present |

### URL convention

`--base-url` must include the versioned API prefix. All SQL endpoints are
resolved relative to it:

```
http://your-server:8080/api/v1          ← correct
http://your-server:8080                 ← wrong — SQL routes will 404
```

### Usage

```bash
# Recommended: run via uv so httpx is available
uv run python scripts/check_remote_api.py \
    --base-url http://localhost:8080/api/v1 \
    --api-key  <your-api-key>

# Or via environment variables
DBS_API_BASE_URL=http://localhost:8080/api/v1 \
DBS_API_KEY=<your-api-key> \
    uv run python scripts/check_remote_api.py

# Optional: increase timeout for slow backends (default 30 s)
uv run python scripts/check_remote_api.py \
    --base-url http://remote-host/api/v1 \
    --api-key  <key> \
    --timeout  60
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed (skips are OK) |
| `1` | One or more checks failed |
| `2` | Argument / setup error (missing `--base-url` or `--api-key`) |

### Wiring the backend into config.yaml

Once the checker returns exit code 0, add an engine block:

```yaml
engines:
  sql_remote:
    description: "Remote slow-query API"
    model_name: "your-embedding-model"
    vector_dimension: 768
    max_token_length: 256
    table_name: "sql_remote"
    mapper_type: "sql"
    chunker_type: "api"
    chunk_max_chars: 0
    api_base_url: "http://localhost:8080/api/v1"
    api_key: "your-api-key"
    api_since_days: 15
    api_page_size: 200
    api_database: ""          # optional: filter to one database
```

Then ingest:

```bash
uv run dbs-vector ingest --type sql_remote
```

### Known backend variations

| Behaviour | `since` future date returns 400 |
|-----------|--------------------------------|
| Compliant? | Yes — `ApiChunker` never sends future dates; both 200+[] and 400 are accepted by the checker. |
