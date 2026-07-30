# Directory Watch — Design Spec

**Date:** 2026-07-30
**Branch:** `feat/directory-watch`
**Status:** Approved design, pre-implementation

## Purpose

Automatically ingest file changes from configured directories while the MCP
server is running. The MCP server already holds the embedding model resident
in memory, so re-ingesting a single changed file becomes a sub-second
in-process operation instead of a multi-second CLI cold start. Primary use
case: Claude (via MCP) searching a documentation vault that is being edited
live.

**v1 scope:** engines with `chunker_type: "document"` (md, md-granite).
The design is engine-generic; other file-based chunkers can be enabled later.

## Hard Rule

> **The watcher watches exactly what the engine ingests.**

CLI ingest, live watch events, and reconciliation all derive their scope from
one engine-owned list — `paths:` — with the same extension filtering,
`ignore_patterns`, and `gitignore` filtering. There is no configuration state
in which search serves files the watcher silently doesn't cover, or the
watcher ingests something the CLI wouldn't.

## Configuration

All filtering/scoping lives at the **engine level**. The `watch:` block holds
watch mechanics **only**.

```yaml
engines:
  md:
    # ... existing fields unchanged ...
    paths: ["/Users/sinanalyuruk/vault"]        # NEW: engine-owned ingestion roots
    ignore_patterns: [".#*", "*~", "*.tmp", ".DS_Store"]  # NEW (defaults shown)
    exclusion_filters: [excalidraw, compressed_json, gitignore]  # gitignore = new built-in
    watch:                                       # NEW, optional
      enabled: true                              # default false
      debounce_seconds: 3.0                      # per-file, default 3.0
      reconcile_on_start: true                   # default true
```

### New engine-level fields

- `paths: list[str]` (default `[]`) — ingestion roots. Used by CLI ingest
  (when no explicit path is given), by the watcher, and by reconciliation.
- `ignore_patterns: list[str]` (default `[".#*", "*~", "*.tmp", ".DS_Store"]`)
  — glob patterns matched against file names during file discovery (CLI walk
  and watch events alike). Fixes the pre-existing gap where CLI ingest picks
  up editor temp files.

### `watch:` block (`WatchConfig` on `EngineConfig`)

- `enabled: bool = false`
- `debounce_seconds: float = 3.0` — per-file window; resets on each new event
  for that file, so a save burst coalesces but an unrelated file changed later
  is not delayed.
- `reconcile_on_start: bool = true`

### Load-time validation

- `watch.enabled: true` requires non-empty engine `paths:` → config error.
- `watch.enabled: true` on an engine whose `chunker_type` is not `"document"`
  → config error (v1 restriction).
- Configured paths that do not exist on disk at watcher startup are a logged
  error and the root is skipped; the MCP server still serves searches.

## CLI behavior

- `dbs-vector ingest --type md` (no path argument, newly valid) → ingests
  **all** roots in engine `paths:`. If `paths:` is missing/empty → usage
  error: `engine 'md' has no paths configured; pass a path or add paths: to
  config.yaml`.
- `dbs-vector ingest "docs/" --type md` → unchanged: the explicit path
  overrides engine `paths:` for that run. When the override path lies outside
  the engine's configured roots, the CLI prints a notice that these rows will
  not be watched or reconciled (see Consequences #3).

## Gitignore filter

Opt-in by listing `gitignore` in the engine's `exclusion_filters`.

- **Semantics:** for each ingestion root (engine path, or CLI override root),
  `<root>/.gitignore` — if present — filters files under that root. Parsed
  with `pathspec` (new dependency), matched against paths relative to the
  root, cached per root for the process lifetime.
- **Enforcement point:** the **file-discovery layer** — the directory walk in
  `IngestionService` and the watcher's pre-debounce event filter — where the
  governing root is known. Files are excluded **before being read**.
- The registered `GitignoreFilter` in `FilterRegistry` exists so the name
  validates and appears in `FilterRegistry.keys()`; its chunk-level hooks
  (`should_skip_file` / `should_drop_block`) are no-ops. This is documented
  in the filter's docstring.
- Out of scope for v1: nested per-subdirectory `.gitignore` files; live
  re-reading of `.gitignore` edits (they take effect on server restart).

## Architecture

`WatcherService` lives in the services layer (`services/watcher.py`),
depending only on core protocols — a second trigger in front of the same
ingestion orchestration. New dependency: `watchdog` (FSEvents-backed on
macOS).

```
watchdog Observer threads (one per engine's root set)
  → filter events (extension, ignore_patterns, gitignore) — path-only, no reads
  → per-file debounce map {path: due_time}, window resets on each event
       ↓ (due)
single global worker thread (one queue across ALL engines)
  → executes actions sequentially: upsert_file / remove_source / reconcile
  → after each drained batch with ≥1 write: create_indices()
  → compact() every 100 processed files and at shutdown
```

- **Startup:** `start_stdio_server()` starts watchers after
  `initialize_services()` and before `mcp.run()`; stops them in a `finally`.
  Watchers start only if ≥1 engine has `watch.enabled: true`.
- **Dependencies:** each watched engine gets its full stack via the existing
  `build_dependencies()`; the embedder comes from the process-level
  `_MODEL_CACHE`, already resident — no second model load.
- **Concurrency:** one global worker serializes all LanceDB writes across
  engines. Concurrent MCP searches are safe: `MLXEmbedder` has a per-model
  lock; LanceDB reads call `checkout_latest()` (MVCC).
- The watcher uses the chunker's `supported_extensions` — same as CLI ingest.
  No separate extension config.

## Upsert model: whole-file replace

Chosen over per-chunk hash diffing. The document chunker propagates the
**file-level** SHA-256 (truncated to 16 chars) to every chunk, so:

- **Unchanged check:** hash the file; if the hash is already in
  `get_existing_hashes()` → no-op. (Works because all of a file's chunks
  share its file hash.)
- **Changed:** `delete_by_source(path)`, then chunk → embed → ingest. Every
  old row for the file is dropped, so chunks "edited away" cannot survive.
- Cost: an edited file re-embeds all its chunks (~5–30 for a typical doc) —
  one batch, sub-second with the resident model.
- Rejected alternative (per-chunk hashes): saves partial-edit embedding work
  but breaks the skip-unchanged-file shortcut, collides on identical chunks
  across files (one file's delete would orphan the other's dedup'd row), and
  forces re-ingesting existing tables.

## Event lifecycle

| Event | Action (after debounce) |
|---|---|
| file created / modified | skip if filtered; unchanged-check; else delete_by_source + re-ingest |
| file deleted | `delete_by_source(path)` |
| file moved / renamed | `delete_by_source(old)`; if new path is under a watched root, has a supported extension, and is not filtered → upsert new path |
| directory created / moved / deleted | schedule a debounced **reconciliation pass** (FSEvents child-event delivery is unreliable; do not enumerate children from the event) |

## Reconciliation pass

Runs on the worker thread at startup (`reconcile_on_start: true`) and when
directory-level events fire. Per engine:

1. **Disk set:** walk engine `paths:` applying extension, `ignore_patterns`,
   and `gitignore` filters.
2. **Store set:** `scan(columns=["source", "content_hash"])` (existing
   method — no new read API).
3. **Prune:** delete rows whose `source` is an absolute path **under a
   watched root** but no longer on disk.
4. **Ingest:** upsert disk files whose file hash is absent from the store.

**Safety:** reconciliation deletes never touch rows whose `source` is outside
the engine's configured roots.

## Path normalization

`delete_by_source` and reconciliation compare stored `source` strings
exactly, so watch-managed engines need canonical paths:

- All ingestion writes (CLI and watch) store `str(Path(p).resolve())` —
  absolute, symlink-resolved — going forward.
- Rows with relative `source` values from older ingests are treated as
  outside the watched roots (unmanaged; never pruned). **Recommendation,
  surfaced in docs and the CLI notice: run one `--rebuild` ingest per watched
  engine when first enabling watch.**
- Dedup is hash-based and unaffected either way.

## Store & service API changes

- `IVectorStore` gains **one** method: `delete_by_source(source: str) -> int`
  (rows deleted). `LanceDBStore` implements it as a SQL predicate delete with
  single-quote escaping, matching existing filter code.
- `IngestionService` gains `upsert_file(path)` and `remove_source(path)`
  implementing the lifecycle above; `ingest_directory` gains multi-root
  support and applies `ignore_patterns` + `gitignore` in its walk.

## Index refresh & compaction

After each drained batch containing ≥1 write, call `create_indices()` — the
Tantivy FTS index must be refreshed for new rows to appear in the hybrid
search's FTS leg (the vector leg finds them immediately, brute-force or IVF).
`compact()` runs every 100 processed files and at watcher shutdown. Bursts
(git checkout, vault sync) coalesce into one drain → one index refresh.

## Error handling

- Every per-file action is wrapped: failures are logged (loguru) and never
  kill the worker loop or the MCP server.
- Non-UTF-8 files are skipped with a warning (existing behavior).
- Observer threads are daemons; shutdown stops observers and the worker in a
  `finally` around `mcp.run()`.

## Consequences of the Hard Rule (accepted)

1. **No scope divergence, ever.** One list (`paths:`) drives CLI ingest,
   watch, and reconciliation with identical filtering.
2. **CLI ingest becomes mostly unnecessary for watched engines.** Startup
   reconciliation is a full delta ingest. CLI ingest remains for initial bulk
   loads, `--rebuild`, and unwatched engines. Do not run CLI ingest against a
   watched engine's table while the MCP server is up — two processes writing
   one LanceDB table can hit commit conflicts (pre-existing limitation; the
   rule makes it easy to avoid).
3. **Out-of-root CLI ingest creates unmanaged rows.** They are never
   re-checked and never pruned — stale the moment the source file changes.
   The protection from deletion and the staleness are the same property. The
   CLI prints a notice in this case.
4. **Shrinking `paths:` orphans rows rather than deleting them.** Deliberate:
   auto-deleting everything outside current roots would turn a config typo
   into a mass delete. Cleanup of an intentionally removed root is manual
   (`--rebuild`; a prune command is out of scope for v1).
5. **`watch.enabled: true` with no `paths:` is a load-time config error**,
   not a runtime surprise. md and md-granite watching the same vault embed
   each change once per engine — the expected price of A/B parity.

## Testing

- **Unit** (mock-based, no I/O): debounce map with a fake clock; extension /
  `ignore_patterns` / gitignore matching; event→action mapping with mock
  store + ingester; reconciliation diff logic; `GitignoreFilter` root
  anchoring and caching; config validation rules.
- **Integration** (tmpdir LanceDB, real chunker/mapper, deterministic fake
  embedder): create/edit/delete/rename files and dispatch events directly
  into the handler (no real timers — hermetic, no sleeps); startup
  reconciliation end-to-end; `LanceDBStore.delete_by_source`; CLI
  no-path/override/paths-missing behaviors.
- One optional smoke test with a real `Observer`, marked slow.

## Out of scope (v1)

- Nested `.gitignore` files; live `.gitignore` tracking.
- Watch for non-document chunkers (duckdb/api engines).
- A prune command for orphaned rows.
- Per-chunk hash diffing.
- Any CLI `watch` command — the watcher exists only inside `dbs-vector mcp`.

## New dependencies

- `watchdog` — filesystem events (FSEvents on macOS).
- `pathspec` — `.gitignore` pattern parsing.
