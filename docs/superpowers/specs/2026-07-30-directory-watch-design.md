# Directory Watch — Design Spec

**Date:** 2026-07-30 (amended same day after adversarial ambiguity review)
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
watcher ingests something the CLI wouldn't. (Corollary: reconciliation prunes
files that become *excluded*, not just files that are deleted — see
Reconciliation step 3.)

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
  Entries must be **absolute paths to directories** — no globs, no files, no
  URLs (the CLI positional argument keeps supporting those). Relative
  entries are a load-time config error (the MCP server's cwd is set by the
  MCP client, so relative roots would resolve differently for CLI and
  server). Roots are `Path(p).resolve()`d at load; nested or duplicate
  roots within one engine are a load-time config error.
- `ignore_patterns: list[str]` (default `[".#*", "*~", "*.tmp", ".DS_Store"]`)
  — `fnmatch` glob patterns applied during file discovery (CLI walk and
  watch events alike). A pattern is tested against **both** the file's
  basename and its root-relative path; matching either excludes the file
  (so `notes/drafts/*` works, and plain `*.tmp` works). Setting the field
  **replaces** the defaults (standard pydantic semantics) — users who add
  patterns must restate the defaults they want to keep. Since discovery is
  extension-gated first, these defaults mostly matter for lockfiles that
  match a supported extension (e.g. emacs `.#foo.md`); they are kept as
  belt-and-braces.

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
- `watch.enabled: true` on an engine whose `table_name` is shared with any
  other engine → config error. Reconciliation prunes by root, not by engine;
  a shared table would let one engine's prune delete another's rows.
  (`sql`/`sql-api` share a table deliberately — they stay unwatched.)
- `paths:` entries must be absolute; nested/duplicate roots rejected (above).
- Configured roots that do not exist on disk at watcher startup are a logged
  error and the root is skipped; the MCP server still serves searches.

## CLI behavior

- `dbs-vector ingest --type md` (no path argument, newly valid) → ingests
  **all** roots in engine `paths:` in a **single ingestion run**: one
  `clear()` (if `--rebuild`), one dedup snapshot, one index/compaction pass
  at the end. Never a per-root loop — with `--rebuild`, a per-root loop
  would clear the table once per root and keep only the last root's data.
  If `paths:` is missing/empty → usage error: `engine 'md' has no paths
  configured; pass a path or add paths: to config.yaml`.
- `dbs-vector ingest "docs/" --type md` → unchanged: the explicit path
  overrides engine `paths:` for that run. If the engine has
  `watch.enabled: true` **and** the override root falls outside all
  configured roots, the CLI prints a notice that these rows will not be
  watched or reconciled (see Consequences #3). No notice for unwatched
  engines (the message would be vacuous) or for overrides inside a root.

### Override root definition

For an explicit CLI path argument, the **root** (used for gitignore
anchoring, `ignore_patterns` relative matching, and the outside-roots
notice) is: the argument itself if it is a directory; its parent directory
if it is a single file; the longest non-glob directory prefix if it is a
glob pattern. URL targets have no root and skip path-based filtering
entirely.

## Gitignore filter

Opt-in by listing `gitignore` in the engine's `exclusion_filters`.

- **Semantics:** for each ingestion root (engine path, or CLI override root
  as defined above), `<root>/.gitignore` — if present — filters files under
  that root. Parsed with `pathspec`, matched against paths relative to the
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

## PathFilter component

Findings from review: `IngestionService` today receives no config and must
not read the `settings` singleton (DI pattern). A new small component owns
all path-level scoping:

- **`PathFilter`** (services layer): built in `bootstrap.build_dependencies`
  from the engine config — roots, `ignore_patterns`, gitignore on/off — and
  injected into **both** `IngestionService` and `WatcherService` (added to
  `EngineDeps`). It answers one question: *is this path ingestable, and
  under which root?* CLI override paths construct a fresh `PathFilter` for
  the override root with the same engine `ignore_patterns`/gitignore
  settings. This is the single enforcement point for the Hard Rule's
  filtering; the walk in `ingest_directory` and the watcher's event filter
  both delegate to it.

## Architecture

`WatcherService` lives in the services layer (`services/watcher.py`),
depending only on core protocols — a second trigger in front of the same
ingestion orchestration. New dependency: `watchdog` (FSEvents-backed on
macOS).

```
watchdog Observer threads (one per engine's root set)
  → PathFilter (extension, ignore_patterns, gitignore) — path-only, no reads
  → per-file debounce map {path: due_time}, window resets on each event
       ↓ (due)
single global worker thread (one queue across ALL engines)
  → executes actions sequentially: upsert_file / remove_source / reconcile
  → after each drained batch with ≥1 write: refresh FTS index
  → full index rebuild + compact() per engine every 100 processed files,
    after each reconciliation pass, and at shutdown
```

- **Startup:** `start_stdio_server()` starts watchers after
  `initialize_services()` and before `mcp.run()`; stops them in a `finally`.
  Watchers start only if ≥1 engine has `watch.enabled: true`.
- **Dependencies:** each watched engine gets its full stack via the existing
  `build_dependencies()`; the embedder comes from the process-level
  `_MODEL_CACHE`, already resident — no second model load. This deliberately
  creates a **second `LanceDBStore` instance** (own connection/table handle)
  distinct from the `SearchService` one: writer and readers on separate
  handles is what makes the MVCC/`checkout_latest()` safety argument valid.
  Do not "optimize" this into a shared store object.
- **Concurrency:** one global worker serializes all LanceDB writes across
  engines. Concurrent MCP searches are safe: `MLXEmbedder` has a per-model
  lock; LanceDB reads call `checkout_latest()` (MVCC). Accepted trade-off:
  search embeds serialize behind watcher embed batches on the same model, so
  a large reconcile adds up-to-one-batch latency to concurrent searches.
- The watcher uses the chunker's `supported_extensions` — same as CLI ingest.
  No separate extension config.

## Upsert model: whole-file replace

Chosen over per-chunk hash diffing. Hash scheme (verified against
`_chunk_content_hash` in `infrastructure/chunking/document.py`): the
document chunker derives every chunk's `content_hash` from the file-level
SHA-256 (16-char truncation) — chunk 0 carries the **bare** file hash,
chunks 1..N carry `{file_hash}_{i}`. The unchanged-check therefore tests the
bare file hash, which is present in the store iff the file was ingested.

`IngestionService.upsert_file(path)` — the exact sequence, which does **not**
reuse the directory-ingest dedup snapshot:

1. Read + hash the file.
2. **Unchanged check:** fresh `get_existing_hashes()` call (per action, not
   cached — see below); bare file hash present → no-op, stop.
3. `delete_by_source(path)`.
4. Chunk → embed → `ingest_chunks`, **unconditionally** — no hash-set
   skipping in this path. (Reusing the snapshot-based dedup would see the
   file's own pre-delete hashes and silently skip every chunk.)

`get_existing_hashes()` gains a `checkout_latest()` call (today it is the
only read path without one — a latent staleness bug this feature would
expose; fixed as part of this work). Its O(rows) single-column scan per
action is accepted (ms-scale).

- Cost: an edited file re-embeds all its chunks (~5–30 for a typical doc) —
  one batch, sub-second with the resident model.
- Rejected alternative (per-chunk content hashes): saves partial-edit
  embedding work but breaks the file-level short-circuit, collides on
  identical chunks across files, and forces re-ingesting existing tables.

### Known limitations (accepted, documented)

- **Identical-content files** dedup to one copy globally (hash-keyed —
  existing CLI behavior, unchanged). If file A is indexed and identical
  file B is created, B's upsert is a no-op; if A is then deleted, the
  content is absent until the **next reconciliation pass** re-ingests it
  under B's source. Event-driven flow alone does not heal this; reconcile
  does.
- **Zero-chunk files** (fully excluded by content filters, or empty) never
  store their hash, so every reconciliation pass re-reads and re-chunks
  them. Harmless, bounded by file count.

## Event lifecycle

| Event | Action (after debounce) |
|---|---|
| file created / modified | skip if `PathFilter` rejects; else `upsert_file` |
| file deleted | `remove_source(path)` → `delete_by_source` |
| file moved / renamed | `delete_by_source(old)`; if `PathFilter` accepts new path → `upsert_file(new)` |
| directory created / moved / deleted | schedule a **reconciliation pass** for that engine (FSEvents child-event delivery is unreliable; do not enumerate children from the event) |

**Directory-event coalescing:** at most one pending reconcile per engine,
debounced with the same `debounce_seconds`. A pending reconcile **absorbs**
that engine's queued per-file actions (drops them — reconciliation subsumes
their work). A burst like `git checkout` touching 40 directories yields one
reconcile, not 40.

## Reconciliation pass

Runs on the worker thread at startup (`reconcile_on_start: true`) and when
directory-level events fire. Per engine:

1. **Disk set:** walk engine `paths:` applying `PathFilter` (extension +
   `ignore_patterns` + gitignore) → map `resolved path → file hash` for
   ingestable files.
2. **Store set:** `scan(columns=["source", "content_hash"])` (existing
   method) → map `source → hashes`.
3. **Prune:** delete rows whose `source` is under a watched root
   (`Path(source).is_relative_to(resolved_root)` — roots are resolved, see
   Configuration) but **not in the step-1 disk set**. This intentionally
   covers more than deletions: files newly matched by `ignore_patterns`,
   newly gitignored, or with a removed extension are pruned too — the Hard
   Rule demands the index track what the engine *would ingest today*, not
   what once existed.
4. **Ingest:** working from the step-2 map **minus the hashes of rows pruned
   in step 3**, upsert disk files whose bare file hash is absent. (Without
   the subtraction, a file renamed while the server was down would be pruned
   under its old source yet skipped as "already present" — ending with zero
   rows.)

**Safety:** reconciliation deletes never touch rows whose `source` is outside
the engine's configured roots (including relative-path sources from older
ingests — see Path normalization).

## Path normalization

`delete_by_source` and reconciliation compare stored `source` strings
exactly, so watch-managed engines need canonical paths:

- The **file-discovery branch** of ingestion (CLI walk and watch alike)
  stores `str(Path(p).resolve())` — absolute, symlink-resolved. This scope
  is deliberate: URL targets (`api` chunker) and `.duckdb` targets are
  untouched — resolving those would corrupt their sources.
- Consequence: `Chunk.id` (derived from filepath) also changes for
  previously relative-ingested files. Rows with relative `source` values
  from older ingests are treated as outside the watched roots (unmanaged;
  never pruned). **Recommendation, surfaced in docs and the CLI notice: run
  one `--rebuild` ingest per watched engine when first enabling watch.**
- Dedup is hash-based and unaffected either way.

## Store & service API changes

- `IVectorStore` gains two methods:
  - `delete_by_source(source: str) -> None` — `LanceDBStore` implements it
    as a SQL predicate delete with single-quote escaping (matching existing
    filter code). Returns nothing: lancedb's `Table.delete` yields a table
    version, not a row count, and a count-then-delete pre-scan per event is
    not worth it. Logging reports the source, not a count.
  - `refresh_fts() -> None` — rebuilds only the Tantivy FTS index (see
    Index maintenance).
- `IngestionService` gains `upsert_file(path)` and `remove_source(path)`
  (sequences above), accepts a `PathFilter`, and `ingest_directory` accepts
  multiple roots in a single run (single clear/snapshot/index cycle) while
  applying the `PathFilter` in its walk.

## Index maintenance & compaction

Split into two tiers — review verified that `create_indices()` retrains
IVF_PQ from scratch and rebuilds FTS with `replace=True`, far too heavy per
save on a real vault:

- **Per drained batch (≥1 write):** `refresh_fts()` only. New rows must
  enter the FTS index to appear in the hybrid search's FTS leg; the vector
  leg finds unindexed rows without retraining.
- **Per engine, every 100 processed files, after each reconciliation pass,
  and at shutdown:** full `create_indices()` + `compact()`. Counters are
  per-engine (the worker queue is global but stores are not).

Bursts (vault sync) coalesce into one drain → one FTS refresh.

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
   CLI prints a notice in this case (watched engines only).
4. **Shrinking `paths:` orphans rows rather than deleting them.** Deliberate:
   auto-deleting everything outside current roots would turn a config typo
   into a mass delete. Cleanup of an intentionally removed root is manual
   (`--rebuild`; a prune command is out of scope for v1).
5. **`watch.enabled: true` with no `paths:` is a load-time config error**,
   not a runtime surprise. md and md-granite watching the same vault embed
   each change once per engine — the expected price of A/B parity.

## Testing

- **Unit** (mock-based, no I/O): debounce map with a fake clock; `PathFilter`
  (extension / `ignore_patterns` basename-and-relative matching / gitignore
  anchoring and caching); event→action mapping with mock store + ingester;
  reconciliation diff logic incl. the renamed-while-down case and
  newly-excluded-file pruning; directory-event coalescing/absorption; config
  validation rules (absolute paths, nested roots, shared table_name, missing
  paths).
- **Integration** (tmpdir LanceDB, real chunker/mapper, deterministic fake
  embedder): create/edit/delete/rename files and dispatch events directly
  into the handler (no real timers — hermetic, no sleeps); startup
  reconciliation end-to-end; identical-content heal-on-reconcile;
  `LanceDBStore.delete_by_source` and `refresh_fts`; CLI
  no-path/override/paths-missing/multi-root-rebuild behaviors.
- One optional smoke test with a real `Observer`, marked slow.

## Out of scope (v1)

- Nested `.gitignore` files; live `.gitignore` tracking.
- Watch for non-document chunkers (duckdb/api engines).
- A prune command for orphaned rows.
- Per-chunk hash diffing.
- Any CLI `watch` command — the watcher exists only inside `dbs-vector mcp`.

## Dependencies

- `watchdog` — filesystem events (FSEvents on macOS). Genuinely new.
- `pathspec` — `.gitignore` pattern parsing. Already in `uv.lock` as a
  transitive dependency (1.0.4); promote to a direct dependency.
