# Directory Watch — Design Spec

**Date:** 2026-07-30 (amended twice: adversarial ambiguity review, then user
review — source-aware unchanged check, scoped invariant, active-root pruning,
event coalescing semantics)
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

## The Invariant

> **Within currently configured roots, CLI discovery, watcher events, and
> reconciliation use the same path filtering.**

One engine-owned list — `paths:` — with the same extension filtering,
`ignore_patterns`, and `gitignore` filtering drives all three. Corollary:
reconciliation prunes files that become *excluded*, not just files that are
deleted (Reconciliation step 3).

The invariant is deliberately scoped to *currently configured, active roots*.
Rows outside it — out-of-root CLI ingests (Consequences #3), orphans from
removed roots (Consequences #4), rows under a root that fails to open
(Reconciliation) — are unmanaged by design: never re-checked, never pruned.

### Configuration-change rebuild rule

Reconciliation compares content hashes, so it detects *file* changes only. It
cannot detect that unchanged files need re-chunking or re-embedding after
changes to the engine's model, prefixes, tuning profile, chunker behavior,
content exclusion filters — or that rows under a removed root need cleanup.
The v1 rule, stated in docs and the README:

> **After changing a watched engine's roots, model, profile, chunking, or
> filtering configuration, run one `ingest --rebuild --force` for that engine
> before starting the MCP server.**

No config fingerprinting, old-root tracking, or surgical migration in v1.

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
      debounce_seconds: 3.0                      # per-file; >= 0, default 3.0
```

Startup reconciliation is **mandatory** for watched engines in v1 — there is
no `reconcile_on_start` knob. A watcher that starts without reconciling could
serve pre-existing divergence indefinitely, silently violating the invariant.

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
  is not delayed. Must be `>= 0`; `0` means immediate processing (useful for
  tests and scripted setups).

### Load-time validation

- `watch.enabled: true` requires non-empty engine `paths:` → config error.
- `watch.enabled: true` on an engine whose `chunker_type` is not `"document"`
  → config error (v1 restriction).
- `watch.enabled: true` on an engine whose `table_name` is shared with any
  other engine → config error. Reconciliation prunes by root, not by engine;
  a shared table would let one engine's prune delete another's rows.
  (`sql`/`sql-api` share a table deliberately — they stay unwatched.)
- `paths:` entries must be absolute; nested/duplicate roots rejected (above).
- `debounce_seconds < 0` → config error.
- Configured roots that do not exist (or cannot be opened) at watcher startup
  are a logged error; the root is skipped as **inactive** — excluded from
  watching *and* from reconciliation (see Reconciliation) — and the MCP
  server still serves searches.

## CLI behavior

- `dbs-vector ingest --type md` (no path argument, newly valid) → ingests
  **all** roots in engine `paths:` in a **single ingestion run**: one
  `clear()` (if `--rebuild`), one dedup snapshot, one index/compaction pass
  at the end. Never a per-root loop — with `--rebuild`, a per-root loop
  would clear the table once per root and keep only the last root's data.
  If `paths:` is missing/empty → usage error: `engine 'md' has no paths
  configured; pass a path or add paths: to config.yaml`. The no-path
  fallback is **document-engine-only in v1**; API/duckdb engines keep their
  existing invocation forms unchanged.
- `dbs-vector ingest "docs/" --type md` → the explicit path **replaces the
  roots for that run, not the filtering rules**: `ignore_patterns` and
  gitignore (via `PathFilter`) apply, and discovered sources are stored as
  resolved absolute paths. This is a deliberate behavior change from today's
  explicit-path ingestion (which had no ignore filtering and stored paths
  as given) — required by the invariant. If the engine has
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

`IngestionService` receives no config and must not read the `settings`
singleton (DI pattern). A new small component owns all path-level scoping:

- **`PathFilter`** (services layer): built in `bootstrap.build_dependencies`
  from the engine config — roots, `ignore_patterns`, gitignore on/off — and
  injected into **both** `IngestionService` and `WatcherService` (added to
  `EngineDeps`). It answers one question: *is this path ingestable, and
  under which root?* CLI override paths construct a fresh `PathFilter` for
  the override root with the same engine `ignore_patterns`/gitignore
  settings. This is the single enforcement point for the invariant's
  filtering; the walk in `ingest_directory` and the watcher's event filter
  both delegate to it.

## Architecture

`WatcherService` lives in the services layer (`services/watcher.py`),
depending only on core protocols — a second trigger in front of the same
ingestion orchestration. New dependency: `watchdog` (FSEvents-backed on
macOS).

```
watchdog Observer threads (one per engine's active root set)
  → PathFilter (extension, ignore_patterns, gitignore) — path-only, no reads
  → per-file debounce map {path: (last_event_kind, due_time)} — last state wins
       ↓ (due)
single global worker thread (one queue across ALL engines)
  → executes actions sequentially: upsert_file / remove_source / reconcile
  → after each drained batch with ≥1 write: refresh FTS index
  → full index rebuild + compact() per engine every 100 processed files,
    after each reconciliation pass, and at shutdown
```

- **Startup:** `start_stdio_server()` starts watchers after
  `initialize_services()` and before `mcp.run()`; stops them in a `finally`.
  Watchers start only if ≥1 engine has `watch.enabled: true`. Startup
  reconciliation is unconditional for watched engines and runs on the worker
  thread (server responds to MCP requests immediately).
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
- **Shutdown (graceful):** stop observers (no new events) → drain pending
  due actions → final per-engine maintenance (`create_indices()` +
  `compact()`) → exit. Abrupt shutdown (kill, crash) is recovered by the
  mandatory startup reconciliation on next launch.
- The watcher uses the chunker's `supported_extensions` — same as CLI ingest.
  No separate extension config.

## Upsert model: whole-file replace, source-aware

Chosen over per-chunk hash diffing. Hash scheme (verified against
`_chunk_content_hash` in `infrastructure/chunking/document.py`): the
document chunker derives every chunk's `content_hash` from the file-level
SHA-256 (16-char truncation) — chunk 0 carries the **bare** file hash,
chunks 1..N carry `{file_hash}_{i}`. A file is "already ingested" iff its
bare file hash is present **under its own source** — the check is on the
`(source, bare_hash)` **pair**, never the hash alone. (A global-hash check
has a correctness hole: edit B.md to duplicate A.md's content and B's hash
"already exists", so B is skipped and its stale old rows survive every
subsequent pass.)

`IngestionService.upsert_file(path)` — the exact sequence, which does **not**
reuse the directory-ingest dedup snapshot:

1. Read + hash the file.
2. **Unchanged check:** fresh `scan(columns=["source", "content_hash"])`
   (per action, not cached; `scan` guarantees `checkout_latest()`). If the
   `(source, bare_hash)` pair exists → no-op, stop.
3. `delete_by_source(path)` — the source's old rows go regardless of what
   happens next.
4. **Cross-source dedup check:** if the bare hash exists under a *different*
   source → stop here. The content is already indexed once; this preserves
   the existing global deduplication behavior (the file is represented by
   the other source's rows).
5. Chunk → embed → `ingest_chunks`, **unconditionally** — no hash-set
   skipping in this path. (Reusing the snapshot-based dedup would see the
   file's own pre-delete hashes and silently skip every chunk.)

`get_existing_hashes()` gains a `checkout_latest()` call (today it is the
only read path without one — a latent staleness bug; fixed as part of this
work, though `upsert_file` itself uses the pair-scan above). The O(rows)
two-column scan per action is accepted (ms-scale).

- Cost: an edited file re-embeds all its chunks (~5–30 for a typical doc) —
  one batch, sub-second with the resident model.
- Rejected alternative (per-chunk content hashes): saves partial-edit
  embedding work but breaks the file-level short-circuit, collides on
  identical chunks across files, and forces re-ingesting existing tables.
- The CLI directory-ingest path keeps its existing global-hash snapshot
  semantics (the same edit-to-duplicate hole exists there today,
  pre-existing); for watched engines the mandatory startup reconciliation
  detects the absent `(source, hash)` pair and repairs it via
  `upsert_file`.

### Known limitations (accepted, documented)

- **Identical-content files** dedup to one copy globally (existing
  behavior, unchanged). If file A is indexed and identical file B is
  created, B's upsert stops at step 4 (zero rows for B); if A is then
  deleted, the content is absent until the **next reconciliation pass**
  detects B's missing `(source, hash)` pair and ingests it. Event-driven
  flow alone does not heal this; reconcile does.
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

### Coalescing: last state wins

The debounce map keeps **one pending action per path** — a later event for
the same path replaces the earlier one (and resets the window):

- modify then delete → **remove** (not upsert-then-remove).
- delete then create → **upsert**.
- rename → two entries: remove(old path) + evaluate/upsert(new path), each
  independently subject to later replacement.
- A **pending reconciliation replaces all earlier per-file actions** for
  that engine (reconciliation subsumes their work). At most one pending
  reconcile per engine, debounced with the same `debounce_seconds` — a
  `git checkout` touching 40 directories yields one reconcile, not 40.
- Events arriving **while a reconciliation is actively running** are queued
  normally and processed after it finishes — changes made after the
  reconciliation's disk/store snapshots are never lost. (`upsert_file` is
  idempotent, so overlap with what reconcile already did is a cheap no-op.)

## Reconciliation pass

Runs on the worker thread at startup (mandatory for watched engines) and
when directory-level events fire. Per engine:

0. **Active roots only.** A configured root that does not exist or cannot be
   opened is *inactive*: excluded from the walk **and from pruning**. A
   missing root must never be interpreted as an empty directory — an
   unmounted volume would otherwise mass-prune every row under it.
   Moving or removing a root permanently is a configuration change: update
   `paths:` and follow the rebuild rule.
1. **Disk set:** walk the engine's *active* roots applying `PathFilter`
   (extension + `ignore_patterns` + gitignore) → map
   `resolved path → bare file hash` for ingestable files.
2. **Store set:** `scan(columns=["source", "content_hash"])` (existing
   method) → map `source → hashes`.
3. **Prune:** delete rows whose `source` is under an **active** root
   (`Path(source).is_relative_to(resolved_root)` — roots are resolved, see
   Configuration) but **not in the step-1 disk set**. This intentionally
   covers more than deletions: files newly matched by `ignore_patterns`,
   newly gitignored, or with a removed extension are pruned too — the
   invariant demands the index track what the engine *would ingest today*,
   not what once existed.
4. **Ingest:** for each disk file whose `(source, bare_hash)` pair is absent
   from the step-2 map **minus the rows pruned in step 3**, call
   `upsert_file` (which re-applies its own live checks, including the
   cross-source dedup stop). The subtraction matters: a file renamed while
   the server was down is pruned under its old source in step 3 and must
   not be skipped as "already present" in step 4.

**Safety:** reconciliation deletes never touch rows whose `source` is outside
the engine's *active* roots (including relative-path sources from older
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

Split into two tiers — `create_indices()` retrains IVF_PQ from scratch and
rebuilds FTS with `replace=True`, far too heavy per save on a real vault:

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
- Observer threads are daemons; graceful shutdown follows the
  stop-observers → drain → final-maintenance order (Architecture), wrapped
  in a `finally` around `mcp.run()`.

## Consequences of the Invariant (accepted)

1. **No scope divergence within active roots.** One list (`paths:`) drives
   CLI ingest, watch, and reconciliation with identical filtering.
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
   into a mass delete. Cleanup of an intentionally removed root follows the
   rebuild rule (`--rebuild --force`; a prune command is out of scope).
5. **`watch.enabled: true` with no `paths:` is a load-time config error**,
   not a runtime surprise. md and md-granite watching the same vault embed
   each change once per engine — the expected price of A/B parity.

## Testing

- **Unit** (mock-based, no I/O): debounce map with a fake clock, including
  the three last-state transitions (modify→delete = remove, delete→create =
  upsert, rename = remove+upsert) and reconcile-absorbs-pending;
  `PathFilter` (extension / `ignore_patterns` basename-and-relative
  matching / gitignore anchoring and caching); event→action mapping with
  mock store + ingester; reconciliation diff logic incl. the
  renamed-while-down case, newly-excluded-file pruning, and inactive-root
  exclusion; config validation rules (absolute paths, nested roots, shared
  table_name, missing paths, negative debounce).
- **Integration** (tmpdir LanceDB, real chunker/mapper, deterministic fake
  embedder): create/edit/delete/rename files and dispatch events directly
  into the handler (no real timers — hermetic, no sleeps); startup
  reconciliation end-to-end; **existing file edited into a duplicate of
  another file** (the source-aware check: old rows deleted, no new rows,
  no stale survivors); identical-content heal-on-reconcile;
  `LanceDBStore.delete_by_source` and `refresh_fts`; CLI
  no-path/override/paths-missing/multi-root-rebuild behaviors.
- One optional smoke test with a real `Observer`, marked slow.

## Out of scope (v1) — explicitly deferred

- Root-move tracking; old-root row migration.
- Ignore-pattern / filter-diff detection; config fingerprints. (Covered by
  the documented rebuild rule.)
- Nested `.gitignore` files; live `.gitignore` tracking.
- Watch for non-document chunkers (duckdb/api engines).
- A prune command for orphaned rows.
- Per-chunk hash diffing / partial-file updates.
- Any CLI `watch` command — the watcher exists only inside `dbs-vector mcp`.

## Dependencies

- `watchdog` — filesystem events (FSEvents on macOS). Genuinely new.
- `pathspec` — `.gitignore` pattern parsing. Already in `uv.lock` as a
  transitive dependency (1.0.4); promote to a direct dependency.
