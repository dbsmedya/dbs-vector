# Directory Watch — Design Spec

**Date:** 2026-07-30 (revision 3: two adversarial ambiguity reviews + user
review. Rev 2 added source-aware upsert, scoped invariant, active-root
pruning, event coalescing. Rev 3 resolves layering, PathFilter scope, map
keying, directory-event routing, CLI missing-root safety, FTS-refresh
semantics, symlink policy.)
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
(Reconciliation step 0), contents of symlinked subdirectories (Path policy) —
are unmanaged by design: never re-checked, never pruned.

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
  roots within one engine are a load-time config error. **`paths:` is only
  valid on document engines in v1** — setting it on any other engine is a
  load-time config error (silently accepting it would be a no-op trap).
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

Pure config-shape rules, checked by `load_settings(validate=True)`:

- `watch.enabled: true` requires non-empty engine `paths:` → config error.
- `watch.enabled: true` on an engine whose `chunker_type` is not `"document"`
  → config error (v1 restriction). Likewise `paths:` on a non-document
  engine (above).
- `watch.enabled: true` on an engine whose `table_name` is shared with any
  other engine → config error. Reconciliation prunes by root, not by engine;
  a shared table would let one engine's prune delete another's rows.
  (`sql`/`sql-api` share a table deliberately — they stay unwatched.)
- `paths:` entries must be absolute; nested/duplicate roots rejected.
- `debounce_seconds < 0` → config error.

**Root *existence* is never checked at config load** — `load_settings` runs
on every CLI command, and an unmounted vault must not break `dbs-vector
search`. Existence is checked at *use* time, with deliberately different
behavior per consumer:

- **Watcher startup** (long-running, unattended): a root that does not exist
  or cannot be opened is logged as an error and skipped as **inactive** —
  excluded from watching *and* from reconciliation (Reconciliation step 0).
  The MCP server still serves searches.
- **CLI no-path ingest** (interactive, foreground): any configured root
  missing → **hard error, nothing ingested**. This asymmetry is deliberate:
  `ingest --type md --rebuild --force` with an unmounted root would
  otherwise `clear()` the table and re-ingest only the surviving roots —
  silent data loss through the CLI door.

## CLI behavior

- `dbs-vector ingest --type md` (no path argument, newly valid) → ingests
  **all** roots in engine `paths:` in a **single ingestion run**: one
  `clear()` (if `--rebuild`), one dedup snapshot, one index/compaction pass
  at the end. Never a per-root loop — with `--rebuild`, a per-root loop
  would clear the table once per root and keep only the last root's data.
  All roots must exist (hard error above). If `paths:` is missing/empty →
  usage error: `engine 'md' has no paths configured; pass a path or add
  paths: to config.yaml`. The no-path fallback is **document-engine-only in
  v1**; API/duckdb engines keep their existing invocation forms unchanged.
- `dbs-vector ingest "docs/" --type md` → the explicit path **replaces the
  roots for that run, not the filtering rules**: extension gating,
  `ignore_patterns`, and gitignore (via `PathFilter`) apply, and discovered
  sources are stored canonically (Path policy). This is a deliberate
  behavior change from today's explicit-path ingestion (which applied no
  ignore filtering, no extension gate on the glob branch, and stored paths
  as given) — required by the invariant. If the engine has
  `watch.enabled: true` **and** the override root falls outside all
  configured roots, the CLI prints a notice that these rows will not be
  watched or reconciled (see Consequences #3). No notice for unwatched
  engines (the message would be vacuous) or for overrides inside a root.

### Override anchoring

- **Override path inside a configured root** → the **configured root stays
  the filtering anchor** (gitignore file, `ignore_patterns` relative
  matching, notice suppression); only the walk's starting point narrows.
  Anything else would let a narrowed CLI run ingest rows (e.g. from a
  gitignored subtree, anchored too deep to see the ignore) that the next
  reconciliation immediately prunes.
- **Override path outside all configured roots** → the override root is:
  the argument itself if it is a directory; its parent directory if it is a
  single file; the longest non-glob directory prefix if it is a glob
  pattern. URL targets have no root and skip path-based filtering entirely.

## Gitignore filter

Opt-in by listing `gitignore` in the engine's `exclusion_filters`.

- **Semantics:** for each ingestion root (engine path, or override root as
  defined above), `<root>/.gitignore` — if present — filters files under
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

- **`PathFilter`** (services layer, pure logic — no watchdog, no I/O beyond
  reading `.gitignore` once per root): built in
  `bootstrap.build_dependencies` from the engine config — roots,
  `ignore_patterns`, gitignore on/off, the chunker's
  `supported_extensions` — and injected into both `IngestionService` and
  `WatcherService` via `EngineDeps`. It answers one question: *is this path
  ingestable, and under which root?*
- **Document engines only (v1):** for engines with any other
  `chunker_type`, `EngineDeps.path_filter` is `None` and every existing
  code path (duckdb file, JSON file, URL) is untouched.
- **Explicit-path runs** construct a run-scoped `PathFilter` (override
  anchoring rules above) with the same engine settings; the engine-default
  filter from `EngineDeps` serves no-path CLI runs and the watcher.
- **Extension gating** is part of `PathFilter` and applies to directory
  walks, glob expansion, and explicit single-file targets alike; a skipped
  explicit file logs a visible warning. Matching is **case-insensitive**
  (`suffix.lower()`), consistent with `DocumentChunker`'s own dispatch —
  a deliberate minor change from today's case-sensitive `rglob` (and the
  glob branch today has no extension gate at all).

## Architecture

Layering follows the repo's composition rule (only `bootstrap` composes
infrastructure):

- **`core/ports.py`** gains a minimal `IWatchBackend` protocol:
  `start(roots, on_event)` / `stop()`, where `on_event` receives a small
  domain event (path, kind: created/modified/deleted/moved(+dest),
  is_directory).
- **`infrastructure/watch/watchdog_backend.py`** implements it with
  `watchdog` (FSEvents on macOS). This is the **only** module importing
  watchdog.
- **`services/watcher.py`** — `WatcherService` — consumes the port plus the
  engine's `PathFilter` and `IngestionService`. Depends only on core
  protocols, like every other service. This is also what lets tests
  dispatch synthetic events directly with no real observer or timers.

```
IWatchBackend (watchdog adapter; Observer threads per engine's active roots)
  → is_directory? ──yes→ schedule engine reconcile (root-membership check only)
  → no: PathFilter (extension, ignore_patterns, gitignore) — path-only, no reads
  → debounce map keyed by (engine, path) → (last_event_kind, due_time)
       ↓ (due)
single global worker thread (one queue across ALL engines)
  → executes actions sequentially: upsert_file / remove_source / reconcile
  → after each drained batch: refresh_fts() on each store written in it
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
- **Shutdown (graceful):** stop the backend (no new events) →
  **force-expire every pending debounce entry** and drain all of them →
  final per-engine maintenance (`create_indices()` + `compact()`) → exit.
  Draining only already-due entries would silently drop a file saved just
  before shutdown, erasing the distinction from abrupt shutdown. Abrupt
  shutdown (kill, crash) is recovered by the mandatory startup
  reconciliation on next launch.

## Path policy (canonical sources & symlinks)

`delete_by_source` and reconciliation compare stored `source` strings
exactly, so watch-managed engines need one canonical form. FSEvents reports
**real** (symlink-resolved) paths, so:

- **Roots** are `Path(p).resolve()`d at config load. Walk, watcher, and
  reconciliation all operate on resolved roots — so config-level root
  symlinks (`/vault` → `/Volumes/data/vault`) are transparent.
- **Below a root, paths are normalized but NOT symlink-resolved:** canonical
  source = `resolved_root / as-seen relative path` (normpath'd). This keeps
  walk paths, event paths, and stored sources identical, keeps every managed
  row's source under its root (a full `resolve()` would relocate a
  symlinked file's source *outside* the root — permanently unmanaged), and
  lets `remove_source` work for deleted symlinks (nothing to resolve
  through).
- **Symlinked subdirectories are not traversed** (pre-existing `rglob`
  behavior, kept): their contents are unmanaged. Documented limitation.
- **Scope:** canonicalization applies to the document-engine file-discovery
  branch only — the branch `PathFilter` governs. URL targets bypass
  discovery entirely; duckdb/JSON SQL targets keep their filepath as given
  (their `SqlChunk.source` is a database name from the log rows, not a
  filepath — canonicalizing the file argument would change nothing and is
  not done).
- Consequence: `Chunk.id` (derived from filepath) changes for previously
  relative-ingested files. Rows with old relative/unnormalized `source`
  values are treated as outside the watched roots (unmanaged; never
  pruned). **Recommendation, surfaced in docs and the CLI notice: run one
  `--rebuild` ingest per watched engine when first enabling watch.**
- Dedup is hash-based and unaffected either way.

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
| directory created / moved / deleted | schedule a **reconciliation pass** for that engine |

**Directory events** are routed *before* `PathFilter` (a directory has no
supported extension — filtering them like files would silently disable the
`git checkout` / bulk-move recovery path). The only check applied is active-
root membership. `ignore_patterns`/gitignore are **not** consulted for
directory events: a reconcile triggered from an ignored subtree is a cheap
no-op, whereas suppression risks missing a rename out of an ignored
directory. FSEvents child-event delivery is unreliable; never enumerate
children from the event itself.

### Coalescing: last state wins

The debounce map keeps **one pending action per `(engine, path)` key** — a
later event for the same key replaces the earlier one (and resets its
window). Keying by path alone would let md and md-granite (both watching one
vault) silently swallow each other's pending actions.

- modify then delete → **remove** (not upsert-then-remove).
- delete then create → **upsert**.
- rename → two entries: remove(old path) + evaluate/upsert(new path), each
  independently subject to later replacement.
- At most one pending reconcile per engine. A pending reconcile **absorbs**
  that engine's per-file actions — both earlier ones and any arriving while
  it is pending (it walks the live disk when it runs, so they are
  redundant). **Only directory events reset the pending reconcile's
  window; per-file events never do** — otherwise continuous editing would
  starve the reconcile indefinitely. A `git checkout` touching 40
  directories yields one reconcile, not 40.
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
   `canonical path → bare file hash` for ingestable files.
2. **Store set:** `scan(columns=["source", "content_hash"])` (existing
   method) → map `source → hashes`.
3. **Prune:** delete rows whose `source` is under an **active** root
   (`Path(source).is_relative_to(resolved_root)`) but **not in the step-1
   disk set**. This intentionally covers more than deletions: files newly
   matched by `ignore_patterns`, newly gitignored, or with a removed
   extension are pruned too — the invariant demands the index track what
   the engine *would ingest today*, not what once existed.
4. **Ingest:** for each disk file whose `(source, bare_hash)` pair is absent
   from the step-2 map **minus the rows pruned in step 3**, call
   `upsert_file` (which re-applies its own live checks, including the
   cross-source dedup stop). The subtraction matters: a file renamed while
   the server was down is pruned under its old source in step 3 and must
   not be skipped as "already present" in step 4.

**Safety:** reconciliation deletes never touch rows whose `source` is outside
the engine's *active* roots (including relative or unnormalized sources from
older ingests — see Path policy).

## Store & service API changes

- `IVectorStore` gains two methods:
  - `delete_by_source(source: str) -> None` — `LanceDBStore` implements it
    as a SQL predicate delete with single-quote escaping (matching existing
    filter code). Returns nothing: lancedb's `Table.delete` yields a table
    version, not a row count, and a count-then-delete pre-scan per event is
    not worth it. Logging reports the source, not a count.
  - `refresh_fts() -> None` — rebuilds the Tantivy FTS index
    (`create_fts_index(..., replace=True)`), and nothing else.
- `core/ports.py` gains `IWatchBackend` (Architecture).
- `IngestionService` gains `upsert_file(path)` and `remove_source(path)`
  (sequences above), accepts a `PathFilter`, and `ingest_directory` accepts
  multiple roots in a single run (single clear/snapshot/index cycle) while
  applying the `PathFilter` in its walk.

## Index maintenance & compaction

Two tiers. To be precise about cost (an earlier revision overstated this):
the FTS rebuild is the *same operation* in both tiers — what tier 1 skips is
`create_indices()`'s **IVF_PQ retrain**, the genuinely heavy part on a real
vault. FTS rebuild cost is proportional to corpus text and is the accepted
price of immediate searchability.

- **Per drained batch:** `refresh_fts()` on **each store written during that
  drain** — and a delete-only drain counts as written (deleted rows must
  leave the FTS leg too). Stores not touched in the drain are skipped.
  New rows must enter the FTS index to appear in the hybrid search's FTS
  leg; the vector leg finds unindexed rows without retraining.
- **Per engine, every 100 processed files, after each reconciliation pass,
  and at shutdown:** full `create_indices()` + `compact()`. Counters are
  per-engine (the worker queue is global but stores are not).

Bursts (vault sync) coalesce into one drain → one FTS refresh per written
store.

## Error handling

- Every per-file action is wrapped: failures are logged (loguru) and never
  kill the worker loop or the MCP server.
- Non-UTF-8 files are skipped with a warning (existing behavior).
- Backend/observer threads are daemons; graceful shutdown follows the
  stop → force-expire → drain → final-maintenance order (Architecture),
  wrapped in a `finally` around `mcp.run()`.

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
  upsert, rename = remove+upsert), `(engine, path)` keying (two engines,
  same path, no swallowing), reconcile-absorbs-pending, and
  only-directory-events-reset-reconcile; `PathFilter` (extension
  case-insensitivity / `ignore_patterns` basename-and-relative matching /
  gitignore anchoring incl. override-inside-root / None for non-document
  engines); directory-event routing (bypasses PathFilter, root-membership
  only); event→action mapping with mock store + ingester; reconciliation
  diff logic incl. the renamed-while-down case, newly-excluded-file
  pruning, and inactive-root exclusion; graceful-shutdown force-expiry;
  config validation rules (absolute paths, nested roots, shared table_name,
  missing paths, negative debounce, `paths:` on non-document engine).
- **Integration** (tmpdir LanceDB, real chunker/mapper, deterministic fake
  embedder): create/edit/delete/rename files and dispatch events directly
  into `WatcherService` (no real backend, no timers — hermetic); startup
  reconciliation end-to-end; **existing file edited into a duplicate of
  another file** (the source-aware check: old rows deleted, no new rows,
  no stale survivors); identical-content heal-on-reconcile;
  `LanceDBStore.delete_by_source` and `refresh_fts`; CLI
  no-path/override/paths-missing (incl. hard error with `--rebuild`)/
  multi-root behaviors.
- One optional smoke test with the real watchdog backend, marked slow.

## Out of scope (v1) — explicitly deferred

- Root-move tracking; old-root row migration.
- Ignore-pattern / filter-diff detection; config fingerprints. (Covered by
  the documented rebuild rule.)
- Nested `.gitignore` files; live `.gitignore` tracking.
- Symlinked-subdirectory traversal.
- Watch for non-document chunkers (duckdb/api engines).
- A prune command for orphaned rows.
- Per-chunk hash diffing / partial-file updates.
- Any CLI `watch` command — the watcher exists only inside `dbs-vector mcp`.

## Dependencies

- `watchdog` — filesystem events (FSEvents on macOS). Genuinely new;
  imported only by `infrastructure/watch/watchdog_backend.py`.
- `pathspec` — `.gitignore` pattern parsing. Already in `uv.lock` as a
  transitive dependency (1.0.4); promote to a direct dependency.
