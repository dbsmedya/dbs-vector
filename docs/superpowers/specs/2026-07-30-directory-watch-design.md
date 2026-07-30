# Directory Watch — Design Spec

**Date:** 2026-07-30 (revision 4 — simplified for v1: the corpus is
rebuildable at any time, so edge cases resolve to "rebuild", not to
surgical guards. Timer-based FTS refresh. Prior revisions: source-aware
upsert, scoped invariant, event coalescing, layering.)
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

## Design philosophy (v1)

The index is a **cache over the filesystem**, rebuildable at any time with
`ingest --rebuild --force`. Nothing in it is irreplaceable. Therefore:

- Correctness bugs that poison the index *silently and permanently* are
  fixed by design (the source-aware upsert).
- Everything else — config changes, moved roots, symlink oddities, missed
  events — resolves to **"run a rebuild"**, documented once, not to
  per-case machinery.
- Performance tuning beyond the timer-based FTS refresh is deferred until
  real usage shows a problem.

## The Invariant

> **Within currently configured roots, CLI discovery, watcher events, and
> reconciliation use the same path filtering.**

One engine-owned list — `paths:` — with the same extension filtering,
`ignore_patterns`, and `gitignore` filtering drives all three.
Reconciliation prunes files that become *excluded*, not just files that are
deleted. Rows outside the active roots (out-of-root CLI ingests, orphans
from removed roots, old unnormalized sources) are unmanaged: never
re-checked, never pruned — cleaned up by the next rebuild.

### The rebuild rule

Reconciliation compares content hashes, so it detects *file* changes only.
Any change to a watched engine's roots, model, prefixes, tuning profile,
chunker behavior, or filters invalidates the corpus. The single v1 answer,
stated in docs and README:

> **After changing a watched engine's configuration, run one
> `ingest --rebuild --force` for that engine before starting the MCP
> server.**

No config fingerprinting, no root-move tracking, no migration logic.

## Configuration

All filtering/scoping lives at the **engine level**. The `watch:` block
holds watch mechanics **only**.

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

Startup reconciliation is **mandatory** for watched engines — no knob.

### New engine-level fields

- `paths: list[str]` (default `[]`) — ingestion roots: absolute directory
  paths (no globs/files/URLs; the CLI positional keeps supporting those).
  Resolved at config load. Only meaningful for document engines in v1;
  documented as ignored elsewhere.
- `ignore_patterns: list[str]` (default `[".#*", "*~", "*.tmp",
  ".DS_Store"]`) — `fnmatch` globs tested against both the basename and the
  root-relative path; either match excludes. Setting the field **replaces**
  the defaults (standard pydantic semantics).

### Load-time validation (essentials only)

- `watch.enabled: true` requires non-empty `paths:` and
  `chunker_type: "document"`.
- A watched engine's `table_name` must not be shared with another engine
  (prune is root-scoped; a shared table would cross-delete —
  `sql`/`sql-api` share one deliberately and stay unwatched).
- `paths:` entries must be absolute; `debounce_seconds >= 0` (`0` = process
  immediately, useful for tests).

Root **existence is checked at use time, never at config load** (an
unmounted vault must not break `dbs-vector search`). Uniform rule for CLI
and watcher alike: **a missing root is a logged warning and is skipped** —
excluded from walking, watching, and pruning (never treated as an empty
directory). If skipping surprised you, the fix is the rebuild rule.

## CLI behavior

- `dbs-vector ingest --type md` (no path, newly valid) → ingests all
  configured roots in **one run** (one `clear()` if `--rebuild`, one dedup
  snapshot, one index pass — never a per-root loop). Empty/missing
  `paths:` → usage error. Document engines only; other engines keep their
  existing forms.
- `dbs-vector ingest "docs/" --type md` → the explicit path replaces the
  roots for that run, **not the filtering rules**: extension gating,
  `ignore_patterns`, and gitignore apply, and sources are stored
  canonically. (Deliberate change: today's explicit-path runs have no
  ignore filtering and no extension gate on globs.) If the engine is
  watched and the path is outside all configured roots, print a one-line
  notice that these rows won't be watched.
- **Anchoring, one rule:** an explicit path is its own filtering anchor
  (directory → itself; file → parent; glob → longest non-glob prefix; URL →
  none). If that anchor loses a `.gitignore` that a configured root would
  have applied, the next reconciliation prunes the difference — acceptable,
  self-healing.

## Gitignore filter

Opt-in via `gitignore` in `exclusion_filters`. `<root>/.gitignore` (if
present) filters files under that root — parsed with `pathspec`, matched
root-relative, cached for the process lifetime (edits apply on restart).
Enforced at the **file-discovery layer** (walk + watcher event filter),
before files are read. The registered `GitignoreFilter` exists so the name
validates in `FilterRegistry`; its chunk-level hooks are no-ops
(documented). Nested `.gitignore` files: out of scope.

## PathFilter component

`IngestionService` receives no config (DI pattern), so one small component
owns path scoping:

- **`PathFilter`** (services layer, pure logic): roots, extensions,
  `ignore_patterns`, gitignore. Built in `bootstrap.build_dependencies`,
  injected into `IngestionService` and `WatcherService` via `EngineDeps`.
  Answers: *is this path ingestable, and under which root?*
- `None` for non-document engines — their code paths are untouched.
- Explicit-path runs build a run-scoped instance (anchoring rule above).
- Extension matching is case-insensitive (`suffix.lower()`), consistent
  with `DocumentChunker`'s dispatch, and now applies to glob expansion and
  explicit files too (skipped explicit file → visible warning).

## Architecture

Layering follows the repo rule (only `bootstrap` composes infrastructure):

- **`core/ports.py`**: minimal `IWatchBackend` protocol —
  `start(roots, on_event)` / `stop()`; `on_event` gets (path, kind,
  is_directory, dest for moves).
- **`infrastructure/watch/watchdog_backend.py`**: the only module importing
  `watchdog` (FSEvents on macOS).
- **`services/watcher.py`**: `WatcherService` — consumes the port,
  `PathFilter`, and `IngestionService`. Tests dispatch synthetic events
  directly; no real observers or timers.

```
IWatchBackend (observer threads over each engine's active roots)
  → directory event? → mark engine for reconcile (root-membership check only)
  → file event: PathFilter → debounce map {(engine, path): (kind, due_time)}
       ↓ due
single global worker thread (serializes ALL LanceDB writes)
  → upsert_file / remove_source / reconcile; marks engine dirty
every 60s: refresh_fts() on each dirty engine's store
after each reconcile + at shutdown: create_indices() + compact()
```

- **Startup:** watchers start in `start_stdio_server()` after
  `initialize_services()`, only if ≥1 engine is watched; startup
  reconciliation runs on the worker thread (server answers MCP requests
  immediately). Watched engines get their stack via `build_dependencies()`
  — the embedder comes from `_MODEL_CACHE`, already resident. This
  intentionally creates a second `LanceDBStore` handle (writer) alongside
  `SearchService`'s (readers) — that separation is what makes
  MVCC/`checkout_latest()` reads safe. Keep it.
- **Concurrency:** one worker; `MLXEmbedder`'s per-model lock covers
  cross-thread embeds. Accepted: searches queue behind at most one watcher
  embed batch during large reconciles.
- **Shutdown:** stop the backend and worker; exit. No drain ceremony —
  anything pending is picked up by the next startup reconciliation.

## Path canonicalization (simple rule)

Stored `source` must match between walk, events, and deletes:

- Roots are resolved at config load (FSEvents reports real paths, so a
  symlinked root works transparently).
- Below a root, paths are used as walked/evented (absolute, normalized; no
  per-file symlink resolution). Symlinked subdirectories are not traversed
  (existing `rglob` behavior). Symlink setups beyond that: unsupported in
  v1 — rebuild if the index drifts.
- Applies to document-engine file discovery only; URL/duckdb/JSON targets
  are untouched (their `source` is not a filesystem path contract).
- Old rows with relative sources are unmanaged. **Docs + CLI notice: run
  one `--rebuild` per watched engine when first enabling watch.**

## Upsert model: whole-file replace, source-aware

Hash scheme (verified in `_chunk_content_hash`): chunk 0 carries the bare
file-level hash, chunks 1..N carry `{file_hash}_{i}`. "Already ingested"
means the `(source, bare_hash)` **pair** exists — never the hash alone.
(The pair check costs nothing extra and closes the one silent-poisoning
hole: edit B.md into a duplicate of A.md and a global-hash check skips B
forever, keeping B's stale rows.)

`IngestionService.upsert_file(path)`:

1. Read + hash the file.
2. `scan(columns=["source", "content_hash"])` (fresh per action; `scan`
   guarantees `checkout_latest()`). Pair exists → no-op.
3. `delete_by_source(path)`.
4. Bare hash exists under a *different* source → stop (preserves existing
   global dedup; the content is indexed once).
5. Chunk → embed → `ingest_chunks`, unconditionally — never the
   snapshot-based dedup here (it would see the file's own pre-delete hashes
   and skip everything).

`remove_source(path)` → `delete_by_source(path)`.

`get_existing_hashes()` gains the missing `checkout_latest()` (latent
staleness bug, one line). CLI directory ingest keeps its existing
global-snapshot dedup — its edit-to-duplicate hole is pre-existing and
healed for watched engines by startup reconciliation.

Known, accepted: identical-content files index once globally (existing
behavior; deleting the indexed copy heals at the next reconcile);
zero-chunk files re-chunk on every reconcile (harmless).

## Event lifecycle

| Event | Action (after debounce) |
|---|---|
| file created / modified | `PathFilter` rejects → skip; else `upsert_file` |
| file deleted | `remove_source` |
| file moved | `remove_source(old)`; `PathFilter` accepts new → `upsert_file(new)` |
| directory created / moved / deleted | mark engine for reconcile |

Directory events bypass `PathFilter` (they have no extension; only active-
root membership is checked — a spurious reconcile is a cheap no-op).
FSEvents child delivery is unreliable; never enumerate children from the
event.

**Coalescing — last state wins:** one pending action per `(engine, path)`
key (path-only keying would let md and md-granite swallow each other's
events); a later event replaces the earlier and resets its window.
modify→delete = remove; delete→create = upsert; move = two entries. At most
one pending reconcile per engine; it absorbs that engine's per-file
actions (it walks live disk when it runs), and only directory events reset
its window — per-file events never do, so editing can't starve it. Events
arriving during a running reconcile queue normally and run after it
(`upsert_file` is idempotent, overlap is a no-op).

## Reconciliation pass

At startup (always) and on directory events. Per engine:

0. **Active roots only** — a missing/unopenable root is skipped for both
   walking *and* pruning (never "empty directory").
1. **Disk:** walk active roots through `PathFilter` → `{canonical path:
   bare hash}`.
2. **Store:** `scan(columns=["source", "content_hash"])` → `{source:
   hashes}`.
3. **Prune:** delete rows under an active root absent from the disk map —
   including files that became excluded (newly ignored/gitignored): the
   index tracks what the engine *would ingest today*.
4. **Ingest:** for disk files whose `(source, bare_hash)` pair is absent
   from the store map **minus step-3 prunes**, call `upsert_file`. (The
   subtraction handles renamed-while-down: pruned under the old source,
   must not be skipped as "present" under the new.)

Deletes never touch sources outside active roots.

## Store & service API changes

- `IVectorStore` + two methods:
  - `delete_by_source(source: str) -> None` — predicate delete with quote
    escaping. No row count (lancedb returns a version, and a pre-count scan
    per event isn't worth it).
  - `refresh_fts() -> None` — `create_fts_index(..., replace=True)` only.
- `core/ports.py` + `IWatchBackend`.
- `IngestionService` + `upsert_file` / `remove_source`, takes `PathFilter`,
  and `ingest_directory` handles multiple roots in one run.

## Index maintenance (timer-based, simple)

- **Every 60 seconds** (fixed in v1): `refresh_fts()` for each engine
  whose store was written since its last refresh (dirty flag). New rows
  reach the FTS leg of hybrid search within a minute; the vector leg finds
  them immediately without any index work. Deletes are filtered from
  results by LanceDB regardless; the refresh just keeps the FTS index
  tight.
- **After each reconciliation pass and at shutdown:** full
  `create_indices()` + `compact()` for that engine.

No per-file, per-drain, or per-N-files index work. If a minute of FTS
staleness or reconcile-time rebuild cost ever becomes a real problem,
tuning knobs are a future development.

## Error handling

Every per-file action is wrapped: failures log (loguru) and never kill the
worker or the MCP server. Non-UTF-8 files are skipped with a warning
(existing behavior). Backend threads are daemons; watcher teardown sits in
a `finally` around `mcp.run()`.

## Consequences (accepted)

1. Within active roots: one list, one filtering, no divergence.
2. CLI ingest becomes mostly unnecessary for watched engines (startup
   reconcile is a full delta ingest). Don't run CLI ingest on a watched
   table while the server is up — concurrent writers can conflict
   (pre-existing).
3. Out-of-root CLI ingest creates unmanaged rows (notice printed for
   watched engines). Stale until a rebuild.
4. Shrinking `paths:` orphans rows; cleanup = rebuild. (Auto-delete would
   turn a config typo into a table wipe.)
5. md + md-granite watching one vault embed each change once per engine —
   the price of A/B parity.

## Testing

- **Unit:** debounce map (fake clock): three last-state transitions,
  `(engine, path)` keying, reconcile-absorbs, only-dir-events-reset;
  `PathFilter`: extensions (case), patterns (basename + relative),
  gitignore anchoring, `None` for non-document engines; reconciliation
  diff: renamed-while-down, newly-excluded pruning, inactive-root skip;
  config validation set.
- **Integration** (tmpdir LanceDB, real chunker/mapper, fake embedder,
  synthetic events — no timers): end-to-end create/edit/delete/rename;
  startup reconcile; **edit-into-duplicate** (old rows deleted, no stale
  survivors); heal-on-reconcile; `delete_by_source` + `refresh_fts`; CLI
  no-path / override / missing-paths / multi-root single-run.
- One slow-marked smoke test with the real watchdog backend.

## Out of scope (v1) — explicitly deferred

Root-move tracking; config fingerprints / filter-diff detection; nested or
live-tracked `.gitignore`; symlinked-subdir traversal; watch for
non-document chunkers; a prune command; per-chunk/partial updates; FTS
timing knobs; any CLI `watch` command (the watcher exists only inside
`dbs-vector mcp`).

## Dependencies

- `watchdog` — new; imported only by the infrastructure backend.
- `pathspec` — already transitive in `uv.lock` (1.0.4); promote to direct.
