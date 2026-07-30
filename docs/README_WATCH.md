# Directory Watch

Automatically re-ingest file changes from configured directories while
`dbs-vector mcp` is running. The MCP server already holds the embedding model
resident, so re-ingesting one changed file is a sub-second in-process
operation instead of a multi-second CLI cold start.

**v1 scope:** engines with `chunker_type: "document"` (`md`, `md-granite`).

## The rebuild rule — read this first

The index is a **cache over the filesystem**, rebuildable at any time.
Reconciliation compares *content hashes*, so it detects file changes only.
Any change to a watched engine's roots, model, prefixes, tuning profile,
chunker behaviour or filters invalidates the corpus.

> **After changing a watched engine's configuration, run one
> `uv run dbs-vector ingest --type <engine> --rebuild --force` for that engine
> before starting the MCP server.**

The same applies the first time you enable watch on an existing table: older
rows may carry relative `source` paths, which the watcher cannot manage.

## Configuration

```yaml
engines:
  md:
    # ... existing fields ...
    paths: ["/Users/you/vault"]                            # engine-owned ingestion roots
    ignore_patterns: [".#*", "*~", "*.tmp", ".DS_Store"]    # defaults shown
    exclusion_filters: [excalidraw, compressed_json, gitignore]
    watch:
      enabled: true          # default false
      debounce_seconds: 3.0  # per file; >= 0, default 3.0
```

- `paths:` — **absolute** directory roots, resolved at config load. No globs,
  files or URLs (the CLI positional keeps supporting those). Existence is
  checked at *use* time, never at load: an unmounted vault must not break
  `dbs-vector search`.
- `ignore_patterns:` — `fnmatch` globs tested against both the basename and
  the root-relative path; either match excludes. Setting the field
  **replaces** the defaults.
- `gitignore` in `exclusion_filters:` — opt-in. `<root>/.gitignore` filters
  files under that root, matched root-relative, cached for the process
  lifetime (edits apply on restart). Nested `.gitignore` files are out of scope.

Startup reconciliation is mandatory for watched engines — there is no knob.

### Validation

- `watch.enabled: true` requires a non-empty `paths:` and
  `chunker_type: "document"`.
- A watched engine's `table_name` must not be shared with another engine
  (prune is root-scoped, so a shared table would cross-delete). `sql` and
  `sql-api` share one deliberately and stay unwatched.

## CLI

| Command | Behaviour |
|---|---|
| `dbs-vector ingest --type md` | Ingests every configured root in **one run**: one `clear()` on `--rebuild`, one dedup snapshot, one index pass. Usage error if `paths:` is empty. |
| `dbs-vector ingest "docs/" --type md` | The explicit path replaces the roots for that run, **not** the filtering rules: extension gating, `ignore_patterns` and gitignore still apply, and sources are stored canonically. |

An explicit path is its own filtering anchor — directory → itself, file →
parent, glob → longest non-glob prefix. If that anchor loses a `.gitignore`
a configured root would have applied, the next reconciliation prunes the
difference. Self-healing, by design.

If the engine is watched and the path is outside all configured roots, the CLI
prints a notice: those rows will not be watched or reconciled.

## What happens at runtime

| Event | Action (after the debounce window) |
|---|---|
| file created / modified | Skipped if the filter rejects it; otherwise whole-file replace |
| file deleted | All rows for that source are deleted |
| file moved | Delete the old source; ingest the new one if it passes the filter |
| directory created / moved / deleted | Mark the engine for reconciliation |

Coalescing is last-state-wins per `(engine, path)`: a later event replaces the
earlier one and resets its window. A pending reconcile absorbs that engine's
per-file actions (it walks live disk when it runs), and only directory events
reset the reconcile window — so continuous editing can never starve it.

**Index maintenance:** every 60 seconds, the FTS index is refreshed for each
engine written since its last refresh, so new rows reach the full-text leg of
hybrid search within a minute. The vector leg finds them immediately. After a
reconciliation pass that changed something, and at shutdown, the full vector +
FTS indices are rebuilt and the dataset is compacted.

**Concurrency:** one worker thread performs every LanceDB write. During a large
reconcile, searches may queue behind at most one watcher embed batch.

## Limits (v1, accepted)

1. Don't run a CLI ingest against a watched table while the server is up —
   concurrent writers can conflict.
2. Out-of-root CLI ingests create unmanaged rows: never re-checked, never
   pruned. Cleaned up by the next rebuild.
3. Shrinking `paths:` orphans rows; cleanup is a rebuild. (Auto-deleting would
   turn a config typo into a table wipe.)
4. Identical-content files are indexed once globally. Deleting the indexed copy
   heals at the next reconcile.
5. `md` + `md-granite` watching one vault embed each change once per engine —
   the price of A/B parity.
6. Symlinked roots work (FSEvents reports real paths). Symlinked
   *subdirectories* are not traversed. Anything beyond that: rebuild if the
   index drifts.

## Out of scope (v1)

Root-move tracking; config fingerprints / filter-diff detection; nested or
live-tracked `.gitignore`; symlinked-subdir traversal; watch for non-document
chunkers; a prune command; per-chunk/partial updates; FTS timing knobs; any
CLI `watch` command — the watcher exists only inside `dbs-vector mcp`.
