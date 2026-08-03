# Configuration Guide

Everything `dbs-vector` does is driven by one YAML file. This guide explains where
that file is found, how it is organised, and what every field means.

If you just want a working setup, run `dbs-vector init` — it interviews you and
writes a valid file. Come back here when you want to change something it chose,
or add a second engine.

---

## 1. Where the config comes from

### Precedence

| How | Wins over | Notes |
|---|---|---|
| `--config-file` / `-c` | everything | A **global** option — it goes *before* the subcommand |
| `./config.yaml` | nothing | The default, relative to your current working directory |

```bash
dbs-vector --config-file ~/corpora/notes.yaml search "chunking strategy"
dbs-vector -c ~/corpora/notes.yaml mcp
```

### `--config-file` must precede the subcommand

This works:

```bash
dbs-vector --config-file /abs/notes.yaml mcp
```

This does **not** do what it looks like:

```bash
dbs-vector mcp --config-file /abs/notes.yaml    # ← wrong
```

The config is loaded and validated before the subcommand runs. In the second
form, `./config.yaml` in the current directory is loaded first — so an unrelated
or malformed file sitting there will fail the command before your override is
ever consulted.

`init` is the one exception: it skips config loading entirely, because it exists
to *generate* the file. A broken `config.yaml` in the current directory never
blocks it.

### `DBS_CONFIG_FILE`

The CLI **exports** this variable so that any process it spawns inherits the same
config path. It is read as an input only by direct library calls that pass no
explicit path.

It does **not** override the command-line default. Setting `DBS_CONFIG_FILE` and
then running `dbs-vector search ...` still loads `./config.yaml`. Use
`--config-file` on the command line.

### Why generated paths are absolute

`init` writes an absolute `db_path`, and an absolute `--config-file` into
`.mcp.json`. An MCP client spawns the server as a subprocess with a working
directory you do not control, so a relative path resolves somewhere unintended —
usually to a database that does not exist, and an empty result set with no error.

---

## 2. The three layers

Configuration is split across three places, by how often each thing changes.

| Layer | Lives in | Holds | Changes when |
|---|---|---|---|
| **Model contract** | code — `ModelRegistry` | vector dimension, context cap, dtype, default prefixes | you add a new embedding model |
| **Tuning profile** | `profiles:` in YAML | five numbers: token and batch budgets | you tune for a corpus or a machine |
| **Engine** | `engines:` in YAML | pipeline shape; which model, which profile | you add a new searchable corpus |

The split exists because these are properties of *different things*. A model's
context cap is true wherever that model runs. A chunk size is a property of your
documents. An engine is a property of your deployment. Merging them means editing
model facts to change a corpus setting.

### Worked example

```yaml
system:
  db_path: /Users/you/lancedb_dbs_vector
  nprobes: 20

profiles:
  notes-medium:                 # ← a profile name you choose
    max_token_length: 2048
    chunk_max_chars: 2560
    batch_size: 16
    chunk_target_tokens: 768
    chunk_max_tokens: 1536

engines:
  notes:                        # ← the engine name; becomes MCP tool search_notes
    description: Personal notes and documentation.
    model: granite-r2           # ← a ModelRegistry key (layer 1)
    tuning_profile: notes-medium # ← a profiles: key (layer 2)
    mapper_type: document
    chunker_type: document
    table_name: notes_vault
    workflow: notes_search
    paths:
      - /Users/you/notes
```

Reading that engine: it embeds with `granite-r2`, whose contract supplies a
768-dimension vector and a 32768-token cap; it sizes chunks per `notes-medium`;
it stores rows in the `notes_vault` LanceDB table; and it exposes an MCP tool
called `search_notes`.

---

## 3. Models

A model is referenced by a short key, not by its Hugging Face path. The key
resolves to a contract in code.

| Key | Model | Context | Dim | Needs prefixes |
|---|---|---|---|---|
| `gemma-bf16` | `mlx-community/embeddinggemma-300m-bf16` | 2 048 tokens | 768 | yes |
| `granite-r2` | `ibm-granite/granite-embedding-311m-multilingual-r2` | 32 768 tokens | 768 | no |

### Which to choose

**`gemma-bf16` is the better default for English prose and documentation.** It is
instruction-tuned, fast on Apple Silicon, and consistently strongest on English
technical writing. Because it is instruction-tuned it needs asymmetric prefixes —
a different instruction for stored passages than for queries — and those are
filled in for you.

**Reach for `granite-r2`** when your corpus has substantial non-English content
(200+ languages, against Gemma's 100+), when individual documents exceed Gemma's
2 048-token context, or when you want to A/B a second engine against a Gemma
baseline on the same corpus. Granite R2 is a symmetric bi-encoder trained
*without* instruction prefixes — leave `passage_prefix` and `query_prefix` empty.

The context cap is not just about long documents: it also bounds how large a
chunk can be. Gemma cannot hold the "large" chunk granularity, because a
2 048-token maximal chunk leaves no room for the prefix that gets prepended to
it. `init` only offers granularities the chosen model can actually hold.

See [README_EMBEDDINGS.md](README_EMBEDDINGS.md) for model internals and
[README_granite.md](README_granite.md) for Granite tuning recipes.

### Adding a model

Models are registered in code — one `ModelRegistry.register()` call declaring the
Hugging Face path, vector dimension, context cap, attention-mask dtype, and any
default prefixes. Nothing else changes: profile derivation, `init`, the CLI and
the MCP layer all read the contract rather than a hardcoded list.

---

## 4. Engine kinds

An engine's `chunker_type` decides what it can read and which fields apply. The
`mapper_type` decides the table schema.

| `chunker_type` | `mapper_type` | Reads | Extra fields it uses |
|---|---|---|---|
| `document` | `document` | `.md` and `.txt` files on disk | `paths`, `ignore_patterns`, `exclusion_filters`, `watch` |
| `sql` | `sql` | MySQL slow-query log as JSON | — |
| `duckdb` | `sql` | slow-query log in a DuckDB file | `duckdb_query` |
| `api` | `sql` | paginated HTTP slow-query API | `api_*` fields |

Fields that do not apply to a kind are inert, not errors — a `duckdb` engine may
carry a `paths:` list, and it will be ignored. Two exceptions are enforced:

- `watch.enabled` requires `chunker_type: document` and a non-empty `paths:`.
- Document engines require `chunk_target_tokens` and `chunk_max_tokens` above
  zero in their profile. For SQL kinds those two fields are unused and are
  conventionally set to `0`.

Markdown is parsed with `markdown-it-py` and split along real document structure —
headings, lists, tables, fenced code — so a chunk ends where a section does and a
code block is never cut in half. Plain `.txt` has no structure to follow and falls
back to naive splitting at `chunk_max_chars`.

Details per source: [README_DOCS.md](README_DOCS.md),
[README_SQL.md](README_SQL.md), [README_duckdb.md](README_duckdb.md),
[README_REMOTE_SQL_API.md](README_REMOTE_SQL_API.md).

---

## 5. Field reference

### `system:`

| Field | Default | Meaning |
|---|---|---|
| `db_path` | `./lancedb_dbs_vector` | Where LanceDB stores its tables. Use an absolute path. |
| `nprobes` | `20` | Vector-index partitions probed per query. Higher is more accurate and slower. |
| `log_level` | `INFO` | Logging threshold. |
| `log_serialize` | `false` | Emit logs as JSON. |
| `memory_budget_gb` | auto | Overrides the detected Metal budget. Leave unset unless you have a reason. |
| `mlx_memory_limit_gb` | unset | Hard cap on the MLX allocator. |
| `mlx_cache_limit_gb` | unset | Cap on the MLX buffer cache. `0` disables caching. |

Unknown keys in `system:` are rejected with the list of accepted ones, so a typo
fails immediately rather than being silently ignored.

### `profiles:`

Five numbers per profile. All are validated against the engine's model and your
machine's memory budget when the config loads.

| Field | Required | Meaning |
|---|---|---|
| `max_token_length` | > 0 | Truncation cap fed to the embedder. Must not exceed the model's context. |
| `chunk_max_chars` | ≥ 0 | Character cap for the naive `.txt` path only. Ignored for Markdown. |
| `batch_size` | > 0 | Passages embedded per forward pass. The main memory lever. |
| `chunk_target_tokens` | > 0 for document engines | The size a chunk aims for. |
| `chunk_max_tokens` | > 0 for document engines | The size a chunk may not exceed. |

Coherence is enforced:
`chunk_target_tokens ≤ chunk_max_tokens ≤ max_token_length ≤ model context`.

Leave headroom between `chunk_max_tokens` and `max_token_length`. The passage
prefix is prepended *after* chunking, so a maximal chunk in a profile with no
headroom truncates the moment a prefix is added.

If a profile will not fit your GPU, loading fails with the numbers that *would*
have fit. See [README_PROFILES.md](README_PROFILES.md) for the memory model.

### `engines:`

The engine name must match `^[a-z0-9][a-z0-9_-]*$` — lowercase letters, digits,
dash, underscore, starting with a letter or digit. It becomes an MCP tool name,
with dashes converted to underscores: `my-docs` exposes `search_my_docs`.

| Field | Default | Meaning |
|---|---|---|
| `description` | required | Shown in MCP tool listings. Write it for the model that will read it. |
| `model` | required | A `ModelRegistry` key. |
| `tuning_profile` | required | A `profiles:` key. |
| `mapper_type` | required | `document` or `sql`. Decides the table schema. |
| `chunker_type` | required | `document`, `sql`, `duckdb`, or `api`. |
| `table_name` | required | LanceDB table. A watched engine needs one all to itself. |
| `workflow` | required | Label recorded on every row; lets you tell A/B variants apart. |
| `family` | `mapper_type` | Which MCP tool set to expose: `document` or `sql`. Rarely set by hand. |
| `passage_prefix` | `""` | Instruction prepended to stored text. Model-specific. |
| `query_prefix` | `""` | Instruction prepended to queries. Model-specific. |
| `similarity_floor` | unset | Minimum cosine similarity for a result to be returned. See below. |
| `exclusion_filters` | `[]` | Content filters: `excalidraw`, `compressed_json`, `gitignore`. |
| `paths` | `[]` | Roots to index when no explicit path is given. Document engines. |
| `ignore_patterns` | `[".#*", "*~", "*.tmp", ".DS_Store"]` | Glob patterns to skip. **See the trap below.** |
| `duckdb_query` | unset | Custom `SELECT` for DuckDB extraction. |
| `api_base_url` | `""` | Endpoint for the `api` chunker. |
| `api_key` | `""` | Bearer token for that endpoint. |
| `api_page_size` | `200` | Records per page. |
| `api_since_days` | `15` | How far back to pull. |
| `api_timeout_sec` | `30` | Per-request timeout. |
| `api_min_execution_ms` | `0.0` | Skip queries faster than this. |
| `api_database` | `""` | Restrict to one database. |

#### The `ignore_patterns` trap

Setting `ignore_patterns` **replaces** the default list — it does not extend it.

```yaml
    # WRONG: silently stops filtering Emacs lock files, backups and .DS_Store
    ignore_patterns:
      - "build/*"

    # RIGHT: repeat the defaults you still want
    ignore_patterns:
      - ".#*"
      - "*~"
      - "*.tmp"
      - ".DS_Store"
      - "build/*"
```

This matters more than it looks. An Emacs lock file named `.#notes.md` has the
suffix `.md`, so it passes the extension gate and gets indexed as a document
unless `.#*` is present.

Patterns are `fnmatch`-style and are tested against both the file's basename and
its path relative to the root. Note that `*` crosses `/` here, unlike in a
`.gitignore` — so `.ayder/archived/*` matches at any depth beneath that
directory.

#### `similarity_floor`

Every engine ships with this unset, and `init` never writes one.

A floor is a claim about *one corpus measured on one machine* — the point below
which a result is noise rather than a weak match. That number does not transfer
between corpora, and it stops being valid the moment you change model, prefixes,
chunker, profile, `nprobes`, or pool geometry. An inherited floor silently
suppresses good results.

If you want one, measure it. The procedure is in
[README_CALIBRATE_CORPUS.md](README_CALIBRATE_CORPUS.md). "No safe floor exists
for this corpus" is a legitimate outcome.

### `watch:`

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Re-ingest changed files while `dbs-vector mcp` runs. |
| `debounce_seconds` | `3.0` | Quiet period before a change is processed. `0` processes immediately. |

Requires `chunker_type: document`, a non-empty `paths:`, and a table no other
engine shares. **After any config change to a watched engine, run one
`ingest --rebuild --force`** — the index is a rebuildable cache, and stale
settings persist in it otherwise. See [README_WATCH.md](README_WATCH.md).

---

## 6. What `init` generates, and what you add by hand

`init` writes **one document engine**: a `system:` block, one profile, one engine,
and a matching `.mcp.json` entry. That is the common case and the one it can
validate end to end.

Everything else is a hand edit to the same file:

- **A second engine over the same corpus** — copy the engine block, give it a new
  name, `table_name`, and `workflow`, and point it at a different model or
  profile. This is how you A/B two models; both appear as separate MCP tools.
- **SQL engines** — add an engine with `chunker_type: sql`, `duckdb`, or `api`
  and `mapper_type: sql`. See the per-source guides.
- **A similarity floor** — only after measuring one.
- **Watch** — set `watch.enabled: true` on a document engine that has `paths:`.

After editing, check the file loads. Every command validates the config before it
runs, so the quickest check is any command at all:

```bash
dbs-vector --config-file config.yaml search "test" --type your-engine-name
```

A validation failure names the engine, the field, and what to do about it — and
happens before any search is attempted. Once the config is valid, use the
`list_engines` MCP tool to inspect loaded engines, their model contracts,
profiles and table names from your client.

---

## 7. Command reference

### `ingest`

```bash
# Index one path explicitly
dbs-vector ingest "docs/" --type md

# Index every configured root for an engine (uses its paths:)
dbs-vector ingest --type md

# Wipe and re-index from scratch
dbs-vector ingest --type md --rebuild --force
```

| Option | Alias | Meaning |
|---|---|---|
| `--type` | `-t` | Engine to ingest into. Default `md`. |
| `--rebuild` | `-r` | Drop the table and recreate it. |
| `--force` | `-f` | Skip the rebuild confirmation prompt. |
| `--query` | `-q` | Custom `SELECT` for DuckDB extraction. |

### `search`

```bash
dbs-vector search "how does dedupe work?" --type md --limit 10
dbs-vector search "SELECT * FROM users" --type sql --json | jq '.[].chunk.source'
```

| Option | Alias | Meaning |
|---|---|---|
| `--type` | `-t` | Engine to search. Default `md`. |
| `--source` | `-s` | Restrict to part of the corpus: a full stored path, a trailing fragment (`specs/api.md`, `api.md`), or a directory to scope to. SQL engines take a database name. Not a glob. |
| `--limit` | `-l` | Maximum results. Default `5`. |
| `--min-time` | | SQL engines only: minimum execution time in ms. |
| `--min-similarity` | | Admission floor for this query only, overriding the engine's. |
| `--no-similarity-floor` | | Disable admission filtering entirely for this query. |
| `--json` | | Emit the full envelope — `floor`, `inspected`, `best_rejected`, and results carrying `similarity` / `retrieved_by` / `rrf_score` — as JSON on stdout. Logs go to stderr, so it pipes cleanly into `jq`. |

An empty result from a floored engine means *low confidence*, not *nothing
indexed*. `--json` shows you `best_rejected`, which is how you tell the two
apart; `--no-similarity-floor` confirms it directly.

### Indexes are built during ingestion

Two indexes are created at the end of every `ingest` run: an `IVF_PQ` vector
index (only once the table exceeds 256 rows) and a Tantivy full-text index, which
hybrid search requires.

If you see **"Cannot perform full text search unless an INVERTED index has been
created"**, the full-text index was never built for that table. Re-run ingestion:

```bash
dbs-vector ingest --type md --rebuild --force
```

---

## See also

- [README_PROFILES.md](README_PROFILES.md) — profile knobs and the memory model
- [README_WATCH.md](README_WATCH.md) — directory watching
- [README_MCP.md](README_MCP.md) — MCP tools and A/B workflows
- [README_EMBEDDINGS.md](README_EMBEDDINGS.md) — embedding models
- [README_CALIBRATE_CORPUS.md](README_CALIBRATE_CORPUS.md) — measuring a similarity floor
- [README_ARCHITECTURE.md](README_ARCHITECTURE.md) — how ingestion and storage work
