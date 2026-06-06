# Header-Aware Section Chunking for Markdown

- **Date:** 2026-06-06
- **Status:** Approved (design) — pending implementation plan
- **Author:** sinan alyuruk (with Claude)
- **Affects:** `md`, `md-granite` engines (`chunker_type: document`)

## 1. Problem & motivation

A performance/quality investigation of markdown ingestion against the
real Obsidian "Ideaspace" vault (168 `.md` files, 18 MB) surfaced that the
*chunking*, not the throughput knobs, is what limits retrieval quality.

Measured chunk distribution under the current `DocumentChunker`:

| profile | chunks | median tok | p90 tok | >2048 tok | <16 tok (noise) |
|---|---|---|---|---|---|
| `md` (max_chars=1000) | 1176 | 46 | 274 | 10 (0.9%) | **309 (26.3%)** |
| `md-granite` (max_chars=3000) | 1089 | 40 | 341 | 10 (0.9%) | **308 (28.3%)** |

Findings:

1. **~27% of chunks are <16 tokens** — bare headings and single lines
   embedded as standalone vectors. These pollute retrieval (a vector for
   `## Setup` matches everything weakly).
2. **The few oversized chunks are almost all junk.** The largest are
   Excalidraw drawings and `compressed-json` machine blobs
   (`Btree index.md` = 11020 tok, `JMX Proxysql Drawing.md` = 10422 tok).
   They are not natural language; embedding them is wasted compute and the
   sole cause of truncation.
3. **Truncation barely touches real content** — only 0.9% of chunks exceed
   2048 tokens, and most of those are the junk in (2).

The current chunker keeps code fences atomic regardless of size and groups
all other top-level blocks greedily by character count. It has no concept
of a heading-scoped section, so headings become standalone noise and large
real blocks (tables, code, lists) are never split at sensible boundaries.

### Quality clarification (recorded to prevent re-litigation)

- **`batch_size` has zero effect on embedding quality.** Each sequence is
  embedded independently; padding is masked by `attention_mask`. It is a
  throughput/memory knob only. (Out of scope for this spec.)
- **`max_token_length` affects quality only via truncation.** Raising it
  never degrades a vector. But cramming more tokens into a single 768-dim
  vector dilutes meaning ("washout"). The fix for oversized real content is
  to *split* it, not to raise the budget and embed it whole.

## 2. Goals & non-goals

**Goals**
- Passage-level retrieval: each chunk is a self-contained, heading-scoped
  passage suitable for returning as "the section that answers the query."
- Eliminate bare-heading / sub-16-token noise vectors.
- Never truncate: any oversized block is split at natural boundaries, with
  a hard token-window fallback as a last resort.
- Make content exclusion **opt-in and configurable per engine** via a
  pluggable filter module (default: exclude nothing).
- Size chunks by **tokens** using the same tokenizer that embeds them.

**Non-goals (YAGNI)**
- Semantic/embedding-similarity chunking.
- Per-engine A/B chunker variants (the registry name `document` is shared).
- Changing the `.txt` naive-splitting path.
- `batch_size` retuning.
- Trimming the truncation-alarm's second tokenizer pass
  (`mlx_engine.py:61`). With this design truncation never fires, so the
  alarm becomes dead overhead — left as an optional future cleanup.

## 3. Design

### 3.1 Section-based algorithm

Rework only the markdown path of `DocumentChunker` (keep `_chunk_text` for
`.txt`). The unit is the **section**: a heading plus all content beneath it
until the next heading of equal-or-higher level. Top-level document content
before the first heading is its own implicit section (heading path empty).

Algorithm:

1. Parse with `markdown-it-py` (already a dependency). Walk top-level
   tokens, tracking the current heading stack to build a **heading path**
   (e.g. `Btree Index > Implementation`).
2. Group blocks under their section.
3. For each section, **greedily pack** its blocks into sub-chunks up to
   `target_tokens`; flush when the next block would exceed it.
4. If a **single block** exceeds `max_tokens`, split it at its natural
   boundary:
   - **code fence** → by lines, each part re-wrapped in ` ```lang ` fences
     and marked `(code, part N/M)`.
   - **list** (`bullet_list`/`ordered_list`) → by items, never mid-item.
   - **table** → by rows, repeating the header row in each part.
   - **paragraph** → by sentence, then newline.
   - **hard-window fallback**: if a single line/item/sentence *still*
     exceeds `max_tokens` (e.g. a one-line `compressed-json` blob that was
     not excluded), window it by tokens. Guarantees no truncation ever.
5. **Tiny-merge**: a sub-chunk whose body is `< min_tokens` (constant 32)
   merges into the previous sibling sub-chunk in the same section; a
   heading-only section merges into its first content chunk. No
   bare-heading vector survives.

### 3.2 Heading context & chunk metadata

Each emitted chunk's `text` is `"<heading path>\n\n<body>"` so the embedded
and displayed passage carries its topical anchor. The existing (currently
null) `Chunk` fields are populated — **no schema change**, the
`DocumentMapper` already declares them (`mappers.py:25-27,46-48,76-78`):

- `parent_scope` = heading path (e.g. `Btree Index > Implementation`)
- `node_type` ∈ {`section`, `code`, `table`, `list`}
- `line_range` = `"<start>-<end>"` from markdown-it `token.map`

`content_hash` stays file-level (whole-file SHA-256), so file-level
deduplication and incremental re-runs are unchanged.

### 3.3 Token budgets (per profile)

Sizing is token-aware via an injected `length_fn`. Token budgets live in
the `profiles:` block, so the two engines diverge intentionally (this is
why they have separate profiles):

| profile | model | `chunk_target_tokens` | `chunk_max_tokens` | `max_token_length` (embedder cap) |
|---|---|---|---|---|
| `gemma-md` | gemma (768-dim, ceiling 2048) | **512** | **1024** | **2048** |
| `granite-md-medium` | granite (768-dim) | **768** | **1536** | **2048** |

Rationale: factor 1 (model ceiling) is the only model-specific bound;
factors 2 (fixed 768-dim vector capacity → washout past a few hundred
tokens) and 3 (passage-level goal) push toward the low end and are
model-agnostic. The divergent granite values (768/1536) are an
**experiment to compare against gemma's 512/1024**, not an assumption —
see §6. `min_tokens` = 32 (code constant, not a profile knob).
`max_token_length` is a truncation safety net only; with §3.1 it should
never be hit.

### 3.4 Pluggable exclusion filters

New module `infrastructure/chunking/filters.py`:

```python
class IContentFilter(Protocol):           # add to core/ports.py
    name: str
    def should_skip_file(self, filepath: str, content: str) -> bool: ...
    def should_drop_block(self, text: str, info_string: str | None) -> bool: ...

class FilterRegistry:                      # open/closed, like ModelRegistry
    @classmethod
    def register(cls, flt: IContentFilter) -> None: ...
    @classmethod
    def resolve(cls, names: list[str]) -> list[IContentFilter]: ...
    #   unknown name -> ValueError at config-load time
```

Built-in registered filters:

| name | `should_skip_file` | `should_drop_block` |
|---|---|---|
| `excalidraw` | filename matches `*.excalidraw.md`, or frontmatter contains `excalidraw-plugin` | `json` fence whose body contains `"type": "excalidraw"` |
| `compressed_json` | — | fence whose info-string is `compressed-json` |

The chunker skips a file if **any** filter's `should_skip_file` is true,
and drops a block if **any** `should_drop_block` is true. Adding a new
filter later = one `register()` call + the name in config; no changes to
chunker, bootstrap, services, or CLI.

### 3.5 Config changes

`TuningProfile` (`config.py:11`) gains two optional fields:

```python
chunk_target_tokens: int = Field(default=0, ge=0)   # 0 => not token-chunked (SQL atomic)
chunk_max_tokens:    int = Field(default=0, ge=0)
```

`EngineConfig` (`config.py:22`) gains:

```python
exclusion_filters: list[str] = []   # per-engine; default: exclude nothing
```

Validation additions (alongside existing checks ~`config.py:199-308`):
- If `chunk_target_tokens > 0`: require `chunk_max_tokens >= chunk_target_tokens`
  and `chunk_max_tokens <= max_token_length`.
- `exclusion_filters` names must all resolve in `FilterRegistry` (else
  load-time `ValueError` listing known names).
- Add `chunk_target_tokens` / `chunk_max_tokens` to the allowed profile-key
  set (`config.py` ~122-125) so the `extra="forbid"` schema accepts them.

`config.yaml` for the two md engines/profiles:

```yaml
profiles:
  gemma-md:          {max_token_length: 2048, chunk_max_chars: 0, batch_size: 64,
                      chunk_target_tokens: 512, chunk_max_tokens: 1024}
  granite-md-medium: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 16,
                      chunk_target_tokens: 768, chunk_max_tokens: 1536}

engines:
  md:
    ...
    exclusion_filters: []        # opt in per your vault, e.g. [excalidraw, compressed_json]
  md-granite:
    ...
    exclusion_filters: []
```

(`chunk_max_chars` drops to 0 for the md profiles — the document path no
longer uses it; SQL/duckdb/api profiles keep their existing values.)

### 3.6 Wiring (`bootstrap.py`)

The embedder is built before the chunker, so token-aware sizing and filter
resolution are injected there:

1. Add `MLXEmbedder.count_tokens(text: str) -> int` (thin wrapper over the
   tokenizer id count) and declare it on `IEmbedder`.
2. `EngineConfig.chunker_kwargs` document branch stops returning
   `max_chars` (returns `{}`); the document chunker no longer takes
   `max_chars`. Token budgets, filters, and `length_fn` are injected by
   bootstrap instead (so config stays free of model/tokenizer coupling).
3. In `build_dependencies`, for the document chunker, extend kwargs:
   - `length_fn = embedder.count_tokens`
   - `filters = FilterRegistry.resolve(engine.exclusion_filters)`
   - `target_tokens = profile.chunk_target_tokens`,
     `max_tokens = profile.chunk_max_tokens`
4. `DocumentChunker` defaults: `length_fn=len` (chars), `filters=[]`, and
   sensible default `target_tokens`/`max_tokens` so it stays standalone and
   unit-testable without a model.

## 4. Component boundaries

- `DocumentChunker` — owns the section algorithm and splitting. Depends on
  `markdown-it-py`, an injected `length_fn`, and a list of `IContentFilter`.
  Testable with a char `length_fn` and fake filters; no model required.
- `filters.py` — owns filter protocol, registry, built-ins. No dependency
  on chunker internals.
- `bootstrap.py` — the only place that couples chunker ↔ embedder tokenizer
  and resolves filter names → instances.
- `config.py` — owns schema + validation of the new knobs.

## 5. Migration

- One-time rebuild per engine: `dbs-vector ingest --type md --rebuild --force`
  and `--type md-granite --rebuild --force`. Chunk boundaries change, so all
  chunk ids/content change; a clean rebuild is required.
- **No LanceDB schema change** — the metadata columns already exist. After
  rebuild, incremental (file-hash dedup) re-runs work as before.

## 6. Testing & comparison

**Unit (`DocumentChunker`, char `length_fn` + fake filters):**
- heading path prepended; bare heading never emitted alone
- section grouping by heading level
- oversized code fence → line-split, each part re-fenced + `(part N/M)`
- oversized list → item-split (never mid-item)
- oversized table → row-split with repeated header
- single over-`max_tokens` line → hard-window fallback (invariant: **no
  chunk exceeds `max_tokens`**)
- tiny-merge: sub-`min_tokens` body merges into previous sibling
- `parent_scope` / `node_type` / `line_range` populated

**Unit (filters):**
- `excalidraw` skips `*.excalidraw.md` and drops excalidraw `json` blocks
- `compressed_json` drops `compressed-json` fences
- `exclusion_filters: []` ⇒ nothing excluded
- unknown filter name ⇒ load-time `ValueError`

**Integration:** ingest a small fixture vault into a tmp LanceDB; assert
no <16-token chunks, no chunk over `max_tokens`, excluded files absent, and
a known query returns the expected section.

**Regression / comparison:** re-run the corpus-distribution analysis used
in §1 before vs after; assert noise (<16 tok) → ~0 and truncation count → 0.
Then **compare gemma 512/1024 vs granite 768/1536**: ingest both, run a
shared set of ~10 real vault queries through `dbs-vector search`, and record
result granularity/relevance side by side to decide whether granite's
larger passages help or wash out. Document the outcome; adjust the granite
profile if 512/1024 wins.

## 6a. SQL engines: no impact (hard guarantee)

The `sql`, `sql-api`, `sql-granite`, and `sql-api-granite` engines must be
**completely unaffected** — their chunking is already optimized. They use
different chunkers (`DuckDBChunker`/`ApiChunker`), a different mapper
(`SqlMapper`), and different tables (`query_vault*`); only `DocumentChunker`
changes here. Every shared file is inert for them, and the implementation
**must preserve** these properties:

- `EngineConfig.chunker_kwargs` (`config.py:69-86`): the `duckdb` and `api`
  branches `return` early, before the document fall-through. Do not move the
  token-budget injection above those early returns.
- New profile fields (`chunk_target_tokens`, `chunk_max_tokens`) are
  optional, default `0`; SQL profiles omit them. `chunk_max_chars` is
  retained for SQL atomic profiles.
- New `EngineConfig.exclusion_filters` defaults to `[]`; SQL engines omit it.
- New validation rules are gated (`chunk_target_tokens > 0`,
  `exclusion_filters` non-empty) exactly like the existing
  `chunk_max_chars > 0` rule (`config.py:301`), so SQL profiles skip them.
- `bootstrap.py` injects `length_fn`/`filters`/token budgets **only for the
  document chunker** (gate on `engine.chunker_type == "document"`).
- `MLXEmbedder.count_tokens` is additive; `embed_batch` is unchanged, so SQL
  embeddings are bit-identical. No `SqlMapper`/`query_vault*` schema change;
  **no SQL re-ingest required.**

**Guard test:** building dependencies for `sql-api` (and `sql`) yields a
chunker constructed with no `filters`/`length_fn`/token-budget kwargs, and
the existing SQL/DuckDB/API chunker unit tests remain green unchanged.

## 7. Risks

- **Token-aware chunking couples chunking to the tokenizer.** Mitigated by
  the `length_fn` default (`len`) and injection only in `bootstrap.py`;
  tests run without a model.
- **markdown-it token edge cases** (nested lists, loose lists, HTML blocks,
  front-matter). Mitigated by fixture-driven unit tests and the hard-window
  fallback guaranteeing the size invariant regardless of parse quirks.
- **Heading-path prefix slightly inflates token counts.** Budgeted: the
  prefix counts toward `length_fn`, so packing accounts for it.

## 8. Out of scope (future)

- Trimming the now-dead truncation-alarm tokenizer pass.
- `batch_size` retuning for `md` (separate, quality-neutral perf win).
- Additional filters (e.g. `mermaid`, `base64_image`) — trivial to add via
  `register()`.
