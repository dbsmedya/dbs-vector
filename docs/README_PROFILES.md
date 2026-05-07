# Model Profiles and `config.yaml`

This guide explains the profile-based configuration used by `dbs-vector`: what to set in `config.yaml`, which values are model contracts, and which values are safe to tune per machine.

## Configuration Layers

`config.yaml` has three relevant layers:

| Layer | Where | Purpose |
|---|---|---|
| System settings | `system:` | Global runtime settings such as LanceDB path, search probes, and optional memory budget override. |
| Tuning profiles | `profiles:` | Hardware-sensitive knobs: `max_token_length`, `chunk_max_chars`, and `batch_size`. |
| Engines | `engines:` | Data pipeline wiring: model key, chunker, mapper, table, workflow, prefixes, and profile selection. |

Model contract details do **not** live in `config.yaml`. The Hugging Face model path, vector dimension, model context cap, and attention-mask dtype are hardcoded in `src/dbs_vector/core/model_registry.py`.

## System Block

Use `system:` for global settings:

```yaml
system:
  db_path: "./lancedb_dbs_vector"
  nprobes: 20
  # Optional. If unset, dbs-vector tries to auto-detect the Metal max buffer.
  # memory_budget_gb: 22.0
```

Fields:

| Field | Default | What to set |
|---|---:|---|
| `db_path` | `./lancedb_dbs_vector` | LanceDB directory. Change this to keep separate indexes. |
| `nprobes` | `20` | LanceDB vector search probe count. Higher can improve recall with more latency. |
| `log_level` | `INFO` | Optional Loguru level override. |
| `log_serialize` | `false` | Optional JSON log serialization flag. |
| `memory_budget_gb` | unset | Optional Metal memory budget override. Set this if auto-detection fails or if you want a lower safety budget. |

Do not set `system.batch_size`; batching moved to profiles.

## Profiles Block

A profile controls how large each embedding batch can get:

```yaml
profiles:
  gemma-md:           {max_token_length: 2048,  chunk_max_chars: 1000, batch_size: 64}
  gemma-sql-atomic:   {max_token_length: 2048,  chunk_max_chars: 0,    batch_size: 64}
  granite-md-large:   {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  granite-sql-atomic: {max_token_length: 8192,  chunk_max_chars: 0,    batch_size: 32}
```

Profile fields:

| Field | Meaning | Practical guidance |
|---|---|---|
| `max_token_length` | Token limit passed to the model tokenizer. | Must be less than or equal to the model contract cap. Gemma cap is 2048. Granite R2 cap is 32768. |
| `chunk_max_chars` | Character target for document chunk accumulation. | Use a positive value for document engines. Use `0` for atomic SQL/API records. |
| `batch_size` | Number of chunks embedded per MLX forward pass. | Lower this first when memory validation fails or you hit runtime memory pressure. |

The validator checks every engine/profile pairing at config load. It rejects profiles that exceed the model context cap or the Metal memory budget and prints safer suggested values.

Validation walks all configured engines, not just the engine selected for the current command. If one local machine cannot load a Granite profile, either lower that profile's batch/context values, set a smaller explicit `system.memory_budget_gb` and follow the suggestion, or remove/comment the engine block on that machine.

## Engines Block

Each engine selects a model contract and a tuning profile:

```yaml
engines:
  md-granite:
    description: "Markdown & Prose Engine (Granite R2 - long context)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"
```

Required engine fields:

| Field | What to set |
|---|---|
| `description` | Human-readable label. |
| `model` | Model registry key, currently `gemma-bf16` or `granite-r2`. |
| `mapper_type` | Storage mapper, usually `document` or `sql`. |
| `chunker_type` | Input chunker, usually `document`, `duckdb`, or `api`. |
| `table_name` | LanceDB table for this engine. Use a different table when changing embedding model or workflow. |
| `workflow` | Stored workflow label for records. |
| `tuning_profile` | Key under `profiles:`. |

Optional engine fields:

| Field | When to set |
|---|---|
| `passage_prefix` / `query_prefix` | Required for asymmetric models like Gemma. Leave empty for Granite R2 in this project. |
| `duckdb_query` | Override the default DuckDB SQL extraction query. |
| `api_base_url`, `api_key`, `api_page_size`, `api_since_days`, `api_timeout_sec`, `api_min_execution_ms`, `api_database` | Remote SQL API chunker settings. |

## Built-In Model Keys

| Model key | Model | Vector dim | Max model context | Typical engines |
|---|---|---:|---:|---|
| `gemma-bf16` | `mlx-community/embeddinggemma-300m-bf16` | 768 | 2048 | `md`, `sql`, `sql-api` |
| `granite-r2` | `ibm-granite/granite-embedding-311m-multilingual-r2` | 768 | 32768 | `md-granite`, `sql-granite`, `sql-api-granite` |

Use Gemma for the default lightweight engines. Use Granite when you need longer context, stronger multilingual behavior, or the Granite-specific SQL/Turkish behavior this project targets.

## Choosing Profile Values

Start from one of the built-in profiles and change only one variable at a time.

For Markdown/document engines:

```yaml
profiles:
  my-granite-docs: {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}

engines:
  md-granite:
    description: "Markdown & Prose Engine (Granite R2 - long context)"
    model: "granite-r2"
    chunker_type: "document"
    mapper_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "my-granite-docs"
```

For SQL/DuckDB engines:

```yaml
profiles:
  my-granite-sql: {max_token_length: 8192, chunk_max_chars: 0, batch_size: 32}

engines:
  sql-granite:
    description: "SQL Slow Query Log Engine (Granite Clustering)"
    model: "granite-r2"
    chunker_type: "duckdb"
    mapper_type: "sql"
    table_name: "query_vault_granite"
    workflow: "sql_clustering_granite"
    tuning_profile: "my-granite-sql"
```

Rules of thumb:

- If validation says the profile would exceed memory, reduce `batch_size` first.
- If `batch_size: 1` still does not fit, reduce `max_token_length`.
- If truncation warnings appear during ingestion, either raise `max_token_length` or lower `chunk_max_chars`.
- For SQL engines, keep `chunk_max_chars: 0`; each SQL record should remain atomic.
- For document engines, keep `chunk_max_chars` well below the token window. The default Granite document profile uses 6000 chars for a 16384-token window to leave tokenizer slack.

## Memory Math and Throughput Tuning

The three profile knobs are not independent. Understanding how they
interact at the validator and at runtime lets you choose between
**conservative** (safe on tight memory budgets) and
**throughput-optimized** profiles for the same chunk size.

### What the validator computes

`_validate_config` Rule 4 (`estimate_peak_buffer_bytes ≤ memory_budget × 0.9`)
estimates GPU memory as a **worst case** based on `max_token_length`,
not on the actual length of any chunk. The pessimistic formula is:

```
peak_bytes ≈ batch_size × max_token_length × hidden_dim × dtype_bytes × overhead
```

For Granite R2 (`hidden_dim = 768`, `dtype_bytes = 2`, overhead ≈ 3×):

| `batch_size` | `max_token_length` | Validator peak (approx) |
|---:|---:|---:|
| 8 | 16384 | ~600 MB |
| 8 | 8192 | ~300 MB |
| 8 | 4096 | ~150 MB |
| 16 | 4096 | ~300 MB |
| 32 | 4096 | ~600 MB |

Two profiles with the same `batch_size × max_token_length` product hit
the validator with the same memory bill — even if their actual chunks
are very different sizes.

### What runtime actually uses

`MLXEmbedder` calls the tokenizer with `padding=True` (longest in
batch), not `padding="max_length"`. Real GPU memory is:

```
runtime_bytes ≈ batch_size × max(actual_token_lengths_in_batch) × hidden_dim × dtype_bytes × overhead
```

When chunks are smaller than `max_token_length`, runtime memory is
smaller than the validator's estimate. The validator pessimism is
intentional — it protects you from a worst-case batch that happens to
fill `max_token_length`. But it also means **lowering
`max_token_length` to match your actual chunk size unlocks a higher
`batch_size` without changing real memory use**.

### The throughput optimization

If your chunks are bounded well below `max_token_length` (e.g.,
`chunk_max_chars: 3000` produces chunks of at most ~1000 tokens, well
under 16384), you have two equivalent-cost profiles:

```yaml
# Conservative: matches granite-md-large's memory budget. Slow ingest.
granite-md-medium-safe:    {max_token_length: 16384, chunk_max_chars: 3000, batch_size: 8}

# Throughput-optimized: same real memory, ~2× faster ingest.
granite-md-medium-fast:    {max_token_length: 4096,  chunk_max_chars: 3000, batch_size: 16}
```

Rule of thumb for picking `max_token_length`: **roughly 4× the typical
token count of your largest chunk**. Tokens-per-character varies by
language and code density, but 1 token ≈ 4 chars is a reasonable
default for English/code-heavy content.

| `chunk_max_chars` | Approx tokens | `max_token_length` (4× headroom) |
|---:|---:|---:|
| 800 | ~200 | 1024 |
| 1500 | ~375 | 2048 |
| 3000 | ~750 | 4096 |
| 6000 | ~1500 | 8192 |
| 10000 | ~2500 | 16384 |

A `max_token_length` larger than 4× your chunks costs you batch_size
headroom without buying anything; smaller than 1× your chunks fires
truncation warnings during ingestion.

### When to keep the pessimistic profile

Use a higher `max_token_length` than the rule of thumb suggests when:

- You're ingesting a heterogeneous corpus and want headroom for
  unusually long chunks.
- You're sharing one profile across multiple engines that have
  different `chunk_max_chars` (the profile must accommodate the
  largest).
- You're explicitly mirroring an existing profile's memory footprint
  for predictable behavior across a fleet.

Otherwise, lower `max_token_length` to ~4× your chunk size and bump
`batch_size` until the validator approves.

## Workload Profile Recipes

Pick the recipe that matches your corpus shape, then point an engine
at it. Each recipe lists both a **safe** profile (mirrors
`granite-md-large` memory) and a **throughput-optimized** profile
(uses the rule from § "Memory Math and Throughput Tuning") where they
differ.

### Long-form prose (whitepapers, design docs, articles)

Each chunk is a coherent ~paragraph cluster. Granite's larger context
shines here.

```yaml
profiles:
  prose-granite-large:      {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  # Throughput-optimized variant (chunks rarely exceed 1500 tokens):
  prose-granite-large-fast: {max_token_length: 8192,  chunk_max_chars: 6000, batch_size: 16}
```

Use `model: "granite-r2"`, `mapper_type: "document"`,
`chunker_type: "document"`. This is the original `granite-md-large`
recipe with a faster sibling.

### Fragmented technical docs (specs, plans, README files with code fences)

Heavily-fragmented markdown with short headings, bullet lists, and code
blocks. Smaller chunks discriminate better here than long ones — see
the `.ayder` corpus benchmark in [README_granite.md](README_granite.md).

```yaml
profiles:
  techdoc-granite:      {max_token_length: 16384, chunk_max_chars: 3000, batch_size: 8}
  # Throughput-optimized — chunks ≤ ~1000 tokens, 4× headroom:
  techdoc-granite-fast: {max_token_length: 4096,  chunk_max_chars: 3000, batch_size: 16}
```

Best baseline for the `.ayder` corpus; usually outperforms
`granite-md-large` on retrieval recall@1 for this content shape.

### Short-form notes (tickets, changelogs, commit messages)

Each chunk is one short item. Tiny `max_token_length` lets you push
`batch_size` very high.

```yaml
profiles:
  shortform-granite:      {max_token_length: 2048, chunk_max_chars: 1500, batch_size: 32}
  shortform-gemma:        {max_token_length: 2048, chunk_max_chars: 1500, batch_size: 64}
```

Gemma is usually the better fit for English-only ticket-style content
(its instruction-tuned task prefixes earn their keep on short queries
matching short passages).

### Code search (atomic functions, classes)

Code chunks bounded by the AST node, not the character count. Granite
R2 supports code in 9 languages.

```yaml
profiles:
  code-granite:      {max_token_length: 8192, chunk_max_chars: 2500, batch_size: 16}
  # Even more aggressive for small functions:
  code-granite-fast: {max_token_length: 4096, chunk_max_chars: 2500, batch_size: 32}
```

If you tag chunks with their language at ingest (see
[README_granite.md § 3.4](README_granite.md)), use `passage_prefix`
and `query_prefix` symmetrically (e.g., `"Python: "`).

### SQL/DuckDB atomic queries

Each SQL log row is a single record. `chunk_max_chars: 0` disables
character-based chunking; the chunker emits one chunk per query.

```yaml
profiles:
  sql-granite-atomic: {max_token_length: 8192, chunk_max_chars: 0, batch_size: 32}
  sql-gemma-atomic:   {max_token_length: 2048, chunk_max_chars: 0, batch_size: 64}
```

Use `mapper_type: "sql"` and `chunker_type: "duckdb"` (or `"api"` for
remote SQL log ingestion). Long SQL statements occasionally exceed the
token cap; truncation warnings here are usually safe to ignore unless
they cluster on specific tables (then bump `max_token_length`).

### Tight-memory workstation (≤ 16 GB Metal budget)

Both Granite and Gemma engines must coexist on a smaller machine.

```yaml
profiles:
  granite-md-tight: {max_token_length: 4096, chunk_max_chars: 2000, batch_size: 4}
  gemma-md-tight:   {max_token_length: 2048, chunk_max_chars: 800,  batch_size: 16}
```

If the validator still rejects after these, set
`system.memory_budget_gb` explicitly to a lower value (the auto-detect
sometimes overestimates available memory) and let the validator's
suggestions guide further cuts.

### Long-context exploration (≤ 32k tokens per chunk)

Edge case for embedding *whole* documents (book chapters, full
specs). Slowest variant; usually retrieval-quality wise inferior to
the fragmented-techdoc recipe but useful for global summarization
pipelines.

```yaml
profiles:
  granite-md-long: {max_token_length: 32768, chunk_max_chars: 10000, batch_size: 1}
```

Use `batch_size: 1` — anything larger will not fit Metal's max buffer
on most consumer Apple Silicon. Ingest is glacial; only worth it for
small corpora (< 1000 documents).

### Wiring an engine to any of the above

```yaml
engines:
  md:
    description: "Markdown & Prose Document Engine (Gemma Search)"
    model: "gemma-bf16"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault"
    workflow: "md_search"
    passage_prefix: "title: none | text: "
    query_prefix: "task: search result | query: "
    tuning_profile: "shortform-gemma"   # pick whichever recipe fits
```

Granite engines should leave `passage_prefix` and `query_prefix`
empty — see [README_granite.md](README_granite.md) for the
explanation.

## Auditing Profile Changes

When someone proposes a new profile or changes an existing one, run
this audit before merging. It catches the most common mis-tuning —
profiles whose `max_token_length` is far larger than any chunk will
ever produce, costing validator headroom for no runtime benefit.

### The check script

```bash
uv run python -c "
from dbs_vector.config import load_settings

s = load_settings('config.yaml', validate=True)
print(f'{\"Profile\":<25} {\"batch×ctx\":>10} {\"Val MB\":>8} {\"Run MB\":>8}  Headroom  Tag')
for name, p in s.profiles.items():
    val_mb = p.batch_size * p.max_token_length * 768 * 2 * 3 / 1024 / 1024
    chunk_tok = max(1, p.chunk_max_chars // 4) if p.chunk_max_chars > 0 else p.max_token_length
    actual_tok = min(chunk_tok, p.max_token_length)
    run_mb = p.batch_size * actual_tok * 768 * 2 * 3 / 1024 / 1024
    ratio = p.max_token_length / max(1, actual_tok)
    if ratio > 12:
        tag = 'over-provisioned max_token_length'
    elif ratio < 1.5 and p.chunk_max_chars > 0:
        tag = 'tight — expect truncation warnings'
    elif 4 <= ratio <= 12:
        tag = 'well-tuned'
    else:
        tag = ''
    print(f'{name:<25} {p.batch_size*p.max_token_length:>10,} {val_mb:>8.0f} {run_mb:>8.0f}  {ratio:>7.1f}×  {tag}')
"
```

The script:

1. Loads `config.yaml` through the real validator (so any rule-1
   through rule-7 failure surfaces here).
2. For each profile, prints the validator's pessimistic memory
   estimate AND a runtime estimate based on actual chunk size.
3. Tags profiles with their tuning state.

**Validator estimate formula:**

```
peak_bytes ≈ batch_size × max_token_length × hidden_dim × dtype_bytes × overhead
            (worst case — assumes every chunk fills max_token_length)
```

**Runtime estimate formula:**

```
peak_bytes ≈ batch_size × min(max_token_length, chunk_max_chars / 4) × hidden_dim × dtype_bytes × overhead
            (real case — padding=True caps the batch at the longest actual chunk)
```

The 4-chars-per-token approximation is a reasonable default for
English plus code. For Asian-language-heavy or unicode-emoji corpora
use 2 chars/token. The 3× overhead matches the calibration constant
in `_validate_config` (`docs/superpowers/specs/2026-05-06-tuning-profiles-design.md`).

### Decision rubric

`headroom = max_token_length / actual_chunk_tokens`

| Headroom | Reading | Action |
|---:|---|---|
| `> 12×` | Over-provisioned. Validator reserves an order of magnitude more memory than runtime uses. | Lower `max_token_length` toward 4–8× the actual chunk-token count, or raise `batch_size` until the validator approves the same total. |
| `4×–12×` | Well-tuned. Comfortable safety margin for outlier chunks (long code blocks, tables) without wasting validator headroom. | Ship as-is. The project's default profiles all live here. |
| `1.5×–4×` | Minimal headroom. Truncation may fire on outlier chunks. | Monitor truncation warnings in ingest logs; if they cluster on specific files, either raise `max_token_length` or lower `chunk_max_chars`. |
| `< 1.5×` | Too tight. Truncation almost certainly firing. | Raise `max_token_length` (within the model contract cap) or lower `chunk_max_chars`. |

For SQL-atomic profiles (`chunk_max_chars: 0`), the script falls back
to using `max_token_length` itself as the chunk-token estimate, so the
ratio is always `1.0×`. Atomic SQL chunks are bounded by the source
data, not by the chunker — review truncation warnings in ingest logs
to catch oversized queries.

### Common review findings

When auditing a proposed profile change, flag any of:

1. **Headroom > 12×** without a justification (heterogeneous corpus,
   shared profile across engines with different `chunk_max_chars`,
   fleet uniformity, deliberate slack for non-English content where
   the 4-chars/token assumption underestimates token count). The
   default fix is to either lower `max_token_length` or raise
   `batch_size`.
2. **Validator MB > 1.5 GB** on profiles intended for laptops. Cap
   the laptop-targeted profile at ~1 GB to leave room for other
   engines and the OS.
3. **Three or more profiles with the same `batch_size × max_token_length`
   product**. Indicates copy-paste rather than per-workload tuning.
   Either consolidate to one profile or differentiate the recipes.
4. **No `granite-*` profile uses `passage_prefix` / `query_prefix`**
   — confirm. Granite engines should leave both empty (see
   [README_granite.md](README_granite.md)).
5. **`max_token_length` exceeds the model contract cap** (Gemma:
   2048, Granite R2: 32768). Caught by Rule 3, but worth eyeballing
   too — the validator only reports the first violation.

### Worked example

Suppose someone proposes `granite-md-medium: {16384, 3000, 16}`.
Running the audit:

```
Profile                   batch×ctx   Val MB   Run MB  Headroom  Tag
granite-md-medium           262,144     1152      264     22.0×  over-provisioned max_token_length
```

Diagnosis: 22× headroom trips the over-provisioned threshold. The
validator reserves 1152 MB but runtime uses ~264 MB. Two corrective
patches — both align with the
[Memory Math](#memory-math-and-throughput-tuning) recommendations:

```yaml
# Option A: throughput-optimized (recommended on most machines)
granite-md-medium: {max_token_length: 4096, chunk_max_chars: 3000, batch_size: 16}
# headroom 5.5× → tag "well-tuned", validator 288 MB, runtime ~264 MB

# Option B: memory parity with granite-md-large (use for fleet uniformity)
granite-md-medium: {max_token_length: 16384, chunk_max_chars: 3000, batch_size: 8}
# headroom 22× — also flagged, but matches granite-md-large's 576 MB validator footprint exactly
```

Choose A unless there's a specific reason to mirror `granite-md-large`'s
memory budget. Document the choice inline in `config.yaml`:

```yaml
profiles:
  # 4× headroom: chunks ≤ 750 tokens, max_token_length=4096 leaves slack
  # for outliers without paying validator overhead.
  granite-md-medium: {max_token_length: 4096, chunk_max_chars: 3000, batch_size: 16}
```

### When to add a new profile vs. tune an existing one

Add a new profile when:

- A new engine has a fundamentally different chunk-size shape
  (long-form prose vs. fragmented techdoc vs. atomic SQL).
- You're A/B testing a tuning hypothesis and need both the baseline
  and the candidate alive simultaneously (see § "A/B testing tuning
  profiles" below).

Tune an existing profile when:

- Truncation warnings fire on the existing engines using it.
- Validator rejects on a new (smaller) target machine.
- An audit shows headroom drift well outside the 4×–8× sweet spot.

Don't add a new profile to "save" a config from a validator
rejection on one machine — fix the profile or set a smaller
`system.memory_budget_gb` instead. Profile sprawl makes future audits
harder.

## Validation and Error Handling

`dbs-vector` validates the full config when a runtime command loads it:

```bash
uv run dbs-vector ingest docs/ --type md-granite
uv run dbs-vector search "slow query" --type sql-granite
```

The validator catches:

- unknown `model` keys;
- unknown `tuning_profile` keys;
- `max_token_length` above the model contract cap;
- profiles that exceed the Metal memory budget;
- old schema fields such as `system.batch_size`, `model_name`, `vector_dimension`, `attention_mask_dtype`, and per-engine `chunk_max_chars`.

If Metal memory cannot be detected, set an explicit budget:

```yaml
system:
  memory_budget_gb: 16.0
```

## Migration From Old Configs

Old config fields moved as follows:

| Old field | New location |
|---|---|
| `system.batch_size` | `profiles.<name>.batch_size` |
| `engines.<name>.model_name` | `engines.<name>.model` registry key |
| `engines.<name>.vector_dimension` | `ModelRegistry` contract |
| `engines.<name>.max_token_length` | `profiles.<name>.max_token_length` |
| `engines.<name>.attention_mask_dtype` | `ModelRegistry` contract |
| `engines.<name>.chunk_max_chars` | `profiles.<name>.chunk_max_chars` |

When an old config is loaded, `dbs-vector` raises one migration-hint error instead of a long Pydantic error list.

## Related Docs

- [README_EMBEDDINGS.md](README_EMBEDDINGS.md) covers supported embedding models and model-contract behavior.
- [README_DOCS.md](README_DOCS.md) covers Markdown chunking behavior.
- [README_SQL.md](README_SQL.md), [README_duckdb.md](README_duckdb.md), and [README_REMOTE_SQL_API.md](README_REMOTE_SQL_API.md) cover SQL ingestion sources.

## A/B testing tuning profiles

Because each engine name maps to a distinct MCP tool and a distinct
LanceDB table, you can run two profile variants of the same model side
by side and compare results without code changes.

### Step 1: Define a new profile

```yaml
profiles:
  granite-md-large:        {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}
  granite-md-experimental: {max_token_length: 8192,  chunk_max_chars: 3000, batch_size: 16}
```

### Step 2: Define a new engine that references the variant profile

```yaml
engines:
  md-granite:
    description: "Granite long-context, 6KB chunks (baseline)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"

  md-granite-experimental:
    description: "Granite, smaller chunks for higher recall (A/B candidate)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite_exp"   # MUST differ from baseline
    workflow: "md_search_granite"
    tuning_profile: "granite-md-experimental"
```

### Step 3: Ingest into both engines

```bash
uv run dbs-vector ingest "./docs/" --type md-granite
uv run dbs-vector ingest "./docs/" --type md-granite-experimental
```

### Step 4: Compare via the MCP tools

Start the server (`uv run dbs-vector mcp`) and run the same query through
both `search_md_granite` and `search_md_granite_experimental`. Use
`list_engines` to dump every engine's profile knobs into your
evaluation report so the comparison is reproducible.

### Memory note

Every engine in `engines:` is loaded eagerly at startup. Each
long-context Granite variant consumes roughly 1–2 GB of GPU memory.
Drop the experimental variant from `engines:` once you've decided
which configuration to keep.
