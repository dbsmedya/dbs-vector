# Granite Embedding R2 — Task Instructions and Prefixes

> **Model:** `ibm-granite/granite-embedding-311m-multilingual-r2`
> **Project engines using it:** `md-granite`, `sql-granite`, `sql-api-granite`

## TL;DR

**Granite R2 is not instruction-tuned.** It was trained as a symmetric
bi-encoder via retrieval-oriented pretraining + contrastive finetuning;
the official model card, the [Granite Embedding paper](https://arxiv.org/abs/2502.20204),
and the model's shipped `config_sentence_transformers.json` (with
`"prompts": {"query": "", "document": ""}`) all agree: the recommended
input format is **raw text, no prefix**.

Concretely, that means:

- Don't copy Gemma-style task strings like `task: search result | query:`
  into Granite's `query_prefix:` / `passage_prefix:`. Granite was never
  trained on those tokens, so they push the embedding *away* from the
  training distribution rather than projecting it onto a task subspace.
- The `passage_prefix:` / `query_prefix:` config slots still work
  mechanically (they're string-concatenated in `MLXEmbedder` —
  `src/dbs_vector/infrastructure/embeddings/mlx_engine.py:127-148`), so
  you *can* experiment, but expect zero or marginal gains, not the
  step-change Gemma sees.

If you want to improve Granite recall on this codebase, the levers in
[§5](#5-non-prefix-levers-likely-to-help-more) (chunk size, Matryoshka,
FTS weights, cross-encoder rerank) are likely to outperform any prefix
engineering by a wide margin.

---

## 1. Why Granite differs from Gemma

| | `embeddinggemma-300m` | `granite-embedding-311m-multilingual-r2` |
|---|---|---|
| Training objective | Instruction-tuned with task descriptors | Contrastive learning on raw query/passage pairs |
| Asymmetry | Yes — different prefixes recommended for queries vs. passages | No — symmetric bi-encoder |
| Documented prompts | `task: search result \| query: …` (queries), `title: none \| text: …` (passages), `task: clustering \| query: …` (clustering), etc. | None. `config_sentence_transformers.json` ships with empty prompts |
| Max input tokens | 2048 | 32768 (project contract); model card lists 8192 for code, longer in practice |
| Languages | English | 200+ with strong support for 52 |
| Matryoshka dimensions | No | Yes — 768 / 512 / 384 / 256 / 128 |

**Implication:** Gemma rewards prefix engineering because the model has
seen those exact prefixes during training and learned to project
prefixed inputs into a task-conditional subspace. Granite has not. The
right mental model for Granite is "general-purpose semantic encoder,"
not "instruction-following retriever."

---

## 2. What the model actually ships

The Granite R2 multilingual model exposes the standard Sentence
Transformers prompt slots, but populates them as empty strings:

```json
{
  "prompts": {
    "query": "",
    "document": ""
  },
  "default_prompt_name": null,
  "similarity_fn_name": "cosine"
}
```

This is documentation-as-defaults: IBM is telling you *the format
exists, but we don't recommend any specific prompt*. If a future
fine-tune ever publishes recommended prompts, they'll show up in this
JSON.

The Sentence Transformers usage examples in the model card use direct
encoding with no prefixes:

```python
input_queries = [' Who made the song My achy breaky heart? ', 'summit define']
input_passages = ["Achy Breaky Heart is a country song …", "Definition of summit …"]

query_embeds   = model.encode(input_queries)
passage_embeds = model.encode(input_passages)
```

---

## 3. Experimental prefixes you *could* try (with caveats)

If you want to A/B-test prefix engineering on Granite, the project
makes this trivial via `config.yaml`. Here are four patterns ordered
from "lowest expected harm" to "actively risky," each with rationale.

### 3.1 No prefix (current default — recommended)

```yaml
md-granite:
  # ... existing config ...
  passage_prefix: ""
  query_prefix: ""
```

Matches the model card. Use as your A/B baseline.

### 3.2 Domain-tag prefix (low risk, possible marginal gain)

```yaml
md-granite:
  passage_prefix: "Documentation: "
  query_prefix: "Documentation: "
```

A short symmetric tag prepended to **both** queries and passages. The
embedding shifts uniformly, so cosine similarity within the corpus is
mostly preserved while the entire vector cloud gets nudged toward a
"documentation" region of latent space. This may help when:

- Your corpus is narrow (one domain) and queries are short and
  ambiguous on their own (e.g., "config" could match anything).
- You're indexing multiple distinct corpora into different tables and
  want to keep their embeddings cleanly separated.

It will **not** make Granite suddenly act like an instruction-tuned
model. Expected impact: 0–2% recall@k swing in either direction.

### 3.3 Symmetric clustering tag (mirrors the project's `sql` engine pattern)

```yaml
md-granite:
  passage_prefix: "concept: "
  query_prefix:   "concept: "
```

Identical pattern to how the existing `sql` engine (Gemma) prefixes
both sides with `task: clustering | query: `. For Granite this is
*structurally* similar to §3.2 — a domain tag — because Granite has no
trained `task: clustering` token. The choice of word is purely
cosmetic; what matters is *symmetry* and *constancy*.

Worth it only if you specifically want clustering-style behavior
(prioritize semantic *similarity* over query-passage *complementarity*).

### 3.4 Code-language tag for code retrieval (use case: Phase 2)

The Granite R2 paper notes the model was trained with code from 9
languages (Python, Go, Java, JavaScript, PHP, Ruby, SQL, C, C++).
Tagging code chunks with their language at ingest time is a plausible
way to let queries disambiguate intent:

```yaml
# Hypothetical future engine
code-granite-py:
  description: "Python source code (Granite)"
  model: "granite-r2"
  mapper_type: "document"      # or a future code-mapper
  chunker_type: "document"
  table_name: "code_vault_py"
  workflow: "code_search_granite"
  tuning_profile: "granite-md-large"
  passage_prefix: "Python: "
  query_prefix:   "Python: "
```

Same caveat as §3.2 — you're using the prefix as a domain tag, not as
an instruction.

### 3.5 Anti-pattern: don't borrow Gemma syntax

```yaml
# DO NOT DO THIS
md-granite:
  passage_prefix: "title: none | text: "
  query_prefix:   "task: search result | query: "
```

These tokens are meaningful to Gemma (instruction-tuned) and noise to
Granite. Adding them takes valid raw text and prepends an
out-of-distribution string the model has never seen during training.
Expected impact: unambiguously *negative* recall.

---

## 4. The *correct* way to A/B test prefixes on Granite

Because the project supports per-engine config without source changes,
prefix experiments are pure config edits. Use this workflow:

### Step 1: Define a baseline + experiment engine

Append to `config.yaml`:

```yaml
profiles:
  granite-md-large:
    {max_token_length: 16384, chunk_max_chars: 6000, batch_size: 8}

engines:
  md-granite:
    description: "Markdown & Prose (Granite, baseline — no prefix)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite"
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"
    # passage_prefix and query_prefix omitted → empty (default)

  md-granite-domaintag:
    description: "Granite + 'Documentation:' symmetric domain tag (A/B candidate)"
    model: "granite-r2"
    mapper_type: "document"
    chunker_type: "document"
    table_name: "knowledge_vault_granite_domaintag"   # MUST differ
    workflow: "md_search_granite"
    tuning_profile: "granite-md-large"
    passage_prefix: "Documentation: "
    query_prefix:   "Documentation: "
```

The two engines must use **different `table_name` values** — the prefix
goes into the *embedded passages*, so the experiment needs a fresh
table or you'll mix prefixed and non-prefixed vectors.

### Step 2: Ingest the same corpus into both

```bash
uv run dbs-vector ingest "./.ayder/" --type md-granite
uv run dbs-vector ingest "./.ayder/" --type md-granite-domaintag
```

### Step 3: Run a fixed query set through both via MCP

Restart the server (`uv run dbs-vector mcp`); both
`search_md_granite` and `search_md_granite_domaintag` will be
available. Score by:

- **Top-k recall** on a held-out set of 10–30 queries you've manually
  judged relevance for.
- **Distance distribution** — within-engine, not cross-engine.
- **Diversity** — does one variant return more distinct sources?

### Step 4: Drop whichever loses

Once you've decided, remove the loser from `engines:`. Each long-context
Granite engine costs ~1–2 GB of GPU memory at startup.

---

## 5. Non-prefix levers likely to help more

Empirically (per the `.ayder` corpus benchmarking that motivated this
doc), the following changes have a higher expected impact than prefix
engineering:

### 5.1 Reduce `chunk_max_chars` for finer discrimination

Granite's profile currently uses `chunk_max_chars: 6000`. For a corpus
of fragmented spec/plan markdown — short headings, code fences, decision
bullet items — that's coarse. A 3000-char profile lets each chunk
embed a more focused topic:

```yaml
profiles:
  granite-md-medium:
    {max_token_length: 16384, chunk_max_chars: 3000, batch_size: 8}

engines:
  md-granite-fine:
    # ... model: "granite-r2", mapper_type: "document", etc. ...
    table_name: "knowledge_vault_granite_fine"
    tuning_profile: "granite-md-medium"
```

This is the single change most likely to improve top-1 recall on the
`.ayder` corpus. Note `batch_size: 8` mirrors `granite-md-large`'s
memory profile — safe, validator-approved, and unchanged from the
project's existing Granite footprint. If ingest throughput matters,
see the "throughput-optimized" pattern in
[README_PROFILES.md § Memory Math and Throughput Tuning](README_PROFILES.md#memory-math-and-throughput-tuning).

### 5.2 Use Matryoshka truncation for speed (not recall)

Granite R2 supports truncating its 768-dim embedding to 512, 384, 256,
or 128 dimensions with documented graceful degradation. This isn't
exposed via `config.yaml` today (the project's `vector_dimension` is
fixed in `ModelRegistry`), but it's a documented model capability worth
noting if you ever ship a low-latency retrieval mode.

### 5.3 Tune hybrid (vector + FTS) weights

`SearchService` performs hybrid retrieval (vector + native FTS). The
weighting is currently implicit in the LanceDB hybrid search; if you
find Granite's vector recall is weaker than FTS for keyword-dominant
queries, biasing the hybrid score toward FTS is more effective than
prefix engineering.

### 5.4 Add a cross-encoder rerank step

The bi-encoder approach Granite uses is fast but lossy. For high-stakes
queries, retrieving top-30 from Granite and reranking with a small
cross-encoder (e.g., `BAAI/bge-reranker-v2-m3`) typically lifts
top-3 precision by 10–20% on technical-doc corpora. Out of scope for
the current architecture but worth roadmapping.

### 5.5 Use the multilingual capability

Granite R2 supports 200+ languages. If your team writes specs/notes in
multiple languages (or queries the codebase in non-English), this
unlocks recall that Gemma (English-only) cannot deliver. No config
change required — just write or query in whatever language; the
embedding space is already shared.

### 5.6 Increase `nprobes` for IVF_PQ recall

`Settings.nprobes` defaults to 20. At larger index sizes (`> 50k`
chunks) bumping this to 40–80 typically lifts vector recall at the
cost of small per-query latency. Orthogonal to prefix choices but
relevant to "Granite results feel weak."

---

## 6. Validation: prove it before you commit

Whatever change you make — prefix, chunk size, profile knobs — measure
it before promoting it to default. A minimal evaluation harness:

```python
# eval_granite.py (sketch — not part of the project)
queries_with_expected_top1 = [
    ("two-registry split for family validation",
     ".ayder/superpowers_20260507/specs/2026-05-07-dynamic-engine-exposure-design.md"),
    ("how to add a new search engine",
     ".ayder/superpowers_20260507/plans/2026-05-07-dynamic-engine-exposure.md"),
    # ... 20+ rows ...
]

import asyncio, json
from dbs_vector.mcp.discovery import _list_engines

async def evaluate(tool_name: str) -> dict:
    # Use the dbs-vector MCP tool over stdio, or call SearchService
    # directly via dbs_vector.services.search.
    ...

# Compare top-1 recall on baseline vs experiment.
```

Use `list_engines` (the discovery MCP tool) to dump every engine's
profile knobs into your evaluation report so the comparison is
reproducible.

---

## 7. References

- IBM Granite Embedding model card (multilingual R2, 311M):
  <https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2>
- IBM Granite Embedding docs (model family overview):
  <https://www.ibm.com/granite/docs/models/embedding>
- Granite Embedding paper (training methodology):
  <https://arxiv.org/abs/2502.20204>
- Hugging Face Granite Embedding collection:
  <https://huggingface.co/collections/ibm-granite/granite-embedding>
- Project model contract:
  `src/dbs_vector/core/model_registry.py:51-60`
- Project prefix wiring:
  `src/dbs_vector/infrastructure/embeddings/mlx_engine.py:127-148`
- A/B testing workflow (general):
  `docs/README_PROFILES.md` — § "A/B testing tuning profiles"
