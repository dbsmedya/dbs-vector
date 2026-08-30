# Markdown Document Engine

The Document engine (`--type md`) in `dbs-vector` is designed to ingest, chunk, and semantically search through prose and code documentation. It is the core engine for building a Retrieval-Augmented Generation (RAG) codebase assistant.

## Overview
This engine utilizes a specialized parsing strategy to ensure that context sent to the LLM is logically coherent. It natively understands Markdown syntax, sizing chunks by token budget while keeping code blocks and container context intact.

---

## The chunking pipeline

`DocumentChunker` is heading-aware and token-sized. It runs four phases over
every `.md` file.

### 1. Parse and descend

`markdown-it-py` turns the file into blocks, and the parser descends into two
container types instead of treating their bodies as opaque:

*   **Admonitions** — `!!! note "Title"`, `!!! warning "Title"`, and the
    `???`/`???+` collapsible variants.
*   **Blockquotes** — including GitHub-style alerts (`> [!WARNING]`, `> [!NOTE]`).

A container can attach a **frame**: a short label carried into the chunk's
breadcrumb, so content retrieved from inside one still says what it was
written in. A chunk pulled from a warning admonition renders a breadcrumb like
`Guide > warning: Data loss risk`.

Ordinary blockquotes are atomic-first — one small enough to fit the budget
keeps its `>` markers and packs with the surrounding prose, so most quotes are
untouched.

### 2. Pack

Blocks are grouped by scope and packed until `chunk_target_tokens` is reached,
then the chunk is sealed. No chunk may exceed `chunk_max_tokens`. Both are set
per tuning profile in `config.yaml` — see
[README_PROFILES.md § Token Budget Knobs](README_PROFILES.md#token-budget-knobs-document-engines).

A fenced code block is **atomic while it fits**: if the whole fence fits the
budget it is never split, so a 150-line function arrives whole.

A fence too large to fit is split rather than dropped, and the split is made
safe rather than silent. Each part is re-fenced with a delimiter long enough
not to collide with the content, and labelled `(code, part 2/5)`, so every
part is valid Markdown on its own. Oversized tables are split the same way
with the **header row repeated** on each part, so a fragment still says what
its columns mean.

### 3. Forward fold

A section whose entire body is too small to be worth retrieving on its own —
a title plus a metadata blockquote, say — would otherwise ship as a standalone
chunk that scores in the corpus's normal band while carrying almost nothing.
Instead it **folds forward** into the next section.

Inside a folded chunk you may see inline `(dbs-vector context: parent="...")`
and `(dbs-vector context: frame="...")` markers showing where the absorbed
section's content begins. These appear **only** in folded chunks. A standalone
chunk — the common case — never contains them.

### 4. Compose

Each chunk is emitted as a breadcrumb followed by its body, with
`parent_scope`, `node_type` and `line_range` stored alongside for filtering
and for `read_md_*` cursor reads.

### Sizing, and what `chunk_max_chars` does not do

Markdown chunk size is governed **entirely** by `chunk_target_tokens` and
`chunk_max_tokens`. The `chunk_max_chars` field applies to the `.txt` fallback
path only and is ignored for `.md` input — lowering it will not produce
smaller Markdown chunks.

### Plain-text fallback

A `.txt` file has no headings to be aware of, so the engine falls back to
splitting on double newlines (`\n\n`) under the `chunk_max_chars` budget. This
keeps raw logs and unformatted notes ingestible without special handling.

---

## Task Prefixes (Asymmetric Embeddings)
Modern embedding models (like `mlx-community/embeddinggemma-300m-bf16`) are instruction-tuned. They require a specific string prepended to the text to understand *what* they are doing. 

The Markdown engine uses an **Asymmetric** retrieval strategy:
1.  **Ingestion:** The engine silently prepends the passage prefix (`title: none | text: `) to your document chunks before calculating the vector.
2.  **Searching:** When you ask a question, the engine prepends the query prefix (`task: search result | query: `).

This teaches the model to project short questions and long, factual answers into the same mathematical space, drastically improving RAG retrieval accuracy compared to naive embeddings.
*   **Important:** These prefixes are strictly used for vector generation. They are *never* stored in the `text` column in LanceDB, ensuring your LLM prompt remains clean and your Full-Text Search index isn't polluted by the word "title" or "query."

> **Granite engines.** The `md-granite` (and `sql-granite`, `sql-api-granite`) engines use IBM's Granite Multilingual R2 model with **empty `passage_prefix` and `query_prefix`**. The model card's examples don't specify task instructions; we treat it as a symmetric encoder. See [README_EMBEDDINGS.md](README_EMBEDDINGS.md) for the supported-models list and the rationale.

---

## Batching & Memory Management

Embedding models (like `embeddinggemma-300m` or Granite R2) require significant GPU memory to process text into vectors. To handle massive codebases (e.g., thousands of files) without crashing, `dbs-vector` employs a **Streaming Batch Architecture**.

1.  **Lazy Generation:** The `DocumentChunker` uses Python generators (`yield`) to extract chunks one by one, meaning the entire codebase is never loaded into RAM simultaneously.
2.  **Configurable Batching:** The `IngestionService` groups these chunks into strict batches (controlled by `batch_size` in the engine's tuning profile in `config.yaml`, default: 64). 
3.  **GPU Offloading:** Only one batch is sent to the Apple MLX GPU at a time. 
4.  **Zero-Copy Storage:** The resulting tensors are instantly mapped to PyArrow `RecordBatch` arrays and flushed to disk via LanceDB.

**GPU** memory is therefore flat and predictable: it is bounded by `batch_size`
and the longest chunk in a batch, not by how large the corpus is.

Host memory is not constant. Ingestion materializes the file list for a run, and
loads every already-stored content hash into a set so it can skip unchanged files
and de-duplicate within a batch. Both grow linearly with corpus size — a few tens
of bytes per file and per chunk, which is negligible next to the model, but it is
not zero. It is the GPU side that the batching architecture keeps flat.

---

## Indexing & Hybrid Search

Once documents are ingested, the engine builds two indices.

### 1. Vector indexing (IVF-PQ)

`LanceDBStore.create_indices()` builds an Inverted File with Product
Quantization index on the `vector` column.

*   **Only above 256 rows.** Below that an index buys nothing and a flat scan
    is exact, so none is created.
*   **Dynamic scaling.** Partitions are `sqrt(N)`, capped at 256.
*   **Cosine.** The engine pins `metric="cosine"` to match the training
    objective of the embedding models it ships.

### 2. Full-text search (FTS)

A native inverted index is built on the `text` column.

When you search:

```bash
uv run dbs-vector search "Unified Memory Architecture" --type md
```

both channels run and their rankings are fused with Reciprocal Rank Fusion.
Each result carries:

*   `similarity` — exact cosine between the query and chunk vectors, computed
    at search time. It is a geometric scale, not a probability of relevance,
    and is comparable only within one engine's configuration.
*   `retrieved_by` — which channel returned the row (`vector`, `fts`, or
    `both`). It reports channel membership, not correctness.

RRF decides the display order, so ordering can disagree with `similarity`.
