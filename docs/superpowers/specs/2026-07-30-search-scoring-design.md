# Search Scoring: Honest Similarity + Relevance Floor — Design

**Date:** 2026-07-30
**Status:** Approved design, pre-implementation
**Source issue:** `.ayder/superpowers_20260730/search_scoring_work.md`
**Breaking changes:** allowed (next release); no index rebuild required.

## Problem

`Score:` in every search surface is LanceDB's Reciprocal Rank Fusion value
(`RRFReranker(K=60)`): a quantized, rank-derived number (max 0.0328) that
carries no absolute relevance signal, cannot be compared across queries, and
never lets a query return "nothing relevant here". Measured consequence: a
corpus with zero relevant content returns five confident-looking rows (e.g. a
`delete_by_source` unit test cited for a beekeeping query at "0.0318"), and an
MCP agent has no way to detect the absence. Meanwhile the signal that would
reveal it — true vector distance — is discarded by LanceDB's hybrid path.

Verified during design (probe on lancedb 0.30.2):

- The hybrid result set already includes each row's `vector` column, so exact
  cosine similarity between query and result costs one NumPy dot product —
  no extra I/O.
- `RRFReranker(K=60, return_score="all")` retains per-leg columns: `_score`
  (FTS/BM25, null when the FTS leg did not return the row) and `_distance`
  (null when the vector leg did not). This yields match provenance.
- Latent bug: `_build_hybrid` never sets `.metric("cosine")`, so on tables
  without an IVF index the hybrid vector leg runs the default L2 metric.
  Ordering is unaffected only when embeddings happen to be unit-norm, which
  no code guarantees. `LanceHybridQueryBuilder.metric()` exists (query.py:2045).

## Outcomes (from the issue, all four preserved)

1. Every result carries an interpretable absolute relevance number.
2. A query with no good answer can return empty — absence is visible.
3. Hybrid retrieval and RRF ordering are preserved (lexical recall matters).
4. Ordering quality does not regress for any engine (unchanged by construction).

## Design

### 1. Retrieval (unchanged + one fix)

- Hybrid search with RRF(K=60) ordering stays exactly as-is.
- `_build_hybrid` adds `.metric("cosine")` and an explicit
  `.rerank(RRFReranker(K=60, return_score="all"))`.
- Over-fetch: `fetch_limit = limit * 3` (the oversized `table_filter`
  full-scan path keeps its existing `count_rows()` fetch). Dedupe by `id` as
  today; floor filtering happens downstream, then truncation to `limit`.
  After floor filtering the result may hold fewer than `limit` rows even when
  the corpus has more above-floor rows past RRF rank `limit * 3`; accepted —
  headers report honest counts.

### 2. Scoring (in `LanceDBStore.search`)

- For every returned row, compute exact cosine similarity between
  `query_vector` and the row's `vector` column in NumPy:
  `sim = dot(q, v) / (|q| * |v|)`, guarding zero norms (either norm 0 →
  similarity 0.0). This is metric-independent and covers FTS-only rows —
  precisely the rows LanceDB's `_distance` leaves null.
- A missing `vector` column in a search result row is a programming error
  (search applies no projection): raise `ValueError`, do not degrade.
- Match provenance from the `return_score="all"` null pattern:
  `_distance` non-null and `_score` non-null → `both`; only `_distance` →
  `semantic`; only `_score` → `keyword`. The pure-vector fallback path
  (`_hybrid_ok is False` or oversized `table_filter`) is always `semantic`,
  and `rrf_score` is `None` there (no fusion ran).
- The store returns the full deduped, RRF-ordered, similarity-annotated
  candidate list (up to `fetch_limit`). It applies no floor: policy lives in
  the service layer.

### 3. Data model (breaking)

`SearchResult` / `SqlSearchResult` replace `score`, `distance`,
`is_fts_match` with:

- `similarity: float` — cosine in [-1, 1], always present.
- `matched_by: Literal["both", "semantic", "keyword"]`.
- `rrf_score: float | None` — kept for JSON/debug output, never rendered in
  text surfaces.

Mapper signatures (`from_polars_row`) change accordingly.

`SearchService.execute_query` gains `min_similarity: float | None = None` and
returns a `SearchResponse` model instead of a bare list:

- `results: list[SearchResult | SqlSearchResult]` — floor-applied, truncated
  to `limit`.
- `floor: float` — the effective floor used.
- `best_rejected: RejectedCandidate | None` — highest-similarity dropped
  candidate (`similarity`, `source`, `matched_by` — no text snippet);
  `None` when nothing was dropped. Formatters use it when `results` is empty.

### 4. Relevance floor — three layers

Effective floor = per-call `min_similarity` if provided, else the engine's
`similarity_floor` from `config.yaml`, else the model's registry default.

- `ModelContract.default_similarity_floor: float` — new required field.
  The similarity distribution is a property of the embedding model, so the
  default lives beside the model's other contracts. Values are set from the
  calibration run (section 7) before merge; registering a future model
  requires choosing a value (the field is required — the calibration script's
  docstring is the pointer).
- `EngineConfig.similarity_floor: float | None = None` — per-engine override,
  validated to [-1, 1] at load.
- MCP `search_*` tools and CLI search gain `min_similarity: float | None`.
  Out-of-range values return an author-controlled error message. `0.0`
  effectively disables the floor for models whose calibrated floor is
  positive (the practical escape hatch; negative similarities are noise for
  these models).

The floor is applied in `SearchService` after RRF ordering, before `limit`
truncation. `count_matching` (the SQL "Showing N of M" denominator) is
unchanged: M remains the prefilter count.

### 5. Presentation

Text surfaces show one number with one meaning. The RRF value never appears.

Result block (both families):

```
--- Result (similarity 0.78, matched: semantic+keyword) ---
```

`matched_by` renders as `semantic+keyword`, `semantic-only`, `keyword-only`.

Headers:

- Document: `Found 3 results with similarity >= 0.55 for 'query' (hybrid-ranked):`
- SQL: `Showing 3 of 250 results that matched your filters for 'query'
  (hybrid-ranked, similarity >= 0.55):`

Empty because floored (the new, load-bearing case):

```
No results with similarity >= 0.55 for 'beehive maintenance'.
Best candidate: 0.38 (tests/integration/test_lancedb.py, keyword-only match).
The corpus likely contains nothing relevant — treat absence as the answer,
or retry with different terms or a lower min_similarity.
```

Empty because no candidates at all (empty table / filters excluded
everything): current messages stay, including the SQL family's
"N rows matched your filters but none ranked…" variant.

CLI `print_results` adopts the same vocabulary
(`[Similarity: 0.78 (semantic+keyword) | DB: …]`). `results_to_json` keeps
full fidelity: `similarity`, `matched_by`, `rrf_score`.

Tool descriptions (`search_description` in both families) are rewritten to
state: similarity is cosine in [-1, 1] and comparable across queries; results
are ordered by hybrid rank fusion (semantic + keyword), so display order may
disagree with similarity order; the engine's default floor and the
`min_similarity` override; that an empty result means the corpus has nothing
relevant and should be treated as the answer; and the `matched_by` semantics
(keyword-only = exact identifier/stem hit; semantic-only = paraphrase).

### 6. Scope

Search surfaces only: MCP `search_<engine>` tools, CLI search. Browse and
triage take no query string — there is no similarity to report — and are
untouched (closes the issue's open question 4).

### 7. Calibration, not guessing

New `scripts/calibrate_similarity_floor.py`:

- Input: engine name, a set of known-relevant queries and a set of
  known-absent queries (seeded from the issue's live cases: risotto, beehive,
  narrowboat + counterpart on-topic queries).
- Runs unfloored searches against the live corpus, prints per-query top
  similarities and percentile distributions for both sets, and suggests a
  floor in the separation gap.
- Run once per model (gemma-bf16 via `md`/`sql`, granite-r2 via
  `md-granite`/`sql-granite`); the chosen values become the
  `default_similarity_floor` registry entries in the same change set.
  Final values are picked by a human from the printed distributions.

### 8. Testing

TDD throughout.

Unit (no I/O):
- cosine helper: exactness on crafted vectors, zero-norm guard.
- provenance mapping from `_distance`/`_score` null patterns, including the
  vector-only fallback path.
- floor filtering in `SearchService`: effective-floor resolution order,
  truncation after floor, `best_rejected` selection, empty-vs-none cases.
- formatters: new headers/blocks, floored-empty rendering with evidence,
  JSON fidelity.
- config: `similarity_floor` range validation; registry field presence.

Integration (tmpdir LanceDB, synthetic vectors — no MLX needed):
- orthogonal query vector → empty result with `best_rejected` populated.
- on-topic query → results with correct similarity ordering semantics.
- FTS-only hit (token overlap, orthogonal vector) → `keyword` provenance and
  a real similarity value.
- vector-only fallback (no FTS index) → `semantic` provenance, floor applies.

### 9. Documentation

- `docs/README_MCP.md`: new output examples and semantics.
- `docs/README_PROFILES.md`: `similarity_floor` engine knob +
  `default_similarity_floor` model contract.
- `CLAUDE.md` key-design-details: replace the RRF score description.

## Non-goals

- No change to the fusion/ordering algorithm (no linear-combination fusion,
  no cross-encoder reranker, no torch dependency).
- No per-result qualitative bands ("strong/weak") — one calibrated number
  (the floor) per model; guidance lives in tool descriptions.
- No browse/triage changes, no schema change, no reingest.

## Migration notes

- Breaking for consumers of `SearchResult`/`SqlSearchResult` JSON
  (`score`/`distance`/`is_fts_match` → `similarity`/`matched_by`/`rrf_score`)
  and of `execute_query`'s return type.
- MCP text output shape changes; downstream skills reading `Score:` lines
  (e.g. find-impacting-queries) should read `similarity` instead.
- No LanceDB schema change: existing indexes remain valid; no
  `--rebuild --force` required.
