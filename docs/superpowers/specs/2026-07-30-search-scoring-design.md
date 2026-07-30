# Search Scoring Baseline: Honest Similarity + Floor Mechanics — Design

**Date:** 2026-07-30 (revised same day after external review)
**Status:** Approved design, pre-implementation
**Source issue:** `.ayder/superpowers_20260730/search_scoring_work.md`
**Companion spec:** `2026-07-30-search-scoring-calibration-design.md` — per-engine
calibration, evaluation on real corpora, and default floor values. This
baseline ships the mechanics; the companion ships the numbers.
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
  (null when the vector leg did not). This yields retrieval-channel
  provenance.
- Latent bug: `_build_hybrid` never sets `.metric("cosine")`, so on tables
  without an IVF index the hybrid vector leg runs the default L2 metric.
  Ordering is unaffected only when embeddings happen to be unit-norm, which
  no code guarantees. `LanceHybridQueryBuilder.metric()` exists (query.py:2045).
- LanceDB applies `limit` to **each leg separately before fusion**
  (query.py `_create_query_builders`: both `_vector_query.limit()` and
  `_fts_query.limit()`), so any change to the fetch limit changes the RRF
  candidate pools and can change realized rankings.

## Outcomes (from the issue)

1. Every result carries a similarity value on a consistent, query-independent
   scale (exact cosine). This is a geometric scale, **not** a calibrated
   probability of relevance: comparisons are meaningful within a single
   engine/configuration, and how much relevance a given value implies varies
   by query shape until the companion spec calibrates it.
2. A query with no good answer can return empty — absence is visible.
3. Hybrid retrieval and RRF ordering are preserved as the ranking method
   (lexical recall matters).
4. Ranking behavior changes are deliberate, minimized in this baseline, and
   measured in the companion spec — not assumed away.

## Design

### 1. Retrieval

- Hybrid search with RRF(K=60) ordering stays the ranking method.
- `_build_hybrid` adds `.metric("cosine")` and an explicit
  `.rerank(RRFReranker(K=60, return_score="all"))`.
- Fetch limit: `limit` when no floor is active (identical candidate pools to
  today); `limit * _FLOOR_OVERSAMPLE` (constant, 3) when a floor is active,
  so floor-filtering doesn't starve the requested `limit`. The oversized
  `table_filter` full-scan path keeps its existing `count_rows()` fetch.
  Dedupe by `id` as today; admission filtering happens downstream, then
  truncation to `limit`.

Ranking honesty: two deliberate behavior changes exist and are stated, not
hidden. (a) The cosine-metric fix can change vector-leg ranks wherever
embeddings are not unit-norm — it is a bug fix, evaluated in the companion
spec. (b) When a floor is active, the enlarged per-leg pools change the RRF
fusion inputs (a row ranked 10th in both enlarged legs scores 2/70 ≈ 0.0286
and outranks a rank-1 single-leg row at 1/61 ≈ 0.0164). With no floor active,
pools are unchanged. The companion spec measures both effects; making
oversampling unconditional is one of its decisions.

### 2. Scoring (in `LanceDBStore.search`)

- For every returned row, compute exact cosine similarity between
  `query_vector` and the row's `vector` column in NumPy:
  `sim = dot(q, v) / (|q| * |v|)`, guarding zero norms (either norm 0 →
  similarity 0.0). This is metric-independent and covers FTS-only rows —
  precisely the rows LanceDB's `_distance` leaves null. Clamp the computed
  value to [-1, 1] (float32 rounding can exceed the declared range) and
  guard non-finite inputs: a row whose vector yields a non-finite similarity
  gets 0.0 plus a warning log — a NaN would otherwise silently fail every
  floor comparison and poison `best_rejected` selection.
- A missing `vector` column in a search result row is a programming error
  (search applies no projection): raise `ValueError`, do not degrade.
- `retrieved_by` — retrieval-channel membership, from the
  `return_score="all"` null pattern: `_distance` non-null and `_score`
  non-null → `both`; only `_distance` → `vector`; only `_score` → `fts`.
  The pure-vector fallback path (`_hybrid_ok is False` or oversized
  `table_filter`) is always `vector`, and `rrf_score` is `None` there (no
  fusion ran). `retrieved_by` states which channel returned the row —
  nothing more. It is not evidence the match is semantically or lexically
  correct, and the docs/tool descriptions must describe it that way.
- The store returns the full deduped, RRF-ordered, similarity-annotated
  candidate list (up to the fetch limit). It applies no floor: policy lives
  in the service layer.

### 3. Data model (breaking)

`SearchResult` / `SqlSearchResult` replace `score`, `distance`,
`is_fts_match` with:

- `similarity: float` — cosine in [-1, 1], always present.
- `retrieved_by: Literal["both", "vector", "fts"]`.
- `rrf_score: float | None` — kept for JSON/debug output, never rendered in
  text surfaces.

Mapper signatures (`from_polars_row`) change accordingly.

`SearchService` gains constructor-injected policy (the engine's
`similarity_floor`), so the service needs no knowledge of `Settings`. A new
`build_search_service(engine_name)` factory in `services/bootstrap.py`
centralizes embedder/store construction plus floor injection, and every
construction site goes through it — today `mcp/state.initialize_services`,
the CLI `search` command, and `scripts/dbs-web.py` each build
`SearchService(deps.embedder, deps.store)` independently, which is exactly
where a hand-wired floor would drift. `execute_query` gains
`min_similarity: float | None = None`, `disable_similarity_floor: bool =
False`, and returns a `SearchResponse` model instead of a bare list:

- `results: list[SearchResult | SqlSearchResult]` — admission-filtered,
  truncated to `limit`.
- `floor: float | None` — the effective floor used (`None` = no floor
  active).
- `inspected: int` — number of deduped candidates examined (for honest
  empty-result wording).
- `best_rejected: RejectedCandidate | None` — highest-similarity dropped
  candidate (`similarity`, `source`, `retrieved_by` — no text snippet);
  `None` when nothing was dropped. Formatters use it when `results` is empty.

### 4. Admission policy — dual channel, floor optional

Floor resolution, in precedence order: per-call
`disable_similarity_floor=True` → no floor **and** the original
candidate-pool size (the exact-baseline state recalibration reruns need —
`min_similarity=0.0` is not equivalent: it still drops negative-similarity
rows and still triggers oversampled pools); else per-call `min_similarity`
if provided; else the engine's `similarity_floor` from `config.yaml`; else
no floor (today's behavior).
In this baseline no engine sets `similarity_floor`; defaults ship with the
companion spec's calibration. Floors are engine-level policy, not model
properties: the same `gemma-bf16` serves `md` (search prefixes) and
`sql`/`sql-api` (clustering prefixes) with different content shapes, so
`ModelContract` is untouched.

When a floor is active, a candidate is admitted when **either**:

1. `similarity >= floor` (semantic channel), **or**
2. the lexical gate passes (protects exact identifier/error-string recall —
   the reason hybrid search exists here; it does **not** guarantee filename
   recall: FTS indexes only the `text` column
   (`create_fts_index("text")`, lancedb_engine.py:92), so a path/filename
   query is protected only when the name also appears in chunk text):

   ```
   tokens   = case-insensitive \w+ matches in the query
              (`delete_by_source` is one token)
   eligible = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
   gate     = bool(eligible)
              and retrieved_by in {"fts", "both"}
              and every eligible token appears in the chunk text on a
                  word boundary (\b<token>\b, case-insensitive, no stemming)
   ```

   `bool(eligible)` is load-bearing: without it, a query whose tokens are
   all stopwords or shorter than three characters (`to be`, `C++`) would
   vacuously admit every FTS candidate. `_STOPWORDS` is frozen for this
   baseline as Lucene's classic 33-word English stop set (module constant);
   tuning it is a companion-spec task. Note this is an **all-terms verbatim**
   match, not phrase equality — token order and adjacency are not checked.

Known limitation, stated openly: the all-terms rule is what rejects the
measured stemming false positives (`beehive` → `stores`/`store` fails: token
absent verbatim; `narrowboat lock` → `uv.lock` fails: `narrowboat` absent)
while admitting `delete_by_source` (single token, present verbatim). A
single-common-token query (`lock`) would pass the gate against `uv.lock` —
the gate trades that residual noise for identifier recall. Tuning the
stopword list and token-length threshold is a companion-spec task driven by
real-corpus evaluation.

Admission runs in `SearchService` after RRF ordering, before `limit`
truncation. Rows failing both channels are dropped; the best of them (by
similarity) becomes `best_rejected`. `count_matching` (the SQL
"Showing N of M" denominator) is unchanged: M remains the prefilter count.

### 5. Presentation

Text surfaces show one number with one meaning. The RRF value never appears.

Result block (both families):

```
--- Result (similarity 0.78, retrieved by: vector+fts) ---
```

`retrieved_by` renders as `vector+fts`, `vector-only`, `fts-only`.

Headers:

- Document, no floor: `Found 3 results for 'query' (hybrid-ranked):`
- Document, floor active: `Found 3 results for 'query' (hybrid-ranked,
  admission: similarity >= 0.55 or all query terms verbatim):`
- SQL adds its existing `Showing N of M results that matched your filters`
  framing with the same admission suffix when a floor is active.

Empty because admission-filtered (the new, load-bearing case) — the message
leads with the only defensible conclusion (this *attempt* had low retrieval
confidence) and never asserts corpus-level absence, because only the
inspected pool is known:

```
No inspected candidate passed admission (similarity >= 0.55 or all query
terms verbatim) for 'beehive maintenance'. Inspected 15 hybrid-ranked
candidates; best was similarity 0.38 (tests/integration/test_lancedb.py,
fts-only). Retrieval confidence for this attempt is low; this does not
establish corpus-level absence. Retry with different terms or a lower
min_similarity if you expected a match.
```

Empty because no candidates at all (empty table / filters excluded
everything): current messages stay, including the SQL family's
"N rows matched your filters but none ranked…" variant.

CLI `print_results` adopts the same vocabulary
(`[Similarity: 0.78 (vector+fts) | DB: …]`).

Tool descriptions (`search_description` in both families) are rewritten to
state: similarity is exact cosine in [-1, 1] — a consistent geometric scale,
not a calibrated probability of relevance; comparisons are meaningful only
within the same engine/configuration and subject to its calibration; results
are ordered by hybrid rank fusion, so display order may disagree with
similarity order; the `min_similarity` and `disable_similarity_floor`
parameters and the engine's configured floor if any; that an empty response
means no inspected candidate passed admission — a low-confidence signal for
this attempt, not proof of absence; and that `retrieved_by` is
retrieval-channel membership only. No uncalibrated quality-band numbers
appear anywhere.

### 6. Response consumers — complete migration inventory

`execute_query`'s return-type change touches every consumer; all migrate in
this baseline:

- MCP families: `document.py`, `sql.py` (handlers + `format_results`
  signatures take `SearchResponse`).
- CLI `search` command (`cli.py`): text path via `print_results`; JSON path
  via `results_to_json`, which becomes an envelope —
  `{"floor": …, "inspected": …, "best_rejected": …, "results": [...]}` —
  with full per-result fidelity (`similarity`, `retrieved_by`, `rrf_score`).
- `scripts/dbs-web.py` `_handle_search` (constructs `SearchService`
  directly and iterates the list today): unwraps the envelope and includes
  `similarity`/`retrieved_by` in its serialized rows. Bootstrap-equivalent
  wiring there passes the engine floor.
- Tests currently touching `execute_query`/`results_to_json`/
  `print_results`: `test_search_service`, `test_sql_family`,
  `test_document_family`, `test_cli_json`, `test_cli_min_time`,
  `integration/test_cli`, `integration/test_granite_engines`,
  `integration/test_ingestion`, `integration/test_embedder_comparison`.
- Tests consuming the result **shape** (`.score`/`.distance`/
  `is_fts_match` attribute access or `from_polars_row` signature), found by
  attribute-level grep: `unit/test_mappers`, `unit/test_lancedb_engine`,
  `integration/test_count_matching_ci`,
  `integration/test_lancedb_filter_bugs`,
  `integration/test_search_table_filter_ci`.

### 7. Config

- `EngineConfig.similarity_floor: float | None = None` — per-engine,
  validated to [-1, 1] at load. Unset in this baseline for every engine.
- MCP `search_*` tools and CLI search gain `min_similarity: float | None`
  (out-of-range values return an author-controlled error message) and
  `disable_similarity_floor: bool = False` (CLI: `--no-similarity-floor`) —
  the true unfloored state: no admission filtering **and** the original
  candidate-pool size, required for exact baseline reruns during
  recalibration.

### 8. Testing

TDD throughout. Synthetic-vector tests validate **mechanics only**; whether
any particular floor value is safe on real corpora is the companion spec's
evaluation, not something these tests can show.

Unit (no I/O):
- cosine helper: exactness on crafted vectors, zero-norm guard, [-1, 1]
  clamping, non-finite input → 0.0.
- `retrieved_by` mapping from `_distance`/`_score` null patterns, including
  the vector-only fallback path.
- admission policy: effective-floor resolution order (disable flag >
  per-call > engine > none); semantic-channel admission; lexical-gate
  admission (all-terms verbatim, stopword and length-3 exclusions,
  FTS-channel requirement); **no-eligible-token queries (all stopwords or
  all short tokens, e.g. `to be`, `C++`) never pass the gate**;
  `disable_similarity_floor` returns unfloored results with the original
  pool size; truncation after admission; `best_rejected` selection;
  `inspected` count; no-floor path returns everything unchanged.
- formatters: new headers/blocks, admission-empty rendering with evidence,
  JSON envelope fidelity.
- config: `similarity_floor` range validation.

Integration (tmpdir LanceDB, synthetic vectors — no MLX needed):
- orthogonal query vector with floor → empty result, `best_rejected`
  populated, `inspected` correct.
- on-topic query → results with exact similarities attached.
- relevant low-cosine identifier: FTS hit whose vector is orthogonal but
  whose text contains the query token verbatim → **admitted** despite the
  floor (`retrieved_by` includes fts) — the lexical-recall protection test.
- stemming over-match: FTS hit via stem with no verbatim token → rejected.
- vector-only fallback (no FTS index) → `retrieved_by="vector"`, floor
  applies, lexical gate never rescues (no FTS channel).

### 9. Documentation

- `docs/README_MCP.md`: new output examples and semantics.
- `docs/README_PROFILES.md`: `similarity_floor` engine knob.
- `CLAUDE.md` key-design-details: replace the RRF score description.

## Non-goals (baseline)

- No change to the fusion/ordering algorithm (no linear-combination fusion,
  no cross-encoder reranker, no torch dependency).
- No default floor values, no calibration, no real-model evaluation — all in
  the companion spec.
- No per-result qualitative bands.
- No browse/triage changes (they take no query string), no schema change,
  no reingest.

## Migration notes

- Breaking for consumers of `SearchResult`/`SqlSearchResult` JSON
  (`score`/`distance`/`is_fts_match` → `similarity`/`retrieved_by`/
  `rrf_score`), of `execute_query`'s return type, and of the CLI JSON shape
  (now an envelope).
- MCP text output shape changes; downstream skills reading `Score:` lines
  (e.g. find-impacting-queries) should read `similarity` instead.
- Default-path behavior change is limited to the cosine-metric bug fix;
  candidate pools are unchanged until a floor is configured or passed.
- No LanceDB schema change: existing indexes remain valid; no
  `--rebuild --force` required.
