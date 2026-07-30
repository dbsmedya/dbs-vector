# Search Scoring Calibration & Evaluation — Design

**Date:** 2026-07-30
**Status:** Approved design, pre-implementation; depends on the baseline spec
`2026-07-30-search-scoring-design.md` being implemented first.
**Purpose:** Turn the baseline's floor *mechanics* into safe per-engine
*defaults*, by measuring on the real corpora instead of guessing. Until this
ships, no engine has a default floor and empty-by-default is not yet active.

## Why a separate spec

The baseline deliberately ships with every `similarity_floor` unset because a
threshold that tells an LLM "conclude low confidence" is a per-engine,
per-corpus property, not a model property (the same `gemma-bf16` serves `md`
with search prefixes and `sql`/`sql-api` with clustering prefixes over
entirely different content shapes). Setting those numbers requires real-model,
real-corpus measurement — synthetic-vector tests prove mechanics, not safety.

## Scope

Per engine (starting with the document engines against the live corpora, per
the working plan; SQL engines follow the same protocol):

1. Build a labeled query set.
2. Run the baseline implementation unfloored; record similarity
   distributions and rankings.
3. Choose `similarity_floor` per engine; record the calibration identity.
4. Evaluate ranking-behavior changes the baseline introduced or deferred.
5. Update tool descriptions with calibrated guidance.

## 1. Labeled query sets

Per engine, two labeled sets:

- **Relevant** — queries with a known-correct answer in the corpus, held out
  from any tuning iteration (write the expected source per query before
  running anything).
- **Absent** — queries whose correct answer is "nothing here", including
  **in-domain hard negatives** (topically adjacent to the corpus, not
  risotto-style off-domain softballs) and the issue's measured live cases
  (beehive, narrowboat) as regression anchors.

Composition requirements — both sets must cover the query shapes agents
actually send:

- exact identifiers (`delete_by_source`), file names, error strings;
- prose/conceptual questions;
- SQL intents for the SQL engines (table names, query-pattern descriptions);
- short (1–2 token) and long (sentence-plus) forms.

Minimum sizes: ≥ 20 relevant and ≥ 20 absent per engine, of which ≥ 10
absent are hard negatives. Sets live in the repo under
`scripts/calibration/<engine>/` as plain text/JSON so reruns are cheap and
diffs reviewable.

Statistical honesty: at these sizes, confidence intervals would be
decorative. We report raw counts, full per-query similarity tables, and
distribution percentiles, and we choose floors conservatively (below the
observed relevant-set minimum for on-target queries, above the absent-set
bulk). If the sets later grow to where interval estimates mean something,
add them then.

## 2. Calibration runs — `scripts/calibrate_similarity_floor.py`

For a given engine:

- loads the labeled sets, runs unfloored searches (baseline code paths, real
  model, live corpus);
- prints per-query: top-5 similarities, `retrieved_by`, and whether the
  expected source ranked first (relevant set);
- prints per-set distribution percentiles and the overlap region;
- suggests a floor as the widest separation gap, but **the final value is
  chosen by a human** reading the tables;
- emits a machine-readable JSON report for the metrics below.

## 3. Metrics — reported per engine, before/after floor

- **recall@k / precision@k** (k = 1, 5) on the relevant set: does the
  expected source appear, and how much noise accompanies it.
- **no-answer precision**: share of absent queries returning empty (the
  metric this whole effort exists to raise; today it is 0 by construction).
- **false-negative rate**: share of relevant queries returning empty — the
  cost side; a floor that empties real questions is worse than no floor.
- **lexical-gate audit**: every admission that passed only via the lexical
  gate, and every rejection the gate failed to rescue, listed for reading.
- **latency**: p50/p95 per search, floored vs. unfloored (oversampling
  triples the per-leg fetch).

Acceptance per engine: recall@k does not regress vs. pre-baseline behavior
on the relevant set; false-negative rate ≤ 1 query in the relevant set (and
that query documented); no-answer precision strictly improves.

## 4. Calibration identity

A floor is only valid for the configuration it was measured on. Each engine's
chosen value is recorded (in a `docs/superpowers/calibration.md` table) with:

- engine name, model registry key, passage/query prefixes, chunker type and
  tuning profile;
- corpus identity: table row count and ingest date;
- script commit hash and run date;
- the chosen floor and the observed relevant/absent percentile summary.

Config changes that alter the embedding space (model, prefixes, chunker,
profile) invalidate the entry — noted in the table header, mirroring the
existing "after any config change, rebuild" rule for watched engines.

## 5. Deferred baseline decisions resolved here

The baseline confined behavior change; this spec measures and decides:

- **Unconditional oversampling**: with per-leg pools at `limit * 3` always
  (not just when floored), does ranking quality improve or regress on the
  relevant sets? Decide and fix one behavior.
- **Cosine-metric fix magnitude**: quantify ranking deltas attributable to
  the L2→cosine correction on each live corpus (compare vector-leg orderings
  old vs. new); expected small if embeddings are near unit-norm, but
  measured, not assumed.
- **Lexical-gate tuning**: stopword list contents and the length-3 token
  threshold, driven by the gate audit (finding the `lock`-style
  single-common-token readmissions and the identifier rescues).

## 6. Rollout

1. Implement baseline; run `uv run poe check`; merge.
2. Live-test on the real document corpus (md-granite first — the issue's own
   evidence base) with per-call `min_similarity` while defaults stay off.
3. Build labeled sets; run calibration; review tables; set
   `similarity_floor` per engine in `config.yaml`.
4. Update tool descriptions with the calibrated floor and guidance wording
   (still "low confidence", never "proof of absence").
5. Repeat for SQL engines.

## Non-goals

- No fusion-algorithm changes, no rerankers, no new dependencies.
- No automatic floor adaptation at runtime (drift handling is a rerun of
  this protocol, not a feedback loop).
- No multilingual query sets: the corpora are English technical docs and
  SQL; revisit only if a corpus changes character.
