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

Per engine, **two separately-authored set pairs** — a set used to choose the
floor cannot also certify it:

- **Development set** — inspected freely: distributions, per-query tables,
  gate audits; the floor is chosen from this set.
- **Locked evaluation set** — written (with expected sources per relevant
  query) *before* calibration begins, run **once** after the floor is
  chosen, and used exclusively for acceptance. If it fails, the floor goes
  back to development and the evaluation set is considered spent — write a
  fresh one before the next acceptance attempt.

Each pair contains:

- **Relevant** queries — a known-correct answer exists in the corpus; the
  expected source is written down before any run.
- **Absent** queries — the correct answer is "nothing here", including
  **in-domain hard negatives** (topically adjacent to the corpus, not
  risotto-style off-domain softballs) and the issue's measured live cases
  (beehive, narrowboat) as regression anchors.

Composition requirements — both sets must cover the query shapes agents
actually send:

- exact identifiers (`delete_by_source`), file names, error strings;
- prose/conceptual questions;
- SQL intents for the SQL engines (table names, query-pattern descriptions);
- short (1–2 token) and long (sentence-plus) forms.

Minimum sizes per engine: development ≥ 20 relevant + ≥ 20 absent (≥ 10
hard negatives); locked evaluation ≥ 15 relevant + ≥ 15 absent (≥ 8 hard
negatives). Sets live in the repo under `scripts/calibration/<engine>/` as
plain text/JSON so reruns are cheap and diffs reviewable.

Statistical honesty: at these sizes, confidence intervals would be
decorative. We report raw counts, full per-query similarity tables, and
distribution percentiles, and we choose floors conservatively (below the
observed relevant-set minimum for on-target queries, above the absent-set
bulk). If the sets later grow to where interval estimates mean something,
add them then.

## 2. Calibration runs — `scripts/calibrate_similarity_floor.py`

For a given engine:

- loads the labeled sets, runs unfloored searches via
  `disable_similarity_floor` (baseline code paths, original candidate-pool
  size, real model, live corpus) — this is what makes recalibration runs
  exactly comparable to baseline behavior;
- prints per-query: top-5 similarities, `retrieved_by`, and whether the
  expected source ranked first (relevant set);
- prints per-set distribution percentiles and the overlap region;
- suggests a floor as the widest separation gap, but **the final value is
  chosen by a human** reading the tables;
- emits a machine-readable JSON report for the metrics below.

## 3. Metrics — reported per engine, before/after floor

Only one expected source is labeled per relevant query, so precision@k over
unjudged candidates is not computable; rank-of-expected-source metrics are
used instead.

- **expected-source hit@k (k = 1, 5), rank, and MRR** on the relevant sets:
  where does the labeled answer land.
- **absent rejection rate** = absent queries returning empty / all absent
  queries — the number this effort exists to raise (0 today by
  construction). This is a specificity-style rate, not a precision.
- **no-answer precision** = absent queries returning empty / all queries
  returning empty — how trustworthy an empty response is when an agent
  sees one.
- **relevant empty rate** = relevant queries returning empty / all relevant
  queries — the cost side; a floor that empties real questions is worse
  than no floor.
- **lexical-gate audit**: every admission that passed only via the lexical
  gate, and every rejection the gate failed to rescue, listed for reading.
- **latency**: p50/p95 per search, floored vs. unfloored (oversampling
  triples the per-leg fetch).

Acceptance per engine — measured **once, on the locked evaluation set**:

- expected-source hit@5 does not regress vs. the unfloored baseline run;
- relevant empty rate ≤ 5% (at most one query at the minimum set size, and
  that query documented);
- absent rejection rate ≥ 60%, including 100% of the off-domain anchor
  queries (risotto-class).

If no floor value satisfies all three, the explicit, recorded outcome is
**"no safe floor found"**: the engine's `similarity_floor` stays unset and
the calibration table says so — an unset floor is a valid end state, not a
failure to finish.

## 4. Calibration identity

A floor is only valid for the configuration it was measured on. Each engine's
chosen value is recorded (in a `docs/superpowers/calibration.md` table) with:

- engine name, model registry key, passage/query prefixes, chunker type and
  tuning profile;
- corpus identity: the Lance table version plus a deterministic content
  digest (sha256 over the sorted `content_hash` column) — row count and
  ingest date cannot distinguish two different corpora of equal size;
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
