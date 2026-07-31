# Calibrating `similarity_floor` for a New Corpus

This guide explains how to measure a deployment-local `similarity_floor` for
the document engines (`md` and `md-granite`).

Do not copy a floor from another corpus or model. A floor is valid only for
the exact corpus, engine configuration, admission policy, and retrieval
geometry used during calibration. It is also valid to finish with
`similarity_floor` unset when no safe value exists.

Commands below use angle-bracket placeholders such as `<corpus-name>`.
Replace every placeholder before running the command.

## What calibration does

Calibration uses two independently authored query sets:

1. A **development set** exposes score distributions and ranking behavior. A
   human reads its report and chooses a candidate floor.
2. A **locked evaluation set** tests that choice once. It must be written,
   reviewed, tracked, and committed before any development query is retrieved.

The runner measures:

- expected-source hit@1, hit@5, rank, and mean reciprocal rank;
- relevant-query empty rate;
- absent-query and off-domain rejection rates;
- no-answer precision;
- relevant and absent similarity distributions;
- original versus oversampled fetch latency;
- candidates admitted only by the lexical gate.

Locked evaluation accepts a floor only when:

- expected-source hit@5 does not regress from the unfloored baseline;
- relevant empty rate is at most 5%;
- at least 60% of absent queries return empty;
- every off-domain anchor returns empty.

## Before you start

You need:

- a working `config.yaml`;
- the new corpus available through absolute filesystem paths;
- both document indexes rebuilt over the same source files;
- a clean Git worktree for the calibration code and query-set files;
- enough time to run each locked evaluation without editing or ingesting the
  corpus concurrently.

Stop MCP servers, watchers, and other ingestion processes before a run. The
runner pins the starting Lance table version and aborts on corpus drift.

The runner also requires these paths to be tracked and clean:

- `scripts/calibrate_similarity_floor.py`;
- `src/dbs_vector/`;
- `pyproject.toml`;
- `uv.lock`;
- the selected query-set file;
- the choice record during evaluation.

If `uv run` changes `uv.lock`, synchronize and commit the lockfile before
calibration. Do not calibrate with an automatically modified dependency state.

## Step 1: Configure identical corpus roots

Set the new absolute roots under both `md` and `md-granite`. The lists must be
identical and in the same order:

```yaml
engines:
  md:
    # existing model, chunker, profile, and prefix fields
    exclusion_filters: [excalidraw, compressed_json]
    paths:
      - "/absolute/path/to/corpus"
      - "/absolute/path/to/additional-docs"

  md-granite:
    # existing model, chunker, and profile fields
    exclusion_filters: [excalidraw, compressed_json]
    paths:
      - "/absolute/path/to/corpus"
      - "/absolute/path/to/additional-docs"
```

Important constraints:

- Do not add `gitignore` to `exclusion_filters` when a configured root is
  itself ignored. That can silently exclude the entire corpus.
- Root directory basenames must be unique. For example, two different roots
  both named `docs` are ambiguous to the calibration identity.
- A source file must belong to exactly one configured root.
- Leave `similarity_floor` unset while calibrating.
- Keep corpus-specific paths and final floors in ignored `config.yaml`.
  `config.yaml.example` should remain generic and unfloored.

Validate the configuration:

```bash
uv run python - <<'PY'
from dbs_vector.config import load_settings

configured = load_settings("config.yaml", validate=True)
print("validated engines:", ", ".join(sorted(configured.engines)))
PY
```

## Step 2: Rebuild both indexes

Changing roots, models, prefixes, chunking, or profiles requires a rebuild:

```bash
uv run dbs-vector ingest --type md --rebuild --force
uv run dbs-vector ingest --type md-granite --rebuild --force
```

Both commands must finish successfully. Calibration preflight later verifies
that the normalized source sets are identical and non-empty; equal source
counts alone are not sufficient.

Do not modify or re-ingest the corpus after this point until development and
evaluation are complete.

## Step 3: Create new query-set files

Create a new directory instead of editing query sets from an earlier corpus:

```text
scripts/calibration/<corpus-name>/
├── dev.json
└── eval.json
```

For shared document-engine calibration, keep `"corpus": "documents"` so the
runner enforces identical `md` and `md-granite` source sets.

Each query has this schema:

```json
{
  "query": "<query text>",
  "kind": "relevant",
  "shape": "prose",
  "expected_source": "<root-name/path/to/file.md>",
  "note": "<optional reason for this label>"
}
```

An absent query uses `absent_kind` and must not contain `expected_source`:

```json
{
  "query": "<query whose answer is genuinely absent>",
  "kind": "absent",
  "shape": "long",
  "absent_kind": "hard_negative",
  "note": "<why this is adjacent to the corpus but unanswered>"
}
```

Allowed values:

| field | values |
|---|---|
| `name` | `dev` or `eval` |
| `kind` | `relevant` or `absent` |
| `shape` | `identifier`, `filename`, `error`, `prose`, `short`, `long` |
| `absent_kind` | `hard_negative` or `off_domain` |

### Development-set requirements

- At least 20 relevant queries.
- At least 20 absent queries.
- At least 10 absent queries must be in-domain hard negatives.
- At least one absent query must be an unambiguous off-domain anchor; using
  several is preferable.
- Relevant queries must cover all six query shapes.
- Relevant labels must cover at least 12 distinct expected sources.
- No expected source may label more than three development queries.

### Locked-evaluation requirements

- At least 15 relevant queries.
- At least 15 absent queries.
- At least 8 absent queries must be hard negatives.
- At least one off-domain anchor is required.
- Relevant queries must cover all six query shapes.
- Do not reuse or paraphrase development queries.
- Prefer expected sources not used by the development set.

### Labeling guidance

- `expected_source` is a readable path suffix, not a machine-specific absolute
  path. It must resolve to exactly one stored corpus source.
- A hard negative should use the corpus's vocabulary while asking about
  something the corpus genuinely does not cover.
- Search or inspect the raw corpus while labeling absent queries. Automated
  validation cannot prove that an answer is absent.
- Watch for evaluation leakage: if an ingested design document literally
  contains an absent query, the lexical gate may correctly retrieve that
  mention. Do not label such a query absent without reviewing what the
  retrieved text actually says.
- Have another person review relevant sources and absent labels when possible.

Query sets are corpus-specific evidence, so this repository ships none to copy
from. The schema above plus the composition rules below are the whole
contract; a set authored for another corpus is never reusable.

## Step 4: Validate without retrieval

Preflight validates schema, set composition, source suffixes, non-empty
tables, and shared source identity. It does not load an embedder or execute a
search:

```bash
for engine in md md-granite; do
  uv run python scripts/calibrate_similarity_floor.py \
    --engine "$engine" \
    --set scripts/calibration/<corpus-name>/dev.json \
    --preflight-only

  uv run python scripts/calibrate_similarity_floor.py \
    --engine "$engine" \
    --set scripts/calibration/<corpus-name>/eval.json \
    --preflight-only
done
```

Fix every error before continuing.

Verify that development and evaluation query text does not overlap:

```bash
uv run python - <<'PY'
from dbs_vector.services.calibration import load_query_set

dev = load_query_set("scripts/calibration/<corpus-name>/dev.json")
evaluation = load_query_set("scripts/calibration/<corpus-name>/eval.json")

normalize = lambda text: " ".join(text.split()).casefold()
dev_text = {normalize(query.query) for query in dev.queries}
eval_text = {normalize(query.query) for query in evaluation.queries}
overlap = dev_text & eval_text
assert not overlap, f"query overlap: {sorted(overlap)}"
print("query sets are valid and disjoint")
PY
```

## Step 5: Seal both query sets before retrieval

Track and commit the development set:

```bash
git add scripts/calibration/<corpus-name>/dev.json
git commit -m "test(calibration): add <corpus-name> development set"
```

Then commit the locked evaluation set:

```bash
git add scripts/calibration/<corpus-name>/eval.json
git commit -m "test(calibration): lock <corpus-name> evaluation set"
```

Do not execute a development query until the evaluation commit exists. The
commit proves the evaluation labels were not adapted after seeing development
scores.

After sealing, do not edit `eval.json`. A replacement must use a new filename,
such as `eval-2.json`.

## Step 6: Run development calibration

Calibrate one engine at a time, normally `md-granite` first:

```bash
uv run python scripts/calibrate_similarity_floor.py \
  --engine md-granite \
  --set scripts/calibration/<corpus-name>/dev.json \
  --limit 5
```

The report is written under ignored `calibration_reports/` with the engine,
set, query-set digest, and timestamp in its name. The command prints the exact
path at the end.

Repeat independently for `md`:

```bash
uv run python scripts/calibrate_similarity_floor.py \
  --engine md \
  --set scripts/calibration/<corpus-name>/dev.json \
  --limit 5
```

Never copy a floor from one engine to another. Their embedding spaces,
prefixes, and chunking profiles differ.

## Step 7: Read the report and choose manually

The runner's `SUGGESTED FLOOR` is advice, not a decision. For each engine,
review:

1. Relevant expected-source similarity minimum and p05.
2. Absent top-similarity p95 and maximum.
3. Queries in the overlap between those distributions.
4. Baseline versus floor-active hit@5 and MRR.
5. Relevant empty rate.
6. Overall and off-domain rejection rates.
7. Every lexical-gate-only admission, especially on absent queries.
8. Whether viable floors form a stable plateau or a narrow cliff.
9. Original versus oversampled latency.

Bias conservatively: retaining a noisy result is usually safer than emptying
a real query and encouraging a false corpus-absence conclusion.

### If no floor passes

Record **no safe floor found** and leave `similarity_floor` unset. This is a
successful calibration outcome. Create a choice record with `"floor": null`,
preserve the development report as described below, and skip locked
evaluation for that engine.

### If a numeric floor passes development

Choose the value yourself from the tables. Do not automatically apply the
suggestion and do not run evaluation yet.

## Step 8: Preserve the development report and seal the choice

Copy the exact development report unchanged into the durable directory. It
stays out of version control: a report measures one deployment's corpus and
means nothing to anyone else, so `calibration_reports/` is gitignored.

```bash
mkdir -p calibration_reports/durable
cp calibration_reports/<exact-development-report>.json \
  calibration_reports/durable/<exact-development-report>.json
```

Get its digest:

```bash
shasum -a 256 \
  calibration_reports/durable/<exact-development-report>.json
```

Read `identity.corpus_digest` and `identity.code_commit` from the report:

```bash
uv run python - <<'PY'
import json

path = "calibration_reports/durable/<exact-development-report>.json"
report = json.load(open(path, encoding="utf-8"))
print(json.dumps(report["identity"], indent=2))
PY
```

Create
`scripts/calibration/choices/<engine>-<development-query-digest12>.json`:

```json
{
  "engine": "<engine>",
  "floor": null,
  "rationale": "<decision based only on development evidence>",
  "dev_report_path": "calibration_reports/durable/<exact-development-report>.json",
  "dev_report_digest": "<sha256>",
  "corpus_digest": "<identity.corpus_digest>",
  "code_commit": "<identity.code_commit>"
}
```

Keep `null` for a no-safe-floor outcome. For a numeric choice, replace `null`
with the JSON number; do not quote it.

Commit the choice record. Sealing means "committed", so this step is what
makes the decision auditable even though the report itself stays local:

```bash
git add scripts/calibration/choices/<choice-record>.json
git commit -m "test(calibration): seal <engine> floor choice"
```

The digest in the record binds the committed decision to the exact local
report, so moving or editing the report invalidates the choice.

Evaluation refuses a choice whose engine, floor, corpus, code identity, or
development-report digest does not match.

## Step 9: Run locked evaluation exactly once

Skip this step when the recorded choice is `null`.

For a numeric choice:

```bash
uv run python scripts/calibrate_similarity_floor.py \
  --engine <engine> \
  --set scripts/calibration/<corpus-name>/eval.json \
  --limit 5 \
  --floor <chosen-floor> \
  --choice-record scripts/calibration/choices/<choice-record>.json
```

Exit codes:

| code | meaning |
|---:|---|
| `0` | development completed or locked evaluation passed |
| `1` | locked evaluation ran and failed acceptance |
| `2` | setup, sealing, identity, or validation error |

The runner creates a spend marker immediately before the first evaluation
search. Once behavior from an evaluation set has been inspected, the set is
spent for that engine—even if the run fails, is interrupted, or detects
corpus drift.

### On PASS

Copy the exact evaluation report unchanged into
`calibration_reports/durable/` and verify its identities. The durable copy is
also what stops the spent set from being rerun for that engine.

### On FAIL

1. Preserve the failed evaluation report in `calibration_reports/durable/`.
2. Return to the development report and choose again, or conclude no safe
   floor exists.
3. Create a new choice record; do not edit the old one.
4. Author and commit a fresh evaluation file such as `eval-2.json`.
5. Run the fresh set once. Never rerun the spent file for that engine.

The same `eval.json` may be used once for `md` and once for `md-granite`
because spending is tracked per engine.

## Step 10: Apply a passing floor

Only after locked evaluation passes, add the value to the calibrated
deployment's ignored `config.yaml`:

```yaml
engines:
  md-granite:
    # Calibrated YYYY-MM-DD. See this deployment's calibration record.
    # Recalibrate after any identity or corpus change.
    similarity_floor: <accepted-floor>
```

For a no-safe-floor outcome, omit the key entirely.

Do not put the numeric value in `config.yaml.example`; it is unsafe for an
arbitrary corpus.

## Step 11: Verify production behavior

Choose a development query that the final report shows becomes empty at the
accepted floor:

```bash
uv run dbs-vector search "<recorded query>" --type <engine>
uv run dbs-vector search "<recorded query>" --type <engine> \
  --no-similarity-floor
```

The first call should report low retrieval confidence for that attempt and
must not claim corpus-level absence. The second should restore the exact
unfloored baseline.

Confirm the MCP tool description contains the configured floor:

```bash
uv run python - <<'PY'
from dbs_vector.config import _populate_singleton_from, load_settings, settings
from dbs_vector.mcp.families.document import DocumentFamily

_populate_singleton_from(load_settings("config.yaml", validate=True))
engine = settings.engines["md-granite"]
print(DocumentFamily().search_description("md-granite", engine))
PY
```

For an unset outcome, confirm the description does not claim that a
configured floor is active.

## Step 12: Record the result

Keep one calibration record per deployment, alongside that deployment's own
notes rather than in this repository. Record:

- floor or “unset — no safe floor found”;
- full engine and corpus identity;
- query-set and choice digests/commits;
- baseline and chosen metrics;
- relevant and absent percentile summaries;
- evaluation result or the reason evaluation was skipped;
- links to every durable development, passing, failed, or invalidated report.

Run the quality gates:

```bash
uv run poe check
uv run pyright src
```

## When calibration becomes invalid

Rebuild, write fresh query sets, and recalibrate after changing any of:

- corpus contents or source labels;
- embedding model;
- query or passage prefix;
- chunker type or chunking profile;
- `nprobes`;
- admission or lexical-gate policy;
- floor-active oversampling geometry;
- LanceDB behavior that changes ranking.

Do not silently carry the old value forward. Compare the recorded corpus
digest and complete calibration identity.

## Troubleshooting

### `unsealed calibration inputs`

The query set, choice, runner, source tree, `pyproject.toml`, or `uv.lock` is
untracked, staged, or modified. Commit the intended state before running.

### `matched 0 corpus sources`

`expected_source` is misspelled or is not a suffix of a stored source path.

### `matched 2 corpus sources`

The suffix is ambiguous. Include more parent path components in
`expected_source`.

### Shared corpus source-set mismatch

Recheck that `md.paths` and `md-granite.paths` are identical, then rebuild
both indexes. Do not compare counts alone.

### Empty table after rebuilding

Check configured roots and exclusion filters. In particular, remove the
`gitignore` exclusion if it hides the corpus root.

### No safe floor found

Leave the key unset and record the outcome. Do not weaken acceptance criteria
or spend the evaluation set merely to force a numeric default.

## Related documentation

- [`scripts/README.md`](../scripts/README.md) — runner command reference
- [`README_PROFILES.md`](README_PROFILES.md) — where `similarity_floor` sits
  among the per-engine tuning knobs
- [Search scoring and admission](README_MCP.md#similarity-ranking-and-admission)
