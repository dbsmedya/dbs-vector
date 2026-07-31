# Similarity-floor calibration

A `similarity_floor` is valid only for the deployment identity on which it was
measured. Changing an engine's model, prefixes, chunker, tuning profile,
`nprobes`, admission policy, retrieval-pool geometry, or corpus invalidates the
record. Compare the source-aware corpus digest, not only the row count.

An unset floor is a valid result. It means no candidate value cleared every
acceptance criterion; it does not mean calibration was incomplete.

These are deployment-local measurements over the ignored local
`config.yaml` roots `.ayder/` and `docs/`. The generic
`config.yaml.example` deliberately leaves floors unset.

## Engines

| engine | floor | model/profile | prefixes | table/version/rows | corpus digest | nprobes/LanceDB | run/code commit | development set digest/commit | choice digest/commit | run date |
|---|---|---|---|---|---|---|---|---|---|---|
| `md-granite` | unset — no safe floor found | `granite-r2` / `granite-md-medium` | empty / empty | `knowledge_vault_granite` / 111 / 1,673 | `d135104db2167efb8f63f85a5a19e8369f65c0039a2f9634d8460d58e6770972` | 20 / 0.30.2 | `55687ff27eb773ad580dc6791d26b36504bd985b` / same | `2fb44e744f59aa5927ee633faff00a9645fde56adc0a230a7a855e37d80e2074` / `65f35023684ea41ce68972a9a69c3b58aed0a712` | `edda1b21e1770807e0d2b170dbd8e49805110b714bcc4eece1515964ade6a107` / `8ae4ce7081702bcafac0bf55e4a82cd51c07031b` | 2026-07-31 |
| `md` | unset — no safe floor found | `gemma-bf16` / `gemma-md` | `title: none \| text: ` / `task: search result \| query: ` | `knowledge_vault` / 37 / 1,984 | `c5c5295b670af8d4225228eee0ace3962ee09312d3222c9514a4ee280f598e43` | 20 / 0.30.2 | `8ae4ce7081702bcafac0bf55e4a82cd51c07031b` / `55687ff27eb773ad580dc6791d26b36504bd985b` | `2fb44e744f59aa5927ee633faff00a9645fde56adc0a230a7a855e37d80e2074` / `65f35023684ea41ce68972a9a69c3b58aed0a712` | `7873c30ce42e66c1ee618f65fcc33457c2733f06ede60cb58c0f02a9432be098` / `965ad78065d75512b826089e6fa9c18010b9f4a4` | 2026-07-31 |
| `sql` | unset — not yet calibrated | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| `sql-api` | unset — not yet calibrated | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| `sql-granite` | unset — not yet calibrated | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| `sql-api-granite` | unset — not yet calibrated | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

## Measured results

The second column below is the exact original-pool baseline. Because neither
engine produced a viable floor, the third column shows the strongest
recall-preserving development candidate for diagnosis only. Those candidates
were not configured and were not evaluated.

### md-granite

- Development set: `scripts/calibration/documents/dev.json`
- Locked evaluation set: `scripts/calibration/documents/eval.json` (sealed,
  unspent, and not run because development found no safe value)
- Development evidence:
  [`md-granite-dev-2fb44e744f59-20260731T011208Z.json`](calibration-reports/md-granite-dev-2fb44e744f59-20260731T011208Z.json)
- Decision:
  `scripts/calibration/choices/md-granite-2fb44e744f59.json`

| metric | unfloored | diagnostic candidate 0.871314914598602 |
|---|---:|---:|
| expected-source hit@1 | 0.200 | 0.250 |
| expected-source hit@5 | 0.700 | 0.700 |
| MRR | 0.388 | 0.433 |
| relevant empty rate | 0.000 | 0.000 |
| absent rejection rate | 0.000 | 0.550 |
| off-domain rejection rate | 0.000 | 0.625 |
| no-answer precision | undefined — nothing empty | 1.000 |
| fetch latency p50 / p95 | 2.625 / 2.923 ms | 3.071 / 3.376 ms |

Relevant expected-source similarity:
min/p05/p50/p95/max = 0.855189/0.855189/0.905842/0.960265/0.960265
(14 of 20 expected sources reached the original top-5 pool).

Absent top similarity:
min/p05/p50/p95/max = 0.787737/0.787737/0.848887/0.896258/0.898014.

Decision: **no safe floor found**. The strongest recall-preserving state
missed both rejection requirements: 55% absent versus the required 60%, and
62.5% off-domain versus the required 100%. The score distributions overlap,
so no evaluation run was authorized.

### md

- Development set: `scripts/calibration/documents/dev.json`
- Locked evaluation set: `scripts/calibration/documents/eval.json` (sealed,
  unspent, and not run because development found no safe value)
- Development evidence:
  [`md-dev-2fb44e744f59-20260731T011349Z.json`](calibration-reports/md-dev-2fb44e744f59-20260731T011349Z.json)
- Decision: `scripts/calibration/choices/md-2fb44e744f59.json`

| metric | unfloored | diagnostic candidate 0.4627230370852818 |
|---|---:|---:|
| expected-source hit@1 | 0.350 | 0.350 |
| expected-source hit@5 | 0.700 | 0.700 |
| MRR | 0.471 | 0.468 |
| relevant empty rate | 0.000 | 0.000 |
| absent rejection rate | 0.000 | 0.650 |
| off-domain rejection rate | 0.000 | 0.625 |
| no-answer precision | undefined — nothing empty | 1.000 |
| fetch latency p50 / p95 | 2.621 / 3.002 ms | 3.102 / 3.466 ms |

Relevant expected-source similarity:
min/p05/p50/p95/max = 0.403023/0.403023/0.575213/0.774308/0.774308
(14 of 20 expected sources reached the original top-5 pool).

Absent top similarity:
min/p05/p50/p95/max = 0.109665/0.109665/0.397840/0.514634/0.536073.

Decision: **no safe floor found**. The strongest recall-preserving state met
the overall absent criterion (65%) but rejected only 62.5% of off-domain
anchors rather than 100%. Raising the floor was not safe for known-answer
behavior, so no evaluation run was authorized.

## Deferred baseline decisions

- **Unconditional oversampling — keep it floor-conditional.** On
  `md-granite`, original versus 3× unfloored pools held hit@5 at 0.700 but
  regressed MRR from 0.388 to 0.364 while p95 fetch latency rose from 2.923 to
  3.376 ms. `md` improved hit@5 from 0.700 to 0.750 and MRR from 0.471 to
  0.485, with p95 rising from 3.002 to 3.466 ms. Because one engine regressed,
  the production fetch geometry remains conditional on an active floor.
- **Cosine-metric fix magnitude — correct and small/inert here.** Both tables
  had a live IVF-PQ vector index; the probe bypassed it for both metrics.
  `md-granite` embedding norms were min/p05/p50/p95/max
  0.9960/0.9978/1.0001/1.0024/1.0043. At k=10, 29/40 query orderings were
  identical, 38/40 had identical membership, top-1 changed for 1/40, and mean
  shared-member Kendall disagreement was 0.0074. `md` norms rounded to
  1.0000 at every percentile; order and membership were identical for 40/40,
  top-1 changed for 0/40, and disagreement was 0.0000.
- **Lexical-gate tuning — unchanged.** Auditing the strongest
  recall-preserving candidates found 11 relevant and 15 absent gate-only rows
  for `md-granite`, and 10 relevant and 17 absent rows for `md`. The affected
  absent queries were the predeclared calibration examples (`beehive`,
  `narrowboat`, `risotto`, Kafka, and cross-encoder), whose exact terms occur
  in design/plan documents in this corpus. No common stopword or
  three-character-token failure pattern appeared. Adding topic-specific
  stopwords would hide corpus mentions and damage the exact-identifier recall
  the gate protects, so `_STOPWORDS` and `_MIN_TOKEN_LEN` remain unchanged.

## Rerunning

```bash
uv run python scripts/calibrate_similarity_floor.py \
  --engine <engine> \
  --set scripts/calibration/documents/dev.json
```

Choose only from the development tables. Certify a numeric choice once on a
committed locked set and never rerun a spent set. See
[`scripts/README.md`](../../scripts/README.md).
