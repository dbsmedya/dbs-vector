---
name: query-rewrite
description: >-
  Rewrite a slow or inefficient SQL query for performance when an index alone won't fix
  it — full-table or per-row aggregates that should be rollups, SELECT * with deep
  pagination, redundant or accidental joins, ORM-generated CASE/subquery noise, or
  queries already using the right index but still scanning millions of rows. Because a
  safe rewrite depends on business meaning the SQL text does NOT carry (what a status
  code means, whether a column is effectively nullable, which columns the caller truly
  consumes, whether stale/approximate results are acceptable), this skill INTERVIEWS the
  domain owner with targeted questions before proposing anything, then validates that
  the rewrite preserves results and improves the plan. Use when the user wants to
  rewrite, refactor, simplify, or speed up a specific SQL query, or when
  find-impacting-queries hands off a "rewrite candidate". Triggers: "rewrite this query",
  "this query is slow and indexing didn't help", "simplify this SQL", "optimize this
  SELECT/JOIN", "make this aggregate faster".
---

# Query Rewrite (domain-informed)

Rewrite a query whose slowness is structural to *how it's written*, not to a missing
index. This is the sibling of `find-impacting-queries`, which finds and ranks offenders
and recommends indexes; when it marks a query **REWRITE CANDIDATE**, it lands here.

## The one rule that shapes everything

**Never change what a query *returns* without the domain owner confirming it's safe.**
The SQL text is ambiguous about intent: `WHERE status IN (2,5,9)` could be "paid +
refunded + cancelled" or any three codes; `SELECT *` might mean "the caller needs all
40 columns" or "the ORM is lazy"; `COUNT(DISTINCT customer_id)` might tolerate an
approximate answer or not. An LLM cannot recover this from the query alone, and a rewrite
that silently changes results is worse than a slow query. So this skill is **interview
first, rewrite second** — and the human owns the final equivalence judgement.

## Workflow

### 1. Establish the query and why an index won't save it
Start from the query + its plan. If handed off from `find-impacting-queries`, reuse its
evidence; otherwise `EXPLAIN` it (read-only) and confirm the lever isn't a missing index
(`list_indexes`, `table_size`). Name the structural cost in one line: *33M-row range
scan + filesort for a GROUP BY*, *SELECT * forcing row fetches a covering index would
avoid*, *correlated subquery per row*, etc.

### 2. Interview the domain owner — one question at a time
Ask only what changes the rewrite. Prefer concrete, multiple-choice questions. Draw from
the bank below; skip what's already known. Wait for each answer before the next.

- **Result shape:** Which columns does the caller actually use? (a `SELECT *` is often
  reducible to 2-3 columns → covering index or smaller payload.)
- **Filter semantics:** What do these status/type codes mean, and is the set stable? Are
  any predicates dead (`1=1`, `x IN (NULL)`, ORM `CASE … ELSE` that always picks one
  branch)?
- **Freshness:** Must this be real-time, or is a cached / nightly-rollup / approximate
  answer acceptable? (Decides rollup-table vs live rewrite.)
- **Cardinality intent:** Is `DISTINCT`/`GROUP BY` essential, or defensive? Is the join
  to table X needed for the result, or only for a filter that could be an `EXISTS`?
- **Pagination:** Is `LIMIT ?,?` real user paging (→ keyset/seek pagination) or an
  export (→ stream/batch)?
- **Volume & access path:** How wide is the typical date/id range? How often is it
  called, by what (scheduled job vs interactive)?

### 3. Propose rewrite options tied to their answers
Offer 1-3 semantically-equivalent rewrites with the tradeoff each makes, each justified
by a specific answer from step 2 (e.g. "you said the caller only reads id+amount, so a
covering index + narrowed projection removes the row fetches"). Common moves: narrow the
projection; replace filtering joins with `EXISTS`/semi-join; keyset pagination instead of
`OFFSET`; pre-aggregate into a summary/rollup table for scheduled reports; hoist
ORM-generated dead predicates; split an OR into a `UNION ALL` of indexable branches.

### 4. Validate — plan and equivalence
- `EXPLAIN` the rewrite (read-only) and show the before/after plan + estimated rows.
- Propose a **result-equivalence check the human runs**, never assume it: e.g. compare
  `SELECT COUNT(*), <checksums>` of old vs new over a bounded, representative window.
  State the assumptions the equivalence rests on (their step-2 answers).

### 5. Output
Deliver the rewritten SQL, the before/after plan, the equivalence-check query, and the
explicit list of domain assumptions it depends on. **Read-only**: only
`SELECT`/`SHOW`/`EXPLAIN` touch the server; the human reviews assumptions and applies the
change. If the answers reveal the rewrite would change results in a way the owner doesn't
want, say so and stop — a correct slow query beats a fast wrong one.

## Anti-patterns
- Rewriting before interviewing — you'll encode a guess about intent.
- Claiming equivalence you didn't (and can't) verify — hand the human a check instead.
- Drifting into index recommendations — that's `find-impacting-queries`. If an index
  turns out to be the real fix, hand it back rather than forcing a rewrite.
