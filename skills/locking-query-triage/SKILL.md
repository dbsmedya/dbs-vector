---
name: locking-query-triage
description: Use when asked to find lock contention, locking queries, "which queries lock the most rows/tables", "who is blocking who", "which services cause lock waits", "row-level lock contention", or to triage SQL lock damage via dbs-vector-stdio MCP. Single-call workflow — one search surfaces the corpus's entire lock universe, then aggregates by table and by service in memory. Companion to [[slow-query-triage]] but optimized for lock_time analysis: includes cause-vs-victim attribution heuristics and architecture-focused remediation patterns (most lock-contending queries do NOT need an index).
---

# Locking Query Triage (token-efficient, single-call survey)

A one-search workflow that surfaces the entire lock-contention universe of a
slow-log corpus, attributes damage to tables and services, and distinguishes
cause-side queries from victim-side ones. Designed as the **lock-dimension
companion** to `[[slow-query-triage]]`.

## Scope

- **In scope:** "find lock contention", "which tables/services lock most",
  "what's holding locks", "investigate lock waits", "who is blocking who"
  (at the slow-log corpus level).
- **Out of scope:**
  - Live-server lock investigation — that needs
    `SHOW ENGINE INNODB STATUS`, `performance_schema.data_locks`, or
    `INFORMATION_SCHEMA.INNODB_LOCK_WAITS`. None of those are surfaced by
    the dbs-vector-stdio MCP.
  - Cumulative execution-time triage (use `[[slow-query-triage]]`).
  - Building blocker-waiter graphs — the slow log has no edge data.

## Key fact — the lock universe is tiny

In a typical slow-log corpus (2,000–3,000 fingerprints), only **20–30**
fingerprints have any measurable cumulative lock-time. **One MCP call**
with `min_lock_time=0.001` and `limit=25` returns essentially the entire
lock universe. No follow-up searches needed.

## Tool contract (reminders — full contract in [[slow-query-triage]])

- `min_lock_time` is in **seconds** — cumulative `lock_time_sec` across
  ALL calls of a fingerprint, NOT per-call avg. (Note: differs from
  `min_time`, which is milliseconds. Easy to swap by mistake.)
- The `Lock Time: X.XXXs` shown in result metadata is also cumulative.
- **Do NOT combine `min_time` and `min_lock_time` for lock investigation**
  — `min_time` will exclude queries that locked heavily but finished
  quickly when not contending. Use `min_lock_time` alone.
- Engine choice doesn't matter with filters applied — default to your
  preferred variant (e.g. `search_sql_api`).
- **`Source Database:` is typically `n/a`** on many corpora — the slow-log
  agent doesn't always tag database names. The `source_filter` arg is
  therefore often inert; do not waste a call trying to narrow by database
  name without confirming the field is populated.

## The fast path (one MCP call)

```
search_sql_api(
    query="update lock contention write",   # placeholder; min_lock_time does the work
    limit=25,
    min_lock_time=0.001,                    # 1 ms cumulative — typically the whole lock universe
)
```

Header should report `Showing N of M results that matched your filters`,
where **M is small (≤50)** — that M *is* your lock universe size. Open with
that number in the report.

If M > limit, escalate `limit` to 50 (one more call). If even that doesn't
cover the universe, the corpus has wider lock damage than usual — note it.

## In-memory aggregations (no extra MCP calls)

From the returned set, produce three views directly from the metadata
visible on each result block:

### A. Tables × cumulative lock-time

Group by each table appearing in the `Tables:` list, sum the `Lock Time`
across all fingerprints touching it. Sort desc.

**Caveat:** a query with `Tables: a, b, c` (e.g. an `INSERT INTO a SELECT
FROM b JOIN c`) contributes its full Lock Time to *all three* tables. This
is **fair for lock analysis** (under default `REPEATABLE-READ`, SELECTs
take shared row locks on JOIN sources) — note shared rows in the report as
"(+N shared)" so the reader knows.

### B. Users / services × cumulative lock-time

Group by `User:` field. The top 2–3 entries are your noisiest services.
Production services typically follow a service-account naming convention
(common patterns: `s_*`, `svc_*`, `app_*`, `worker_*`). Named human
accounts (DBAs, analysts) and hosts matching bastion / VPN / cloud-SQL-
proxy patterns almost never appear in lock data — UPDATE-heavy paths are
overwhelmingly service-driven.

### C. Consolidate ORM-emitted near-duplicates BEFORE classifying

ORMs (Hibernate, ActiveRecord, similar) emit different normalized SQL when
a field is `?` vs explicit `NULL` in the SET list. So a single logical
write to one table from one service can surface as **4–6 fingerprints**
that differ only in which columns are `?` vs `NULL`:

```
update <table> set col_a=?,    col_b=null, ..., col_z=null where id=?
update <table> set col_a=?,    col_b=?,    ..., col_z=?    where id=?
update <table> set col_a=null, col_b=?,    ..., col_z=?    where id=?
```

**Before aggregating, collapse these:**

- Same table set + same `WHERE` clause + overlapping column list (≥70% of
  columns identical) + same `User` → one logical fingerprint.
- Sum `Lock Time` and `Calls` across the collapsed group.
- Report as a single row with the SET column list deduplicated.

Failing to collapse will under-rank a service whose ORM emits many shape
variants and over-count the "fingerprint count" column. Five wide-row
rewrites that sum to 20 s of lock are one operation, not five.

### D. Per-fingerprint pattern classification

Look at `Calls`, `Lock Time`, and the SQL shape:

| Pattern | Signature | What it means |
|---|---|---|
| **Hot-row contention** | `Calls > 5` AND `Lock Time / Calls < 2 s` | Many small writers piling onto the same row. Distributed pain. |
| **Medium contention / wide write** | `Calls > 5` AND `Lock Time / Calls` in 2–5 s | Point write that's neither cheap nor a long block. Often a wide-row UPDATE setting 20+ columns; per-call wait dominates. Treat as a **borderline victim** unless the SQL touches multiple tables. |
| **Long single block** | `Calls = 1` AND `Lock Time > 10 s` | One query waited a long time. Usually a **victim**, not a cause. |
| **Multi-table INSERT…SELECT** | `INSERT INTO x SELECT FROM y JOIN z` in normalized SQL | Holds X-locks on `x`, S-locks on `y/z`. Usually a **cause**. |
| **Large batch write** | `UPDATE t SET … WHERE tenant_id = ? AND status IN (…)` | Locks many rows in one statement. Common cause of brief mass blocks. |
| **OTP / token / counter write** | `UPDATE user SET someCounter = ? WHERE id = ?` from a single service | Hot-row pattern; usually fixable by moving the field off the canonical row. |

## Cause vs victim — which to investigate first

Lock contention is a producer-consumer game. The fingerprint with the
biggest cumulative `Lock Time` isn't always the culprit — it might be the
one that waited longest.

- **Cause-side queries to investigate first:**
  - Multi-table `INSERT … SELECT` or `UPDATE … JOIN`
  - Long-running `SELECT` (large rows examined) under `REPEATABLE-READ`
  - Large-batch writes touching many rows in one statement
- **Victim-side queries** (don't fix these directly):
  - Point UPDATE/INSERT with `Calls = 1` and multi-second `Lock Time` —
    the wait dominates; the query itself is fine
  - High-frequency point UPDATEs on a hot row — each call sees brief waits

**Investigate the cause first.** Fixing it unblocks the victims for free.

## Remediation patterns (lock contention is rarely an index problem)

Unlike `[[slow-query-triage]]` where the answer is usually a composite
index, lock contention almost never benefits from a new index — most
contending queries are already `WHERE id = ?` point lookups; the lock,
not the read, is what's slow. Real fixes are architectural:

1. **Decompose hot rows.** If a single row (e.g., `users.id`) is updated
   by N services for N different field-groups (login timestamp, refresh
   token, OTP, locale, device token), split those fields into side
   tables (`user_session`, `user_otp`, `user_locale`). Each service
   holds its own lock.
2. **Shrink transaction scope on INSERT…SELECT.** Replace
   `INSERT INTO X SELECT FROM Y JOIN Z` with a two-step prepare-then-
   write — the SELECT runs in autocommit (releases its shared locks
   immediately), and only the small INSERT is transactional.
3. **Skip locked rows for queue patterns.**
   `SELECT … FOR UPDATE SKIP LOCKED` or `UPDATE … SKIP LOCKED` lets
   concurrent workers grab different rows instead of queuing.
4. **Lower isolation on read-heavy paths.** `READ COMMITTED` drops gap
   locks. Worth checking whether the workload tolerates non-repeatable
   reads. Often yes for analytics / reporting paths.
5. **Demote durable single-field counters.** `UPDATE user SET
   last_login_at = NOW() WHERE id = ?` on every login is a candidate
   for cache-write, async-flush, or counter-table patterns. Ask whether
   the field must be durable synchronously.
6. **Consider primary-key sharding for write-hot tables.** If a single
   table's PK is the lock target for many writers, hash-partitioning or
   moving status-flags to a side table can disperse contention.

**Don't recommend an index** unless `EXPLAIN` shows a wide scan
(`type: ALL` or `index`) on a contending query. For point updates it
won't help.

## Trap: shared-row locks make `Tables:` attribution *more* accurate, not less

The `[[slow-query-triage]]` skill warns against attributing a query to
JOIN-source tables when recommending indexes — most of the time the
JOIN target gets no useful predicate.

**For lock analysis the opposite holds.** Under default `REPEATABLE-READ`,
a SELECT from a table acquires shared row locks on every read row, and
those locks survive until the transaction ends. So if a query reads from
table `A` while INSERTing into table `B`, **both tables are genuinely
part of the lock surface**. Attribute cumulative lock-time to all tables
in `Tables:` — but note shared rows (mark JOIN-source contributions with
"(+N shared)") so the reader can distinguish.

## What good output looks like

A one-screen report:

1. **Lock universe size** — "M fingerprints have cumulative lock_time > 1 ms (out of K in corpus)".
2. **Tables table** — table name, cumulative lock-time, fingerprint count, dominant pattern.
3. **Services table** — service/user, cumulative lock-time, the 1–2 fingerprints driving their total.
4. **Two prime suspects** — name one cause-side fingerprint and one victim-side fingerprint, with the SQL snippet and a one-sentence "what to fix" each.
5. **Caveats** — `Lock Time` is cumulative; the slow log only captures locks above the slow threshold; brief locks are invisible; the corpus has no blocker-waiter graph.

## Anti-patterns

- ❌ Don't run multiple probes for lock investigation — one call with
  `min_lock_time=0.001` covers the universe.
- ❌ Don't combine `min_time` and `min_lock_time` — `min_time` filters out
  fingerprints that locked heavily but finished fast.
- ❌ Don't recommend a composite index for `UPDATE t SET x WHERE id = ?`
  — it's already a point lookup; locks are the bottleneck, not access.
- ❌ Don't claim "query X is blocking query Y" from this corpus alone.
  The slow log records lock-time but not lock-target graph. Use
  `performance_schema.data_locks` or `SHOW ENGINE INNODB STATUS` for
  blocker-waiter direction.
- ❌ Don't deprioritize a fingerprint just because its `Tables:` field
  includes other tables. Under REPEATABLE-READ those *are* part of the
  lock surface; attribute the cumulative time to all of them.
- ❌ Don't propose `READ COMMITTED` or `SKIP LOCKED` without knowing the
  workload's correctness expectations. These are architectural changes
  with semantics implications.
