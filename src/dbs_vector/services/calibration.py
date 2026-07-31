"""Pure calibration domain for similarity-floor measurement.

Retrieval stays in ``scripts/calibrate_similarity_floor.py``. Keeping the
models and arithmetic here makes calibration behavior typechecked and fast to
exercise without loading MLX or LanceDB.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from dbs_vector.services.admission import is_admitted

QueryKind = Literal["relevant", "absent"]
AbsentKind = Literal["hard_negative", "off_domain"]
QueryShape = Literal["identifier", "filename", "error", "prose", "short", "long"]
RetrievedBy = Literal["both", "vector", "fts"]

MIN_SIZES: dict[str, dict[str, int]] = {
    "dev": {"relevant": 20, "absent": 20, "hard_negative": 10},
    "eval": {"relevant": 15, "absent": 15, "hard_negative": 8},
}
REQUIRED_SHAPES: tuple[QueryShape, ...] = (
    "identifier",
    "filename",
    "error",
    "prose",
    "short",
    "long",
)


class LabeledQuery(BaseModel):
    """One query and the ground truth recorded before retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str
    kind: QueryKind
    shape: QueryShape
    expected_source: str | None = None
    absent_kind: AbsentKind | None = None
    note: str = ""

    @model_validator(mode="after")
    def _check_ground_truth(self) -> Self:
        if self.kind == "relevant":
            if not self.expected_source:
                raise ValueError(f"relevant query {self.query!r} needs an expected_source")
            if self.absent_kind is not None:
                raise ValueError(f"relevant query {self.query!r} must not set absent_kind")
        else:
            if self.absent_kind is None:
                raise ValueError(
                    f"absent query {self.query!r} needs an absent_kind (hard_negative | off_domain)"
                )
            if self.expected_source:
                raise ValueError(f"absent query {self.query!r} must not set expected_source")
        return self


class QuerySet(BaseModel):
    """A development or locked-evaluation set for one corpus."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["dev", "eval"]
    corpus: str
    queries: list[LabeledQuery]

    @property
    def relevant(self) -> list[LabeledQuery]:
        return [query for query in self.queries if query.kind == "relevant"]

    @property
    def absent(self) -> list[LabeledQuery]:
        return [query for query in self.queries if query.kind == "absent"]

    @property
    def hard_negatives(self) -> list[LabeledQuery]:
        return [query for query in self.absent if query.absent_kind == "hard_negative"]

    @property
    def off_domain(self) -> list[LabeledQuery]:
        return [query for query in self.absent if query.absent_kind == "off_domain"]


def load_query_set(path: Path | str) -> QuerySet:
    """Parse and validate a labeled query set's composition."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    query_set = QuerySet.model_validate(payload)

    normalized_queries: set[str] = set()
    for query in query_set.queries:
        normalized = " ".join(query.query.split()).casefold()
        if not normalized:
            raise ValueError("query text must not be blank")
        if normalized in normalized_queries:
            raise ValueError(f"duplicate query text after normalization: {query.query!r}")
        normalized_queries.add(normalized)

    minimums = MIN_SIZES[query_set.name]
    counts = {
        "relevant": len(query_set.relevant),
        "absent": len(query_set.absent),
        "hard_negative": len(query_set.hard_negatives),
    }
    for key, needed in minimums.items():
        if counts[key] < needed:
            raise ValueError(
                f"{query_set.name} set needs at least {needed} {key} queries; got {counts[key]}"
            )
    if not query_set.off_domain:
        raise ValueError(
            f"{query_set.name} set needs at least one off_domain anchor query "
            "(acceptance demands 100% rejection of them)"
        )

    present_shapes = {query.shape for query in query_set.relevant}
    missing_shapes = [shape for shape in REQUIRED_SHAPES if shape not in present_shapes]
    if missing_shapes:
        raise ValueError(
            f"{query_set.name} set is missing query shapes: {', '.join(missing_shapes)}"
        )

    if query_set.name == "dev":
        source_counts: dict[str, int] = {}
        for query in query_set.relevant:
            expected = query.expected_source or ""
            source_counts[expected] = source_counts.get(expected, 0) + 1
        if len(source_counts) < 12:
            raise ValueError(
                f"dev set needs at least 12 distinct expected sources; got {len(source_counts)}"
            )
        overused = sorted(source for source, count in source_counts.items() if count > 3)
        if overused:
            raise ValueError(
                "no expected source may label more than 3 dev queries: " + ", ".join(overused)
            )
    return query_set


def source_matches(stored_source: str, expected_source: str) -> bool:
    """Return whether a stored path ends with the labeled path suffix."""
    if not expected_source:
        return False
    stored = Path(stored_source).as_posix()
    expected = Path(expected_source).as_posix().lstrip("/")
    return stored == expected or stored.endswith("/" + expected)


def source_resolution_errors(
    query_set: QuerySet,
    corpus_sources: Iterable[str],
) -> list[str]:
    """Return labeled suffixes that do not resolve to exactly one source."""
    sources = set(corpus_sources)
    expected_sources = dict.fromkeys(query.expected_source or "" for query in query_set.relevant)
    errors: list[str] = []
    for expected in expected_sources:
        matches = sum(source_matches(source, expected) for source in sources)
        if matches != 1:
            errors.append(f"{expected}: matched {matches} corpus sources")
    return errors


class CandidateRecord(BaseModel):
    """A flattened retrieved candidate suitable for durable reports."""

    model_config = ConfigDict(extra="forbid")

    source: str
    chunk_id: str
    similarity: float
    retrieved_by: RetrievedBy
    rrf_score: float | None = None
    lexical_gate: bool


class QueryRun(BaseModel):
    """Both candidate-pool geometries produced for one labeled query."""

    model_config = ConfigDict(extra="forbid")

    labeled: LabeledQuery
    candidates: list[CandidateRecord]
    baseline: list[CandidateRecord]
    fetch_ms_unfloored: float
    fetch_ms_floored: float
    eligible_tokens: list[str]

    def rank_of_expected(self, results: Sequence[CandidateRecord]) -> int | None:
        """Return the 1-based rank of the expected source, if present."""
        expected = self.labeled.expected_source
        if not expected:
            return None
        for rank, candidate in enumerate(results, start=1):
            if source_matches(candidate.source, expected):
                return rank
        return None

    def best_expected_similarity(self) -> float | None:
        """Return the highest score among candidates from the expected source."""
        expected = self.labeled.expected_source
        if not expected:
            return None
        scores = [
            candidate.similarity
            for candidate in self.candidates
            if source_matches(candidate.source, expected)
        ]
        return max(scores, default=None)

    def top_similarity(self) -> float | None:
        """Return the highest score in the oversampled pool."""
        return max((candidate.similarity for candidate in self.candidates), default=None)


class CalibrationIdentity(BaseModel):
    """The deployment state for which a measured floor is valid."""

    model_config = ConfigDict(extra="forbid")

    engine: str
    model: str
    passage_prefix: str
    query_prefix: str
    chunker_type: str
    tuning_profile: str
    table_name: str
    table_version: int
    row_count: int
    corpus_digest: str
    nprobes: int
    lancedb_version: str
    run_commit: str
    code_commit: str
    query_set_path: str
    query_set_digest: str
    query_set_commit: str
    choice_record_path: str | None
    choice_record_digest: str | None
    choice_record_commit: str | None
    run_date: str


def corpus_digest(source_hash_pairs: Iterable[tuple[str, str]]) -> str:
    """Return a source-aware sha256 retaining row multiplicity."""
    joined = "\n".join(
        f"{source}\0{content_hash}" for source, content_hash in sorted(source_hash_pairs)
    )
    return hashlib.sha256(joined.encode()).hexdigest()


def file_digest(path: Path | str) -> str:
    """Return the sha256 of a file's exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Distribution(BaseModel):
    """A nearest-rank percentile summary."""

    model_config = ConfigDict(extra="forbid")

    n: int
    minimum: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    maximum: float


def percentiles(values: Sequence[float]) -> Distribution:
    """Compute deterministic nearest-rank percentiles."""
    if not values:
        raise ValueError("cannot summarise an empty distribution")
    ordered = sorted(values)
    count = len(ordered)

    def at(percentile: float) -> float:
        index = max(
            0,
            min(count - 1, math.ceil(percentile / 100 * count) - 1),
        )
        return ordered[index]

    return Distribution(
        n=count,
        minimum=ordered[0],
        p05=at(5),
        p25=at(25),
        p50=at(50),
        p75=at(75),
        p95=at(95),
        maximum=ordered[-1],
    )


def admit_records(
    candidates: Sequence[CandidateRecord],
    floor: float,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    """Apply the production predicate to flattened candidate records."""
    admitted: list[CandidateRecord] = []
    rejected: list[CandidateRecord] = []
    for candidate in candidates:
        target = (
            admitted
            if is_admitted(candidate.similarity, candidate.lexical_gate, floor)
            else rejected
        )
        target.append(candidate)
    return admitted, rejected


class SetMetrics(BaseModel):
    """All metrics for one query set at one floor."""

    model_config = ConfigDict(extra="forbid")

    floor: float | None
    n_relevant: int
    n_absent: int
    n_off_domain: int
    hit_at_1: float
    hit_at_5: float
    mrr: float
    ranks: dict[str, int | None]
    relevant_empty_rate: float
    absent_rejection_rate: float
    off_domain_rejection_rate: float
    no_answer_precision: float | None
    unresolved_expected: list[str]
    relevant_distribution: Distribution | None
    absent_distribution: Distribution | None
    latency_p50_ms: float
    latency_p95_ms: float


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(
    runs: Sequence[QueryRun],
    floor: float | None,
    limit: int,
    *,
    use_oversampled_pool: bool = False,
) -> SetMetrics:
    """Score a baseline, oversampled baseline, or floor-active run."""
    relevant = [run for run in runs if run.labeled.kind == "relevant"]
    absent = [run for run in runs if run.labeled.kind == "absent"]
    off_domain = [run for run in absent if run.labeled.absent_kind == "off_domain"]

    def distribution_pool(run: QueryRun) -> list[CandidateRecord]:
        if floor is None and use_oversampled_pool:
            return run.candidates
        return run.baseline if floor is None else run.candidates

    def admitted_results(run: QueryRun) -> list[CandidateRecord]:
        if floor is None:
            return list(run.candidates[:limit]) if use_oversampled_pool else list(run.baseline)
        return admit_records(run.candidates, floor)[0][:limit]

    def best_expected_score(run: QueryRun) -> float | None:
        expected = run.labeled.expected_source or ""
        scores = [
            candidate.similarity
            for candidate in distribution_pool(run)
            if source_matches(candidate.source, expected)
        ]
        return max(scores, default=None)

    ranks: dict[str, int | None] = {}
    hits_1 = 0
    hits_5 = 0
    reciprocal = 0.0
    relevant_empty = 0
    unresolved: list[str] = []
    for run in relevant:
        admitted = admitted_results(run)
        rank = run.rank_of_expected(admitted)
        ranks[run.labeled.query] = rank
        hits_1 += rank == 1
        hits_5 += rank is not None and rank <= 5
        reciprocal += 1.0 / rank if rank is not None else 0.0
        relevant_empty += not admitted
        if best_expected_score(run) is None:
            unresolved.append(run.labeled.expected_source or "")

    absent_empty = sum(not admitted_results(run) for run in absent)
    off_domain_empty = sum(not admitted_results(run) for run in off_domain)
    total_empty = absent_empty + relevant_empty

    relevant_scores = [score for run in relevant if (score := best_expected_score(run)) is not None]
    absent_scores = [
        score
        for run in absent
        if (
            score := max(
                (candidate.similarity for candidate in distribution_pool(run)),
                default=None,
            )
        )
        is not None
    ]
    latencies = [
        (
            run.fetch_ms_unfloored
            if floor is None and not use_oversampled_pool
            else run.fetch_ms_floored
        )
        for run in runs
    ]

    return SetMetrics(
        floor=floor,
        n_relevant=len(relevant),
        n_absent=len(absent),
        n_off_domain=len(off_domain),
        hit_at_1=_rate(hits_1, len(relevant)),
        hit_at_5=_rate(hits_5, len(relevant)),
        mrr=reciprocal / len(relevant) if relevant else 0.0,
        ranks=ranks,
        relevant_empty_rate=_rate(relevant_empty, len(relevant)),
        absent_rejection_rate=_rate(absent_empty, len(absent)),
        off_domain_rejection_rate=_rate(off_domain_empty, len(off_domain)),
        no_answer_precision=absent_empty / total_empty if total_empty else None,
        unresolved_expected=unresolved,
        relevant_distribution=percentiles(relevant_scores) if relevant_scores else None,
        absent_distribution=percentiles(absent_scores) if absent_scores else None,
        latency_p50_ms=percentiles(latencies).p50 if latencies else 0.0,
        latency_p95_ms=percentiles(latencies).p95 if latencies else 0.0,
    )


class Acceptance(BaseModel):
    """Locked-evaluation acceptance criteria."""

    model_config = ConfigDict(extra="forbid")

    max_relevant_empty_rate: float = 0.05
    min_absent_rejection_rate: float = 0.60
    min_off_domain_rejection_rate: float = 1.0


ACCEPTANCE = Acceptance()


def candidate_floors(runs: Sequence[QueryRun]) -> list[float]:
    """Return one floor for every distinct valid admission state."""
    observed = sorted({candidate.similarity for run in runs for candidate in run.candidates})
    if not observed:
        raise ValueError("cannot sweep an empty candidate pool")
    if any(not math.isfinite(score) or not -1.0 <= score <= 1.0 for score in observed):
        raise ValueError("candidate similarities must be finite and within [-1, 1]")

    floors = [-1.0]
    floors.extend(
        lower + (upper - lower) / 2 for lower, upper in zip(observed, observed[1:], strict=False)
    )
    if observed[-1] < 1.0:
        floors.append(math.nextafter(observed[-1], 1.0))
    return sorted(set(floors))


def sweep(
    runs: Sequence[QueryRun],
    floors: Sequence[float],
    limit: int,
) -> list[SetMetrics]:
    """Compute exact metrics at every candidate floor."""
    return [compute_metrics(runs, floor=floor, limit=limit) for floor in floors]


def meets_acceptance(
    metrics: SetMetrics,
    baseline: SetMetrics,
) -> tuple[bool, list[str]]:
    """Return whether metrics pass and a reason for every failed criterion."""
    reasons: list[str] = []
    if metrics.hit_at_5 < baseline.hit_at_5:
        reasons.append(
            f"hit@5 regressed: {metrics.hit_at_5:.3f} < baseline {baseline.hit_at_5:.3f}"
        )
    if metrics.relevant_empty_rate > ACCEPTANCE.max_relevant_empty_rate:
        reasons.append(
            f"relevant empty rate {metrics.relevant_empty_rate:.3f} > "
            f"{ACCEPTANCE.max_relevant_empty_rate:.3f}"
        )
    if metrics.absent_rejection_rate < ACCEPTANCE.min_absent_rejection_rate:
        reasons.append(
            f"absent rejection rate {metrics.absent_rejection_rate:.3f} < "
            f"{ACCEPTANCE.min_absent_rejection_rate:.3f}"
        )
    if metrics.off_domain_rejection_rate < ACCEPTANCE.min_off_domain_rejection_rate:
        reasons.append(
            f"off-domain rejection rate {metrics.off_domain_rejection_rate:.3f} < "
            f"{ACCEPTANCE.min_off_domain_rejection_rate:.3f}"
        )
    return not reasons, reasons


def suggest_floor(
    swept: Sequence[SetMetrics],
    baseline: SetMetrics,
    runs: Sequence[QueryRun],
) -> float | None:
    """Suggest the center of the widest viable admission plateau."""
    viable = [
        metrics
        for metrics in swept
        if metrics.floor is not None and meets_acceptance(metrics, baseline)[0]
    ]
    if not viable:
        return None
    observed = sorted({candidate.similarity for run in runs for candidate in run.candidates})

    def plateau_width(floor: float) -> float:
        lower = max((score for score in observed if score < floor), default=-1.0)
        upper = min((score for score in observed if score >= floor), default=1.0)
        return upper - lower

    def ranking_key(metrics: SetMetrics) -> tuple[float, float, float, float]:
        floor = metrics.floor
        if floor is None:
            raise AssertionError("viable metrics must have a floor")
        return (
            plateau_width(floor),
            metrics.absent_rejection_rate,
            metrics.hit_at_5,
            -floor,
        )

    return max(viable, key=ranking_key).floor


class GateAdmission(BaseModel):
    """One candidate decided by the lexical gate."""

    model_config = ConfigDict(extra="forbid")

    query: str
    kind: QueryKind
    source: str
    similarity: float
    retrieved_by: RetrievedBy
    eligible_tokens: list[str]


class GateAudit(BaseModel):
    """Gate-only admissions and rejections the gate did not rescue."""

    model_config = ConfigDict(extra="forbid")

    rescued: list[GateAdmission]
    unrescued: list[GateAdmission]


def audit_gate(
    runs: Sequence[QueryRun],
    floor: float,
    limit: int,
) -> GateAudit:
    """List gate-only admissions and all rejected candidates."""
    rescued: list[GateAdmission] = []
    unrescued: list[GateAdmission] = []
    for run in runs:
        admitted, rejected = admit_records(run.candidates, floor)
        for candidate in admitted[:limit]:
            if candidate.similarity < floor and candidate.lexical_gate:
                rescued.append(
                    GateAdmission(
                        query=run.labeled.query,
                        kind=run.labeled.kind,
                        source=candidate.source,
                        similarity=candidate.similarity,
                        retrieved_by=candidate.retrieved_by,
                        eligible_tokens=run.eligible_tokens,
                    )
                )
        for candidate in rejected:
            unrescued.append(
                GateAdmission(
                    query=run.labeled.query,
                    kind=run.labeled.kind,
                    source=candidate.source,
                    similarity=candidate.similarity,
                    retrieved_by=candidate.retrieved_by,
                    eligible_tokens=run.eligible_tokens,
                )
            )
    return GateAudit(rescued=rescued, unrescued=unrescued)


class CalibrationReport(BaseModel):
    """The machine-readable artifact of one calibration run."""

    model_config = ConfigDict(extra="forbid")

    identity: CalibrationIdentity
    set_name: str
    corpus: str
    limit: int
    acceptance: Acceptance
    baseline: SetMetrics
    oversampled_unfloored: SetMetrics
    swept: list[SetMetrics]
    suggested_floor: float | None
    chosen_floor: float | None
    chosen_metrics: SetMetrics | None
    accepted: bool | None
    acceptance_failures: list[str]
    gate_audit: GateAudit | None
    runs: list[QueryRun]
