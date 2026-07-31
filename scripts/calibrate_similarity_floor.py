#!/usr/bin/env python
"""Measure a deployment-local similarity floor on a real corpus.

Development mode sweeps every distinct admission state. Evaluation mode
scores exactly one human-sealed choice and consumes the locked set before the
first search.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
CONFIG_FILE = Path(os.environ.get("DBS_CONFIG_FILE", REPO_ROOT / "config.yaml"))

from dbs_vector.config import (  # noqa: E402
    _populate_singleton_from,
    load_settings,
    settings,
)

_populate_singleton_from(load_settings(str(CONFIG_FILE), validate=True))

from dbs_vector.services.admission import eligible_tokens, lexical_gate  # noqa: E402
from dbs_vector.services.bootstrap import build_dependencies, build_store  # noqa: E402
from dbs_vector.services.calibration import (  # noqa: E402
    ACCEPTANCE,
    CalibrationIdentity,
    CalibrationReport,
    CandidateRecord,
    LabeledQuery,
    QueryRun,
    SetMetrics,
    audit_gate,
    candidate_floors,
    compute_metrics,
    corpus_digest,
    file_digest,
    load_query_set,
    meets_acceptance,
    source_resolution_errors,
    suggest_floor,
    sweep,
)
from dbs_vector.services.search import FLOOR_OVERSAMPLE  # noqa: E402

CALIBRATION_CODE_PATHS = (
    "scripts/calibrate_similarity_floor.py",
    "src/dbs_vector",
    "pyproject.toml",
    "uv.lock",
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"calibration input must live inside the repository: {path}") from exc


def _require_committed_inputs(
    query_path: Path,
    choice_path: Path | None,
) -> tuple[str, str, str, str | None]:
    """Return commit identities after rejecting dirty or untracked inputs."""
    query_relative = _repo_relative(query_path)
    sealed_paths = [query_relative]
    choice_relative: str | None = None
    if choice_path is not None:
        choice_relative = _repo_relative(choice_path)
        sealed_paths.append(choice_relative)
    for path in sealed_paths:
        _git("ls-files", "--error-unmatch", "--", path)
    compared_paths = [*sealed_paths, *CALIBRATION_CODE_PATHS]
    _git("diff", "--quiet", "--", *compared_paths)
    _git("diff", "--cached", "--quiet", "--", *compared_paths)

    run_commit = _git("rev-parse", "HEAD")
    code_commit = _git("log", "-1", "--format=%H", "--", *CALIBRATION_CODE_PATHS)
    query_commit = _git("log", "-1", "--format=%H", "--", query_relative)
    if not query_commit:
        raise ValueError(f"query set has no commit: {query_relative}")
    choice_commit = (
        _git("log", "-1", "--format=%H", "--", choice_relative)
        if choice_relative is not None
        else None
    )
    if choice_relative is not None and not choice_commit:
        raise ValueError(f"choice record has no commit: {choice_relative}")
    return run_commit, code_commit, query_commit, choice_commit


def _normalized_source(source: str, roots: list[str]) -> str:
    resolved = Path(source).resolve()
    resolved_roots = [Path(root).resolve() for root in roots]
    root_names = [root.name for root in resolved_roots]
    if len(root_names) != len(set(root_names)):
        raise ValueError("configured corpus roots must have unique basenames")
    matches = [
        f"{root.name}/{resolved.relative_to(root).as_posix()}"
        for root in resolved_roots
        if resolved.is_relative_to(root)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"source {source!r} belongs to {len(matches)} configured roots; expected exactly one"
        )
    return matches[0]


def _document_source_set(engine_name: str) -> set[str]:
    engine = settings.engines[engine_name]
    store = build_store(engine_name)
    sources = store.scan(["source"]).column("source").to_pylist()
    return {_normalized_source(source, engine.paths) for source in sources}


def _require_shared_document_corpus() -> None:
    md_sources = _document_source_set("md")
    granite_sources = _document_source_set("md-granite")
    if not md_sources or md_sources != granite_sources:
        difference = sorted(md_sources ^ granite_sources)
        raise ValueError(
            "md and md-granite must expose the same non-empty normalized source "
            f"set; symmetric difference: {difference}"
        )


def _consume_eval_once(engine: str, query_digest: str, floor: float) -> None:
    """Create the spend marker before the first evaluation search."""
    durable_dir = REPO_ROOT / "docs" / "superpowers" / "calibration-reports"
    durable = list(durable_dir.glob(f"{engine}-*-{query_digest[:12]}-*.json"))
    if durable:
        raise ValueError(
            "this evaluation set already has durable evidence for this engine: "
            + ", ".join(str(path) for path in durable)
        )

    marker = REPO_ROOT / "calibration_reports" / "consumed" / f"{engine}-{query_digest}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": engine,
        "query_set_digest": query_digest,
        "floor": floor,
        "started_at": datetime.now(UTC).isoformat(),
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except FileExistsError as exc:
        raise ValueError(
            f"this evaluation set is already spent for this engine; marker: {marker}"
        ) from exc


def _records(results: list[Any], eligible: list[str]) -> list[CandidateRecord]:
    return [
        CandidateRecord(
            source=result.chunk.source,
            chunk_id=result.chunk.id,
            similarity=result.similarity,
            retrieved_by=result.retrieved_by,
            rrf_score=result.rrf_score,
            lexical_gate=lexical_gate(
                eligible,
                result.retrieved_by,
                result.chunk.text,
            ),
        )
        for result in results
    ]


def run_one(
    embedder: Any,
    store: Any,
    labeled: LabeledQuery,
    limit: int,
) -> QueryRun:
    """Fetch original and oversampled candidate-pool geometries."""
    eligible = eligible_tokens(labeled.query)
    query_vector = embedder.embed_query(labeled.query)
    large_limit = limit * FLOOR_OVERSAMPLE

    store.search(query=labeled.query, query_vector=query_vector, limit=large_limit)
    timings: dict[int, list[float]] = {limit: [], large_limit: []}
    latest: dict[int, list[Any]] = {}
    for small_first in (True, False, True, False):
        sizes = (limit, large_limit) if small_first else (large_limit, limit)
        for fetch_limit in sizes:
            started = time.perf_counter()
            latest[fetch_limit] = store.search(
                query=labeled.query,
                query_vector=query_vector,
                limit=fetch_limit,
            )
            timings[fetch_limit].append((time.perf_counter() - started) * 1000)

    return QueryRun(
        labeled=labeled,
        candidates=_records(latest[large_limit], eligible),
        baseline=_records(latest[limit], eligible),
        fetch_ms_unfloored=statistics.median(timings[limit]),
        fetch_ms_floored=statistics.median(timings[large_limit]),
        eligible_tokens=eligible,
    )


def _print_metrics(title: str, metrics: SetMetrics) -> None:
    print(f"\n--- {title} (floor={metrics.floor}) ---")
    print(f"  hit@1={metrics.hit_at_1:.3f}  hit@5={metrics.hit_at_5:.3f}  MRR={metrics.mrr:.3f}")
    print(
        f"  relevant empty={metrics.relevant_empty_rate:.3f}   "
        f"absent rejection={metrics.absent_rejection_rate:.3f}   "
        f"off-domain rejection={metrics.off_domain_rejection_rate:.3f}"
    )
    no_answer = (
        "n/a (nothing empty)"
        if metrics.no_answer_precision is None
        else f"{metrics.no_answer_precision:.3f}"
    )
    print(f"  no-answer precision={no_answer}")
    print(f"  fetch latency p50={metrics.latency_p50_ms:.1f}ms  p95={metrics.latency_p95_ms:.1f}ms")
    distributions = (
        ("relevant (expected-source sim)", metrics.relevant_distribution),
        ("absent (top sim)", metrics.absent_distribution),
    )
    for label, distribution in distributions:
        if distribution is not None:
            print(
                f"  {label:<32} n={distribution.n:<3} "
                f"min={distribution.minimum:+.3f} p05={distribution.p05:+.3f} "
                f"p50={distribution.p50:+.3f} p95={distribution.p95:+.3f} "
                f"max={distribution.maximum:+.3f}"
            )


def _print_per_query(runs: list[QueryRun], floor: float | None) -> None:
    print(f"\n{'=' * 100}\nPER-QUERY TOP-5\n{'=' * 100}")
    for run in runs:
        labeled = run.labeled
        tag = labeled.kind if labeled.kind == "relevant" else f"absent/{labeled.absent_kind}"
        header = f"[{tag}] {labeled.query!r}  shape={labeled.shape}"
        if labeled.kind == "relevant":
            header += f"  expected={labeled.expected_source}"
        print(f"\n{header}")
        pools = (
            ("baseline", run.baseline),
            (f"oversampled x{FLOOR_OVERSAMPLE}", run.candidates),
        )
        for pool_name, candidates in pools:
            print(f"  {pool_name} expected-rank={run.rank_of_expected(candidates)}")
            for rank, candidate in enumerate(candidates[:5], start=1):
                markers = []
                if candidate.lexical_gate:
                    markers.append("GATE")
                if floor is not None and candidate.similarity >= floor:
                    markers.append("SEM")
                short_source = "/".join(Path(candidate.source).parts[-2:])
                print(
                    f"   {rank}. {candidate.similarity:+.4f}  "
                    f"{candidate.retrieved_by:<6}  {short_source:<48} "
                    f"{','.join(markers)}"
                )


def _load_choice(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("choice record must contain a JSON object")
    return value


def _choice_mismatches(
    choice: dict[str, Any],
    *,
    engine_name: str,
    floor: float,
    corpus: str,
    code_commit: str,
) -> list[str]:
    mismatches = []
    expected_fields = {
        "engine": engine_name,
        "floor": floor,
        "corpus_digest": corpus,
        "code_commit": code_commit,
    }
    mismatches.extend(
        field for field, expected in expected_fields.items() if choice.get(field) != expected
    )
    if not str(choice.get("rationale", "")).strip():
        mismatches.append("rationale")

    dev_report_value = choice.get("dev_report_path")
    if not isinstance(dev_report_value, str):
        mismatches.append("dev_report_path")
        return mismatches
    try:
        raw_path = Path(dev_report_value)
        dev_report_path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        _repo_relative(dev_report_path)
        if file_digest(dev_report_path) != choice.get("dev_report_digest"):
            mismatches.append("dev_report_digest")
        report = json.loads(dev_report_path.read_text(encoding="utf-8"))
        if report.get("set_name") != "dev":
            mismatches.append("dev_report_set")
        identity = report.get("identity", {})
        if identity.get("corpus_digest") != corpus:
            mismatches.append("dev_report_corpus")
        if identity.get("code_commit") != code_commit:
            mismatches.append("dev_report_code")
    except (OSError, ValueError, json.JSONDecodeError):
        mismatches.append("dev_report_path")
    return mismatches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="engine name from config.yaml")
    parser.add_argument(
        "--set",
        dest="set_path",
        required=True,
        help="labeled query-set JSON",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--floor", type=float)
    parser.add_argument("--choice-record")
    parser.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.engine not in settings.engines:
        print(
            f"Unknown engine {args.engine!r}. Known: {sorted(settings.engines)}",
            file=sys.stderr,
        )
        return 2
    if args.out is not None and Path(args.out).exists():
        print(f"refusing to overwrite existing report: {args.out}", file=sys.stderr)
        return 2
    if not 1 <= args.limit <= 100:
        print(f"--limit must be within [1, 100]; got {args.limit}", file=sys.stderr)
        return 2
    if args.floor is not None and not -1.0 <= args.floor <= 1.0:
        print(f"--floor must be within [-1, 1]; got {args.floor}", file=sys.stderr)
        return 2

    try:
        query_set = load_query_set(args.set_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid query set: {exc}", file=sys.stderr)
        return 2
    query_path = Path(args.set_path)
    choice_path = Path(args.choice_record) if args.choice_record is not None else None
    if args.preflight_only and (args.floor is not None or choice_path is not None):
        print(
            "--preflight-only does not accept --floor or --choice-record",
            file=sys.stderr,
        )
        return 2
    if not args.preflight_only:
        if query_set.name == "dev" and (args.floor is not None or choice_path is not None):
            print(
                "development sets reject --floor/--choice-record; run the sweep",
                file=sys.stderr,
            )
            return 2
        if query_set.name == "eval" and (args.floor is None or choice_path is None):
            print(
                "evaluation sets require --floor and --choice-record",
                file=sys.stderr,
            )
            return 2

    engine = settings.engines[args.engine]
    try:
        store = build_store(args.engine)
        table = store.scan(["source", "content_hash"])
        if table.num_rows == 0:
            raise ValueError(f"engine {args.engine!r} table {engine.table_name!r} is empty")
        if query_set.corpus == "documents":
            _require_shared_document_corpus()
        sources = set(table.column("source").to_pylist())
        source_errors = source_resolution_errors(query_set, sources)
        if source_errors:
            raise ValueError("invalid expected_source labels: " + "; ".join(source_errors))
    except ValueError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2

    if args.preflight_only:
        print(
            f"preflight passed: {args.engine}, {len(query_set.queries)} labels, "
            f"{len(sources)} sources; no model loaded and no search executed"
        )
        return 0

    try:
        run_commit, code_commit, query_commit, choice_commit = _require_committed_inputs(
            query_path, choice_path
        )
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"unsealed calibration inputs: {exc}", file=sys.stderr)
        return 2

    deps = build_dependencies(args.engine)
    store = deps.store
    source_table = store.scan(["source", "content_hash"])
    source_values = source_table.column("source").to_pylist()
    hash_values = source_table.column("content_hash").to_pylist()
    digest = corpus_digest(
        (
            _normalized_source(source, engine.paths),
            content_hash,
        )
        for source, content_hash in zip(source_values, hash_values, strict=True)
    )

    try:
        choice = _load_choice(choice_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid choice record: {exc}", file=sys.stderr)
        return 2
    if choice is not None:
        if args.floor is None:
            raise AssertionError("evaluation validation requires a floor")
        mismatches = _choice_mismatches(
            choice,
            engine_name=args.engine,
            floor=args.floor,
            corpus=digest,
            code_commit=code_commit,
        )
        if mismatches:
            print(
                "choice record does not match evaluation run: " + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 2

    identity = CalibrationIdentity(
        engine=args.engine,
        model=engine.model,
        passage_prefix=engine.passage_prefix,
        query_prefix=engine.query_prefix,
        chunker_type=engine.chunker_type,
        tuning_profile=engine.tuning_profile,
        table_name=engine.table_name,
        table_version=store.table.version,
        row_count=source_table.num_rows,
        corpus_digest=digest,
        nprobes=settings.nprobes,
        lancedb_version=importlib.metadata.version("lancedb"),
        run_commit=run_commit,
        code_commit=code_commit,
        query_set_path=_repo_relative(query_path),
        query_set_digest=file_digest(query_path),
        query_set_commit=query_commit,
        choice_record_path=(_repo_relative(choice_path) if choice_path is not None else None),
        choice_record_digest=file_digest(choice_path) if choice_path is not None else None,
        choice_record_commit=choice_commit,
        run_date=datetime.now(UTC).date().isoformat(),
    )
    print(
        f"engine={identity.engine} model={identity.model} "
        f"profile={identity.tuning_profile} rows={identity.row_count} "
        f"version={identity.table_version}\ndigest={identity.corpus_digest}"
    )
    print(
        f"queries: {len(query_set.relevant)} relevant / "
        f"{len(query_set.absent)} absent "
        f"({len(query_set.hard_negatives)} hard negative, "
        f"{len(query_set.off_domain)} off-domain)"
    )

    if query_set.name == "eval":
        if args.floor is None:
            raise AssertionError("evaluation mode requires a floor")
        try:
            _consume_eval_once(args.engine, identity.query_set_digest, args.floor)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    pinned_version = identity.table_version
    runs = []
    for query in query_set.queries:
        runs.append(run_one(deps.embedder, store, query, args.limit))
        if store.table.version != pinned_version:
            spent = " and the evaluation set is spent" if query_set.name == "eval" else ""
            print(
                f"corpus changed during calibration; this run is invalid{spent}",
                file=sys.stderr,
            )
            return 2

    baseline = compute_metrics(runs, floor=None, limit=args.limit)
    oversampled = compute_metrics(
        runs,
        floor=None,
        limit=args.limit,
        use_oversampled_pool=True,
    )
    _print_metrics("UNFLOORED BASELINE (original pool)", baseline)
    _print_metrics("UNFLOORED, OVERSAMPLED POOL", oversampled)

    chosen = args.floor
    accepted: bool | None = None
    failures: list[str] = []
    chosen_metrics: SetMetrics | None = None
    swept: list[SetMetrics] = []
    suggested: float | None = None
    if chosen is None:
        swept = sweep(runs, candidate_floors(runs), args.limit)
        suggested = suggest_floor(swept, baseline, runs)
        print(f"\n{'=' * 100}\nFLOOR SWEEP\n{'=' * 100}")
        print(
            f"{'floor':>8} {'hit@5':>7} {'MRR':>7} {'rel-empty':>10} "
            f"{'abs-rej':>8} {'off-rej':>8} {'accept':>7}"
        )
        for metrics in swept:
            passed = meets_acceptance(metrics, baseline)[0]
            print(
                f"{metrics.floor:>8.4f} {metrics.hit_at_5:>7.3f} "
                f"{metrics.mrr:>7.3f} {metrics.relevant_empty_rate:>10.3f} "
                f"{metrics.absent_rejection_rate:>8.3f} "
                f"{metrics.off_domain_rejection_rate:>8.3f} "
                f"{'YES' if passed else '-':>7}"
            )
        if suggested is None:
            print(
                "\nNO SAFE FLOOR FOUND: no candidate clears all acceptance "
                "criteria. Leaving similarity_floor unset is the correct outcome."
            )
        else:
            print(f"\nSUGGESTED FLOOR: {suggested:.4f} (a suggestion — a human chooses)")
            _print_metrics(
                "AT SUGGESTED FLOOR",
                compute_metrics(runs, suggested, args.limit),
            )
    else:
        chosen_metrics = compute_metrics(runs, chosen, args.limit)
        accepted, failures = meets_acceptance(chosen_metrics, baseline)
        _print_metrics(f"CHOSEN FLOOR {chosen}", chosen_metrics)
        print(f"\nACCEPTANCE: {'PASS' if accepted else 'FAIL'}")
        for reason in failures:
            print(f"  - {reason}")
        if not accepted:
            print(
                "\nThis evaluation set is now SPENT. Return to development and "
                "author a fresh evaluation set before retrying."
            )

    audit_floor = chosen if chosen is not None else suggested
    audit = audit_gate(runs, audit_floor, args.limit) if audit_floor is not None else None
    if audit is not None:
        print(f"\n{'=' * 100}\nLEXICAL-GATE AUDIT at floor {audit_floor:.4f}\n{'=' * 100}")
        print(f"gate-only admissions ({len(audit.rescued)}):")
        for admission in audit.rescued:
            print(
                f"  [{admission.kind}] {admission.query!r} -> "
                f"{Path(admission.source).name} sim={admission.similarity:+.4f} "
                f"tokens={admission.eligible_tokens}"
            )
        print(f"rejections the gate did not rescue ({len(audit.unrescued)}): see the report JSON")
    _print_per_query(runs, audit_floor)

    report = CalibrationReport(
        identity=identity,
        set_name=query_set.name,
        corpus=query_set.corpus,
        limit=args.limit,
        acceptance=ACCEPTANCE,
        baseline=baseline,
        oversampled_unfloored=oversampled,
        swept=swept,
        suggested_floor=suggested,
        chosen_floor=chosen,
        chosen_metrics=chosen_metrics,
        accepted=accepted,
        acceptance_failures=failures,
        gate_audit=audit,
        runs=runs,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(
        args.out
        or REPO_ROOT
        / "calibration_reports"
        / (f"{args.engine}-{query_set.name}-{identity.query_set_digest[:12]}-{timestamp}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2))
    except FileExistsError:
        print(f"refusing to overwrite existing report: {out}", file=sys.stderr)
        return 2
    print(f"\nreport written: {out}")
    return 0 if accepted is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
