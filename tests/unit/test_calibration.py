import json

import pytest

from dbs_vector.services.calibration import (
    CandidateRecord,
    LabeledQuery,
    QueryRun,
    admit_records,
    audit_gate,
    candidate_floors,
    compute_metrics,
    corpus_digest,
    file_digest,
    load_query_set,
    meets_acceptance,
    percentiles,
    source_matches,
    source_resolution_errors,
    suggest_floor,
    sweep,
)

SHAPES = ["identifier", "filename", "error", "prose", "short", "long"]


def _relevant(index: int, shape: str = "prose") -> dict[str, str]:
    return {
        "query": f"relevant query {index}",
        "kind": "relevant",
        "shape": shape,
        "expected_source": f".ayder/file_{index}.md",
    }


def _absent(index: int, absent_kind: str = "hard_negative") -> dict[str, str]:
    return {
        "query": f"absent query {index}",
        "kind": "absent",
        "shape": "prose",
        "absent_kind": absent_kind,
    }


def _valid_dev_payload() -> dict[str, object]:
    relevant = [_relevant(index, SHAPES[index % len(SHAPES)]) for index in range(20)]
    absent = [
        _absent(index, "hard_negative" if index < 10 else "off_domain") for index in range(20)
    ]
    return {"name": "dev", "corpus": "documents", "queries": relevant + absent}


def _write(tmp_path, payload: dict[str, object]):
    path = tmp_path / "set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cand(source: str, similarity: float, gate: bool = False) -> CandidateRecord:
    return CandidateRecord(
        source=source,
        chunk_id=f"{source}#0",
        similarity=similarity,
        retrieved_by="fts" if gate else "vector",
        rrf_score=0.03,
        lexical_gate=gate,
    )


def _run(
    labeled: LabeledQuery,
    candidates,
    baseline=None,
    *,
    unfloored_ms: float = 1.0,
    floored_ms: float = 2.0,
) -> QueryRun:
    return QueryRun(
        labeled=labeled,
        candidates=list(candidates),
        baseline=list(baseline if baseline is not None else candidates),
        fetch_ms_unfloored=unfloored_ms,
        fetch_ms_floored=floored_ms,
        eligible_tokens=[],
    )


def _mixed_runs() -> list[QueryRun]:
    runs = []
    for index in range(20):
        query = LabeledQuery(
            query=f"rel {index}",
            kind="relevant",
            shape="prose",
            expected_source=f".ayder/r{index}.md",
        )
        runs.append(_run(query, [_cand(f"/c/.ayder/r{index}.md", 0.60 + index * 0.01)]))
    for index in range(20):
        query = LabeledQuery(
            query=f"abs {index}",
            kind="absent",
            shape="prose",
            absent_kind="off_domain" if index < 5 else "hard_negative",
        )
        runs.append(_run(query, [_cand(f"/c/.ayder/n{index}.md", 0.25 + index * 0.01)]))
    return runs


def test_load_query_set_accepts_a_conforming_dev_set(tmp_path):
    query_set = load_query_set(_write(tmp_path, _valid_dev_payload()))
    assert query_set.name == "dev"
    assert len(query_set.relevant) == 20
    assert len(query_set.absent) == 20
    assert len(query_set.hard_negatives) == 10
    assert len(query_set.off_domain) == 10


def test_load_query_set_rejects_undersized_dev_set(tmp_path):
    payload = _valid_dev_payload()
    payload["queries"] = payload["queries"][:30]  # type: ignore[index]
    with pytest.raises(ValueError, match="absent"):
        load_query_set(_write(tmp_path, payload))


def test_load_query_set_rejects_missing_relevant_query_shape(tmp_path):
    payload = _valid_dev_payload()
    for query in payload["queries"]:  # type: ignore[union-attr]
        if query["kind"] == "relevant" and query["shape"] == "error":
            query["shape"] = "prose"
    with pytest.raises(ValueError, match="error"):
        load_query_set(_write(tmp_path, payload))


def test_labeled_queries_enforce_ground_truth_fields():
    with pytest.raises(ValueError, match="expected_source"):
        LabeledQuery(query="q", kind="relevant", shape="prose")
    with pytest.raises(ValueError, match="absent_kind"):
        LabeledQuery(query="q", kind="absent", shape="prose")
    with pytest.raises(ValueError, match="expected_source"):
        LabeledQuery(
            query="q",
            kind="absent",
            shape="prose",
            absent_kind="off_domain",
            expected_source=".ayder/x.md",
        )


def test_source_matches_by_path_suffix_with_separator_guard():
    stored = "/Users/someone/repo/.ayder/claude_api_contract.md"
    assert source_matches(stored, ".ayder/claude_api_contract.md")
    assert not source_matches(stored, ".ayder/other.md")
    assert not source_matches(stored, "api_contract.md")


def test_source_resolution_errors_flag_missing_and_ambiguous_labels(tmp_path):
    query_set = load_query_set(_write(tmp_path, _valid_dev_payload()))
    corpus = {f"/repo/.ayder/file_{index}.md" for index in range(19)}
    assert source_resolution_errors(query_set, corpus) == [
        ".ayder/file_19.md: matched 0 corpus sources"
    ]
    corpus.add("/repo/.ayder/file_19.md")
    corpus.add("/other/.ayder/file_19.md")
    assert source_resolution_errors(query_set, corpus) == [
        ".ayder/file_19.md: matched 2 corpus sources"
    ]


@pytest.mark.parametrize("query", [" ", "\n\t"])
def test_load_query_set_rejects_blank_query(tmp_path, query):
    payload = _valid_dev_payload()
    payload["queries"][0]["query"] = query  # type: ignore[index]
    with pytest.raises(ValueError, match="blank"):
        load_query_set(_write(tmp_path, payload))


def test_load_query_set_rejects_normalized_duplicate_query(tmp_path):
    payload = _valid_dev_payload()
    payload["queries"][1]["query"] = "  RELEVANT   QUERY 0 "  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate query"):
        load_query_set(_write(tmp_path, payload))


def test_dev_set_enforces_source_diversity_and_reuse_limit(tmp_path):
    payload = _valid_dev_payload()
    for query in payload["queries"]:  # type: ignore[union-attr]
        if query["kind"] == "relevant":
            query["expected_source"] = ".ayder/one_file.md"
    with pytest.raises(ValueError, match="distinct expected sources"):
        load_query_set(_write(tmp_path, payload))

    payload = _valid_dev_payload()
    for index in range(4):
        payload["queries"][index]["expected_source"] = ".ayder/shared.md"  # type: ignore[index]
    with pytest.raises(ValueError, match="more than 3"):
        load_query_set(_write(tmp_path, payload))


def test_corpus_digest_captures_order_content_source_and_multiplicity():
    original = [("/c.md", "h3"), ("/a.md", "h1"), ("/b.md", "h2")]
    assert corpus_digest(original) == corpus_digest(list(reversed(original)))
    assert corpus_digest(original) != corpus_digest(
        [("/c.md", "h4"), ("/a.md", "h1"), ("/b.md", "h2")]
    )
    assert corpus_digest([("/a.md", "h1")]) != corpus_digest([("/b.md", "h1")])
    assert corpus_digest([("/a.md", "h1")]) != corpus_digest([("/a.md", "h1"), ("/a.md", "h1")])
    assert len(corpus_digest(original)) == 64


def test_file_digest_is_content_sensitive(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text("one", encoding="utf-8")
    first = file_digest(path)
    path.write_text("two", encoding="utf-8")
    assert file_digest(path) != first


def test_query_run_exposes_expected_rank_and_similarity_helpers():
    labeled = LabeledQuery(
        query="q",
        kind="relevant",
        shape="prose",
        expected_source=".ayder/b.md",
    )
    run = _run(
        labeled,
        [_cand("/r/.ayder/a.md", 0.9), _cand("/r/.ayder/b.md", 0.6)],
        baseline=[_cand("/r/.ayder/a.md", 0.9)],
    )
    assert run.rank_of_expected(run.candidates) == 2
    assert run.rank_of_expected(run.baseline) is None
    assert run.best_expected_similarity() == 0.6
    assert run.top_similarity() == 0.9


def test_percentiles_uses_nearest_rank_and_rejects_empty():
    distribution = percentiles([0.1, 0.2, 0.3, 0.4, 0.5])
    assert distribution.n == 5
    assert distribution.minimum == 0.1
    assert distribution.p50 == 0.3
    assert distribution.maximum == 0.5
    with pytest.raises(ValueError, match="empty"):
        percentiles([])


def test_admit_records_uses_both_channels_and_preserves_order():
    high = _cand("/a.md", 0.9)
    gated = _cand("/b.md", 0.01, gate=True)
    low = _cand("/c.md", 0.02)
    admitted, rejected = admit_records([high, gated, low], floor=0.5)
    assert admitted == [high, gated]
    assert rejected == [low]


def test_compute_metrics_scores_clean_separation():
    relevant = LabeledQuery(
        query="how does batching work",
        kind="relevant",
        shape="prose",
        expected_source=".ayder/BATCHING.md",
    )
    absent = LabeledQuery(
        query="risotto recipe",
        kind="absent",
        shape="prose",
        absent_kind="off_domain",
    )
    runs = [
        _run(
            relevant,
            [
                _cand("/r/.ayder/BATCHING.md", 0.80),
                _cand("/r/.ayder/x.md", 0.60),
            ],
        ),
        _run(absent, [_cand("/r/.ayder/x.md", 0.20)]),
    ]
    metrics = compute_metrics(runs, floor=0.5, limit=5)
    assert metrics.hit_at_1 == metrics.hit_at_5 == metrics.mrr == 1.0
    assert metrics.relevant_empty_rate == 0.0
    assert metrics.absent_rejection_rate == 1.0
    assert metrics.off_domain_rejection_rate == 1.0
    assert metrics.no_answer_precision == 1.0


def test_compute_metrics_counts_relevant_empty_and_truncates_to_limit():
    relevant = LabeledQuery(
        query="q",
        kind="relevant",
        shape="short",
        expected_source=".ayder/tail.md",
    )
    below = compute_metrics(
        [_run(relevant, [_cand("/r/.ayder/tail.md", 0.30)])],
        floor=0.5,
        limit=5,
    )
    assert below.relevant_empty_rate == 1.0
    assert below.no_answer_precision == 0.0

    candidates = [_cand(f"/r/.ayder/f{index}.md", 0.9) for index in range(5)]
    candidates.append(_cand("/r/.ayder/tail.md", 0.9))
    truncated = compute_metrics(
        [_run(relevant, candidates)],
        floor=0.5,
        limit=5,
    )
    assert truncated.hit_at_5 == 0.0
    assert truncated.ranks["q"] is None


def test_compute_metrics_unfloored_and_no_empty_precision_semantics():
    absent = LabeledQuery(
        query="risotto",
        kind="absent",
        shape="short",
        absent_kind="off_domain",
    )
    baseline = compute_metrics(
        [_run(absent, [_cand("/r/.ayder/x.md", -0.4)])],
        floor=None,
        limit=5,
    )
    assert baseline.floor is None
    assert baseline.absent_rejection_rate == 0.0
    assert baseline.no_answer_precision is None


def test_unfloored_distribution_uses_selected_pool_geometry():
    relevant = LabeledQuery(
        query="q",
        kind="relevant",
        shape="prose",
        expected_source=".ayder/deep.md",
    )
    candidates = [
        _cand("/c/.ayder/a.md", 0.9),
        _cand("/c/.ayder/b.md", 0.8),
        _cand("/c/.ayder/deep.md", 0.7),
    ]
    run = _run(relevant, candidates, baseline=candidates[:2])
    baseline = compute_metrics([run], floor=None, limit=5)
    oversampled = compute_metrics(
        [run],
        floor=None,
        limit=5,
        use_oversampled_pool=True,
    )
    assert baseline.hit_at_5 == 0.0
    assert baseline.relevant_distribution is None
    assert baseline.unresolved_expected == [".ayder/deep.md"]
    assert oversampled.hit_at_5 == 1.0
    assert oversampled.relevant_distribution is not None
    assert oversampled.relevant_distribution.minimum == 0.7
    assert oversampled.latency_p50_ms == 2.0


def test_candidate_floors_cover_every_admission_state():
    floors = candidate_floors(_mixed_runs())
    assert floors == sorted(set(floors))
    assert floors[0] == -1.0
    assert any(0.44 < floor < 0.60 for floor in floors)
    assert floors[-1] > 0.79


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.1, 1.1])
def test_candidate_floors_rejects_invalid_similarities(score):
    absent = LabeledQuery(
        query="q",
        kind="absent",
        shape="short",
        absent_kind="off_domain",
    )
    with pytest.raises(ValueError, match="finite"):
        candidate_floors([_run(absent, [_cand("/x.md", score)])])


def test_sweep_suggests_floor_inside_clean_gap():
    runs = _mixed_runs()
    baseline = compute_metrics(runs, floor=None, limit=5)
    chosen = suggest_floor(
        sweep(runs, candidate_floors(runs), limit=5),
        baseline,
        runs,
    )
    assert chosen is not None
    assert 0.44 < chosen <= 0.60


def test_suggest_floor_returns_none_when_scores_overlap():
    runs = []
    for index in range(20):
        relevant = LabeledQuery(
            query=f"rel {index}",
            kind="relevant",
            shape="prose",
            expected_source=f".ayder/r{index}.md",
        )
        runs.append(_run(relevant, [_cand(f"/c/.ayder/r{index}.md", 0.50)]))
        absent = LabeledQuery(
            query=f"abs {index}",
            kind="absent",
            shape="prose",
            absent_kind="off_domain" if index < 5 else "hard_negative",
        )
        runs.append(_run(absent, [_cand(f"/c/.ayder/n{index}.md", 0.50)]))
    baseline = compute_metrics(runs, floor=None, limit=5)
    assert (
        suggest_floor(
            sweep(runs, candidate_floors(runs), limit=5),
            baseline,
            runs,
        )
        is None
    )


def test_meets_acceptance_reports_all_failures():
    runs = _mixed_runs()
    baseline = compute_metrics(runs, floor=None, limit=5)
    passed, reasons = meets_acceptance(
        compute_metrics(runs, floor=0.99, limit=5),
        baseline,
    )
    assert passed is False
    assert any("hit@5" in reason for reason in reasons)
    assert any("relevant empty rate" in reason for reason in reasons)


def test_audit_gate_separates_rescues_from_unrescued_candidates():
    rescued_query = LabeledQuery(
        query="delete_by_source",
        kind="relevant",
        shape="identifier",
        expected_source=".ayder/store.md",
    )
    missed_query = LabeledQuery(
        query="narrowboat lock",
        kind="absent",
        shape="long",
        absent_kind="off_domain",
    )
    rescued_run = _run(
        rescued_query,
        [_cand("/c/.ayder/store.md", 0.02, gate=True)],
    )
    rescued_run.eligible_tokens = ["delete_by_source"]
    audit = audit_gate(
        [rescued_run, _run(missed_query, [_cand("/c/.ayder/uv.md", 0.31)])],
        floor=0.5,
        limit=5,
    )
    assert [admission.query for admission in audit.rescued] == ["delete_by_source"]
    assert [admission.query for admission in audit.unrescued] == ["narrowboat lock"]
