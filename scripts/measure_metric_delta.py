#!/usr/bin/env python
"""Quantify raw vector-ranking changes from L2 to cosine distance."""

from __future__ import annotations

import argparse
import os
import sys
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

import numpy as np  # noqa: E402

from dbs_vector.services.bootstrap import build_dependencies  # noqa: E402
from dbs_vector.services.calibration import load_query_set, percentiles  # noqa: E402


def _ids(table: Any, query_vector: Any, metric: str, limit: int) -> list[str]:
    operation = table.search(query_vector).metric(metric).bypass_vector_index().limit(limit)
    return [row["id"] for row in operation.to_polars().iter_rows(named=True)]


def kendall_disagreement(left: list[str], right: list[str]) -> float:
    """Return the discordant-pair fraction among shared top-k members."""
    right_items = set(right)
    shared = [item for item in left if item in right_items]
    if len(shared) < 2:
        return 0.0
    right_rank = {item: rank for rank, item in enumerate(right)}
    discordant = 0
    pairs = 0
    for index, first in enumerate(shared):
        for second in shared[index + 1 :]:
            pairs += 1
            discordant += right_rank[first] > right_rank[second]
    return discordant / pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--set", dest="set_path", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    if args.engine not in settings.engines:
        print(f"Unknown engine {args.engine!r}", file=sys.stderr)
        return 2
    if not 2 <= args.k <= 100:
        print(f"--k must be within [2, 100]; got {args.k}", file=sys.stderr)
        return 2
    try:
        queries = load_query_set(args.set_path).queries
    except (OSError, ValueError) as exc:
        print(f"invalid query set: {exc}", file=sys.stderr)
        return 2

    dependencies = build_dependencies(args.engine)
    table = dependencies.store.table
    table.checkout_latest()
    vector_indices = [
        (index.name, index.index_type, index.columns)
        for index in table.list_indices()
        if "vector" in index.columns
    ]
    print(f"live vector indices: {vector_indices or 'none'}")

    vectors = np.asarray(
        table.search().select(["vector"]).to_arrow().column("vector").to_pylist(),
        dtype=np.float32,
    )
    norms = np.linalg.norm(vectors, axis=1)
    distribution = percentiles([float(norm) for norm in norms])
    print(
        f"embedding norms: n={distribution.n} min={distribution.minimum:.4f} "
        f"p05={distribution.p05:.4f} p50={distribution.p50:.4f} "
        f"p95={distribution.p95:.4f} max={distribution.maximum:.4f}"
    )
    print("(all norms == 1.0 means the metric fix cannot change ordering)")

    identical = 0
    same_set = 0
    top1_changed = 0
    disagreements: list[float] = []
    for labeled in queries:
        query_vector = dependencies.embedder.embed_query(labeled.query)
        l2_ids = _ids(table, query_vector, "l2", args.k)
        cosine_ids = _ids(table, query_vector, "cosine", args.k)
        identical += l2_ids == cosine_ids
        same_set += set(l2_ids) == set(cosine_ids)
        disagreements.append(kendall_disagreement(l2_ids, cosine_ids))
        if l2_ids[:1] != cosine_ids[:1]:
            top1_changed += 1
            print(f"  top-1 moved: {labeled.query!r} l2={l2_ids[:1]} cosine={cosine_ids[:1]}")

    count = len(queries)
    mean_disagreement = sum(disagreements) / count if count else 0.0
    print(
        f"\nover {count} queries at k={args.k}: "
        f"identical ordering {identical}/{count}, "
        f"same membership {same_set}/{count}, "
        f"top-1 changed {top1_changed}/{count}, "
        f"mean shared-member Kendall disagreement {mean_disagreement:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
