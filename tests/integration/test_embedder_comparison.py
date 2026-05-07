"""E2E comparison tests for Gemma vs Granite embedders on markdown documents.

Collects ingestion latency, search latency, and accuracy (top-1 & hit-rate@3)
for English and Turkish queries against the same fixture documents.

Opt-in via RUN_SLOW_TESTS=1 (downloads ~2 GB of models on first run).

    RUN_SLOW_TESTS=1 uv run pytest tests/integration/test_embedder_comparison.py -v
"""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest


def slow(fn):
    """Combine the registered `slow` marker with an env-gated skipif."""
    fn = pytest.mark.slow(fn)
    fn = pytest.mark.skipif(
        os.getenv("RUN_SLOW_TESTS") != "1",
        reason="Embedder comparison tests download real models; set RUN_SLOW_TESTS=1 to enable.",
    )(fn)
    return fn


def _patch_settings_singleton(monkeypatch, fixture_path: str) -> None:
    from dbs_vector import config as config_module
    from dbs_vector.services import bootstrap as bootstrap_module
    from dbs_vector.services import ingestion as ingestion_module

    fixture_settings = config_module.load_settings(fixture_path)
    monkeypatch.setattr(config_module, "settings", fixture_settings)
    monkeypatch.setattr(bootstrap_module, "settings", fixture_settings)
    monkeypatch.setattr(ingestion_module, "settings", fixture_settings)


def _build_engine_deps(
    tmp_path: Path,
    monkeypatch,
    engine_name: str,
    model_key: str,
    table_name: str,
    passage_prefix: str,
    query_prefix: str,
    chunk_max_chars: int,
    max_token_length: int = 2048,
    batch_size: int = 8,
) -> tuple:
    """Build embedder / store / chunker stack for a single engine config."""
    fixture = tmp_path / f"config_{engine_name}.yaml"
    profile_name = f"{engine_name}-profile"
    fixture.write_text(
        textwrap.dedent(f"""
        system:
          db_path: {tmp_path / f"lancedb_{engine_name}"}
          nprobes: 5
          memory_budget_gb: 96
        profiles:
          {profile_name}:
            max_token_length: {max_token_length}
            chunk_max_chars: {chunk_max_chars}
            batch_size: {batch_size}
        engines:
          {engine_name}:
            description: "test"
            model: "{model_key}"
            table_name: "{table_name}"
            mapper_type: "document"
            chunker_type: "document"
            passage_prefix: "{passage_prefix}"
            query_prefix: "{query_prefix}"
            workflow: "md_search"
            tuning_profile: "{profile_name}"
    """)
    )

    _patch_settings_singleton(monkeypatch, str(fixture))

    from dbs_vector.services.bootstrap import build_dependencies
    from dbs_vector.services.ingestion import IngestionService
    from dbs_vector.services.search import SearchService

    deps = build_dependencies(engine_name)
    ingest = IngestionService(deps.chunker, deps.embedder, deps.store, deps.workflow)
    search = SearchService(deps.embedder, deps.store)
    return ingest, search


def _make_fixture_docs(docs_dir: Path) -> None:
    """Write focused markdown fixtures with unambiguous topics."""
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "architecture.md").write_text(
        "# Unified Memory Architecture\n\n"
        "Apple Silicon uses a Unified Memory Architecture (UMA) that allows the CPU, GPU, "
        "and Neural Engine to share a single pool of high-bandwidth, low-latency memory. "
        "This eliminates the need for data copies between discrete memory pools, which is "
        "a major bottleneck in traditional PC architectures.\n\n"
        "In production RAG pipelines, UMA enables zero-copy tensor extraction from MLX "
        "to NumPy and then directly into PyArrow RecordBatches.\n"
    )

    (docs_dir / "embeddings.md").write_text(
        "# Embedding Models for Retrieval\n\n"
        "Choosing the right embedding model is critical for search quality. "
        "Dense retrieval models map text into high-dimensional vectors where semantic "
        "similarity corresponds to vector proximity.\n\n"
        "Popular families include sentence-transformers, BGE, E5, and modern general-purpose "
        "encoders such as Granite and Gemma. Each model differs in context length, language "
        "coverage, and asymmetric prefix requirements.\n"
    )

    (docs_dir / "multilingual.md").write_text(
        "# Multilingual Language Support\n\n"
        "Multilingual embedding models can map text from many languages into a shared vector "
        "space. This enables cross-lingual retrieval: a query in Turkish can retrieve relevant "
        "English documents because both languages occupy overlapping regions in the embedding space.\n\n"
        "Turkish is an agglutinative language with suffix-based morphology, so high-quality "
        "tokenization and subword coverage are essential for Turkish language support.\n"
    )

    (docs_dir / "cooking.md").write_text(
        "# Cooking Risotto\n\n"
        "Risotto is a northern Italian rice dish cooked with broth until it reaches a creamy "
        "consistency. The technique involves toasting arborio rice in butter, then adding warm "
        "stock one ladle at a time while stirring constantly.\n\n"
        "Common additions include mushrooms, saffron, or seafood.\n"
    )


def _run_queries(search, queries: list[tuple[str, str]], metrics: dict, label: str) -> None:
    """Execute queries, record latency and accuracy, and merge into metrics[label]."""
    hits_at_3 = 0
    top1_hits = 0
    total_latency = 0.0
    results_log: list[dict] = []

    for query, expected_file in queries:
        t0 = time.perf_counter()
        results = search.execute_query(query, source_filter=None, limit=3, extra_filters={})
        latency = time.perf_counter() - t0
        total_latency += latency

        sources = [r.chunk.source for r in results]
        top1_hit = bool(results) and (expected_file in sources[0])
        hit3 = any(expected_file in s for s in sources)

        top1_hits += int(top1_hit)
        hits_at_3 += int(hit3)

        results_log.append(
            {
                "query": query,
                "latency_ms": round(latency * 1000, 2),
                "top1": sources[0] if sources else None,
                "top1_correct": top1_hit,
                "hit3": hit3,
                "sources": sources,
            }
        )

    n = len(queries)
    metrics.setdefault(label, {})
    metrics[label].update(
        {
            "queries": results_log,
            "top1_accuracy": round(top1_hits / n, 2) if n else 0.0,
            "hit_rate_at_3": round(hits_at_3 / n, 2) if n else 0.0,
            "avg_latency_ms": round(total_latency / n * 1000, 2) if n else 0.0,
            "total_latency_ms": round(total_latency * 1000, 2),
        }
    )


def _print_report(metrics: dict) -> None:
    """Pretty-print the collected metrics to stdout."""
    print("\n" + "=" * 70)
    print("EMBEDDER COMPARISON REPORT")
    print("=" * 70)

    for section, data in metrics.items():
        print(f"\n{section}")
        print("-" * 40)
        if "ingestion_time_ms" in data:
            print(f"  Ingestion time : {data['ingestion_time_ms']:.1f} ms")
        print(f"  Top-1 accuracy : {data['top1_accuracy']}")
        print(f"  Hit-rate@3     : {data['hit_rate_at_3']}")
        print(f"  Avg query lat. : {data['avg_latency_ms']:.1f} ms")
        print(f"  Total query lat.: {data['total_latency_ms']:.1f} ms")
        for q in data["queries"]:
            status = "✅" if q["top1_correct"] else ("🟡" if q["hit3"] else "❌")
            print(
                f"    {status} {q['query']!r:30} ({q['latency_ms']:>6.1f} ms) -> top1={q['top1']}"
            )
    print("=" * 70 + "\n")


@slow
@pytest.mark.e2e
def test_english_queries_both_embedders(tmp_path: Path, monkeypatch, capsys):
    """Both Gemma and Granite should retrieve the correct English documents.

    Collects ingestion latency, per-query latency, top-1 accuracy and hit-rate@3.
    """
    docs_dir = tmp_path / "fixture_docs"
    _make_fixture_docs(docs_dir)

    queries = [
        ("Unified Memory Architecture", "architecture.md"),
        ("embedding models comparison", "embeddings.md"),
        ("cross-lingual retrieval Turkish", "multilingual.md"),
    ]

    metrics: dict[str, dict] = {}

    # --- Gemma ---
    gemma_ingest, gemma_search = _build_engine_deps(
        tmp_path,
        monkeypatch,
        engine_name="md-gemma-test",
        model_key="gemma-bf16",
        table_name="knowledge_vault_gemma_test",
        passage_prefix="title: none | text: ",
        query_prefix="task: search result | query: ",
        chunk_max_chars=1000,
        batch_size=8,
    )
    t0 = time.perf_counter()
    gemma_ingest.ingest_directory(str(docs_dir), rebuild=True)
    metrics["Gemma (English)"] = {"ingestion_time_ms": (time.perf_counter() - t0) * 1000}
    _run_queries(gemma_search, queries, metrics, "Gemma (English)")

    # --- Granite ---
    granite_ingest, granite_search = _build_engine_deps(
        tmp_path,
        monkeypatch,
        engine_name="md-granite-test",
        model_key="granite-r2",
        table_name="knowledge_vault_granite_test",
        passage_prefix="",
        query_prefix="",
        chunk_max_chars=2000,
        max_token_length=8192,
        batch_size=8,
    )
    t0 = time.perf_counter()
    granite_ingest.ingest_directory(str(docs_dir), rebuild=True)
    metrics["Granite (English)"] = {"ingestion_time_ms": (time.perf_counter() - t0) * 1000}
    _run_queries(granite_search, queries, metrics, "Granite (English)")

    _print_report(metrics)

    # Final assertions (top-1 must be perfect for a 4-doc corpus)
    assert metrics["Gemma (English)"]["top1_accuracy"] == 1.0, "Gemma English top-1 accuracy failed"
    assert metrics["Granite (English)"]["top1_accuracy"] == 1.0, (
        "Granite English top-1 accuracy failed"
    )


@slow
@pytest.mark.e2e
def test_turkish_queries_multilingual_comparison(tmp_path: Path, monkeypatch, capsys):
    """Granite (multilingual) should retrieve English docs for Turkish queries.

    Collects ingestion latency, per-query latency, top-1 accuracy and hit-rate@3.
    """
    docs_dir = tmp_path / "fixture_docs"
    _make_fixture_docs(docs_dir)

    queries = [
        ("Apple bellek", "architecture.md"),
        ("embedding modelleri", "embeddings.md"),
        ("Türkçe dil desteği", "multilingual.md"),
        ("çok dilli arama", "multilingual.md"),
    ]

    metrics: dict[str, dict] = {}

    # --- Granite ---
    granite_ingest, granite_search = _build_engine_deps(
        tmp_path,
        monkeypatch,
        engine_name="md-granite-test",
        model_key="granite-r2",
        table_name="knowledge_vault_granite_test",
        passage_prefix="",
        query_prefix="",
        chunk_max_chars=2000,
        max_token_length=8192,
        batch_size=8,
    )
    t0 = time.perf_counter()
    granite_ingest.ingest_directory(str(docs_dir), rebuild=True)
    metrics["Granite (Turkish)"] = {"ingestion_time_ms": (time.perf_counter() - t0) * 1000}
    _run_queries(granite_search, queries, metrics, "Granite (Turkish)")

    _print_report(metrics)

    assert metrics["Granite (Turkish)"]["top1_accuracy"] == 1.0, (
        "Granite Turkish top-1 accuracy failed"
    )
