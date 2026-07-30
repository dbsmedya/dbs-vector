# Search Scoring Baseline (Honest Similarity + Floor Mechanics) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-07-30-search-scoring-design.md` (approved). Companion spec (calibration/defaults) is explicitly out of scope.

**Goal:** Replace the RRF `Score:` on every search surface with exact cosine `similarity` + `retrieved_by` channel provenance, add an optional dual-channel admission floor (semantic floor OR all-query-terms-verbatim lexical gate), and return a `SearchResponse` envelope so empty results carry evidence.

**Architecture:** The store (`LanceDBStore.search`) annotates every candidate with exact cosine similarity, channel membership, and the raw RRF value — it applies no policy. `SearchService` owns admission policy (floor resolution, lexical gate, oversampling, truncation) and returns a `SearchResponse`. Presentation layers (MCP families, CLI, dbs-web) render one number with one meaning; the RRF value never appears in text surfaces. A new `build_search_service()` factory in `services/bootstrap.py` centralizes floor injection for all three construction sites.

**Tech Stack:** Python 3.12, pydantic v2, LanceDB 0.30.2 (`RRFReranker(K=60, return_score="all")` verified available), NumPy, Polars, FastMCP, typer, pytest.

## Global Constraints

- Breaking changes allowed (next release); **no LanceDB schema change, no reingest, no `--rebuild`** — verify no mapper `schema` edits sneak in.
- **No engine sets `similarity_floor` in `config.yaml` in this baseline** — defaults ship with the companion calibration spec. Do not edit `config.yaml`.
- RRF stays the ranking method: `RRFReranker(K=60, return_score="all")`, explicit in `_build_hybrid`. `_FLOOR_OVERSAMPLE = 3` (constant, service layer).
- `_STOPWORDS` is frozen as Lucene's classic 33-word English stop set (module constant); token length threshold is 3. Tuning either is companion-spec work.
- The RRF value (`rrf_score`) never renders in any text surface; it survives only in JSON/debug output.
- `count_matching` semantics unchanged: the SQL "Showing N of M" denominator M remains the prefilter count.
- No browse/triage changes (they take no query string). No torch, no cross-encoder, no fusion-algorithm change.
- Every task must end with `uv run poe check` green (format, lint, **mypy**, tests) AND `uv run pyright src` reporting 0 errors — `poe check` does NOT run pyright; it is a separate dev dependency. `tests/` has 13 known-intentional pyright errors — do not add new ones.
- Write shell commands plain (`uv`, `git`, `grep`); the session hook rewrites them through `rtk` automatically (RTK.md) — do not hand-prefix.
- Stage commits with the explicit file lists given per task — never `git add -A` (subagent workflows can have unrelated work in the tree).
- Ruff line length 100.
- Default-path behavior change is limited to the cosine-metric bug fix; candidate pools are unchanged until a floor is configured or passed.

## File Structure

**New files:**
- `src/dbs_vector/infrastructure/storage/scoring.py` — pure scoring helpers: `cosine_similarity()`, `classify_retrieved_by()`. No I/O, no LanceDB import.
- `src/dbs_vector/services/admission.py` — pure admission-policy helpers: `_STOPWORDS`, `eligible_tokens()`, `lexical_gate()`. No I/O.
- `tests/unit/test_scoring.py`, `tests/unit/test_admission.py`, `tests/integration/test_similarity_floor_ci.py`.

**Modified files (ownership after this plan):**
- `core/models.py` — `RetrievedBy` type alias; `SearchResult`/`SqlSearchResult` carry `similarity`/`retrieved_by`/`rrf_score`; new `RejectedCandidate`, `SearchResponse`.
- `core/ports.py` — `IStoreMapper.from_polars_row` keyword-only new signature.
- `infrastructure/storage/mappers.py` — both mappers' `from_polars_row` new signature.
- `infrastructure/storage/lancedb_engine.py` — cosine metric fix, explicit RRF rerank, per-row similarity/channel annotation.
- `services/search.py` — floor policy, `SearchResponse` return, presentation helpers (`retrieved_by_label`, `admission_phrase`, `format_admission_empty`), envelope JSON, new `print_results`.
- `services/bootstrap.py` — new `build_search_service()` factory.
- `config.py` — `EngineConfig.similarity_floor`.
- `mcp/families/base.py`, `document.py`, `sql.py` — protocol + formatting + handlers + rewritten descriptions.
- `mcp/state.py`, `cli.py`, `scripts/dbs-web.py` — construction via factory; envelope consumers; new CLI flags.
- Docs: `docs/README_MCP.md`, `docs/README_PROFILES.md`, `CLAUDE.md`.

**Migration strategy:** expand → cut over → contract. Task 4 adds the new fields alongside the old ones (suite stays green); Task 5 switches every reader and deletes the old fields; Task 6 introduces the envelope; Tasks 7–8 add policy and surfaces. Every task commits green.

---

### Task 1: Scoring helpers (`scoring.py`)

**Files:**
- Create: `src/dbs_vector/infrastructure/storage/scoring.py`
- Create: `tests/unit/test_scoring.py`

**Interfaces:**
- Consumes: nothing (pure module; imports `RetrievedBy` from `core/models.py` — added in this task as a standalone alias, used by later tasks).
- Produces: `cosine_similarity(query_vector, row_vector) -> float`; `classify_retrieved_by(distance: float | None, fts_score: float | None) -> RetrievedBy`; `RetrievedBy = Literal["both", "vector", "fts"]` in `core/models.py`.

- [ ] **Step 1: Add the `RetrievedBy` alias to `core/models.py`**

At the top of `src/dbs_vector/core/models.py` (after the imports, before `class Chunk`), add:

```python
from typing import Any, Literal

# Retrieval-channel membership: which hybrid leg(s) returned a row.
# States which channel returned the row — nothing more; not evidence the
# match is semantically or lexically correct.
RetrievedBy = Literal["both", "vector", "fts"]
```

(`Any` is already imported; merge into the existing `typing` import line.)

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_scoring.py`:

```python
"""Unit tests for pure scoring helpers: exact cosine + channel provenance."""

import math

import numpy as np
import pytest

from dbs_vector.infrastructure.storage.scoring import (
    classify_retrieved_by,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_unit_vectors_give_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_give_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_give_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_scale_invariant(self):
        assert cosine_similarity([2.0, 0.0], [0.5, 0.0]) == pytest.approx(1.0)

    def test_known_angle(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(math.sqrt(2) / 2)

    def test_zero_query_norm_gives_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_row_norm_gives_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_result_clamped_to_declared_range(self):
        v = np.asarray([0.1] * 768, dtype=np.float32)
        assert -1.0 <= cosine_similarity(v, v) <= 1.0

    def test_non_finite_input_gives_zero(self):
        assert cosine_similarity([np.inf, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_similarity([np.nan, 0.0], [1.0, 0.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_accepts_python_lists_and_numpy_arrays(self):
        q = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(q, [1.0, 2.0, 3.0]) == pytest.approx(1.0)


class TestClassifyRetrievedBy:
    def test_both_legs(self):
        assert classify_retrieved_by(0.12, 3.4) == "both"

    def test_vector_only(self):
        assert classify_retrieved_by(0.12, None) == "vector"

    def test_fts_only(self):
        assert classify_retrieved_by(None, 3.4) == "fts"

    def test_neither_leg_raises(self):
        with pytest.raises(ValueError, match="neither"):
            classify_retrieved_by(None, None)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dbs_vector.infrastructure.storage.scoring'`

- [ ] **Step 4: Implement `scoring.py`**

Create `src/dbs_vector/infrastructure/storage/scoring.py`:

```python
"""Pure scoring helpers: exact cosine similarity + retrieval-channel provenance.

No LanceDB import, no I/O — unit-testable in isolation. Used by
LanceDBStore.search to annotate every returned row.
"""

import math
from typing import Any

import numpy as np
from loguru import logger

from dbs_vector.core.models import RetrievedBy


def cosine_similarity(query_vector: Any, row_vector: Any) -> float:
    """Exact cosine similarity in [-1, 1] between two vectors.

    Metric-independent and defined even for FTS-only rows (whose LanceDB
    `_distance` is null). Guards: either norm 0 -> 0.0; non-finite result
    (inf/nan inputs) -> 0.0 with a warning, because a NaN would silently
    fail every floor comparison and poison best_rejected selection.
    """
    q = np.asarray(query_vector, dtype=np.float64).ravel()
    v = np.asarray(row_vector, dtype=np.float64).ravel()
    q_norm = float(np.linalg.norm(q))
    v_norm = float(np.linalg.norm(v))
    if q_norm == 0.0 or v_norm == 0.0:
        return 0.0
    sim = float(np.dot(q, v) / (q_norm * v_norm))
    if not math.isfinite(sim):
        logger.warning("Non-finite cosine similarity computed; substituting 0.0")
        return 0.0
    return max(-1.0, min(1.0, sim))


def classify_retrieved_by(distance: float | None, fts_score: float | None) -> RetrievedBy:
    """Map the RRFReranker(return_score="all") null pattern to channel membership.

    `distance` is the row's `_distance` (vector leg), `fts_score` its `_score`
    (FTS/BM25 leg); null means that leg did not return the row.
    """
    if distance is not None and fts_score is not None:
        return "both"
    if distance is not None:
        return "vector"
    if fts_score is not None:
        return "fts"
    raise ValueError(
        "Hybrid result row carries neither _distance nor _score; "
        "cannot classify retrieval channel (programming error)."
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: all PASS

- [ ] **Step 6: Full check + commit**

```bash
uv run poe check
git add src/dbs_vector/core/models.py src/dbs_vector/infrastructure/storage/scoring.py tests/unit/test_scoring.py
git commit -m "feat(scoring): exact cosine + retrieval-channel helpers"
```

---

### Task 2: Admission helpers (`admission.py`)

**Files:**
- Create: `src/dbs_vector/services/admission.py`
- Create: `tests/unit/test_admission.py`

**Interfaces:**
- Consumes: nothing (pure module).
- Produces: `eligible_tokens(query: str) -> list[str]`; `lexical_gate(eligible: list[str], retrieved_by: str, chunk_text: str) -> bool`. Task 7's `SearchService` calls both.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_admission.py`:

```python
"""Unit tests for the admission-policy lexical gate (spec section 4)."""

from dbs_vector.services.admission import eligible_tokens, lexical_gate


class TestEligibleTokens:
    def test_identifier_with_underscores_is_one_token(self):
        assert eligible_tokens("delete_by_source") == ["delete_by_source"]

    def test_stopwords_excluded(self):
        assert eligible_tokens("the delete_by_source of it") == ["delete_by_source"]

    def test_short_tokens_excluded(self):
        # 'go', 'to', 'db' all shorter than 3 chars or stopwords
        assert eligible_tokens("go to db now") == ["now"]

    def test_all_stopword_query_yields_empty(self):
        assert eligible_tokens("to be or not to be") == []

    def test_symbols_and_short_tokens_yield_empty(self):
        assert eligible_tokens("C++") == []

    def test_tokens_are_lowercased(self):
        assert eligible_tokens("MagentoOrders JOIN") == ["magentoorders", "join"]


class TestLexicalGate:
    def test_all_terms_verbatim_fts_row_passes(self):
        assert lexical_gate(["delete_by_source"], "fts", "def delete_by_source(x): ...") is True

    def test_both_channel_passes(self):
        assert lexical_gate(["lock"], "both", "the uv.lock file") is True

    def test_vector_only_row_never_passes(self):
        assert lexical_gate(["delete_by_source"], "vector", "def delete_by_source(x)") is False

    def test_no_eligible_tokens_never_passes(self):
        # Load-bearing: without this, an all-stopword query would vacuously
        # admit every FTS candidate.
        assert lexical_gate([], "fts", "anything at all") is False

    def test_one_missing_token_fails_all_terms_rule(self):
        # 'narrowboat lock' vs a uv.lock chunk: 'lock' verbatim, 'narrowboat' absent
        assert lexical_gate(["narrowboat", "lock"], "fts", "uv.lock lockfile contents") is False

    def test_stemmed_variant_is_not_verbatim(self):
        # measured stemming false positive: 'stores' must not match 'store'
        assert lexical_gate(["stores"], "fts", "the store opens at nine") is False

    def test_word_boundary_not_substring(self):
        assert lexical_gate(["lock"], "fts", "clockwork mechanism") is False

    def test_boundary_across_punctuation_passes(self):
        assert lexical_gate(["lock"], "fts", "uv.lock") is True

    def test_match_is_case_insensitive(self):
        assert lexical_gate(["magentoorders"], "fts", "JOIN MagentoOrders ON x") is True

    def test_underscore_identifier_not_matched_inside_longer_identifier(self):
        # \b sees no boundary between word chars: embedded identifier is no match
        assert lexical_gate(["delete_by_source"], "fts", "x_delete_by_source_y") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_admission.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dbs_vector.services.admission'`

- [ ] **Step 3: Implement `admission.py`**

Create `src/dbs_vector/services/admission.py`:

```python
"""Admission-policy helpers: the lexical gate for floor-active search.

The gate protects exact identifier/error-string recall — the reason hybrid
search exists here. It is an ALL-TERMS VERBATIM match (word boundary, case-
insensitive, no stemming), not phrase equality: token order and adjacency
are not checked. It does not guarantee filename recall: FTS indexes only the
`text` column, so a path/filename query is protected only when the name also
appears in chunk text.
"""

import re

# Lucene's classic 33-word English stop set. Frozen for this baseline;
# tuning the list and the length threshold is companion-spec work driven by
# real-corpus evaluation.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
        "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
        "that", "the", "their", "then", "there", "these", "they", "this",
        "to", "was", "will", "with",
    }
)

_TOKEN_RE = re.compile(r"\w+")
_MIN_TOKEN_LEN = 3


def eligible_tokens(query: str) -> list[str]:
    """Case-folded \\w+ tokens of the query, minus stopwords and short tokens.

    `delete_by_source` is one token (\\w includes underscore).
    """
    tokens = _TOKEN_RE.findall(query.lower())
    return [t for t in tokens if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS]


def lexical_gate(eligible: list[str], retrieved_by: str, chunk_text: str) -> bool:
    """True when the FTS-channel row contains every eligible token verbatim.

    `bool(eligible)` is load-bearing: a query whose tokens are all stopwords
    or shorter than three characters must never vacuously admit candidates.
    """
    if not eligible:
        return False
    if retrieved_by not in ("fts", "both"):
        return False
    return all(
        re.search(rf"\b{re.escape(token)}\b", chunk_text, re.IGNORECASE) is not None
        for token in eligible
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_admission.py -v`
Expected: all PASS

- [ ] **Step 5: Full check + commit**

```bash
uv run poe check
git add src/dbs_vector/services/admission.py tests/unit/test_admission.py
git commit -m "feat(admission): lexical gate + frozen Lucene stopword set"
```

---

### Task 3: Config knob (`EngineConfig.similarity_floor`)

**Files:**
- Modify: `src/dbs_vector/config.py:34-70` (`EngineConfig` field block)
- Test: `tests/unit/test_config_validation.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `EngineConfig.similarity_floor: float | None` (default `None`), range-validated to [-1, 1] at load. Task 7's `build_search_service` reads it; Task 8's descriptions read it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_validation.py`:

```python
import pytest
from pydantic import ValidationError

from dbs_vector.config import EngineConfig


def _minimal_engine(**overrides):
    base = dict(
        description="d",
        model="gemma-bf16",
        mapper_type="document",
        chunker_type="document",
        table_name="t",
        workflow="w",
        tuning_profile="p",
    )
    base.update(overrides)
    return EngineConfig(**base)


class TestSimilarityFloor:
    def test_default_is_none(self):
        assert _minimal_engine().similarity_floor is None

    def test_valid_floor_accepted(self):
        assert _minimal_engine(similarity_floor=0.55).similarity_floor == 0.55

    def test_boundaries_accepted(self):
        assert _minimal_engine(similarity_floor=-1.0).similarity_floor == -1.0
        assert _minimal_engine(similarity_floor=1.0).similarity_floor == 1.0

    def test_above_range_rejected(self):
        with pytest.raises(ValidationError):
            _minimal_engine(similarity_floor=1.5)

    def test_below_range_rejected(self):
        with pytest.raises(ValidationError):
            _minimal_engine(similarity_floor=-1.5)
```

(Reuse the file's existing imports/builders if an equivalent minimal-engine helper already exists there; do not duplicate.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_validation.py -k SimilarityFloor -v`
Expected: FAIL — `ValidationError: Extra inputs are not permitted` (model has `extra="forbid"`)

- [ ] **Step 3: Add the field**

In `src/dbs_vector/config.py`, inside `EngineConfig` after the `query_prefix: str = ""` line:

```python
    # Engine-level admission floor (policy, not a model property): minimum
    # exact cosine similarity for the semantic admission channel. None (the
    # baseline default for every engine) = no floor = today's behavior.
    # Default values ship with the calibration companion spec.
    similarity_floor: float | None = Field(default=None, ge=-1.0, le=1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config_validation.py -v`
Expected: all PASS

- [ ] **Step 5: Full check + commit**

```bash
uv run poe check
git add src/dbs_vector/config.py tests/unit/test_config_validation.py
git commit -m "feat(config): per-engine similarity_floor knob (unset by default)"
```

---

### Task 4: Expand — store annotates similarity/channel/rrf (old fields kept)

**Files:**
- Modify: `src/dbs_vector/core/models.py:114-129` (`SearchResult`, `SqlSearchResult`)
- Modify: `src/dbs_vector/core/ports.py:43-54` (`IStoreMapper.from_polars_row`)
- Modify: `src/dbs_vector/infrastructure/storage/mappers.py:67-87, 168-195`
- Modify: `src/dbs_vector/infrastructure/storage/lancedb_engine.py:1-13, 227-234, 242-266, 275-313`
- Test: `tests/unit/test_mappers.py`, `tests/unit/test_lancedb_engine.py`, `tests/integration/test_count_matching_ci.py:37`, `tests/integration/test_lancedb_filter_bugs.py:46-54`, `tests/integration/test_search_table_filter_ci.py:31`

**Interfaces:**
- Consumes: `cosine_similarity`, `classify_retrieved_by`, `RetrievedBy` (Task 1).
- Produces: `SearchResult`/`SqlSearchResult` additionally carry `similarity: float = 0.0`, `retrieved_by: RetrievedBy = "vector"`, `rrf_score: float | None = None` (old fields still present and populated). `from_polars_row(row, score=None, distance=None, *, similarity=0.0, retrieved_by="vector", rrf_score=None)`. `LanceDBStore.search` passes all five per row. This is the transitional shape; Task 5 contracts it.

- [ ] **Step 1: Write the failing tests — mappers**

In `tests/unit/test_mappers.py`, add to the DocumentMapper test class (near `test_from_polars_row`, L117):

```python
    def test_from_polars_row_carries_similarity_channel_and_rrf(self, mapper, row):
        result = mapper.from_polars_row(
            row, similarity=0.73, retrieved_by="both", rrf_score=0.0164
        )
        assert result.similarity == 0.73
        assert result.retrieved_by == "both"
        assert result.rrf_score == 0.0164

    def test_from_polars_row_new_fields_default_when_omitted(self, mapper, row):
        result = mapper.from_polars_row(row, score=0.9)
        assert result.similarity == 0.0
        assert result.retrieved_by == "vector"
        assert result.rrf_score is None
```

Add the analogous two tests to `TestSqlMapper` (asserting on a `SqlSearchResult` built from the existing SQL `row` fixture). Reuse each class's existing `mapper`/`row` fixtures — if the current tests build rows inline instead of via fixtures, follow the file's existing pattern and build the row dict inline.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_mappers.py -v`
Expected: FAIL — `TypeError: ... got an unexpected keyword argument 'similarity'`

- [ ] **Step 3: Expand the models and mapper signatures**

`src/dbs_vector/core/models.py` — both result classes become (keep old fields for now):

```python
class SearchResult(BaseModel):
    """A matched chunk returned from the vector store."""

    chunk: Chunk
    score: float | None = None
    distance: float | None = None
    is_fts_match: bool = False
    similarity: float = 0.0
    retrieved_by: RetrievedBy = "vector"
    rrf_score: float | None = None


class SqlSearchResult(BaseModel):
    """A matched SQL chunk returned from the vector store."""

    chunk: SqlChunk
    score: float | None = None
    distance: float | None = None
    is_fts_match: bool = False
    similarity: float = 0.0
    retrieved_by: RetrievedBy = "vector"
    rrf_score: float | None = None
```

`src/dbs_vector/core/ports.py:43` — protocol signature becomes:

```python
    def from_polars_row(
        self,
        row: dict[str, Any],
        score: float | None = None,
        distance: float | None = None,
        *,
        similarity: float = 0.0,
        retrieved_by: RetrievedBy = "vector",
        rrf_score: float | None = None,
    ) -> Any:
        """Convert one Polars result row into a domain search-result model.

        `similarity` is exact query-to-row cosine; `retrieved_by` is the
        retrieval channel that returned the row; `rrf_score` is the fused RRF
        value (None when no fusion ran — the pure-vector fallback path).
        `score`/`distance` are the legacy RRF/L2 fields, removed in the next
        task.
        """
        ...
```

with `from dbs_vector.core.models import RetrievedBy` added to `ports.py` imports.

`src/dbs_vector/infrastructure/storage/mappers.py` — both `from_polars_row` methods take the same expanded signature and pass the new values through:

```python
    def from_polars_row(
        self,
        row: dict[str, Any],
        score: float | None = None,
        distance: float | None = None,
        *,
        similarity: float = 0.0,
        retrieved_by: RetrievedBy = "vector",
        rrf_score: float | None = None,
    ) -> Any:
        chunk = Chunk(...)  # unchanged chunk construction
        return SearchResult(
            chunk=chunk,
            score=score,
            distance=distance,
            is_fts_match=(score is None and distance is None),
            similarity=similarity,
            retrieved_by=retrieved_by,
            rrf_score=rrf_score,
        )
```

(same for `SqlMapper` with `SqlChunk`/`SqlSearchResult`; add the `RetrievedBy` import.)

- [ ] **Step 4: Run mapper tests to verify they pass**

Run: `uv run pytest tests/unit/test_mappers.py -v`
Expected: all PASS (old-field tests untouched and still green)

- [ ] **Step 5: Write the failing tests — store annotation**

In `tests/unit/test_lancedb_engine.py`:

1. Update the three mapper stubs to keyword-tolerant form and pin the new values:
   - L509 (`test_search_reads_relevance_score_for_hybrid`): replace
     `side_effect=lambda row, score, distance: (row["id"], score, distance)` with
     `side_effect=lambda row, **kw: (row["id"], kw["rrf_score"], kw["retrieved_by"], round(kw["similarity"], 4))`
     and update the expected tuple to match the frame's values (see step 6 for the wire columns).
   - L589 (`test_search_handles_fts_only_match`): same pattern; an FTS-only row (null `_distance`, non-null `_score`) must yield `retrieved_by == "fts"`.
   - L770 (`test_search_dedupes_results_by_id`): `side_effect=lambda row, **kw: row["id"]`.
2. Every **hybrid-path** search-op mock (the `mock_search = MagicMock()` blocks that today chain only `vector`/`text`/`nprobes`/`limit` back to themselves) additionally needs:

   ```python
   mock_search.metric.return_value = mock_search
   mock_search.rerank.return_value = mock_search
   ```

   `_build_hybrid` now calls both; without these, the chain derails onto an unconfigured child mock and `to_polars()` returns a MagicMock instead of the frame. (Vector-path mocks already chain `.metric` — `_build_vector` has always called it.)
3. Every fake result DataFrame handed to the mocked `to_polars()` (the frames at L325, 363, 395, 428, 463, 541, 582, 609, 636, 665, 689, 725, 765, 791, 841) gains a `"vector"` column, e.g. `"vector": [[1.0, 0.0, 0.0]] * n_rows` (the store now raises on a missing vector column), and every **hybrid** frame must become faithful to `return_score="all"`: a finite `_relevance_score` on every row plus a non-null `_distance` and/or `_score` per row — `classify_retrieved_by(None, None)` raises, and a hybrid row without `_relevance_score` now raises too (see step 7). Concretely:
   - L495 frame (only `_relevance_score`) gains `"_distance": [0.12], "_score": [None]`.
   - L582 frame's premise changes: the docstring becomes "null `_distance` + non-null `_score` = FTS-only" and the frame becomes `"_distance": [0.9, None], "_score": [None, 2.1], "_relevance_score": [0.9, 0.8]` (row 0 → `vector`, row 1 → `fts`).
   - L765 dedupe frame gains `"_distance": [0.1, 0.2, 0.3]`.
   - Vector-fallback frames keep bare `_distance` (no `_relevance_score`, no `_score`) — that path never classifies.
4. Add new tests (in the search test class):

```python
    def test_search_annotates_exact_cosine_similarity(self, store_with_hybrid_frame):
        """Query [1,0,0] against a row vector [1,1,0] -> cos 0.7071."""
        # build frame with vector=[[1.0, 1.0, 0.0]]; query_vector=[1,0,0]
        # stub mapper: lambda row, **kw: kw["similarity"]
        results = store.search("q", np.array([1.0, 0.0, 0.0], dtype=np.float32), limit=5)
        assert results[0] == pytest.approx(0.7071, abs=1e-4)

    def test_search_raises_on_missing_vector_column(self, ...):
        """A result frame without 'vector' is a programming error."""
        # frame WITHOUT the vector column
        with pytest.raises(ValueError, match="vector"):
            store.search("q", query_vector, limit=5)

    def test_vector_fallback_rows_are_vector_channel_with_no_rrf(self, ...):
        """_hybrid_ok False -> retrieved_by='vector', rrf_score=None."""
        # force store._hybrid_ok = False; stub mapper returns (kw["retrieved_by"], kw["rrf_score"])
        assert results == [("vector", None)]

    def test_hybrid_builder_sets_cosine_metric_and_explicit_rrf_rerank(self, ...):
        """_build_hybrid must call .metric('cosine') and .rerank(_RRF_RERANKER)."""
        # follow the file's existing query-op mock pattern; extend the op mock
        # so .metric/.rerank return the op (as .nprobes/.limit already do), then:
        op_mock.metric.assert_called_once_with("cosine")
        from dbs_vector.infrastructure.storage.lancedb_engine import _RRF_RERANKER
        op_mock.rerank.assert_called_once_with(_RRF_RERANKER)

    def test_hybrid_row_without_relevance_score_raises(self, ...):
        """A successful hybrid path must never yield rrf_score=None — None is
        reserved for the pure-vector fallback (no fusion ran)."""
        # hybrid frame with vector + _distance columns but NO _relevance_score
        with pytest.raises(ValueError, match="_relevance_score"):
            store.search("q", query_vector, limit=5)
```

Write these against the file's existing fixture/mock conventions (mock `lancedb.connect`, mock table, `search_op.to_polars.return_value = pl.DataFrame({...})`) — the four sketches above show the assertions and frame shapes; flesh each out to a full test exactly like its neighbors in the file.

- [ ] **Step 6: Run to verify failure**

Run: `uv run pytest tests/unit/test_lancedb_engine.py -v`
Expected: new tests FAIL (no `.metric` call, no `similarity` kwarg); updated stubs FAIL with `KeyError: 'rrf_score'`

- [ ] **Step 7: Implement the store changes**

In `src/dbs_vector/infrastructure/storage/lancedb_engine.py`:

1. Imports (top of file):

```python
from lancedb.rerankers import RRFReranker  # type: ignore[import-untyped]

from dbs_vector.core.models import RetrievedBy
from dbs_vector.infrastructure.storage.scoring import classify_retrieved_by, cosine_similarity

# Explicit reranker: keeps today's RRF(K=60) ordering while retaining the
# per-leg _score/_distance columns needed for retrieved_by provenance.
_RRF_RERANKER = RRFReranker(K=60, return_score="all")
```

2. `_build_hybrid` (L227) gains the metric fix and explicit rerank:

```python
        def _build_hybrid() -> Any:
            op = self.table.search(query_type="hybrid").vector(query_vector).text(query)
            op = _apply_filters(op)
            if bypass_index:
                op = op.bypass_vector_index()
            else:
                op = op.nprobes(self.nprobes)
            # Bug fix: without an explicit metric, tables without an IVF index
            # run the vector leg under L2; nothing guarantees unit-norm
            # embeddings, so ordering could differ from cosine.
            op = op.metric("cosine")
            return op.limit(fetch_limit).rerank(_RRF_RERANKER)
```

3. Track which path produced the frame (L242-266): set `used_fallback = True` in the `if self._hybrid_ok is False or oversized:` branch and in the `except` fallback branch; `used_fallback = False` on hybrid success.

4. Replace the row-mapping block (L307-311) inside the loop:

```python
            row_vector = row.get("vector")
            if row_vector is None:
                raise ValueError(
                    "Search result row is missing the 'vector' column; search "
                    "applies no projection, so this is a programming error."
                )
            sim = cosine_similarity(query_vector, row_vector)
            if used_fallback:
                retrieved_by: RetrievedBy = "vector"
                rrf_score = None
            else:
                rel = row.get("_relevance_score")
                if not isinstance(rel, float) or not math.isfinite(rel):
                    raise ValueError(
                        "Hybrid result row has no finite _relevance_score; the "
                        "explicit RRF rerank guarantees one (programming error). "
                        "rrf_score=None is reserved for the vector-fallback path."
                    )
                rrf_score = rel
                dist = row.get("_distance")
                fts = row.get("_score")
                retrieved_by = classify_retrieved_by(
                    float(dist) if isinstance(dist, float) else None,
                    float(fts) if isinstance(fts, float) else None,
                )
            rel_legacy = row.get("_relevance_score")
            dist_legacy = row.get("_distance")
            mapped_results.append(
                self.mapper.from_polars_row(
                    row,
                    score=float(rel_legacy) if isinstance(rel_legacy, float) else None,
                    distance=float(dist_legacy) if isinstance(dist_legacy, float) else None,
                    similarity=sim,
                    retrieved_by=retrieved_by,
                    rrf_score=rrf_score,
                )
            )
```

The trailing `return mapped_results[:limit]` stays: `limit` is the fetch limit; final truncation to the caller's limit moves to the service in Task 7.

5. Update the three hand-rolled integration-test mappers to keyword-tolerant form (final shape, so Task 5 does not touch them again):
   - `tests/integration/test_count_matching_ci.py:37` and `tests/integration/test_search_table_filter_ci.py:31`: `def from_polars_row(self, row, *args, **kwargs): return {"id": row["id"], "tables": row.get("tables")}`
   - `tests/integration/test_lancedb_filter_bugs.py:46-54`: same `*args, **kwargs` signature; drop the dead `"score"`/`"distance"` dict keys (no consumer reads them).

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest tests/unit/test_lancedb_engine.py tests/unit/test_mappers.py tests/integration/test_count_matching_ci.py tests/integration/test_lancedb_filter_bugs.py tests/integration/test_search_table_filter_ci.py -v`
Expected: all PASS. Then `uv run poe check` — the whole suite must stay green (readers still consume the old fields, which are still populated).

- [ ] **Step 9: Commit**

```bash
git add src/dbs_vector/core/models.py src/dbs_vector/core/ports.py \
  src/dbs_vector/infrastructure/storage/mappers.py \
  src/dbs_vector/infrastructure/storage/lancedb_engine.py \
  tests/unit/test_mappers.py tests/unit/test_lancedb_engine.py \
  tests/integration/test_count_matching_ci.py \
  tests/integration/test_lancedb_filter_bugs.py \
  tests/integration/test_search_table_filter_ci.py
git commit -m "feat(store): annotate results with exact cosine, channel provenance, rrf_score; fix hybrid cosine metric"
```

---

### Task 5: Cut over readers to similarity/retrieved_by and delete the legacy fields

**Files:**
- Modify: `src/dbs_vector/core/models.py` (drop legacy fields; make new fields required)
- Modify: `src/dbs_vector/core/ports.py:43-54` (final keyword-only signature; also fix the L100 docstring "returns mapped SearchResult models" if it names score/distance)
- Modify: `src/dbs_vector/infrastructure/storage/mappers.py` (final signature)
- Modify: `src/dbs_vector/infrastructure/storage/lancedb_engine.py` (stop passing score/distance)
- Modify: `src/dbs_vector/services/search.py:57-102` (`results_to_json` docstring, `print_results`, new `retrieved_by_label`)
- Modify: `src/dbs_vector/mcp/families/document.py:39-51` (`_block`)
- Modify: `src/dbs_vector/mcp/families/sql.py:207-234` (`_block`)
- Modify: `scripts/dbs-web.py:93-151` (`_score_str` -> similarity, `_serialize` rows)
- Test: `tests/unit/test_search_service.py`, `tests/unit/test_document_family.py`, `tests/unit/test_sql_family.py`, `tests/unit/test_cli_json.py`, `tests/unit/test_mappers.py`, `tests/integration/test_ingestion.py:97`

**Interfaces:**
- Consumes: expanded models from Task 4.
- Produces: final model shape — `SearchResult`/`SqlSearchResult` have exactly `chunk`, `similarity: float` (required), `retrieved_by: RetrievedBy` (required), `rrf_score: float | None = None`. Final mapper signature `from_polars_row(self, row, *, similarity, retrieved_by, rrf_score)`. `retrieved_by_label(value: str) -> str` in `services/search.py` mapping `both -> "vector+fts"`, `vector -> "vector-only"`, `fts -> "fts-only"`. Result blocks render `--- Result (similarity 0.78, retrieved by: vector+fts) ---`.

- [ ] **Step 1: Contract the models and mappers**

`core/models.py` final result classes:

```python
class SearchResult(BaseModel):
    """A matched chunk returned from the vector store."""

    chunk: Chunk
    similarity: float  # exact cosine in [-1, 1], always present
    retrieved_by: RetrievedBy
    rrf_score: float | None = None  # fused RRF value; JSON/debug only, never rendered


class SqlSearchResult(BaseModel):
    """A matched SQL chunk returned from the vector store."""

    chunk: SqlChunk
    similarity: float
    retrieved_by: RetrievedBy
    rrf_score: float | None = None
```

`core/ports.py` + both mappers — final signature (drop `score`/`distance` params and the `is_fts_match` computation):

```python
    def from_polars_row(
        self,
        row: dict[str, Any],
        *,
        similarity: float,
        retrieved_by: RetrievedBy,
        rrf_score: float | None,
    ) -> Any:
```

`lancedb_engine.py`: delete the `rel_legacy`/`dist_legacy` lines and the `score=`/`distance=` arguments from the `from_polars_row` call.

- [ ] **Step 2: Add the label helper and cut over `services/search.py`**

Module-level in `services/search.py`:

```python
_RETRIEVED_BY_LABELS = {"both": "vector+fts", "vector": "vector-only", "fts": "fts-only"}


def retrieved_by_label(value: str) -> str:
    """Render channel membership for text surfaces (vector+fts / vector-only / fts-only)."""
    return _RETRIEVED_BY_LABELS.get(value, value)
```

`print_results` body — replace the L74-79 fallback chain and both log lines:

```python
        for res in results:
            sim_str = f"{res.similarity:.2f} ({retrieved_by_label(res.retrieved_by)})"
            if hasattr(res.chunk, "raw_query"):
                logger.info(
                    "[Similarity: {} | DB: {} | Calls: {} | Time: {}ms]",
                    sim_str,
                    res.chunk.source,
                    res.chunk.calls,
                    res.chunk.execution_time_ms,
                )
                snippet = res.chunk.raw_query[:100].replace("\n", " ")
            else:
                logger.info(
                    "[Similarity: {} | Source: {} | Hash: {}]",
                    sim_str,
                    res.chunk.source,
                    res.chunk.content_hash,
                )
                snippet = res.chunk.text[:100].replace("\n", " ")
            logger.info('  --> "{}..."', snippet)
```

`results_to_json` docstring: "every result carries its similarity, retrieved_by, rrf_score, source, full text, and all chunk metadata" (body unchanged this task — still a top-level array).

- [ ] **Step 3: Cut over both family `_block`s**

`mcp/families/document.py` `_block` (replacing L39-51):

```python
        def _block(res: Any) -> str:
            chunk = res.chunk
            return (
                f"--- Result (similarity {res.similarity:.2f}, "
                f"retrieved by: {retrieved_by_label(res.retrieved_by)}) ---\n"
                f"Source: {chunk.source}\n"
                f"Content:\n{chunk.text}\n"
            )
```

with `from dbs_vector.services.search import SearchService, retrieved_by_label` as the import.

`mcp/families/sql.py` `_block` first line (replacing the L208-213 fallback chain and the L219 header line):

```python
            block_parts = [
                f"--- Result (similarity {res.similarity:.2f}, "
                f"retrieved by: {retrieved_by_label(res.retrieved_by)}) ---",
                ...
```

Headers stay as they are in this task (Task 8 rewords them).

- [ ] **Step 4: Cut over `scripts/dbs-web.py`**

Replace `_score_str` (L93-95):

```python
def _similarity_str(res: object) -> str:
    sim = res.similarity  # type: ignore[attr-defined]
    label = retrieved_by_label(res.retrieved_by)  # type: ignore[attr-defined]
    return f"{sim:.3f} ({label})"
```

(import `retrieved_by_label` from `dbs_vector.services.search`). In `_serialize`, rename the `("Score / Dist", sc)` rows to `("Similarity", sc)` and add a `("Retrieved by", retrieved_by_label(res.retrieved_by))` row to both the SQL and document row lists; `sc = _similarity_str(res)`.

- [ ] **Step 5: Migrate the tests**

Mechanical, per the inventory:

- `tests/unit/test_mappers.py`: change every `from_polars_row(row, score=..., distance=...)` call to the final keyword form (e.g. `from_polars_row(row, similarity=0.95, retrieved_by="both", rrf_score=0.0125)`); replace assertions on `score`/`distance`/`is_fts_match` (L136-138, 154-156, 321-322, 340-341) with assertions on `similarity`/`retrieved_by`/`rrf_score`; delete Task 4's `test_from_polars_row_new_fields_default_when_omitted` (defaults no longer exist); `test_from_polars_row_with_none_score` becomes `test_from_polars_row_fts_row` pinning `retrieved_by="fts"`, `rrf_score` as passed.
- `tests/unit/test_search_service.py`: builders at L42-51, 165-177, 194-207, 223-232, 244-253, 265-285, 304-316, 342-356 replace `score=/distance=/is_fts_match=` kwargs with `similarity=<same number>, retrieved_by=<"both"|"vector"|"fts">, rrf_score=<value or None>`. The three fallback-pin tests are replaced:
  - `test_print_hybrid_result_shows_relevance_score` (L219-239) → `test_print_result_shows_similarity_and_channel`: build `similarity=0.78, retrieved_by="both"`, assert log contains `"Similarity: 0.78 (vector+fts)"`.
  - `test_print_fts_match_result` (L241-260) → `test_print_fts_only_result_labels_channel`: `similarity=0.05, retrieved_by="fts"`, assert `"(fts-only)"` in output and `"N/A"` NOT in output.
  - JSON tests (L302-336, 338-370): replace `payload[0]["score"]/["distance"]/["is_fts_match"]` assertions with `payload[0]["similarity"]`, `payload[0]["retrieved_by"]`, `payload[0]["rrf_score"]`.
- `tests/unit/test_document_family.py`: builder L12-19 and inline constructions L35-39, 52-56, 67-71, 114-118 use the new kwargs. `test_format_results_includes_source_and_text` asserts `"similarity 0.12"` and `"retrieved by: vector-only"`; `test_format_results_uses_score_when_distance_none` → `test_format_results_renders_both_channel_label` asserting `"vector+fts"`; `test_format_results_marks_fts_match_...` → `test_format_results_renders_fts_only_label` asserting `"fts-only"` and that the RRF value does not appear.
- `tests/unit/test_sql_family.py`: `_make_sql_result` (L22-39) and `_make_full_sql_result` (L42-66) → `similarity=0.5678, retrieved_by="vector", rrf_score=None` (keep the 0.5678 magic number so the L280 `assert "0.5678" in out` keeps working — update it to `assert "similarity 0.57" in out` since rendering is `:.2f`). The MagicMock duck-typed results at L331-332 and L369-370 set `r.similarity = 0.5; r.retrieved_by = "vector"` instead of `.distance`/`.score`.
- `tests/unit/test_cli_json.py`: `_doc_result` builder (L27-36) uses new kwargs; L63 asserts `payload[0]["similarity"] == 0.9`.
- `tests/integration/test_ingestion.py:97`: replace `assert isinstance(first_result.is_fts_match, bool)` with:

```python
    assert first_result.retrieved_by in ("both", "vector", "fts")
    assert -1.0 <= first_result.similarity <= 1.0
```

- [ ] **Step 6: Run the full suite**

Run: `uv run poe check`
Expected: green. Grep-verify the legacy names are gone from src:
`grep -rn "is_fts_match\|\.distance\b" src/ scripts/` → no hits (excluding LanceDB wire-column `_distance` strings in `lancedb_engine.py`).

- [ ] **Step 7: Commit**

```bash
git add src/dbs_vector/core/models.py src/dbs_vector/core/ports.py \
  src/dbs_vector/infrastructure/storage/mappers.py \
  src/dbs_vector/infrastructure/storage/lancedb_engine.py \
  src/dbs_vector/services/search.py \
  src/dbs_vector/mcp/families/document.py src/dbs_vector/mcp/families/sql.py \
  scripts/dbs-web.py \
  tests/unit/test_search_service.py tests/unit/test_document_family.py \
  tests/unit/test_sql_family.py tests/unit/test_cli_json.py \
  tests/unit/test_mappers.py tests/integration/test_ingestion.py
git commit -m "feat(search)!: replace score/distance/is_fts_match with similarity + retrieved_by + rrf_score"
```

---

### Task 6: `SearchResponse` envelope

**Files:**
- Modify: `src/dbs_vector/core/models.py` (add `RejectedCandidate`, `SearchResponse`)
- Modify: `src/dbs_vector/services/search.py:20-64` (`execute_query` return type, `results_to_json`, `print_results` signature)
- Modify: `src/dbs_vector/mcp/families/base.py:95-113` (Protocol signatures)
- Modify: `src/dbs_vector/mcp/families/document.py:23-95` (run_search/format_results/handler)
- Modify: `src/dbs_vector/mcp/families/sql.py:151-243, 297-360` (run_search/format_results/handler)
- Modify: `src/dbs_vector/cli.py:265-271`
- Modify: `scripts/dbs-web.py:164-181`
- Test: `tests/unit/test_search_service.py`, `tests/unit/test_document_family.py`, `tests/unit/test_sql_family.py`, `tests/unit/test_cli_json.py`, `tests/unit/test_family_registry.py:21`, `tests/integration/test_granite_engines.py:88-94, 170-177`, `tests/integration/test_ingestion.py:85-97`, `tests/integration/test_embedder_comparison.py:145-160`

**Interfaces:**
- Consumes: final result models (Task 5).
- Produces:

```python
class RejectedCandidate(BaseModel):
    """Evidence about the best candidate dropped by admission (no text snippet)."""

    similarity: float
    source: str
    retrieved_by: RetrievedBy


class SearchResponse(BaseModel):
    """Admission-filtered search outcome plus the evidence formatters need."""

    results: list[SearchResult | SqlSearchResult]
    floor: float | None = None  # effective floor used; None = no floor active
    # Required (no default): deduped candidates examined. Constructors must
    # state it — a silent 0 would fabricate the empty-result evidence.
    inspected: int
    best_rejected: RejectedCandidate | None = None
```

`SearchService.execute_query(...) -> SearchResponse`. `SearchService.results_to_json(response: SearchResponse) -> str` (envelope: `{"floor", "inspected", "best_rejected", "results"}`). `SearchService.print_results(response: SearchResponse, query: str = "") -> None`. Family protocol: `run_search(...) -> SearchResponse`, `format_results(response: SearchResponse, query: str, total_matching: int = 0) -> str`.

- [ ] **Step 1: Write the failing service tests**

In `tests/unit/test_search_service.py`:

- `test_basic_query_execution` (L38-71): change the final assertion to

```python
        response = service.execute_query(...)
        assert response.results == expected_results
        assert response.floor is None
        assert response.inspected == len(expected_results)
        assert response.best_rejected is None
```

- `test_empty_results_is_empty_array` (L299-300) →

```python
    def test_empty_response_serializes_to_envelope(self):
        service = ...
        payload = json.loads(
            service.results_to_json(SearchResponse(results=[], floor=None, inspected=0))
        )
        assert payload == {
            "floor": None,
            "inspected": 0,
            "best_rejected": None,
            "results": [],
        }
```

- The two `results_to_json` fidelity tests (L302-370): wrap the built result in `SearchResponse(results=[res], inspected=1)` and index `payload["results"][0][...]`; add `assert payload["results"][0]["rrf_score"]` fidelity.
- All `print_results` tests: call `service.print_results(SearchResponse(results=[...], inspected=len(...)), "some query")`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_search_service.py -v`
Expected: FAIL — `NameError: SearchResponse` / attribute errors.

- [ ] **Step 3: Implement models + service**

Add `RejectedCandidate` and `SearchResponse` to `core/models.py` exactly as in the Interfaces block above (after `SqlSearchResult`).

`services/search.py`:

```python
from dbs_vector.core.models import RejectedCandidate, SearchResponse  # noqa: F401 (RejectedCandidate used in Task 7)

    def execute_query(
        self,
        query: str,
        source_filter: str | None = None,
        limit: int = 5,
        extra_filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Embeds the query, fetches candidates, and wraps them in a SearchResponse."""
        logger.info("Executing query: {}", query)
        if extra_filters is None:
            extra_filters = {}
        query_vector = self.embedder.embed_query(query)
        candidates = self.vector_store.search(
            query=query,
            query_vector=query_vector,
            source_filter=source_filter,
            limit=limit,
            **extra_filters,
        )
        return SearchResponse(
            results=candidates[:limit],
            floor=None,
            inspected=len(candidates),
            best_rejected=None,
        )

    def results_to_json(self, response: SearchResponse) -> str:
        """Serialize a SearchResponse to a JSON envelope with full fidelity."""
        payload = {
            "floor": response.floor,
            "inspected": response.inspected,
            "best_rejected": (
                response.best_rejected.model_dump(mode="json")
                if response.best_rejected is not None
                else None
            ),
            "results": [res.model_dump(mode="json") for res in response.results],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def print_results(self, response: SearchResponse, query: str = "") -> None:
        results = response.results
        if not results:
            logger.info("No results found")
            return
        ...  # loop body unchanged from Task 5
```

(`query` is unused until Task 8 adds the admission-empty message — accept it now so the CLI call site changes once.)

- [ ] **Step 4: Migrate the families, CLI, and dbs-web**

`mcp/families/base.py`: `run_search(...) -> "SearchResponse"`, `format_results(self, response: "SearchResponse", query: str, total_matching: int = 0) -> str` (import `SearchResponse` from `dbs_vector.core.models`; keep docstrings).

`document.py`:

```python
    def run_search(...) -> SearchResponse:
        return service.execute_query(query, source_filter, limit, extra_filters={})

    def format_results(self, response: SearchResponse, query: str, total_matching: int = 0) -> str:
        results = response.results
        if not results:
            return f"No results found for query: '{query}'"
        header = f"Found {len(results)} results for '{query}':\n"
        ...  # _block unchanged; render_with_budget(header, (_block(r) for r in results), ..., total=len(results))
```

Handler (L84-91): `response = await asyncio.to_thread(...)`; `return family.format_results(response, query)`.

`sql.py`: `run_search` returns the response; `format_results(self, response, query, total_matching=0, include_raw=False)` with `results = response.results` at the top and all `results` references below unchanged; `_search_then_count` returns `(response, total)`; the handler passes `family.format_results(response, query, total_matching=total, include_raw=effective_include_raw)`.

`cli.py` search command (L265-271):

```python
    response = service.execute_query(
        query, source_filter=filter_source, limit=limit, extra_filters=extra_filters
    )
    if json_output:
        typer.echo(service.results_to_json(response))
    else:
        service.print_results(response, query)
```

`scripts/dbs-web.py` `_handle_search` (L180-181):

```python
    response = service.execute_query(query, limit=limit)
    return {
        "results": [_serialize(r) for r in response.results],
        "floor": response.floor,
        "inspected": response.inspected,
        "best_rejected": (
            response.best_rejected.model_dump(mode="json") if response.best_rejected else None
        ),
    }
```

- [ ] **Step 5: Migrate the remaining tests**

- `tests/unit/test_document_family.py` / `test_sql_family.py`: every `format_results([...], ...)` call wraps the list — `format_results(SearchResponse(results=[...], inspected=N), ...)`; every mocked `execute_query`/`run_search` `.return_value` becomes a `SearchResponse` (the `assert_called_once_with` mock-argument assertions at test_sql_family L182-249 are unchanged — only return values change, e.g. L489/507/525/554/562/592/611).
- `tests/unit/test_cli_json.py`: mock `results_to_json.return_value` becomes an envelope dict dump; assertions index `payload["results"][0]["chunk"]["source"]` etc.
- `tests/unit/test_family_registry.py:21`: stub signature becomes `def format_results(self, response: Any, query: str, total_matching: int = 0) -> str`.
- `tests/unit/test_cli_min_time.py`: the `execute_query.return_value = []` stubs (L45/64) keep working (the CLI never attribute-accesses the response itself — it hands it to the also-mocked `print_results`); verify, and if the run proves otherwise, return `SearchResponse(results=[], inspected=0)` instead.
- Integration unwraps: `test_granite_engines.py` L89-93/L171-176, `test_ingestion.py` L85-91, `test_embedder_comparison.py` L145-160 — `results = search.execute_query(...).results`.

- [ ] **Step 6: Run the full suite + commit**

Run: `uv run poe check` — green (MLX-dependent integration tests run per their existing markers).

```bash
git add src/dbs_vector/core/models.py src/dbs_vector/services/search.py \
  src/dbs_vector/mcp/families/base.py src/dbs_vector/mcp/families/document.py \
  src/dbs_vector/mcp/families/sql.py src/dbs_vector/cli.py scripts/dbs-web.py \
  tests/unit/test_search_service.py tests/unit/test_document_family.py \
  tests/unit/test_sql_family.py tests/unit/test_cli_json.py \
  tests/unit/test_cli_min_time.py tests/unit/test_family_registry.py \
  tests/integration/test_granite_engines.py tests/integration/test_ingestion.py \
  tests/integration/test_embedder_comparison.py
git commit -m "feat(search)!: execute_query returns a SearchResponse envelope"
```

---

### Task 7: Floor policy in `SearchService` + `build_search_service` factory

**Files:**
- Modify: `src/dbs_vector/services/search.py` (ctor, `execute_query` policy)
- Modify: `src/dbs_vector/services/bootstrap.py` (new factory)
- Modify: `src/dbs_vector/mcp/state.py:11-18`
- Modify: `src/dbs_vector/cli.py:258-259` (+ a `_build_search_service` wrapper next to `_build_store`)
- Modify: `scripts/dbs-web.py:176-180`
- Test: `tests/unit/test_search_service.py` (new policy class), `tests/unit/test_bootstrap.py` (factory)

**Interfaces:**
- Consumes: `eligible_tokens`/`lexical_gate` (Task 2), `similarity_floor` config (Task 3), `SearchResponse`/`RejectedCandidate` (Task 6).
- Produces:

```python
class SearchService:
    def __init__(self, embedder: IEmbedder, vector_store: IVectorStore,
                 similarity_floor: float | None = None) -> None: ...

    def execute_query(self, query: str, source_filter: str | None = None, limit: int = 5,
                      extra_filters: dict[str, Any] | None = None,
                      min_similarity: float | None = None,
                      disable_similarity_floor: bool = False) -> SearchResponse: ...
```

`build_search_service(engine_name: str, deps: EngineDeps | None = None) -> SearchService` in `services/bootstrap.py`. Every `SearchService(...)` construction site outside tests goes through it.

- [ ] **Step 1: Write the failing policy tests**

Add to `tests/unit/test_search_service.py`:

```python
import numpy as np
import pytest

from dbs_vector.core.models import Chunk, SearchResult


def _floor_result(sim, rb="vector", text="body text", source="doc.md"):
    return SearchResult(
        chunk=Chunk(id="c1", text=text, source=source, content_hash="h1"),
        similarity=sim,
        retrieved_by=rb,
        rrf_score=None,
    )


class TestFloorPolicy:
    def _service(self, results, floor=None):
        embedder = MagicMock()
        embedder.embed_query.return_value = np.zeros(4, dtype=np.float32)
        store = MagicMock()
        store.search.return_value = results
        return SearchService(embedder, store, similarity_floor=floor), store

    def test_no_floor_returns_everything_unchanged(self):
        svc, store = self._service([_floor_result(0.1), _floor_result(-0.5)])
        resp = svc.execute_query("q", limit=5)
        assert resp.floor is None
        assert len(resp.results) == 2
        assert resp.inspected == 2
        assert resp.best_rejected is None
        assert store.search.call_args.kwargs["limit"] == 5

    def test_engine_floor_oversamples_and_filters(self):
        svc, store = self._service([_floor_result(0.9), _floor_result(0.2)], floor=0.5)
        resp = svc.execute_query("q", limit=5)
        assert store.search.call_args.kwargs["limit"] == 15  # limit * _FLOOR_OVERSAMPLE
        assert [r.similarity for r in resp.results] == [0.9]
        assert resp.floor == 0.5
        assert resp.inspected == 2
        assert resp.best_rejected is not None
        assert resp.best_rejected.similarity == 0.2
        assert resp.best_rejected.source == "doc.md"

    def test_per_call_min_similarity_overrides_engine_floor(self):
        svc, _ = self._service([_floor_result(0.4)], floor=0.9)
        resp = svc.execute_query("q", min_similarity=0.3)
        assert resp.floor == 0.3
        assert len(resp.results) == 1

    def test_min_similarity_zero_is_a_real_floor_not_disable(self):
        svc, store = self._service([_floor_result(-0.2)])
        resp = svc.execute_query("q", limit=5, min_similarity=0.0)
        assert resp.floor == 0.0
        assert resp.results == []
        assert store.search.call_args.kwargs["limit"] == 15  # still oversampled

    def test_disable_flag_beats_everything_and_keeps_original_pool(self):
        svc, store = self._service([_floor_result(-0.9)], floor=0.9)
        resp = svc.execute_query(
            "q", limit=5, min_similarity=0.8, disable_similarity_floor=True
        )
        assert resp.floor is None
        assert len(resp.results) == 1
        assert store.search.call_args.kwargs["limit"] == 5  # exact-baseline pool

    def test_lexical_gate_admits_fts_verbatim_row_below_floor(self):
        row = _floor_result(0.0, rb="fts", text="def delete_by_source(): ...")
        svc, _ = self._service([row], floor=0.5)
        resp = svc.execute_query("delete_by_source")
        assert resp.results == [row]
        assert resp.best_rejected is None

    def test_lexical_gate_requires_fts_channel(self):
        row = _floor_result(0.0, rb="vector", text="def delete_by_source(): ...")
        svc, _ = self._service([row], floor=0.5)
        assert svc.execute_query("delete_by_source").results == []

    def test_truncation_happens_after_admission(self):
        rows = [
            _floor_result(0.9),
            _floor_result(0.2),
            _floor_result(0.8),
            _floor_result(0.7),
        ]
        svc, _ = self._service(rows, floor=0.5)
        resp = svc.execute_query("q", limit=2)
        assert [r.similarity for r in resp.results] == [0.9, 0.8]  # RRF order kept, gaps dropped
        assert resp.inspected == 4

    def test_out_of_range_min_similarity_raises(self):
        svc, _ = self._service([])
        with pytest.raises(ValueError, match="min_similarity"):
            svc.execute_query("q", min_similarity=1.5)

    def test_range_validation_is_unconditional_even_with_disable_flag(self):
        # The chosen rule for conflicting controls: input validation always
        # runs; disable_similarity_floor only wins the FLOOR resolution.
        # Garbage input fails loudly instead of being masked by the flag.
        svc, _ = self._service([])
        with pytest.raises(ValueError, match="min_similarity"):
            svc.execute_query("q", min_similarity=2.0, disable_similarity_floor=True)

    def test_invalid_limit_raises(self):
        # LanceDB treats a non-positive limit as "no limit" (unbounded fetch)
        # on an LLM-callable surface; floor mode also multiplies limit by 3.
        svc, _ = self._service([])
        for bad in (0, -1, 101):
            with pytest.raises(ValueError, match="limit"):
                svc.execute_query("q", limit=bad)

    def test_best_rejected_is_highest_similarity_among_all_rejected(self):
        rows = [
            _floor_result(0.1),
            _floor_result(0.4, source="close.md"),
            _floor_result(0.2),
        ]
        svc, _ = self._service(rows, floor=0.5)
        resp = svc.execute_query("q")
        assert resp.best_rejected is not None
        assert resp.best_rejected.similarity == 0.4
        assert resp.best_rejected.source == "close.md"
```

And to `tests/unit/test_bootstrap.py`, using the file's existing `mock_settings` fixture (it yields `(settings, engine_config, profile)` with **MagicMock** engines — `similarity_floor` MUST be set explicitly per test, otherwise attribute access returns another MagicMock). Extend the file's import line to include `build_search_service`:

```python
class TestBuildSearchService:
    def _deps(self):
        return EngineDeps(
            embedder=MagicMock(),
            store=MagicMock(),
            chunker=MagicMock(),
            workflow="default",
            batch_size=64,
        )

    def test_injects_engine_floor(self, mock_settings):
        _, engine_config, _ = mock_settings
        engine_config.similarity_floor = 0.4
        deps = self._deps()
        with patch(
            "dbs_vector.services.bootstrap.build_dependencies", return_value=deps
        ) as mock_build:
            svc = build_search_service("md")
        mock_build.assert_called_once_with("md")
        assert svc.similarity_floor == 0.4
        assert svc.embedder is deps.embedder
        assert svc.vector_store is deps.store

    def test_prebuilt_deps_skip_dependency_build(self, mock_settings):
        _, engine_config, _ = mock_settings
        engine_config.similarity_floor = None
        deps = self._deps()
        with patch("dbs_vector.services.bootstrap.build_dependencies") as mock_build:
            svc = build_search_service("md", deps=deps)
        mock_build.assert_not_called()
        assert svc.vector_store is deps.store
        assert svc.similarity_floor is None

    def test_unknown_engine_raises_value_error(self, mock_settings):
        with pytest.raises(ValueError, match="Unknown engine"):
            build_search_service("no-such-engine")
```

(`EngineDeps` field name is `store`; `SearchService` exposes it as `vector_store`. Patching `build_dependencies` avoids constructing a real `MLXEmbedder`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_search_service.py tests/unit/test_bootstrap.py -k "FloorPolicy or BuildSearchService" -v` (one combined `-k` — pytest applies only the last `-k` option given)
Expected: FAIL — unexpected kwarg `similarity_floor` / no `build_search_service`.

- [ ] **Step 3: Implement the service policy**

`services/search.py`:

```python
from dbs_vector.core.models import RejectedCandidate, SearchResponse
from dbs_vector.services.admission import eligible_tokens, lexical_gate

# When a floor is active, fetch this multiple of `limit` so floor-filtering
# doesn't starve the requested limit. Enlarged per-leg pools change RRF
# fusion inputs — a deliberate, spec-stated behavior change (floor-active
# paths only; measured in the companion spec).
_FLOOR_OVERSAMPLE = 3

# LLM-callable surface guard: LanceDB treats a non-positive limit as "no
# limit" (unbounded fetch before app-side truncation), and floor mode
# multiplies the fetch by _FLOOR_OVERSAMPLE.
_MAX_LIMIT = 100


class SearchService:
    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        similarity_floor: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_floor = similarity_floor

    def execute_query(
        self,
        query: str,
        source_filter: str | None = None,
        limit: int = 5,
        extra_filters: dict[str, Any] | None = None,
        min_similarity: float | None = None,
        disable_similarity_floor: bool = False,
    ) -> SearchResponse:
        """Embed, fetch hybrid-ranked candidates, apply admission policy.

        Floor precedence: disable_similarity_floor (no floor AND the original
        candidate-pool size — the exact-baseline state; min_similarity=0.0 is
        NOT equivalent) > per-call min_similarity > engine similarity_floor >
        no floor. Input validation (limit and min_similarity ranges) is
        unconditional — it runs even when disable_similarity_floor=True, so
        garbage input fails loudly instead of being masked by the flag.
        """
        if not 1 <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be within [1, {_MAX_LIMIT}]; got {limit}")
        if min_similarity is not None and not (-1.0 <= min_similarity <= 1.0):
            raise ValueError(f"min_similarity must be within [-1, 1]; got {min_similarity}")
        logger.info("Executing query: {}", query)
        if extra_filters is None:
            extra_filters = {}

        if disable_similarity_floor:
            floor: float | None = None
        elif min_similarity is not None:
            floor = min_similarity
        else:
            floor = self.similarity_floor
        fetch_limit = limit if floor is None else limit * _FLOOR_OVERSAMPLE

        query_vector = self.embedder.embed_query(query)
        candidates = self.vector_store.search(
            query=query,
            query_vector=query_vector,
            source_filter=source_filter,
            limit=fetch_limit,
            **extra_filters,
        )
        inspected = len(candidates)
        if floor is None:
            return SearchResponse(
                results=candidates[:limit], floor=None, inspected=inspected, best_rejected=None
            )

        eligible = eligible_tokens(query)
        admitted: list[Any] = []
        rejected: list[Any] = []
        for cand in candidates:
            if cand.similarity >= floor or lexical_gate(
                eligible, cand.retrieved_by, cand.chunk.text
            ):
                admitted.append(cand)
            else:
                rejected.append(cand)
        best = max(rejected, key=lambda c: c.similarity, default=None)
        best_rejected = (
            RejectedCandidate(
                similarity=best.similarity,
                source=best.chunk.source,
                retrieved_by=best.retrieved_by,
            )
            if best is not None
            else None
        )
        return SearchResponse(
            results=admitted[:limit],
            floor=floor,
            inspected=inspected,
            best_rejected=best_rejected,
        )
```

- [ ] **Step 4: Implement the factory and rewire all construction sites**

`services/bootstrap.py` (top-level import `from dbs_vector.services.search import SearchService` is cycle-free):

```python
def build_search_service(engine_name: str, deps: EngineDeps | None = None) -> SearchService:
    """SearchService with the engine's similarity_floor injected.

    The single construction path for every search surface (MCP, CLI,
    dbs-web) — a hand-wired SearchService is exactly where a floor would
    drift. Pass prebuilt `deps` to reuse cached embedder/store handles.
    """
    if engine_name not in settings.engines:
        raise ValueError(
            f"Unknown engine: '{engine_name}'. "
            f"Check {os.environ.get('DBS_CONFIG_FILE', 'config.yaml')}."
        )
    if deps is None:
        deps = build_dependencies(engine_name)
    return SearchService(
        deps.embedder,
        deps.store,
        similarity_floor=settings.engines[engine_name].similarity_floor,
    )
```

`mcp/state.py`:

```python
from dbs_vector.services.bootstrap import build_search_service

def initialize_services() -> dict[str, SearchService]:
    _services.clear()
    for engine_name in settings.engines.keys():
        logger.info("Loading engine: {}", engine_name)
        _services[engine_name] = build_search_service(engine_name)
    return _services
```

`cli.py` — add next to `_build_store`:

```python
def _build_search_service(engine_name: str) -> SearchService:
    """CLI-facing search-service builder: converts schema-mismatch to a typer exit."""
    try:
        return build_search_service(engine_name)
    except ValueError as e:
        if "Schema mismatch" in str(e):
            typer.echo(f"\n[!] Database Error: {e}", err=True)
            raise typer.Exit(code=1) from e
        raise
```

and in the `search` command replace L258-259 with `service = _build_search_service(engine_name)` (import `build_search_service` from bootstrap).

`scripts/dbs-web.py` `_handle_search` (L176-180):

```python
    from dbs_vector.services.bootstrap import build_search_service

    deps = _get_deps(engine)
    service = build_search_service(engine, deps=deps)
```

- [ ] **Step 5: Run the full suite**

Run: `uv run poe check`
Expected: green. Note: `tests/integration/test_cli.py` patches `dbs_vector.cli.SearchService` — the CLI no longer constructs `SearchService` directly, so patch `dbs_vector.cli.build_search_service` there instead (fixture at L124-130) and keep the same MagicMock behavior.

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/services/search.py src/dbs_vector/services/bootstrap.py \
  src/dbs_vector/mcp/state.py src/dbs_vector/cli.py scripts/dbs-web.py \
  tests/unit/test_search_service.py tests/unit/test_bootstrap.py \
  tests/integration/test_cli.py
git commit -m "feat(search): dual-channel admission floor + build_search_service factory"
```

---

### Task 8: Surfaces — MCP params, headers, admission-empty message, descriptions, CLI flags

**Files:**
- Modify: `src/dbs_vector/services/search.py` (add `admission_phrase`, `format_admission_empty`; wire into `print_results`)
- Modify: `src/dbs_vector/mcp/families/document.py` (handler params, header, empty case, `search_description`)
- Modify: `src/dbs_vector/mcp/families/sql.py` (handler params, headers, empty case, `search_description`)
- Modify: `src/dbs_vector/cli.py` (search command: `--min-similarity`, `--no-similarity-floor`, ValueError handling, `--json` help text)
- Test: `tests/unit/test_document_family.py`, `tests/unit/test_sql_family.py`, `tests/unit/test_search_service.py`, `tests/unit/test_browse_descriptions.py`, `tests/unit/test_dynamic_tools.py:86-87`, `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: `SearchResponse` with populated `floor`/`inspected`/`best_rejected` (Task 7).
- Produces (in `services/search.py`, imported by both families):

```python
def admission_phrase(floor: float) -> str:
    return f"similarity >= {floor:g} or all query terms verbatim"


def format_admission_empty(query: str, response: SearchResponse) -> str: ...
```

MCP search handlers gain `min_similarity: float | None = None, disable_similarity_floor: bool = False`. CLI gains `--min-similarity` / `--no-similarity-floor`.

- [ ] **Step 1: Write the failing formatter tests**

`tests/unit/test_document_family.py` additions:

```python
def _floor_response(results, floor, inspected, best=None):
    return SearchResponse(results=results, floor=floor, inspected=inspected, best_rejected=best)


class TestFloorPresentation:
    def test_no_floor_header_says_hybrid_ranked(self):
        out = DocumentFamily().format_results(
            _floor_response([_fake_doc_result()], None, 1), "q"
        )
        assert "Found 1 results for 'q' (hybrid-ranked):" in out

    def test_floor_header_carries_admission_phrase(self):
        out = DocumentFamily().format_results(
            _floor_response([_fake_doc_result()], 0.55, 3), "q"
        )
        assert "(hybrid-ranked, admission: similarity >= 0.55 or all query terms verbatim)" in out

    def test_admission_empty_leads_with_low_confidence_not_absence(self):
        best = RejectedCandidate(similarity=0.38, source="tests/x.py", retrieved_by="fts")
        out = DocumentFamily().format_results(
            _floor_response([], 0.55, 15, best), "beehive maintenance"
        )
        assert "No inspected candidate passed admission" in out
        assert "similarity >= 0.55 or all query terms verbatim" in out
        assert "'beehive maintenance'" in out
        assert "Inspected 15 hybrid-ranked candidates" in out
        assert "0.38" in out and "tests/x.py" in out and "fts-only" in out
        assert "does not establish corpus-level absence" in out

    def test_empty_with_no_candidates_keeps_current_message(self):
        out = DocumentFamily().format_results(_floor_response([], 0.55, 0), "q")
        assert out == "No results found for query: 'q'"

    def test_rrf_score_never_rendered(self):
        res = _fake_doc_result()  # builder sets rrf_score=0.0328
        out = DocumentFamily().format_results(_floor_response([res], None, 1), "q")
        assert "0.0328" not in out
```

`tests/unit/test_sql_family.py` additions (same shape): floor-active header asserts
`"Showing 1 of 4 results that matched your filters for 'q' (hybrid-ranked, admission: similarity >= 0.55 or all query terms verbatim):"`;
no-floor header asserts `"(hybrid-ranked):"` replaces `"(ranked by similarity):"` (update the existing header assertions at L274-449 accordingly); admission-empty case identical to the document family; `total_matching>0` with `inspected == 0` keeps the existing "rows matched your filters but none ranked" message; the anomaly-warning header (L325-358) keeps its WARNING text with `(hybrid-ranked)` wording.

Handler tests: extend the signature test (test_sql_family L252-269 pattern) so both families' handlers accept `min_similarity`/`disable_similarity_floor` and forward them to `execute_query`; add one test per family asserting an out-of-range `min_similarity=2.0` returns the string `"min_similarity must be within [-1, 1]; got 2.0."` without calling the service.

`tests/unit/test_search_service.py`: one test that `print_results` with an admission-empty response logs the `format_admission_empty` text.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_document_family.py tests/unit/test_sql_family.py -v`
Expected: new tests FAIL on old header/empty wording.

- [ ] **Step 3: Implement helpers + formatting**

`services/search.py` module level:

```python
def admission_phrase(floor: float) -> str:
    """The one-line admission rule, rendered identically on every surface."""
    return f"similarity >= {floor:g} or all query terms verbatim"


def format_admission_empty(query: str, response: SearchResponse) -> str:
    """Empty-because-admission-filtered message: low retrieval confidence for
    THIS attempt — never an assertion of corpus-level absence (only the
    inspected pool is known)."""
    floor = response.floor
    if floor is None:  # defensive; callers gate on floor being active
        return f"No results found for query: '{query}'"
    msg = (
        f"No inspected candidate passed admission ({admission_phrase(floor)}) "
        f"for '{query}'. Inspected {response.inspected} hybrid-ranked candidates"
    )
    best = response.best_rejected
    if best is not None:
        msg += (
            f"; best was similarity {best.similarity:.2f} "
            f"({best.source}, {retrieved_by_label(best.retrieved_by)})"
        )
    msg += (
        ". Retrieval confidence for this attempt is low; this does not "
        "establish corpus-level absence. Retry with different terms or a "
        "lower min_similarity if you expected a match."
    )
    return msg
```

`print_results` empty branch:

```python
        if not results:
            if response.floor is not None and response.inspected > 0:
                logger.info("{}", format_admission_empty(query, response))
            else:
                logger.info("No results found")
            return
```

`document.py` `format_results`:

```python
        results = response.results
        if not results:
            if response.floor is not None and response.inspected > 0:
                return format_admission_empty(query, response)
            return f"No results found for query: '{query}'"
        suffix = "" if response.floor is None else f", admission: {admission_phrase(response.floor)}"
        header = f"Found {len(results)} results for '{query}' (hybrid-ranked{suffix}):\n"
```

`sql.py` `format_results` — empty branch first checks admission (`response.floor is not None and response.inspected > 0` → `format_admission_empty(query, response)`), then the existing `total_matching > 0` message, then the plain no-results message. Both non-empty headers replace `(ranked by similarity)` with `(hybrid-ranked{suffix})` using the same `suffix` expression.

- [ ] **Step 4: Add handler params + rewritten descriptions**

Both `make_handler` inner handlers append parameters and validation:

```python
            min_similarity: float | None = None,
            disable_similarity_floor: bool = False,
        ) -> str:
            ...
            if min_similarity is not None and not (-1.0 <= min_similarity <= 1.0):
                return f"min_similarity must be within [-1, 1]; got {min_similarity}."
```

and forward both to `run_search`, whose implementations pass them to `execute_query(query, source_filter, limit, extra_filters=..., min_similarity=family_kwargs.get("min_similarity"), disable_similarity_floor=bool(family_kwargs.get("disable_similarity_floor", False)))`.

`document.py` `search_description` — full replacement:

```python
    def search_description(self, engine_name: str, engine: "EngineConfig") -> str:
        emb = embeddings_phrase(engine.model)
        floor = engine.similarity_floor
        floor_clause = (
            f"This engine has a configured admission floor of {floor:g}; "
            if floor is not None
            else "This engine has no configured admission floor; "
        )
        return (
            f"Hybrid semantic + full-text search over Markdown documentation "
            f"chunks ({emb}). Each result carries `similarity`: exact cosine "
            f"similarity in [-1, 1] between query and chunk embeddings — a "
            f"consistent geometric scale, NOT a calibrated probability of "
            f"relevance; comparisons are meaningful only within this engine/"
            f"configuration. Results are ordered by hybrid rank fusion, so "
            f"display order may disagree with similarity order. `retrieved_by` "
            f"reports only which retrieval channel(s) returned the row "
            f"(vector, fts, or both) — not evidence the match is correct. "
            f"{floor_clause}`min_similarity` sets a per-call floor and "
            f"`disable_similarity_floor=true` disables admission filtering "
            f"entirely (exact unfloored baseline). An empty response means no "
            f"inspected candidate passed admission — a low-confidence signal "
            f"for this attempt, NOT proof the corpus lacks relevant content."
        )
```

`sql.py` `search_description` — keep the existing source/filters/browse-pointer sentences, replace the first sentence and the "ranked by cosine similarity" claim with the same semantics block: hybrid rank fusion ordering, `similarity` = exact cosine (geometric, uncalibrated, engine-local), `retrieved_by` = channel membership only, `min_similarity`/`disable_similarity_floor` params, engine-floor clause, and the empty-response meaning. Keep the "'Showing N of M results…' so callers can tell when results are similarity-truncated" sentence but reword to "admission- or rank-truncated".

Test updates: `tests/unit/test_browse_descriptions.py` L21-46 — update asserted substrings (keep asserting `min_time`/`min_lock_time`/`table_filter`/`browse_sql_api` for SQL; for the document family assert `"exact cosine"` and `"min_similarity"`); `tests/unit/test_dynamic_tools.py:86` — change `"Semantic search" in tool.description` to `"Hybrid semantic" in tool.description`.

- [ ] **Step 5: CLI flags**

`cli.py` search command — add options after `min_time`:

```python
    min_similarity: Annotated[
        float | None,
        typer.Option(
            "--min-similarity",
            help="Admission floor: only return results with cosine similarity >= this "
            "value (or all query terms verbatim). Overrides the engine's configured floor.",
        ),
    ] = None,
    no_similarity_floor: Annotated[
        bool,
        typer.Option(
            "--no-similarity-floor",
            help="Disable admission filtering entirely (exact unfloored baseline: "
            "no floor AND the original candidate-pool size).",
        ),
    ] = False,
```

call site:

```python
    try:
        response = service.execute_query(
            query,
            source_filter=filter_source,
            limit=limit,
            extra_filters=extra_filters,
            min_similarity=min_similarity,
            disable_similarity_floor=no_similarity_floor,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
```

Update the `--json` help text (L247): "Emit the full envelope (floor, inspected, best_rejected, results with similarity/retrieved_by/rrf_score) as JSON to stdout."

Add to `tests/integration/test_cli.py` (following `test_search_with_options`): `test_search_forwards_similarity_flags` asserting `execute_query` receives `min_similarity=0.4, disable_similarity_floor=False` for `--min-similarity 0.4`, and `disable_similarity_floor=True` for `--no-similarity-floor`.

- [ ] **Step 6: Run the full suite + commit**

Run: `uv run poe check`
Expected: green.

```bash
git add src/dbs_vector/services/search.py \
  src/dbs_vector/mcp/families/document.py src/dbs_vector/mcp/families/sql.py \
  src/dbs_vector/cli.py \
  tests/unit/test_document_family.py tests/unit/test_sql_family.py \
  tests/unit/test_search_service.py tests/unit/test_browse_descriptions.py \
  tests/unit/test_dynamic_tools.py tests/integration/test_cli.py
git commit -m "feat(mcp,cli): admission-aware headers, honest empty-result evidence, min_similarity controls"
```

---

### Task 9: Integration tests — floor mechanics on a real tmpdir LanceDB

**Files:**
- Create: `tests/integration/test_similarity_floor_ci.py`

**Interfaces:**
- Consumes: everything shipped in Tasks 1–8. No MLX: a fake embedder supplies synthetic vectors.

These tests validate **mechanics only** (spec section 8): whether any particular floor value is safe on real corpora is the companion spec's evaluation.

- [ ] **Step 1: Write the tests**

Create `tests/integration/test_similarity_floor_ci.py`:

```python
"""Floor-mechanics integration tests: real tmpdir LanceDB, synthetic vectors, no MLX.

Vector geometry (dimension 4):
  axis 0 = beekeeping topic, axis 1 = code topic,
  axis 2 = off-corpus query axis (near-orthogonal to the corpus: only d3
  carries a 0.1 component on it), axis 3 = shop/store topic.
"""

import numpy as np
import pytest

from dbs_vector.core.models import Chunk
from dbs_vector.infrastructure.storage.lancedb_engine import LanceDBStore
from dbs_vector.infrastructure.storage.mappers import DocumentMapper
from dbs_vector.services.search import SearchService

DIM = 4

DOCS = [
    # (id, text, source, vector)
    ("d1", "beekeeping hive maintenance in spring", "bees.md", [1.0, 0.0, 0.0, 0.0]),
    ("d2", "def delete_by_source(x): removes all rows for a source", "store.py", [0.0, 1.0, 0.0, 0.0]),
    ("d3", "the uv.lock lockfile pins python dependencies", "uv.md", [0.0, 0.9, 0.1, 0.0]),
    ("d4", "arrow record batches stream to lancedb", "arch.md", [0.7, 0.7, 0.0, 0.0]),
    ("d5", "pydantic models validate configuration", "config.md", [0.6, 0.8, 0.0, 0.0]),
    ("d6", "the store opens at nine", "shop.md", [0.0, 0.0, 0.0, 1.0]),
]

QUERY_VECTORS = {
    "beekeeping spring": [1.0, 0.0, 0.0, 0.0],
    "delete_by_source": [0.0, 0.0, 1.0, 0.0],       # off-corpus axis
    "narrowboat lock": [0.0, 0.0, 1.0, 0.0],        # off-corpus axis
    "quantum chromodynamics": [0.0, 0.0, 1.0, 0.0],  # off-corpus axis
    # cos(d6) ~= 0.30: makes d6 the best REJECTED candidate, proving the
    # FTS stem-hit happened while staying below the 0.5 floor.
    "stores": [0.0, 0.0, 0.95, 0.3],
}


class _FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(QUERY_VECTORS[query], dtype=np.float32)


def _make_store(tmp_path, with_fts: bool = True) -> LanceDBStore:
    store = LanceDBStore(
        db_path=str(tmp_path / "db"),
        table_name="floor_ci",
        vector_dimension=DIM,
        mapper=DocumentMapper(vector_dimension=DIM),
    )
    chunks = [
        Chunk(id=i, text=t, source=s, content_hash=f"hash_{i}") for i, t, s, _ in DOCS
    ]
    vectors = np.asarray([v for *_, v in DOCS], dtype=np.float32)
    store.ingest_chunks(chunks, vectors, workflow="test")
    if with_fts:
        store.create_indices()
    return store


@pytest.fixture()
def store(tmp_path):
    return _make_store(tmp_path)


def _service(store, floor):
    return SearchService(_FakeEmbedder(), store, similarity_floor=floor)


def test_on_topic_query_carries_exact_similarity(store):
    resp = _service(store, floor=None).execute_query("beekeeping spring", limit=5)
    assert resp.floor is None
    by_id = {r.chunk.id: r for r in resp.results}
    assert by_id["d1"].similarity == pytest.approx(1.0, abs=1e-4)
    # d4 = [0.7, 0.7]: cos with [1, 0] = 0.7071
    assert by_id["d4"].similarity == pytest.approx(0.7071, abs=1e-3)
    # FTS index exists, so hybrid ran — fail fast if the environment degraded
    assert store._hybrid_ok is True


def test_floor_orthogonal_query_returns_empty_with_evidence(store):
    resp = _service(store, floor=0.5).execute_query("quantum chromodynamics", limit=5)
    assert resp.results == []
    assert resp.floor == 0.5
    # vector leg returns all 6 docs (flat search, fetch limit 15); FTS matches none
    assert resp.inspected == 6
    assert resp.best_rejected is not None
    assert resp.best_rejected.similarity < 0.5


def test_lexical_gate_rescues_verbatim_identifier(store):
    # Vector-orthogonal query whose token appears verbatim in d2's text: the
    # FTS leg returns it and the gate admits it despite similarity ~0.
    resp = _service(store, floor=0.5).execute_query("delete_by_source", limit=5)
    ids = [r.chunk.id for r in resp.results]
    assert ids == ["d2"]
    assert resp.results[0].retrieved_by in ("fts", "both")
    assert resp.results[0].similarity < 0.5


def test_all_terms_rule_rejects_partial_verbatim_match(store):
    # 'lock' is verbatim in d3 (FTS returns it) but 'narrowboat' is absent:
    # the all-terms rule rejects — the measured-stemming-noise defense.
    resp = _service(store, floor=0.5).execute_query("narrowboat lock", limit=5)
    assert resp.results == []
    assert resp.inspected > 0
    assert resp.best_rejected is not None


def test_stemming_overmatch_rejected_by_gate(store):
    # The measured false-positive class, end to end: FTS stemming makes the
    # query 'stores' retrieve d6 ('store' in text) — verified live on the
    # installed LanceDB/Tantivy — but the gate demands the token VERBATIM,
    # so the row is rejected. best_rejected's channel proves the stem hit
    # actually happened (a 'vector'-only channel here would mean the FTS
    # premise failed — investigate the FTS backend, don't loosen the assert).
    resp = _service(store, floor=0.5).execute_query("stores", limit=5)
    assert resp.results == []
    assert resp.best_rejected is not None
    assert resp.best_rejected.source == "shop.md"
    assert resp.best_rejected.retrieved_by == "both"
    assert resp.best_rejected.similarity < 0.5


def test_vector_only_fallback_floor_applies_no_lexical_rescue(tmp_path):
    # No FTS index: hybrid degrades to pure vector; retrieved_by is always
    # 'vector', so the lexical gate can never rescue a below-floor row.
    store = _make_store(tmp_path, with_fts=False)
    resp = _service(store, floor=0.5).execute_query("delete_by_source", limit=5)
    assert store._hybrid_ok is False
    assert resp.results == []  # verbatim token can't rescue: no FTS channel
    resp_unfloored = _service(store, floor=None).execute_query("delete_by_source", limit=5)
    assert resp_unfloored.results  # rows exist; the floor was what dropped them
    assert all(r.retrieved_by == "vector" for r in resp_unfloored.results)
    assert all(r.rrf_score is None for r in resp_unfloored.results)


def test_disable_similarity_floor_restores_baseline_pool(store):
    floored = _service(store, floor=0.5)
    baseline = floored.execute_query(
        "beekeeping spring", limit=2, disable_similarity_floor=True
    )
    assert baseline.floor is None
    assert baseline.inspected == 2  # original pool: fetch limit == limit
    active = floored.execute_query("beekeeping spring", limit=2)
    assert active.inspected > 2  # oversampled pool (limit * 3 per leg)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/integration/test_similarity_floor_ci.py -v`
Expected: all PASS. If `create_indices()` fails to build FTS in the test environment, `test_on_topic_query_carries_exact_similarity`'s `_hybrid_ok is True` assertion fails loudly — investigate the FTS backend rather than skipping (existing hybrid-path integration tests imply FTS works in tmpdir).

- [ ] **Step 3: Full check + commit**

```bash
uv run poe check
git add tests/integration/test_similarity_floor_ci.py
git commit -m "test(integration): floor mechanics on real LanceDB with synthetic vectors"
```

---

### Task 10: Documentation

**Files:**
- Modify: `docs/README_MCP.md`
- Modify: `docs/README_PROFILES.md`
- Modify: `CLAUDE.md` (Key Design Details)

- [ ] **Step 1: `docs/README_MCP.md`** — add a "Similarity, ranking, and admission" section (near the existing search-tool description around L171):

  - Result block example: `--- Result (similarity 0.78, retrieved by: vector+fts) ---`.
  - Semantics: `similarity` is exact cosine in [-1, 1] — a consistent geometric scale, not a calibrated probability of relevance; engine-local comparisons only. Ordering is hybrid RRF rank fusion, so display order may disagree with similarity order. `retrieved_by` is retrieval-channel membership only.
  - Admission: when a floor is active (engine `similarity_floor` or per-call `min_similarity`), a result is admitted when `similarity >= floor` OR every eligible query term appears verbatim in the chunk text (word-boundary, case-insensitive, no stemming; stopwords and tokens under 3 chars excluded; FTS-channel rows only). Note the stated limitation: a single-common-token query can still pass the gate; FTS indexes only `text`, so filenames are protected only when they appear in chunk text.
  - Empty responses: reproduce the spec's example message verbatim and state it signals low retrieval confidence for that attempt, not corpus-level absence. Document `min_similarity` (range [-1, 1]) and `disable_similarity_floor` (exact-baseline rerun state: no floor AND original pool size — `min_similarity=0` is not equivalent).
  - CLI JSON envelope: `{"floor", "inspected", "best_rejected", "results"}`; `rrf_score` appears only here, never in text output.
  - Migration note: consumers parsing `Score:` lines (e.g. the find-impacting-queries skill) must read `similarity` from the new result block instead.

- [ ] **Step 2: `docs/README_PROFILES.md`** — in the engine-block reference, document the new engine key:

  ```yaml
  engines:
    md:
      # Optional admission floor (exact cosine, [-1, 1]). Unset = no floor
      # (baseline default for every engine). Floors are engine-level policy,
      # not model properties: the same model serves engines with different
      # prefixes and content shapes. Calibrated default values ship with the
      # calibration spec — do NOT copy a number from documentation; leave
      # this unset until calibration produces one for your engine.
      # similarity_floor: <calibrated value>
  ```

  plus one paragraph: floor-active searches oversample per-leg candidate pools (`limit * 3`), which changes RRF fusion inputs — a deliberate, spec-stated trade; `disable_similarity_floor` restores the exact unfloored baseline.

- [ ] **Step 3: `CLAUDE.md`** — in "Key Design Details", add one bullet (and reconcile any existing wording that claims results are "ranked by cosine similarity"):

  > - **Search scoring**: every result carries `similarity` (exact cosine between query and chunk vectors, computed in NumPy at search time — metric-independent, covers FTS-only rows), `retrieved_by` (channel membership: `both`/`vector`/`fts`), and `rrf_score` (fused RRF value, JSON/debug only). Ranking stays hybrid RRF(K=60); `_build_hybrid` pins `.metric("cosine")`. Admission policy lives in `SearchService` (engine `similarity_floor` / per-call `min_similarity` / `disable_similarity_floor`), which returns a `SearchResponse` envelope (`results`, `floor`, `inspected`, `best_rejected`). Construct services via `build_search_service()` — never hand-wire `SearchService`.

- [ ] **Step 4: Commit**

```bash
uv run poe check
git add docs/README_MCP.md docs/README_PROFILES.md CLAUDE.md
git commit -m "docs: honest similarity semantics, admission floor knob, envelope shape"
```

---

## Post-merge follow-ups (outside this repo, do not fold into tasks)

- The `find-impacting-queries` and related skills in `~/.claude/skills/` read `Score:` lines from MCP output; after this ships they should read `similarity` (spec migration note).
- Auto-memory `search-score-semantics` (RRF-score description) becomes stale on merge and should be rewritten to the new semantics.

## Self-review record

- **Spec coverage:** §1 retrieval → Tasks 4, 7 (oversample). §2 scoring → Tasks 1, 4. §3 data model → Tasks 4, 5, 6, 7 (factory). §4 admission → Tasks 2, 7. §5 presentation → Tasks 5, 8. §6 consumer inventory → Tasks 5, 6, 7, 8 (every file from the spec's list appears in a task). §7 config → Tasks 3, 8. §8 testing → Tasks 1–4, 7–9. §9 docs → Task 10. Non-goals respected: no fusion change, no default floors, no browse/triage edits, no schema change.
- **Deliberate decision beyond spec letter:** the SQL family's `(ranked by similarity)` header text becomes `(hybrid-ranked…)` — the spec mandates only the admission suffix, but the old text is precisely the dishonest claim this spec removes; flagged for reviewer sign-off.
- **Type consistency check:** `from_polars_row` final signature (Tasks 4→5) matches store call and all test doubles; `SearchResponse` fields (`results`/`floor`/`inspected`/`best_rejected`) used identically in Tasks 6, 7, 8, 9; `retrieved_by_label` defined once (Task 5) and imported by families/dbs-web; `_FLOOR_OVERSAMPLE` referenced only inside `services/search.py`.
- **External review round 2 applied (all findings verified against the codebase first):** hybrid mock chains need `.metric`/`.rerank` self-returns and frames must be faithful to `return_score="all"` (Task 4 — confirmed at `test_lancedb_engine.py:312/495/582/765`); hybrid rows without a finite `_relevance_score` now raise (None reserved for vector fallback); input validation (limit ∈ [1, 100], min_similarity range) is unconditional and tested against the disable flag — LanceDB treats non-positive limit as unbounded; bootstrap tests rewritten against the real `mock_settings` fixture with `similarity_floor` set explicitly (MagicMock engines); pytest `-k "A or B"` fix; `uv run pyright src` added alongside `poe check` (which runs mypy, not pyright — confirmed in pyproject); explicit per-task staging replaces `git add -A`; true stemming-overmatch integration case added (`stores` → `store`, FTS stem hit proven via `best_rejected.retrieved_by`); `SearchResponse.inspected` made required; docs floor example is a commented placeholder. **Pushed back on one item:** hand-prefixing `rtk` — the session hook rewrites shell commands automatically (RTK.md), so plan commands stay plain.
