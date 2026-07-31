"""Unit tests for the admission policy (spec section 4)."""

from dbs_vector.core.models import Chunk, SearchResult
from dbs_vector.services.admission import (
    apply_admission,
    eligible_tokens,
    is_admitted,
    lexical_gate,
)


def _result(text: str, similarity: float, retrieved_by: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=text[:8], text=text, source="s.md", content_hash="h"),
        similarity=similarity,
        retrieved_by=retrieved_by,
    )


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


def test_apply_admission_splits_on_semantic_channel():
    high = _result("unrelated prose", 0.80, "vector")
    low = _result("unrelated prose", 0.10, "vector")
    admitted, rejected = apply_admission([high, low], "anything at all", floor=0.5)
    assert admitted == [high]
    assert rejected == [low]


def test_apply_admission_lexical_gate_rescues_below_floor():
    rescued = _result("call delete_by_source to purge", 0.02, "fts")
    admitted, rejected = apply_admission([rescued], "delete_by_source", floor=0.5)
    assert admitted == [rescued]
    assert rejected == []


def test_apply_admission_preserves_input_order_in_both_lists():
    a = _result("alpha", 0.9, "vector")
    b = _result("bravo", 0.1, "vector")
    c = _result("charlie", 0.8, "vector")
    d = _result("delta", 0.2, "vector")
    admitted, rejected = apply_admission([a, b, c, d], "zzz qqq", floor=0.5)
    assert [result.chunk.text for result in admitted] == ["alpha", "charlie"]
    assert [result.chunk.text for result in rejected] == ["bravo", "delta"]


def test_shared_admission_predicate_uses_semantic_or_lexical_channel():
    assert is_admitted(0.8, False, 0.5)
    assert is_admitted(0.1, True, 0.5)
    assert not is_admitted(0.1, False, 0.5)
