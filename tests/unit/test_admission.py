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
