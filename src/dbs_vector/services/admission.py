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
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
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
