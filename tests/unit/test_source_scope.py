"""Resolution semantics for `source_filter`."""

from __future__ import annotations

import pytest

from dbs_vector.core.source_scope import resolve_source_filter

SOURCES = [
    "/root/docs/specs/api.md",
    "/root/docs/specs/storage.md",
    "/root/docs/decisions/0001-use-lancedb.md",
    "/root/docs/README.md",
    "/root/notes/phase-030/plan.md",
    "/root/notes/phase-030/review.md",
    "/root/notes/phase-031/plan.md",
]


class TestPrecedence:
    def test_exact_full_path_wins(self):
        res = resolve_source_filter("/root/docs/specs/api.md", SOURCES)
        assert res.kind == "exact"
        assert res.matched == ["/root/docs/specs/api.md"]

    def test_suffix_resolves_a_named_file(self):
        res = resolve_source_filter("specs/api.md", SOURCES)
        assert res.kind == "suffix"
        assert res.matched == ["/root/docs/specs/api.md"]

    def test_bare_filename_resolves_by_suffix(self):
        res = resolve_source_filter("api.md", SOURCES)
        assert res.kind == "suffix"
        assert res.matched == ["/root/docs/specs/api.md"]

    def test_a_suffix_shared_by_several_files_returns_all_of_them(self):
        res = resolve_source_filter("plan.md", SOURCES)
        assert res.kind == "suffix"
        assert res.matched == ["/root/notes/phase-030/plan.md", "/root/notes/phase-031/plan.md"]

    def test_exact_is_not_widened_by_a_looser_rule(self):
        """A caller naming one file must never receive its siblings."""
        res = resolve_source_filter("/root/notes/phase-030/plan.md", SOURCES)
        assert res.kind == "exact"
        assert res.matched == ["/root/notes/phase-030/plan.md"]


class TestDirectoryScope:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("specs", ["/root/docs/specs/api.md", "/root/docs/specs/storage.md"]),
            ("decisions", ["/root/docs/decisions/0001-use-lancedb.md"]),
            (
                "phase-030",
                ["/root/notes/phase-030/plan.md", "/root/notes/phase-030/review.md"],
            ),
            ("docs/specs", ["/root/docs/specs/api.md", "/root/docs/specs/storage.md"]),
        ],
    )
    def test_directory_names_scope_to_everything_beneath(self, value, expected):
        """The exact inputs that silently returned nothing before."""
        res = resolve_source_filter(value, SOURCES)
        assert res.kind == "scope"
        assert res.matched == expected

    def test_leading_and_trailing_slashes_are_tolerated(self):
        assert resolve_source_filter("/specs/", SOURCES).matched == [
            "/root/docs/specs/api.md",
            "/root/docs/specs/storage.md",
        ]

    def test_component_run_must_be_contiguous(self):
        assert resolve_source_filter("docs/phase-030", SOURCES).kind == "none"

    def test_a_partial_component_is_not_a_scope(self):
        """'phase' is a prefix of 'phase-030', not a path component."""
        assert resolve_source_filter("phase", SOURCES).kind == "none"


class TestUnmatched:
    def test_unknown_value_reports_none_with_suggestions(self):
        res = resolve_source_filter("api.mdx", SOURCES)
        assert res.kind == "none"
        assert res.matched == []
        assert res.is_unmatched
        assert "/root/docs/specs/api.md" in res.suggestions

    def test_suggestions_are_capped(self):
        res = resolve_source_filter("plan", SOURCES)
        assert len(res.suggestions) <= 3

    def test_empty_filter_resolves_to_nothing(self):
        res = resolve_source_filter("   ", SOURCES)
        assert res.kind == "none"
        assert res.matched == []

    def test_empty_corpus_never_raises(self):
        res = resolve_source_filter("anything", [])
        assert res.kind == "none"
        assert res.suggestions == []


class TestSqlStyleSources:
    """SQL engines store a bare database name, not a path."""

    def test_database_name_matches_exactly(self):
        res = resolve_source_filter("odeal", ["odeal", "magento", "reporting"])
        assert res.kind == "exact"
        assert res.matched == ["odeal"]

    def test_unknown_database_suggests_near_names(self):
        res = resolve_source_filter("odeall", ["odeal", "magento"])
        assert res.kind == "none"
        assert res.suggestions == ["odeal"]
