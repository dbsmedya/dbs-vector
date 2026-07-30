# tests/unit/test_path_filter.py
from pathlib import Path

import pytest

from dbs_vector.services.path_filter import PathFilter, anchor_for

DEFAULT_IGNORES = [".#*", "*~", "*.tmp", ".DS_Store"]


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "notes" / "deep").mkdir(parents=True)
    (root / "notes" / "a.md").write_text("# A")
    (root / "notes" / "deep" / "b.MD").write_text("# B")
    (root / "notes" / "c.txt").write_text("c")
    (root / "notes" / "d.pdf").write_bytes(b"%PDF")
    (root / "notes" / ".#lock.md").write_text("lock")
    (root / "notes" / "scratch.tmp").write_text("tmp")
    (root / "build").mkdir()
    (root / "build" / "gen.md").write_text("# generated")
    return root


def _filter(root, **kwargs):
    return PathFilter(
        roots=[str(root)],
        extensions=[".md", ".txt"],
        ignore_patterns=kwargs.pop("ignore_patterns", DEFAULT_IGNORES),
        **kwargs,
    )


class TestExtensions:
    def test_supported_extension_is_ingestable(self, vault):
        assert _filter(vault).is_ingestable(vault / "notes" / "a.md") is True

    def test_extension_match_is_case_insensitive(self, vault):
        assert _filter(vault).is_ingestable(vault / "notes" / "deep" / "b.MD") is True

    def test_unsupported_extension_is_rejected(self, vault):
        assert _filter(vault).is_ingestable(vault / "notes" / "d.pdf") is False


class TestIgnorePatterns:
    def test_basename_pattern_excludes(self, vault):
        assert _filter(vault).is_ingestable(vault / "notes" / ".#lock.md") is False

    def test_root_relative_pattern_excludes(self, vault):
        pf = _filter(vault, ignore_patterns=["build/*"])
        assert pf.is_ingestable(vault / "build" / "gen.md") is False
        assert pf.is_ingestable(vault / "notes" / "a.md") is True


class TestRootMembership:
    def test_path_outside_every_root_is_rejected(self, vault, tmp_path):
        outside = tmp_path / "elsewhere.md"
        outside.write_text("# x")
        assert _filter(vault).is_ingestable(outside) is False

    def test_root_for_returns_the_containing_root(self, vault):
        pf = _filter(vault)
        assert pf.root_for(vault / "notes" / "a.md") == Path(str(vault))
        assert pf.root_for("/nowhere/x.md") is None


class TestActiveRoots:
    def test_missing_root_is_skipped_not_treated_as_empty(self, vault, tmp_path, caplog):
        pf = PathFilter(
            roots=[str(vault), str(tmp_path / "unmounted")],
            extensions=[".md"],
            ignore_patterns=DEFAULT_IGNORES,
        )
        assert pf.active_roots() == [Path(str(vault))]
        assert "unmounted" in caplog.text

    def test_all_roots_missing_yields_no_active_roots(self, tmp_path):
        pf = PathFilter(
            roots=[str(tmp_path / "gone")], extensions=[".md"], ignore_patterns=[]
        )
        assert pf.active_roots() == []
        assert list(pf.iter_files()) == []


class TestIterFiles:
    def test_walk_yields_only_ingestable_files(self, vault):
        found = {p.name for p in _filter(vault).iter_files()}
        assert found == {"a.md", "b.MD", "c.txt", "gen.md"}

    def test_nested_roots_do_not_yield_duplicates(self, vault):
        pf = PathFilter(
            roots=[str(vault), str(vault / "notes")],
            extensions=[".md"],
            ignore_patterns=DEFAULT_IGNORES,
        )
        found = [p for p in pf.iter_files()]
        assert len(found) == len(set(found))


class TestGitignore:
    def test_gitignore_excludes_matching_files_when_enabled(self, vault):
        (vault / ".gitignore").write_text("build/\n")
        pf = _filter(vault, use_gitignore=True)
        assert pf.is_ingestable(vault / "build" / "gen.md") is False
        assert pf.is_ingestable(vault / "notes" / "a.md") is True

    def test_gitignore_is_ignored_when_not_opted_in(self, vault):
        (vault / ".gitignore").write_text("build/\n")
        assert _filter(vault).is_ingestable(vault / "build" / "gen.md") is True

    def test_gitignore_is_matched_root_relative(self, vault):
        (vault / ".gitignore").write_text("/notes/a.md\n")
        pf = _filter(vault, use_gitignore=True)
        assert pf.is_ingestable(vault / "notes" / "a.md") is False
        assert pf.is_ingestable(vault / "notes" / "deep" / "b.MD") is True

    def test_gitignore_is_cached_for_the_process_lifetime(self, vault):
        (vault / ".gitignore").write_text("build/\n")
        pf = _filter(vault, use_gitignore=True)
        assert pf.is_ingestable(vault / "build" / "gen.md") is False
        (vault / ".gitignore").write_text("")  # edits apply on restart only
        assert pf.is_ingestable(vault / "build" / "gen.md") is False


class TestAnchorFor:
    def test_directory_anchors_to_itself(self, vault):
        assert anchor_for(str(vault)) == str(vault.resolve())

    def test_file_anchors_to_its_parent(self, vault):
        assert anchor_for(str(vault / "notes" / "a.md")) == str((vault / "notes").resolve())

    def test_glob_anchors_to_the_longest_non_glob_prefix(self, vault):
        assert anchor_for(f"{vault}/notes/**/*.md") == str((vault / "notes").resolve())

    def test_url_has_no_anchor(self):
        assert anchor_for("http://host/api/v1") is None
