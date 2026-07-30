"""Path scoping for file-based ingestion.

THE INVARIANT: within currently configured roots, CLI discovery, watcher
events and reconciliation use the same path filtering. This module is the one
place that filtering lives — every caller asks the same object.

`PathFilter` is pure logic (services layer, no infrastructure imports). It is
built in `bootstrap.build_dependencies` and injected into `IngestionService`
and `WatcherService`. Non-document engines get `None` and keep their existing
code paths untouched.
"""

import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pathspec import GitIgnoreSpec

_GLOB_CHARS = "*?["


class PathFilter:
    """Answers: is this path ingestable, and under which root?"""

    def __init__(
        self,
        roots: list[str],
        extensions: list[str],
        ignore_patterns: list[str],
        use_gitignore: bool = False,
    ) -> None:
        self.roots = [Path(r) for r in roots]
        self.extensions = {e.lower() for e in extensions}
        self.ignore_patterns = list(ignore_patterns)
        self.use_gitignore = use_gitignore
        # Parsed once per root, for the process lifetime: .gitignore edits
        # apply on restart (spec: nested / live-tracked .gitignore out of scope).
        self._gitignore_cache: dict[str, GitIgnoreSpec | None] = {}

    # ---- roots ---------------------------------------------------------

    def active_roots(self) -> list[Path]:
        """Configured roots that currently exist as directories.

        A missing root is a logged warning and is SKIPPED — excluded from
        walking, watching and pruning, never treated as an empty directory
        (which would delete the whole root's rows on the next reconcile).
        """
        active: list[Path] = []
        for root in self.roots:
            if root.is_dir():
                active.append(root)
            else:
                logger.warning(
                    "Ingestion root missing or unreadable, skipping (rows under it "
                    "are left untouched): {}",
                    root,
                )
        return active

    def root_for(self, path: str | Path, roots: list[Path] | None = None) -> Path | None:
        """The configured root lexically containing `path`, or None.

        Pass `roots` (usually `active_roots()`) to reuse one existence check
        across a whole pass instead of re-stat'ing per file.
        """
        candidate = Path(path)
        for root in self.roots if roots is None else roots:
            if candidate == root or root in candidate.parents:
                return root
        return None

    # ---- membership ----------------------------------------------------

    def is_ingestable(self, path: str | Path, roots: list[Path] | None = None) -> bool:
        """Extension + root membership + ignore_patterns + gitignore."""
        candidate = Path(path)
        if candidate.suffix.lower() not in self.extensions:
            return False
        root = self.root_for(candidate, roots)
        if root is None:
            return False
        relative = candidate.relative_to(root).as_posix()
        if self._is_ignored(candidate.name, relative):
            return False
        return not (self.use_gitignore and self._gitignore_matches(root, relative))

    def _is_ignored(self, name: str, relative: str) -> bool:
        """fnmatch globs tested against BOTH the basename and the
        root-relative path; either match excludes."""
        return any(
            fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relative, pattern)
            for pattern in self.ignore_patterns
        )

    def _gitignore_matches(self, root: Path, relative: str) -> bool:
        spec = self._spec_for(root)
        return bool(spec and spec.match_file(relative))

    def _spec_for(self, root: Path) -> GitIgnoreSpec | None:
        key = str(root)
        if key in self._gitignore_cache:
            return self._gitignore_cache[key]
        spec: GitIgnoreSpec | None = None
        gitignore = root / ".gitignore"
        if gitignore.is_file():
            try:
                spec = GitIgnoreSpec.from_lines(gitignore.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError, ValueError) as e:
                logger.warning("Could not parse {}: {}", gitignore, e)
        self._gitignore_cache[key] = spec
        return spec

    # ---- discovery -----------------------------------------------------

    def iter_files(self) -> Iterator[Path]:
        """Every ingestable file under the ACTIVE roots, deduped.

        Absolute and normalized: roots are already resolved at config load and
        children are joined, so no per-file symlink resolution happens.
        Symlinked subdirectories are not traversed (rglob behaviour).
        """
        roots = self.active_roots()
        seen: set[Path] = set()
        for root in roots:
            for candidate in root.rglob("*"):
                if candidate in seen or not candidate.is_file():
                    continue
                if self.is_ingestable(candidate, roots):
                    seen.add(candidate)
                    yield candidate


def anchor_for(target: str) -> str | None:
    """The filtering anchor for an explicit CLI path — ONE rule.

    directory -> itself; file -> parent; glob -> longest non-glob prefix;
    URL -> none. An explicit path is its own anchor: if that loses a
    `.gitignore` a configured root would have applied, the next reconciliation
    prunes the difference. Self-healing, by design.
    """
    if target.startswith(("http://", "https://")):
        return None
    if os.path.isdir(target):
        return str(Path(target).resolve())
    if any(ch in target for ch in _GLOB_CHARS):
        prefix: list[str] = []
        for part in Path(target).parts:
            if any(ch in part for ch in _GLOB_CHARS):
                break
            prefix.append(part)
        return str(Path(*prefix).resolve()) if prefix else str(Path.cwd())
    return str(Path(target).resolve().parent)
