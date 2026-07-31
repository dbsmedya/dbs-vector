"""Resolve a caller's `source_filter` against the sources actually stored.

A stored `source` is a full path for document engines and a database name for
SQL engines. Requiring callers to reproduce it verbatim made the most common
inputs — a bare filename, a directory, a path fragment — match nothing and
return an empty result set indistinguishable from "the corpus has nothing".

Resolution is tried in precedence order and stops at the first rule that
matches anything, so an exact path can never be widened by a looser rule:

1. `exact`  — the stored value verbatim (`/root/docs/specs/api.md`)
2. `suffix` — a trailing path fragment (`specs/api.md`, `api.md`)
3. `scope`  — a directory component run (`specs`, `docs/specs`, `phase-030`),
              matching every source beneath it
4. `none`   — nothing matched; carries suggestions instead of failing silently
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel

SourceMatchKind = Literal["exact", "suffix", "scope", "none"]

_MAX_SUGGESTIONS = 3


class SourceResolution(BaseModel):
    """What a `source_filter` resolved to, and how."""

    value: str
    kind: SourceMatchKind
    matched: list[str] = []
    # Populated only when kind == "none": the closest stored sources, so a
    # caller that guessed wrong is told what it could have asked for.
    suggestions: list[str] = []

    @property
    def is_unmatched(self) -> bool:
        return self.kind == "none"


def _parts(path: str) -> tuple[str, ...]:
    """Path components with any root marker dropped."""
    return tuple(part for part in PurePosixPath(path).parts if part != "/")


def _is_contiguous_run(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1)
    )


def matches_suffix(stored_source: str, value: str) -> bool:
    """Whether a stored path ends with `value` on a component boundary."""
    if not value:
        return False
    stored = PurePosixPath(stored_source).as_posix()
    expected = PurePosixPath(value).as_posix().lstrip("/")
    return stored == expected or stored.endswith("/" + expected)


def matches_scope(stored_source: str, value: str) -> bool:
    """Whether `value` names a directory run containing the stored source.

    Only DIRECTORY components are considered, so a filename never resolves as
    a scope — that case belongs to `matches_suffix`.
    """
    needle = _parts(value.strip("/"))
    if not needle:
        return False
    directories = _parts(stored_source)[:-1]
    return _is_contiguous_run(needle, directories)


def _suggest(value: str, sources: list[str]) -> list[str]:
    """Closest stored sources for a filter that matched nothing."""
    by_name: dict[str, list[str]] = {}
    for source in sources:
        by_name.setdefault(PurePosixPath(source).name, []).append(source)

    suggestions: list[str] = []
    for name in difflib.get_close_matches(value, sorted(by_name), n=_MAX_SUGGESTIONS, cutoff=0.6):
        suggestions.extend(sorted(by_name[name]))
    if not suggestions:
        suggestions = difflib.get_close_matches(
            value, sorted(sources), n=_MAX_SUGGESTIONS, cutoff=0.3
        )
    return suggestions[:_MAX_SUGGESTIONS]


def resolve_source_filter(value: str, sources: Iterable[str]) -> SourceResolution:
    """Resolve `value` against the distinct sources stored in one table."""
    known = sorted(set(sources))
    cleaned = value.strip()
    if not cleaned:
        return SourceResolution(value=value, kind="none", matched=[], suggestions=[])

    exact = [source for source in known if source == cleaned]
    if exact:
        return SourceResolution(value=cleaned, kind="exact", matched=exact)

    suffix = [source for source in known if matches_suffix(source, cleaned)]
    if suffix:
        return SourceResolution(value=cleaned, kind="suffix", matched=suffix)

    scope = [source for source in known if matches_scope(source, cleaned)]
    if scope:
        return SourceResolution(value=cleaned, kind="scope", matched=scope)

    return SourceResolution(
        value=cleaned, kind="none", matched=[], suggestions=_suggest(cleaned, known)
    )
