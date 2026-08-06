"""Markdown token interpretation for the document chunker.

Owns markdown-it configuration and token dispatch. `document.py` consumes
only `_Block` values from here and never sees a `markdown_it.Token`.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import markdown_it
import yaml
from markdown_it.token import Token
from mdit_py_plugins.admon import admon_plugin
from mdit_py_plugins.front_matter import front_matter_plugin

_ATX_RE = re.compile(r"^#{1,6}\s*")
# NOTE: _LIST_MARKER stays in document.py — it is used only by _list_items(),
# which is part of the oversized-block splitter and does NOT move.

_NOISE_HTML = re.compile(
    r"\A(?:<!--.*?-->|<(style|script)\b[^>]*>.*?</\1\s*>)\Z",
    re.DOTALL | re.IGNORECASE,
)


def _is_noise_html(text: str) -> bool:
    """True for html_block content that is only a comment, <style> or <script>.

    Everything else — tables, <img>, <div> wrappers — routinely carries real
    content and is kept as text. Full HTML parsing is out of scope.
    """
    return bool(_NOISE_HTML.match(text.strip()))


_BACKTICK_RUN = re.compile(r"^\s*(`{3,})", re.MULTILINE)
_TILDE_RUN = re.compile(r"^\s*(~{3,})", re.MULTILINE)


def choose_fence(body: str, info: str = "") -> str:
    """Pick a fence delimiter that `body` cannot terminate early.

    Two independent hazards:

    * CommonMark closes a fence at the first line whose delimiter run is at
      least as long as the opener's, so the opener must OUTRUN every matching
      run in the body. A literal ``` line inside a three-backtick fence closes
      it and spills the remainder into prose.
    * A backtick-fenced block's info string may not contain a backtick. Source
      like ``~~~py`variant`` is a valid tilde fence, but re-rendering it with
      backticks produces something markdown-it will not parse as a fence at
      all — so a backtick in `info` forces tildes regardless of the body.

    Chosen ONCE from a complete body, then reused for every part it is split
    into: a delimiter safe for the whole is safe for each piece, and choosing
    per-part would be circular, since `_split_code` must reserve wrapper
    overhead before `_pack_atoms` has produced any parts.
    """
    if "`" in info:
        longest = max((len(m.group(1)) for m in _TILDE_RUN.finditer(body)), default=2)
        return "~" * max(3, longest + 1)
    longest = max((len(m.group(1)) for m in _BACKTICK_RUN.finditer(body)), default=2)
    return "`" * max(3, longest + 1)


def render_fence(body: str, info: str = "", delim: str | None = None) -> str:
    """Wrap `body`. Pass `delim` to reuse a delimiter chosen from a larger body."""
    d = delim or choose_fence(body, info)
    return f"{d}{info}\n{body}\n{d}"


def _strip_indent_unit(lines: list[str]) -> list[str]:
    """Remove ONE markdown indentation unit: one tab, or up to four spaces.

    Used here to dedent indented code, and reused in Task 5 as the admonition
    container's unprefix. Removing a literal four-character prefix would
    corrupt tab-indented content, which 7 of 357 admonitions in the reference
    corpus use.
    """
    out: list[str] = []
    for ln in lines:
        if ln.startswith("\t"):
            out.append(ln[1:])
            continue
        i = 0
        while i < 4 and i < len(ln) and ln[i] == " ":
            i += 1
        out.append(ln[i:])
    return out


@dataclass(frozen=True)
class _ScopedBlock:
    node_type: str  # "heading" | "section" | "code" | "list" | "table"
    text: str  # raw source for a container wrapper
    start_line: int
    end_line: int
    level: int = 0  # heading level (1-6) when node_type == "heading"
    info: str = ""  # fence language when node_type == "code"
    scope: tuple[tuple[str, int], ...] = ()  # (container_type, start_line), outermost first
    frames: tuple[str, ...] = ()  # INHERITED display labels, outermost first
    expansion: tuple["_ScopedBlock", ...] = ()  # descended children, if any
    # --- container-wrapper metadata; all default-empty for ordinary blocks ---
    container_type: str = ""  # e.g. "blockquote_open"
    self_frame: str | None = None  # this container's OWN frame, if any
    always_expand: bool = False  # admonitions: packing never keeps atomic
    expanded_fallback_frame: str | None = None  # frame applied to children iff self_frame is None


_Block = _ScopedBlock  # transitional alias; remove once document.py is updated


@dataclass(frozen=True)
class _ContainerSpec:
    token_type: str
    frame: Callable[[Token], str | None]
    unprefix: Callable[[list[str]], list[str]]
    always_expand: bool = False
    # Applied to children when the container has no frame of its own — an
    # expanded ordinary blockquote loses its `>` markers, so `quote` is what
    # keeps the quotation semantics visible.
    expanded_fallback_frame: str | None = None


def _admonition_frame(t: Token) -> str | None:
    tag = str(t.meta.get("tag", "")).strip().casefold()
    if not tag:
        return None
    title = (t.content or "").strip()
    return f"{tag}: {title}" if title else tag


_CONTAINERS: dict[str, _ContainerSpec] = {
    "admonition_open": _ContainerSpec(
        token_type="admonition_open",
        frame=_admonition_frame,
        unprefix=_strip_indent_unit,
        always_expand=True,
    ),
}


def _flatten_always_expand(blocks: list[_ScopedBlock]) -> list[_ScopedBlock]:
    out: list[_ScopedBlock] = []
    for b in blocks:
        if b.always_expand:
            # Note: NOT `and b.expansion`. An always-expand container is
            # replaced by its children unconditionally — an EMPTY admonition
            # must vanish, not survive atomically as raw `!!! note "x"` text
            # with no frame attached.
            out.extend(_flatten_always_expand(list(b.expansion)))
        else:
            out.append(b)
    return out


def _slice(
    lines: list[str],
    start: int,
    end: int,
    unprefixers: tuple[Callable[[list[str]], list[str]], ...],
) -> list[str]:
    """One token's source window with every ancestor container's framing removed.

    Applies `unprefixers` OUTERMOST FIRST to the window only. An earlier draft
    materialised a normalized copy of the whole document per container, which
    is O(containers x document lines) — a page with 40 admonitions copied the
    whole file 40 times. Composing functions over the mapped slice avoids
    whole-document copies and preserves indices identically. (Container and
    child maps overlap, so total work is not strictly linear in document
    length, but it is bounded by nesting depth rather than by container count.)
    """
    window = lines[start:end]
    for fn in unprefixers:
        window = fn(window)
    return window


@dataclass(frozen=True)
class _ParsedDocument:
    title: str | None
    blocks: tuple[_Block, ...]


def _extract_title(raw: str) -> str | None:
    """Lift a scalar top-level `title:` out of YAML front matter.

    Malformed front matter must never fail an ingest, so every failure mode
    collapses to None.
    """
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("title")
    if value is None or isinstance(value, (list, dict, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


class MarkdownBlockParser:
    """Parses markdown source into flat semantic blocks."""

    def __init__(self) -> None:
        self._md = (
            markdown_it.MarkdownIt("commonmark")
            .enable("table")
            .use(front_matter_plugin)
            .use(admon_plugin)
        )

    def parse(self, content: str) -> _ParsedDocument:
        tokens = self._md.parse(content)
        lines = content.splitlines(keepends=True)
        front_raw = next((t.content for t in tokens if t.type == "front_matter"), None)
        blocks = self._walk(tokens, 0, len(tokens), 0, lines, (), (), (), in_container=False)
        blocks = _flatten_always_expand(blocks)
        title = _extract_title(front_raw) if front_raw else None
        # H1 de-duplication: `title: X` + `# X` must not render as "X > X".
        first_h1 = next(
            (b.text for b in blocks if b.node_type == "heading" and b.level == 1), None
        )
        if title and first_h1 and title.strip().casefold() == first_h1.strip().casefold():
            title = None
        return _ParsedDocument(title=title, blocks=tuple(blocks))

    def _walk(
        self,
        tokens: list[Token],
        lo: int,
        hi: int,
        level: int,
        lines: list[str],
        unprefixers: tuple[Callable[[list[str]], list[str]], ...],
        scope: tuple[tuple[str, int], ...],
        frames: tuple[str, ...],
        in_container: bool = False,
    ) -> list[_ScopedBlock]:
        """`lines` is the ORIGINAL source and is never copied.

        `unprefixers` holds every enclosing container's strip function,
        OUTERMOST FIRST. `_slice` applies them in that order to one token's
        mapped window, so the outer framing is gone before the inner matcher
        ever sees the line. Indices stay valid because each strip is per-line
        and never changes the line count.
        """
        blocks: list[_ScopedBlock] = []
        i = lo
        while i < hi:
            t = tokens[i]
            if t.level != level or t.map is None:
                i += 1
                continue
            start, end = t.map
            text = "".join(_slice(lines, start, end, unprefixers)).strip()
            spec = _CONTAINERS.get(t.type)
            if spec is not None:
                close = self._matching_close(tokens, i, level)
                # Frame detection reads the ancestor-normalized first line, so
                # `> > [!WARNING]` is `> [!WARNING]` here and the inner descent
                # sees `[!WARNING]` — nesting depth cannot confuse the match.
                window = _slice(lines, start, min(start + 1, end), unprefixers)
                t.meta["first_line"] = window[0] if window else ""
                frame = spec.frame(t)
                child_scope = scope + ((t.type, start),)
                child_frames = frames + ((frame,) if frame else ())
                children = self._walk(
                    tokens,
                    i + 1,
                    close,
                    level + 1,
                    lines,
                    unprefixers + (spec.unprefix,),  # appended = applied last = innermost
                    child_scope,
                    child_frames,
                    in_container=True,
                )
                blocks.append(
                    _ScopedBlock(
                        node_type="section",
                        text=text,
                        start_line=start,
                        end_line=end,
                        scope=scope,
                        frames=frames,
                        expansion=tuple(children),
                        container_type=t.type,
                        self_frame=frame,
                        always_expand=spec.always_expand,
                        expanded_fallback_frame=spec.expanded_fallback_frame,
                    )
                )
                i = close + 1
                continue
            if not text:
                i += 1
                continue
            blocks.extend(
                self._dispatch(tokens, i, text, start, end, scope, frames, in_container)
            )
            i += 1
        return blocks

    @staticmethod
    def _matching_close(tokens: list[Token], open_idx: int, level: int) -> int:
        for j in range(open_idx + 1, len(tokens)):
            if tokens[j].level == level and tokens[j].nesting == -1:
                return j
        return len(tokens)

    def _dispatch(
        self,
        tokens: list[Token],
        i: int,
        text: str,
        start: int,
        end: int,
        scope: tuple[tuple[str, int], ...],
        frames: tuple[str, ...],
        in_container: bool,
    ) -> list[_ScopedBlock]:
        t = tokens[i]
        if t.type == "front_matter" or t.type == "admonition_title_open":
            # front_matter: collected separately in parse(). admonition_title_open:
            # its text is already the frame (_admonition_frame reads t.content on
            # the admonition_open token) — emitting it too would duplicate the title.
            return []
        if t.type == "heading_open":
            if in_container:
                # Spec §4.3: a heading inside a container is CONTENT, not a
                # scope change. Emitting "heading" here would push it onto
                # _build_specs' heading stack and corrupt the breadcrumb of
                # every following top-level block.
                return [_ScopedBlock("section", text, start, end, scope=scope, frames=frames)]
            heading_title = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                heading_title = tokens[i + 1].content.strip()
            if not heading_title:
                heading_title = _ATX_RE.sub("", text).strip().rstrip("#").strip()
            level = int(t.tag[1]) if t.tag[:1] == "h" and t.tag[1:].isdigit() else 1
            return [
                _ScopedBlock(
                    "heading", heading_title, start, end, level=level, scope=scope, frames=frames
                )
            ]
        if t.type == "fence":
            return [
                _ScopedBlock(
                    "code", text, start, end, info=t.info.strip(), scope=scope, frames=frames
                )
            ]
        if t.type == "hr":
            # CommonMark defines thematic breaks as purely presentational;
            # they carry no content and must not reach chunk text.
            return []
        if t.type == "code_block":
            # Indented code. The generic `text` above is WRONG here: its
            # .strip() removes the first line's indentation while leaving
            # every other line indented, and because _split_code only runs
            # for OVERSIZED blocks, an under-budget indented block would be
            # stored in that mangled half-indented state forever. Normalize
            # to an explicit fence at parse time instead, so both the under-
            # and over-budget paths store valid code.
            # Use t.content, NOT a source slice: markdown-it already supplies
            # dedented code content for code_block tokens at top level and
            # nested inside blockquotes/admonitions alike.
            body = t.content.rstrip("\n")
            return [
                _ScopedBlock(
                    "code", render_fence(body), start, end, info="", scope=scope, frames=frames
                )
            ]
        if t.type == "html_block" and _is_noise_html(text):
            return []
        if t.type in ("bullet_list_open", "ordered_list_open"):
            return [_ScopedBlock("list", text, start, end, scope=scope, frames=frames)]
        if t.type == "table_open":
            return [_ScopedBlock("table", text, start, end, scope=scope, frames=frames)]
        # paragraphs, blockquotes (not yet a registered container), thematic
        # breaks, etc. -> "section"
        return [_ScopedBlock("section", text, start, end, scope=scope, frames=frames)]
