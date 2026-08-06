"""Markdown token interpretation for the document chunker.

Owns markdown-it configuration and token dispatch. `document.py` consumes
only `_Block` values from here and never sees a `markdown_it.Token`.
"""

import re
from dataclasses import dataclass

import markdown_it
import yaml
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


@dataclass
class _Block:
    node_type: str  # "heading" | "section" | "code" | "list" | "table"
    text: str
    start_line: int
    end_line: int
    level: int = 0  # heading level (1-6) when node_type == "heading"
    info: str = ""  # fence language when node_type == "code"


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
        self._md = markdown_it.MarkdownIt("commonmark").enable("table").use(front_matter_plugin)

    def parse(self, content: str) -> _ParsedDocument:
        tokens = self._md.parse(content)
        lines = content.splitlines(keepends=True)
        blocks: list[_Block] = []
        n = len(tokens)
        front_raw: str | None = None
        for i, t in enumerate(tokens):
            if t.level != 0 or t.map is None:
                continue
            if t.type == "front_matter":
                front_raw = t.content
                continue
            start, end = t.map
            text = "".join(lines[start:end]).strip()
            if not text:
                continue
            if t.type == "heading_open":
                heading_title = ""
                if i + 1 < n and tokens[i + 1].type == "inline":
                    heading_title = tokens[i + 1].content.strip()
                if not heading_title:
                    heading_title = _ATX_RE.sub("", text).strip().rstrip("#").strip()
                level = int(t.tag[1]) if t.tag[:1] == "h" and t.tag[1:].isdigit() else 1
                blocks.append(_Block("heading", heading_title, start, end, level=level))
            elif t.type == "fence":
                blocks.append(_Block("code", text, start, end, info=t.info.strip()))
            elif t.type == "hr":
                # CommonMark defines thematic breaks as purely presentational;
                # they carry no content and must not reach chunk text.
                continue
            elif t.type == "code_block":
                # Indented code. The generic `text` above is WRONG here: its
                # .strip() removes the first line's indentation while leaving
                # every other line indented, and because _split_code only runs
                # for OVERSIZED blocks, an under-budget indented block would be
                # stored in that mangled half-indented state forever.
                # Normalize to an explicit fence at parse time instead, so both
                # the under- and over-budget paths store valid code.
                # Use t.content, NOT a source slice: markdown-it already
                # supplies dedented code content for code_block tokens at top
                # level and nested inside blockquotes/admonitions alike. Task 5
                # moves this branch into _dispatch(), which receives neither
                # `lines` nor `unprefixers` — a slice here would reference an
                # undefined name once that move happens.
                body = t.content.rstrip("\n")
                blocks.append(_Block("code", render_fence(body), start, end, info=""))
            elif t.type == "html_block" and _is_noise_html(text):
                continue
            elif t.type in ("bullet_list_open", "ordered_list_open"):
                blocks.append(_Block("list", text, start, end))
            elif t.type == "table_open":
                blocks.append(_Block("table", text, start, end))
            else:
                # paragraphs, blockquotes, thematic breaks, etc. -> "section"
                blocks.append(_Block("section", text, start, end))
        title = _extract_title(front_raw) if front_raw else None
        # H1 de-duplication: `title: X` + `# X` must not render as "X > X".
        first_h1 = next((b.text for b in blocks if b.node_type == "heading" and b.level == 1), None)
        if title and first_h1 and title.strip().casefold() == first_h1.strip().casefold():
            title = None
        return _ParsedDocument(title=title, blocks=tuple(blocks))
