"""Markdown token interpretation for the document chunker.

Owns markdown-it configuration and token dispatch. `document.py` consumes
only `_Block` values from here and never sees a `markdown_it.Token`.
"""

import re
from dataclasses import dataclass

import markdown_it

_ATX_RE = re.compile(r"^#{1,6}\s*")
# NOTE: _LIST_MARKER stays in document.py — it is used only by _list_items(),
# which is part of the oversized-block splitter and does NOT move.


@dataclass
class _Block:
    node_type: str  # "heading" | "section" | "code" | "list" | "table"
    text: str
    start_line: int
    end_line: int
    level: int = 0  # heading level (1-6) when node_type == "heading"
    info: str = ""  # fence language when node_type == "code"


class MarkdownBlockParser:
    """Parses markdown source into flat semantic blocks."""

    def __init__(self) -> None:
        self._md = markdown_it.MarkdownIt("commonmark").enable("table")

    def parse(self, content: str) -> list[_Block]:
        tokens = self._md.parse(content)
        lines = content.splitlines(keepends=True)
        blocks: list[_Block] = []
        n = len(tokens)
        for i, t in enumerate(tokens):
            if t.level != 0 or t.map is None:
                continue
            start, end = t.map
            text = "".join(lines[start:end]).strip()
            if not text:
                continue
            if t.type == "heading_open":
                title = ""
                if i + 1 < n and tokens[i + 1].type == "inline":
                    title = tokens[i + 1].content.strip()
                if not title:
                    title = _ATX_RE.sub("", text).strip().rstrip("#").strip()
                level = int(t.tag[1]) if t.tag[:1] == "h" and t.tag[1:].isdigit() else 1
                blocks.append(_Block("heading", title, start, end, level=level))
            elif t.type == "fence":
                blocks.append(_Block("code", text, start, end, info=t.info.strip()))
            elif t.type in ("bullet_list_open", "ordered_list_open"):
                blocks.append(_Block("list", text, start, end))
            elif t.type == "table_open":
                blocks.append(_Block("table", text, start, end))
            else:
                # paragraphs, blockquotes, thematic breaks, etc. -> "section"
                blocks.append(_Block("section", text, start, end))
        return blocks
