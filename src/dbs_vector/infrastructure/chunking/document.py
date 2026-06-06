import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import markdown_it

from dbs_vector.core.models import Chunk, Document
from dbs_vector.core.ports import IContentFilter

_ATX_RE = re.compile(r"^#{1,6}\s*")
_LIST_MARKER = re.compile(r"^(\s*)([-*+]|\d+[.)])\s")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Block:
    node_type: str  # "heading" | "section" | "code" | "list" | "table"
    text: str
    start_line: int
    end_line: int
    level: int = 0  # heading level (1-6) when node_type == "heading"
    info: str = ""  # fence language when node_type == "code"


@dataclass
class _Spec:
    text: str
    node_type: str
    parent_scope: str | None
    line_range: str


class DocumentChunker:
    """Heading-aware, token-sized markdown chunker. Falls back to naive
    paragraph splitting for .txt files."""

    def __init__(
        self,
        *,
        max_chars: int = 1000,
        target_tokens: int = 512,
        max_tokens: int = 1024,
        min_tokens: int = 32,
        length_fn: Callable[[str], int] = len,
        filters: list[IContentFilter] | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self._len = length_fn
        self._filters = list(filters) if filters else []
        self._md = markdown_it.MarkdownIt("commonmark").enable("table")

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".txt"]

    def process(self, document: Document) -> Iterator[Chunk]:
        if document.filepath.lower().endswith(".md"):
            yield from self._chunk_markdown(document)
        else:
            yield from self._chunk_text(document)

    # ---- markdown path -------------------------------------------------

    def _chunk_markdown(self, document: Document) -> Iterator[Chunk]:
        if any(f.should_skip_file(document.filepath, document.content) for f in self._filters):
            return
        blocks = self._parse_blocks(document.content)
        specs = self._build_specs(blocks)
        for i, s in enumerate(specs):
            yield Chunk(
                id=f"{document.filepath}_chunk_{i}",
                text=s.text,
                source=document.filepath,
                content_hash=document.content_hash,
                node_type=s.node_type,
                parent_scope=s.parent_scope,
                line_range=s.line_range,
            )

    def _parse_blocks(self, content: str) -> list[_Block]:
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

    def _build_specs(self, blocks: list[_Block]) -> list[_Spec]:
        specs: list[_Spec] = []
        stack: list[tuple[int, str]] = []
        section: list[_Block] = []

        def path() -> str:
            return " > ".join(title for _, title in stack)

        def flush() -> None:
            if section:
                specs.extend(self._emit_section(path(), section))
                section.clear()

        for b in blocks:
            if b.node_type == "heading":
                flush()
                while stack and stack[-1][0] >= b.level:
                    stack.pop()
                stack.append((b.level, b.text))
            else:
                info = b.info if b.node_type == "code" else None
                if any(f.should_drop_block(b.text, info) for f in self._filters):
                    continue
                section.append(b)
        flush()
        return specs

    def _emit_section(self, path: str, blocks: list[_Block]) -> list[_Spec]:
        # The size invariant is on the FINAL Chunk.text (heading path + any
        # re-fence labels included). Reserve the per-chunk prefix cost up front
        # so packing/splitting target the *rendered* size, and apply a
        # char-window safety net at the end as an absolute guarantee.
        prefix = f"{path}\n\n" if path else ""
        plen = self._len(prefix)
        eff_target = max(1, self.target_tokens - plen)
        eff_max = max(1, self.max_tokens - plen)

        # 1) expand oversized blocks into <= eff_max units (body-only sizing)
        units: list[tuple[str, str, int, int]] = []  # text, node_type, start, end
        for b in blocks:
            if self._len(b.text) <= eff_max:
                units.append((b.text, b.node_type, b.start_line, b.end_line))
            else:
                for piece in self._split_block(b, eff_target, eff_max):
                    units.append((piece, b.node_type, b.start_line, b.end_line))

        # 2) greedy pack to eff_target
        packed: list[list[Any]] = []
        for text, ntype, start, end in units:
            if packed:
                cur = packed[-1]
                cand = str(cur[0]) + "\n\n" + str(text)
                if self._len(cand) <= eff_target:
                    cur[0] = cand
                    cur[3] = end
                    if cur[1] != ntype:
                        cur[1] = "section"
                    continue
            packed.append([text, ntype, start, end])

        # 3) tiny-merge: a chunk below min_tokens folds into previous (same section)
        merged: list[list[Any]] = []
        for item in packed:
            if merged and self._len(str(item[0])) < self.min_tokens:
                p = merged[-1]
                p[0] = str(p[0]) + "\n\n" + str(item[0])
                p[1] = "section"
                p[3] = item[3]
            else:
                merged.append(item)

        # 4) compose final text (prefix once per chunk); 1-based inclusive line
        #    range; safety-net char-window guarantees len(text) <= max_tokens
        #    even if overhead accounting under-reserved (e.g. 3-digit part count).
        out: list[_Spec] = []
        for text, ntype, start, end in merged:
            rng = f"{int(start) + 1}-{end}"  # markdown-it map is 0-based, end-exclusive
            body = prefix + str(text) if prefix else str(text)
            if self._len(body) <= self.max_tokens:
                out.append(_Spec(body, str(ntype), path or None, rng))
            else:
                for window in self._char_window(body, self.max_tokens):
                    out.append(_Spec(window, str(ntype), path or None, rng))
        return out

    # ---- oversized-block splitting -------------------------------------

    def _split_block(self, b: _Block, target: int, max_: int) -> list[str]:
        if b.node_type == "code":
            return self._split_code(b, target, max_)
        if b.node_type == "table":
            return self._split_table(b, target, max_)
        if b.node_type == "list":
            return self._pack_atoms(self._list_items(b.text), "\n", target, max_)
        return self._pack_atoms(_SENTENCE.split(b.text), " ", target, max_)

    def _split_code(self, b: _Block, target: int, max_: int) -> list[str]:
        lines = b.text.split("\n")
        if len(lines) >= 2 and lines[0].lstrip().startswith("```"):
            inner = lines[1:-1]
        else:
            inner = lines
        # Reserve the fence + part-marker overhead so each WRAPPED piece fits.
        overhead = self._len(f"(code, part 99/99)\n```{b.info}\n\n```")
        bt = max(1, target - overhead)
        bm = max(1, max_ - overhead)
        parts = self._pack_atoms(inner, "\n", bt, bm)
        m = len(parts)
        if m <= 1:
            return [f"```{b.info}\n{parts[0] if parts else ''}\n```"]
        return [f"(code, part {k}/{m})\n```{b.info}\n{p}\n```" for k, p in enumerate(parts, 1)]

    def _split_table(self, b: _Block, target: int, max_: int) -> list[str]:
        rows = [ln for ln in b.text.split("\n") if ln.strip()]
        if len(rows) <= 2:
            return [b.text]
        header = "\n".join(rows[:2])
        # Reserve the repeated header so each part fits.
        hlen = self._len(header + "\n")
        bt = max(1, target - hlen)
        bm = max(1, max_ - hlen)
        groups = self._pack_atoms(rows[2:], "\n", bt, bm)
        return [f"{header}\n{g}" for g in groups]

    def _list_items(self, text: str) -> list[str]:
        base: int | None = None
        items: list[str] = []
        cur: list[str] = []
        for ln in text.split("\n"):
            m = _LIST_MARKER.match(ln)
            if m and (base is None or len(m.group(1)) <= base):
                base = len(m.group(1)) if base is None else base
                if cur:
                    items.append("\n".join(cur))
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            items.append("\n".join(cur))
        return items

    def _pack_atoms(self, atoms: list[str], joiner: str, target: int, max_: int) -> list[str]:
        out: list[str] = []
        cur = ""
        for a in atoms:
            pieces = [a] if self._len(a) <= max_ else self._char_window(a, max_)
            for p in pieces:
                cand = p if not cur else cur + joiner + p
                if cur and self._len(cand) > target:
                    out.append(cur)
                    cur = p
                else:
                    cur = cand
        if cur:
            out.append(cur)
        return out

    def _char_window(self, text: str, max_: int) -> list[str]:
        """Adaptive CHARACTER window whose size is measured by the injected
        tokenizer (`length_fn`): grow on character indices until adding more
        would exceed `max_` tokens, then flush. This is NOT a true token
        encode/decode split (it slices characters, not token ids), but it
        guarantees every window measures <= max_ tokens — the truncation
        safety net for atoms with no internal boundary (e.g. a one-line
        compressed-json blob)."""
        windows: list[str] = []
        i, n = 0, len(text)
        step = max(1, max_)
        while i < n:
            j = min(n, i + step)
            while j < n and self._len(text[i:j]) <= max_:
                j = min(n, j + step)
            while j > i + 1 and self._len(text[i:j]) > max_:
                j -= max(1, (j - i) // 4)
            j = max(i + 1, j)
            windows.append(text[i:j])
            i = j
        return windows

    # ---- .txt fallback (unchanged behavior) ----------------------------

    def _chunk_text(self, document: Document) -> Iterator[Chunk]:
        paragraphs = document.content.split("\n\n")
        chunks_text: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) > self.max_chars and current:
                chunks_text.append(current.strip())
                current = paragraph
            else:
                current = current + "\n\n" + paragraph if current else paragraph
        if current.strip():
            chunks_text.append(current.strip())
        valid = [t for t in chunks_text if len(t.strip()) >= 5]
        for i, text in enumerate(valid):
            yield Chunk(
                id=f"{document.filepath}_chunk_{i}",
                text=text,
                source=document.filepath,
                content_hash=document.content_hash,
            )
