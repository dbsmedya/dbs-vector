import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from dbs_vector.core.models import Chunk, Document
from dbs_vector.core.ports import IContentFilter
from dbs_vector.infrastructure.chunking.markdown_blocks import MarkdownBlockParser, _Block

_LIST_MARKER = re.compile(r"^(\s*)([-*+]|\d+[.)])\s")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _chunk_content_hash(file_hash: str, index: int) -> str:
    """Per-chunk content hash for a document chunk.

    ``content_hash`` carries two jobs at once and they must not collide:

    * Chunk 0 keeps the RAW file hash so ``IngestionService``'s file-level
      short-circuit (``if file_hash in existing_hashes``) still recognises an
      already-indexed file and skips it on re-ingest.
    * Chunks 1..N get a suffixed hash so the chunk-level dedup (which also keys
      on ``content_hash``) does NOT treat them as duplicates of chunk 0 and
      collapse a multi-chunk file down to its first chunk.

    Stable across runs: identical file content -> identical ``file_hash`` ->
    identical per-chunk hashes, so an unchanged file still re-dedups cleanly.
    """
    return file_hash if index == 0 else f"{file_hash}_{index}"


@dataclass
class _Spec:
    text: str
    node_type: str
    parent_scope: str | None
    line_range: str


@dataclass
class _PackedUnit:
    text: str
    node_type: str
    start: int
    end: int
    est: int = 0  # running token ESTIMATE for text (see _emit_section step 2)


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
        self.max_chars = max_chars  # only used by the .txt fallback path
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        if target_tokens > max_tokens:
            raise ValueError(
                f"target_tokens ({target_tokens}) must not exceed max_tokens ({max_tokens})"
            )
        self._len = length_fn
        self._filters = list(filters) if filters else []
        self._parser = MarkdownBlockParser()
        # Lazy caches for the running-estimate math; length_fn may be a full
        # tokenizer pass under the MLX model lock, so constants are measured
        # at most once per instance.
        self._special_overhead: int | None = None
        self._jcost_cache: dict[str, int] = {}

    def _overhead(self) -> int:
        """Token cost of the EMPTY string — the special tokens (BOS/EOS) the
        production tokenizer adds to every measurement. The joined text pays
        this cost once, not once per atom, so running sums deduct it from
        every measurement after the first (0 for plain `len`)."""
        if self._special_overhead is None:
            self._special_overhead = self._len("")
        return self._special_overhead

    def _joiner_cost(self, joiner: str) -> int:
        """Net token cost of a joiner string (special-token overhead deducted),
        memoized — joiners are a handful of constants."""
        cost = self._jcost_cache.get(joiner)
        if cost is None:
            cost = max(0, self._len(joiner) - self._overhead())
            self._jcost_cache[joiner] = cost
        return cost

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
        blocks = self._parser.parse(document.content)
        specs = self._build_specs(blocks)
        for i, s in enumerate(specs):
            yield Chunk(
                id=f"{document.filepath}_chunk_{i}",
                text=s.text,
                source=document.filepath,
                content_hash=_chunk_content_hash(document.content_hash, i),
                node_type=s.node_type,
                parent_scope=s.parent_scope,
                line_range=s.line_range,
            )

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

        # 1) expand oversized blocks into <= eff_max units (body-only sizing),
        #    carrying each unit's measured length so step 2 never re-tokenizes
        #    it. Split pieces ARE re-measured: _split_code/_split_table add
        #    fence/header wrappers AFTER packing, so lengths measured inside
        #    _pack_atoms don't survive the wrapping.
        units: list[tuple[str, str, int, int, int]] = []  # text, ntype, start, end, tokens
        for b in blocks:
            blen = self._len(b.text)
            if blen <= eff_max:
                units.append((b.text, b.node_type, b.start_line, b.end_line, blen))
            else:
                for piece in self._split_block(b, eff_target, eff_max):
                    units.append((piece, b.node_type, b.start_line, b.end_line, self._len(piece)))

        # 2) greedy pack to eff_target, using a RUNNING TOKEN ESTIMATE instead
        #    of re-tokenizing the growing candidate each step. Tokenization is
        #    not additive: EVERY measurement includes the tokenizer's special
        #    tokens (BOS/EOS), which the joined text pays exactly once — so the
        #    first unit keeps its full count and every ADDED unit (and the
        #    joiner) is netted by _overhead(). Estimate drift is corrected
        #    exactly in step 4 (emit-time re-measure + char-window), which
        #    preserves the hard <= max_tokens guarantee.
        # "\n\n" matches the `cur.text + "\n\n" + text` concat in steps 2/3.
        jcost = self._joiner_cost("\n\n")
        ov = self._overhead()
        packed: list[_PackedUnit] = []
        for text, ntype, start, end, tlen in units:
            if packed:
                cur = packed[-1]
                est = cur.est + jcost + max(0, tlen - ov)
                if est <= eff_target:
                    cur.text = cur.text + "\n\n" + text
                    cur.end = end
                    if cur.node_type != ntype:
                        cur.node_type = "section"
                    cur.est = est
                    continue
            packed.append(_PackedUnit(text, ntype, start, end, est=tlen))

        # 3) tiny-merge: a chunk whose ESTIMATE is below min_tokens folds into
        #    the previous one (same section) — but ONLY if the merged estimate
        #    still fits eff_max, so the merge never forces a later char-window
        #    re-split (which would defeat the merge by fragmenting the combined
        #    content).
        merged: list[_PackedUnit] = []
        for item in packed:
            if merged and item.est < self.min_tokens:
                p = merged[-1]
                cand_est = p.est + jcost + max(0, item.est - ov)
                if cand_est <= eff_max:
                    p.text = p.text + "\n\n" + item.text
                    p.node_type = "section"
                    p.end = item.end
                    p.est = cand_est
                    continue
            merged.append(item)

        # 4) compose final text (prefix once per chunk); 1-based inclusive line
        #    range; safety-net char-window guarantees len(text) <= max_tokens
        #    even if overhead accounting under-reserved (e.g. 3-digit part count)
        #    or a tiny-merge edge case slips through.
        out: list[_Spec] = []
        for u in merged:
            rng = f"{u.start + 1}-{u.end}"  # markdown-it map is 0-based, end-exclusive
            body = prefix + u.text if prefix else u.text
            if self._len(body) <= self.max_tokens:
                out.append(_Spec(body, u.node_type, path or None, rng))
            else:
                for window in self._char_window(body, self.max_tokens):
                    out.append(_Spec(window, u.node_type, path or None, rng))
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
        first = lines[0].lstrip()
        if len(lines) >= 2 and (first.startswith("```") or first.startswith("~~~")):
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
        # RUNNING-SUM estimate with special-token netting — see _emit_section
        # step 2 for the full rationale. The hard <= max_ guarantee is
        # preserved by the per-atom _char_window split for oversized atoms and
        # (for full chunks) by step 4's emit-time re-measure.
        out: list[str] = []
        cur = ""
        cur_est = 0
        jcost = self._joiner_cost(joiner)
        ov = self._overhead()
        for a in atoms:
            alen = self._len(a)
            if alen <= max_:
                pieces: list[tuple[str, int]] = [(a, alen)]
            else:
                pieces = [(w, self._len(w)) for w in self._char_window(a, max_)]
            for p, plen in pieces:
                if not cur:
                    cur, cur_est = p, plen
                    continue
                est = cur_est + jcost + max(0, plen - ov)
                if est > target:
                    out.append(cur)
                    cur, cur_est = p, plen
                else:
                    cur = cur + joiner + p
                    cur_est = est
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
                content_hash=_chunk_content_hash(document.content_hash, i),
            )
