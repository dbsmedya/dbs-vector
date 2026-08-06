import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

from dbs_vector.core.models import Chunk, Document
from dbs_vector.core.ports import IContentFilter
from dbs_vector.infrastructure.chunking.markdown_blocks import (
    MarkdownBlockParser,
    _Block,
    _ParsedDocument,
    _ScopedBlock,
    choose_fence,
    render_fence,
)

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


@dataclass(frozen=True)
class _PackedFragment:
    text: str
    node_type: str
    verbatim: bool  # true for fenced and indented code


@dataclass(frozen=True)
class _PackedUnit:
    fragments: tuple[_PackedFragment, ...]
    node_type: str
    start: int
    end: int
    scope: tuple[tuple[str, int], ...]
    frames: tuple[str, ...]
    eff_target: int
    eff_max: int
    est: int  # running token ESTIMATE of the body, carried from the packing loop

    @property
    def text(self) -> str:
        return "\n\n".join(f.text for f in self.fragments)


@dataclass(frozen=True)
class _PackedSection:
    document_title: str | None
    ancestors: tuple[tuple[int, str], ...]  # (level, title), outermost first
    heading: tuple[int, str, int] | None  # level, title, source line
    units: tuple[_PackedUnit, ...]  # invariant: at least one


@dataclass
class _UnitBuilder:
    """Mutable accumulator for one in-progress unit during greedy packing.

    `_PackedUnit` is frozen (Tasks 5-8 rely on that immutability), so the
    greedy-pack loop accumulates into this instead and converts to
    `_PackedUnit` once a group's packing is finished.
    """

    fragments: list[_PackedFragment]
    node_type: str
    start: int
    end: int
    est: int  # running token ESTIMATE, same netting rule as _PackedUnit.est


_CONTEXT_INTRODUCER = "(dbs-vector context: "


def _context_marker(key: str, value: str) -> str:
    return f"{_CONTEXT_INTRODUCER}{key}={json.dumps(value, ensure_ascii=False)})"


def _render_boundary(section: _PackedSection, unit: _PackedUnit) -> str:
    """Inline structural boundary for a folded chunk.

    Fold-only by design: a standalone chunk has no cross-section join to
    disambiguate, so it keeps the plain `breadcrumb + body` shape.
    """
    lines: list[str] = []
    if section.heading is not None:
        level, title, _ = section.heading
        lines.append("#" * level + " " + title)
    parent_parts: list[str] = []
    if section.document_title:
        parent_parts.append(section.document_title)
    parent_parts.extend(t for _, t in section.ancestors)
    if parent_parts:
        lines.append(_context_marker("parent", " > ".join(parent_parts)))
    for frame in unit.frames:
        lines.append(_context_marker("frame", frame))
    return "\n".join(lines)


def _escape_collisions(unit: _PackedUnit) -> _PackedUnit:
    """Disambiguate authored lines that mimic a generated context marker.

    Read-time disambiguation only — generated markers carry zero leading
    backslashes, authored lookalikes carry at least one. Nothing decodes this;
    there is no reverse pass and none is planned.

    Verbatim fragments (fenced and re-fenced indented code) are exempt: a
    marker inside a fence is already unambiguous, and adding a backslash would
    corrupt the code sample. This repo's own docs describe the marker format
    and are in an ingested path, so the case is real, not hypothetical.
    """
    out: list[_PackedFragment] = []
    for frag in unit.fragments:
        if frag.verbatim:
            out.append(frag)
            continue
        lines = []
        for ln in frag.text.split("\n"):
            if ln.lstrip("\\").startswith(_CONTEXT_INTRODUCER):
                ln = "\\" + ln
            lines.append(ln)
        out.append(replace(frag, text="\n".join(lines)))
    return replace(unit, fragments=tuple(out))


class DocumentChunker:
    """Heading-aware, token-sized markdown chunker. Falls back to naive
    paragraph splitting for .txt files."""

    def __init__(
        self,
        *,
        max_chars: int = 1000,
        target_tokens: int = 512,
        max_tokens: int = 1024,
        length_fn: Callable[[str], int] = len,
        filters: list[IContentFilter] | None = None,
    ) -> None:
        self.max_chars = max_chars  # only used by the .txt fallback path
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
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
        parsed = self._parser.parse(document.content)
        specs = self._build_specs(parsed)
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

    def _build_specs(self, parsed: _ParsedDocument) -> list[_Spec]:
        sections: list[_PackedSection] = []
        # (level, title, start_line) — the start_line rides along so a folded
        # section's heading (Task 8) can recover its source line; nothing
        # reads it yet.
        stack: list[tuple[int, str, int]] = []
        section: list[_Block] = []

        def ancestry() -> tuple[tuple[tuple[int, str], ...], tuple[int, str, int] | None]:
            if not stack:
                return (), None
            ancestors = tuple((lvl, t) for lvl, t, _ in stack[:-1])
            lvl, t, line = stack[-1]
            return ancestors, (lvl, t, line)

        def flush() -> None:
            if section:
                ancestors, heading = ancestry()
                packed = self._pack_section(parsed.title, ancestors, heading, section)
                if packed is not None:
                    sections.append(packed)
                section.clear()

        for b in parsed.blocks:
            if b.node_type == "heading":
                flush()
                while stack and stack[-1][0] >= b.level:
                    stack.pop()
                stack.append((b.level, b.text, b.start_line))
            else:
                section.append(b)
        flush()
        return self._compose(self._fold(tuple(sections)))

    def _effective_path(
        self,
        title: str | None,
        ancestors: tuple[tuple[int, str], ...],
        heading: tuple[int, str, int] | None,
        frames: tuple[str, ...],
    ) -> str:
        """document title > ancestors > THIS section's heading > unit frames.

        The section's own heading IS part of the breadcrumb — `Guide > Setup`,
        not `Guide`. It is excluded only from the generated `parent=` marker in
        Task 7, where the ATX line already carries it.
        """
        parts: list[str] = []
        if title:
            parts.append(title)
        parts.extend(t for _, t in ancestors)
        if heading is not None:
            parts.append(heading[1])
        parts.extend(frames)
        return " > ".join(parts)

    def _prefix(
        self,
        title: str | None,
        ancestors: tuple[tuple[int, str], ...],
        heading: tuple[int, str, int] | None,
        frames: tuple[str, ...],
    ) -> str:
        """The rendered prefix — EMPTY when there is no path.

        Today's `prefix = f"{path}\\n\\n" if path else ""` must be preserved.
        Appending "\\n\\n" unconditionally would charge a true preamble (no
        title, no heading, no frames) two tokens it does not carry, shifting
        its budget and its atomic-vs-expanded selection. Every call site uses
        this helper; none builds the prefix inline.
        """
        path = self._effective_path(title, ancestors, heading, frames)
        return f"{path}\n\n" if path else ""

    def _scope_groups(self, blocks: list[_Block]) -> list[list[_Block]]:
        """Maximal contiguous runs of blocks sharing one structural scope.

        Returning to an earlier scope after an intervening container starts a
        NEW group; groups are never reopened. `_Block` (Tasks 1-4) has no
        `scope` attribute at all, so every block defaults to `()` and this
        always yields exactly one group until Task 5 introduces `_ScopedBlock`.
        """
        groups: list[list[_Block]] = []
        for b in blocks:
            scope = getattr(b, "scope", ())
            if groups and getattr(groups[-1][0], "scope", ()) == scope:
                groups[-1].append(b)
            else:
                groups.append([b])
        return groups

    def _select_form(
        self,
        b: _ScopedBlock,
        title: str | None,
        ancestors: tuple[tuple[int, str], ...],
        heading: tuple[int, str, int] | None,
    ) -> list[_ScopedBlock]:
        """Atomic-first, applied RECURSIVELY to reachable expansions.

        Measure the ATOMIC prefix and raw body against max_tokens. Only on
        failure select the expansion — and each wrapped child of that expansion
        gets the same treatment, so an oversized quote inside an oversized quote
        expands twice while a small one inside an oversized one stays atomic.
        Never use the expanded form's (longer) prefix to decide atomic fit.
        """
        # ORDER MATTERS: always_expand is checked FIRST, before the
        # empty-expansion guard. An empty admonition has no expansion, so
        # leading with `if not b.expansion: return [b]` would hand back its raw
        # `!!! note "x"` source. _flatten_always_expand catches that at the top
        # level, but never runs on children uncovered by an expanded blockquote.
        if b.always_expand:
            return [
                out_child
                for child in b.expansion
                for out_child in self._select_form(child, title, ancestors, heading)
            ]  # empty expansion -> [] : the container disappears, as it should

        if not b.expansion:
            return [b]

        atomic_frames = b.frames + ((b.self_frame,) if b.self_frame else ())
        atomic_prefix = self._prefix(title, ancestors, heading, atomic_frames)
        if self._len(atomic_prefix + b.text) <= self.max_tokens:
            # A FRAMED atomic container enters its own scope so its frame cannot
            # leak onto adjacent prose. An ORDINARY one does not, preserving
            # today's ability to pack it with neighbours.
            scope = b.scope + (((b.container_type, b.start_line),) if b.self_frame else ())
            return [replace(b, frames=atomic_frames, scope=scope, expansion=())]

        # Expanded. Children already carry the inherited frame stack; when this
        # container had no frame of its own, apply its fallback so an expanded
        # ordinary quote keeps its quotation semantics after `>` removal.
        out: list[_ScopedBlock] = []
        for child in b.expansion:
            c = child
            if b.self_frame is None and b.expanded_fallback_frame:
                c = replace(c, frames=c.frames + (b.expanded_fallback_frame,))
            out.extend(self._select_form(c, title, ancestors, heading))
        return out

    def _pack_section(
        self,
        title: str | None,
        ancestors: tuple[tuple[int, str], ...],
        heading: tuple[int, str, int] | None,
        blocks: list[_Block],
    ) -> _PackedSection | None:
        # Form selection happens BEFORE filtering: a filter must see whichever
        # form (atomic or expanded) packing will actually use, so a dropped
        # block inside an expanded container (e.g. a compressed_json fence
        # inside a blockquote) is caught too — filtering the unexpanded
        # wrapper would never see its descended children.
        selected = [out for b in blocks for out in self._select_form(b, title, ancestors, heading)]
        chosen = [
            b
            for b in selected
            if not any(
                f.should_drop_block(b.text, b.info if b.node_type == "code" else None)
                for f in self._filters
            )
        ]
        if not chosen:
            return None
        units: list[_PackedUnit] = []
        for group in self._scope_groups(chosen):
            units.extend(self._pack_group(title, ancestors, heading, group))
        if not units:
            return None
        return _PackedSection(
            document_title=title, ancestors=ancestors, heading=heading, units=tuple(units)
        )

    def _pack_group(
        self,
        title: str | None,
        ancestors: tuple[tuple[int, str], ...],
        heading: tuple[int, str, int] | None,
        group: list[_Block],
    ) -> list[_PackedUnit]:
        # The size invariant is on the FINAL Chunk.text (heading path + any
        # re-fence labels included). Reserve the per-chunk prefix cost up front
        # so packing/splitting target the *rendered* size, and apply a
        # char-window safety net at compose time as an absolute guarantee.
        scope = getattr(group[0], "scope", ())
        frames = getattr(group[0], "frames", ())
        plen = self._len(self._prefix(title, ancestors, heading, frames))
        eff_target = max(1, self.target_tokens - plen)
        eff_max = max(1, self.max_tokens - plen)

        # 1) expand oversized blocks into <= eff_max fragments (body-only
        #    sizing), carrying each fragment's measured length so step 2 never
        #    re-tokenizes it. Split pieces ARE re-measured: _split_code/
        #    _split_table add fence/header wrappers AFTER packing, so lengths
        #    measured inside _pack_atoms don't survive the wrapping.
        frags: list[
            tuple[str, str, int, int, int, bool]
        ] = []  # text, ntype, start, end, tok, verbatim
        for b in group:
            blen = self._len(b.text)
            verbatim = b.node_type == "code"
            if blen <= eff_max:
                frags.append((b.text, b.node_type, b.start_line, b.end_line, blen, verbatim))
            else:
                for piece in self._split_block(b, eff_target, eff_max):
                    frags.append(
                        (piece, b.node_type, b.start_line, b.end_line, self._len(piece), verbatim)
                    )

        # 2) greedy pack to eff_target, using a RUNNING TOKEN ESTIMATE instead
        #    of re-tokenizing the growing candidate each step. Tokenization is
        #    not additive: EVERY measurement includes the tokenizer's special
        #    tokens (BOS/EOS), which the joined text pays exactly once — so the
        #    first fragment keeps its full count and every ADDED fragment (and
        #    the joiner) is netted by _overhead(). Estimate drift is corrected
        #    exactly at compose time (emit-time re-measure + char-window),
        #    which preserves the hard <= max_tokens guarantee.
        # "\n\n" matches _PackedUnit.text's fragment join.
        jcost = self._joiner_cost("\n\n")
        ov = self._overhead()
        packed: list[_UnitBuilder] = []
        for text, ntype, start, end, tlen, verbatim in frags:
            frag = _PackedFragment(text, ntype, verbatim)
            if packed:
                cur = packed[-1]
                est = cur.est + jcost + max(0, tlen - ov)
                if est <= eff_target:
                    cur.fragments.append(frag)
                    cur.end = end
                    if cur.node_type != ntype:
                        cur.node_type = "section"
                    cur.est = est
                    continue
            packed.append(_UnitBuilder([frag], ntype, start, end, est=tlen))

        # A section-local tiny-merge used to run here (step 3). It is gone:
        # `_fold` (called between packing and composition) replaces it with a
        # forward fold across WHOLE sections, which is the only shape #2
        # actually needed — a section-local merge could not help an entire
        # undersized section with nothing to merge into within its own group.
        return [
            _PackedUnit(
                fragments=tuple(u.fragments),
                node_type=u.node_type,
                start=u.start,
                end=u.end,
                scope=scope,
                frames=frames,
                eff_target=eff_target,
                eff_max=eff_max,
                est=u.est,
            )
            for u in packed
        ]

    def _folded_body(self, absorbed: _PackedSection, receiving: _PackedSection) -> str:
        """Body only — NO breadcrumb prefix. _compose adds that later, once."""
        return "\n\n".join(
            part
            for part in (
                _render_boundary(absorbed, absorbed.units[0]),
                _escape_collisions(absorbed.units[0]).text,
                _render_boundary(receiving, receiving.units[0]),
                _escape_collisions(receiving.units[0]).text,
            )
            if part
        )

    def _replace_first_unit(
        self, receiving: _PackedSection, body: str, absorbed: _PackedSection
    ) -> _PackedSection:
        recv_unit = receiving.units[0]
        start = absorbed.heading[2] if absorbed.heading is not None else absorbed.units[0].start
        new_unit = _PackedUnit(
            fragments=(_PackedFragment(text=body, node_type="section", verbatim=False),),
            node_type="section",  # the merge spans two sections; a finer label would be a lie
            start=start,
            end=recv_unit.end,
            scope=recv_unit.scope,
            frames=recv_unit.frames,  # the chunk renders under the receiving breadcrumb
            eff_target=recv_unit.eff_target,
            eff_max=recv_unit.eff_max,
            est=self._len(body),  # exact, since it was just measured
        )
        return replace(receiving, units=(new_unit, *receiving.units[1:]))

    def _fold(self, sections: tuple[_PackedSection, ...]) -> tuple[_PackedSection, ...]:
        """Fold a wholly-undersized section forward into the next one.

        Option A: eligibility is computed ONCE from the original sections and
        each section participates in at most one fold (absorber or absorbed,
        never both), so three tiny sections yield A+B and C — never A+B+C, and
        the result never depends on evaluation order.
        """
        eligible = [
            i
            for i, s in enumerate(sections)
            if len(s.units) == 1 and s.units[0].est < s.units[0].eff_target
        ]
        spent: set[int] = set()
        merges: dict[int, str] = {}  # receiving index -> merged BODY
        dropped: set[int] = set()  # absorbed indices
        for i in eligible:
            j = i + 1
            if i in spent or j >= len(sections) or j in spent:
                continue
            recv = sections[j]
            body = self._folded_body(sections[i], recv)
            prefix = self._prefix(
                recv.document_title, recv.ancestors, recv.heading, recv.units[0].frames
            )
            # Exact measurement of the FINISHED text, never an estimate: the
            # only correction available downstream is the char-window, which
            # would fragment exactly what the fold set out to keep together.
            if self._len(prefix + body) > self.max_tokens:
                continue
            merges[j] = body
            dropped.add(i)
            spent.update({i, j})

        out: list[_PackedSection] = []
        for k, s in enumerate(sections):
            if k in dropped:
                continue
            if k in merges:
                s = self._replace_first_unit(s, merges[k], sections[k - 1])
            out.append(s)
        return tuple(out)

    def _compose(self, sections: tuple[_PackedSection, ...]) -> list[_Spec]:
        # 4) compose final text (prefix once per chunk); 1-based inclusive line
        #    range; safety-net char-window guarantees len(text) <= max_tokens
        #    even if overhead accounting under-reserved. A folded unit's body
        #    is already exact-measured (see _fold), but this stays as the
        #    absolute guarantee for every other path.
        out: list[_Spec] = []
        for sec in sections:
            for u in sec.units:
                path = self._effective_path(
                    sec.document_title, sec.ancestors, sec.heading, u.frames
                )
                prefix = self._prefix(sec.document_title, sec.ancestors, sec.heading, u.frames)
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
        body = "\n".join(inner)
        delim = choose_fence(body, b.info)
        # Upper bound on part-count digits. NOT len(inner): _pack_atoms splits
        # one oversized atom into many char-windowed parts, so a single long
        # line can produce hundreds of parts from one atom. Every part holds at
        # least one character, so total characters IS a sound bound.
        digits = len(str(max(1, len(body))))
        widest = "9" * digits
        overhead = self._len(f"(code, part {widest}/{widest})\n{delim}{b.info}\n\n{delim}")
        bt = max(1, target - overhead)
        bm = max(1, max_ - overhead)
        # `inner` is still the atom list _pack_atoms consumes; only the bound
        # above is computed from the joined body.
        parts = self._pack_atoms(inner, "\n", bt, bm)
        m = len(parts)
        if m <= 1:
            return [render_fence(parts[0] if parts else "", b.info, delim)]
        return [
            f"(code, part {k}/{m})\n" + render_fence(p, b.info, delim)
            for k, p in enumerate(parts, 1)
        ]

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
