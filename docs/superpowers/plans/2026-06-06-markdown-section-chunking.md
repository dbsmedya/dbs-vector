# Header-Aware Section Chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `DocumentChunker`'s markdown path with a heading-aware, section-based, token-sized chunker that prepends heading context, splits oversized blocks at natural boundaries (never truncates), and supports per-engine pluggable content-exclusion filters.

**Architecture:** Markdown is parsed with `markdown-it-py` into top-level blocks tagged with a heading path. Blocks are grouped per heading section, packed to a token target, and oversized blocks are split by code-line / list-item / table-row / sentence with a hard token-window fallback. Token sizing uses an injected `length_fn` (the embedder's tokenizer, wired in `bootstrap.py`). Exclusion filters are a small open/closed registry resolved per engine.

**Tech stack:** Python 3.12, `markdown-it-py`, Pydantic v2 (`pydantic-settings`), MLX embeddings, LanceDB, pytest, `uv`/`poe`.

**Spec:** `docs/superpowers/specs/2026-06-06-markdown-section-chunking-design.md`

---

## File structure

**Create**
- `src/dbs_vector/infrastructure/chunking/filters.py` — `IContentFilter` impls, `FilterRegistry`, built-ins `excalidraw`, `compressed_json`.
- `tests/unit/test_filters.py` — filter + registry tests.

**Modify**
- `src/dbs_vector/core/ports.py` — add `IContentFilter` protocol; add `count_tokens` to `IEmbedder`.
- `src/dbs_vector/infrastructure/embeddings/mlx_engine.py` — add `MLXEmbedder.count_tokens`.
- `src/dbs_vector/infrastructure/chunking/document.py` — rewrite markdown path (section-based, token-sized, filter-aware); keep `.txt` path.
- `src/dbs_vector/config.py` — `TuningProfile` gains `chunk_target_tokens`/`chunk_max_tokens`; `EngineConfig` gains `exclusion_filters`; drop `chunk_max_chars` arg from `chunker_kwargs`; add validation; extend allowed profile keys.
- `src/dbs_vector/services/bootstrap.py` — build document chunker with `length_fn`/`filters`/token budgets (gated on `chunker_type == "document"`).
- `config.yaml` — md/md-granite profiles + engines.
- `tests/unit/test_chunker.py` — replace markdown-specific tests with new-behavior tests; keep `.txt` + id tests.
- `tests/unit/test_mlx_engine.py` — add `count_tokens` test.
- `tests/unit/test_bootstrap.py` — add SQL-unaffected guard test.
- `docs/README_PROFILES.md` — document new knobs + filters.

**Invariants the implementation MUST preserve (SQL untouched — spec §6a):**
- `chunker_kwargs` keeps `duckdb`/`api` early-return branches; token/filter injection lives only in the `document` branch of `bootstrap.py`.
- New profile/engine fields are optional with inert defaults (`0` / `[]`).
- `MLXEmbedder.embed_batch` is unchanged.

---

## Task 1: Add `count_tokens` to the embedder

**Files:**
- Modify: `src/dbs_vector/core/ports.py`
- Modify: `src/dbs_vector/infrastructure/embeddings/mlx_engine.py`
- Test: `tests/unit/test_mlx_engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_mlx_engine.py`:

```python
def test_count_tokens_returns_input_id_length(mock_load):
    _, _, mock_tokenizer = mock_load
    mock_tokenizer._tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

    embedder = MLXEmbedder(
        model_name="m", max_token_length=2048, dimension=768
    )
    assert embedder.count_tokens("hello world") == 5
    # add_special_tokens must be requested so the count matches embedding time
    _, kwargs = mock_tokenizer._tokenizer.call_args
    assert kwargs.get("add_special_tokens") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_mlx_engine.py::test_count_tokens_returns_input_id_length -v`
Expected: FAIL — `AttributeError: 'MLXEmbedder' object has no attribute 'count_tokens'`.

- [ ] **Step 3: Add `count_tokens` to the protocol**

In `src/dbs_vector/core/ports.py`, inside the `IEmbedder` Protocol, add:

```python
    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens (incl. special tokens) for `text`."""
        ...
```

- [ ] **Step 4: Implement on `MLXEmbedder`**

In `src/dbs_vector/infrastructure/embeddings/mlx_engine.py`, add a method to the class (after `dimension`):

```python
    def count_tokens(self, text: str) -> int:
        """Token count for a single string, using the same tokenizer (and
        special tokens) the model embeds with."""
        with self._lock:
            ids = self.tokenizer._tokenizer(text, add_special_tokens=True)["input_ids"]
        return len(ids)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_mlx_engine.py::test_count_tokens_returns_input_id_length -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/core/ports.py src/dbs_vector/infrastructure/embeddings/mlx_engine.py tests/unit/test_mlx_engine.py
git commit -m "feat(embeddings): add count_tokens for token-aware chunking"
```

---

## Task 2: Exclusion-filter module + registry

**Files:**
- Modify: `src/dbs_vector/core/ports.py`
- Create: `src/dbs_vector/infrastructure/chunking/filters.py`
- Test: `tests/unit/test_filters.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_filters.py`:

```python
import pytest

from dbs_vector.infrastructure.chunking.filters import FilterRegistry


def test_resolve_empty_returns_no_filters():
    assert FilterRegistry.resolve([]) == []


def test_unknown_filter_name_raises():
    with pytest.raises(ValueError, match="Unknown exclusion filter"):
        FilterRegistry.resolve(["does_not_exist"])


def test_excalidraw_skips_excalidraw_files():
    flt = FilterRegistry.resolve(["excalidraw"])[0]
    assert flt.should_skip_file("notes/Diagram.excalidraw.md", "anything") is True
    assert flt.should_skip_file("notes/Real.md", "# Title") is False


def test_excalidraw_skips_files_with_plugin_frontmatter():
    flt = FilterRegistry.resolve(["excalidraw"])[0]
    content = "---\nexcalidraw-plugin: parsed\n---\n# Drawing\n"
    assert flt.should_skip_file("notes/Real.md", content) is True


def test_excalidraw_drops_excalidraw_json_block():
    flt = FilterRegistry.resolve(["excalidraw"])[0]
    block = '{\n  "type": "excalidraw",\n  "version": 2\n}'
    assert flt.should_drop_block(block, "json") is True
    assert flt.should_drop_block("normal text", None) is False


def test_compressed_json_drops_by_info_string():
    flt = FilterRegistry.resolve(["compressed_json"])[0]
    assert flt.should_drop_block("N4KAkARAL...", "compressed-json") is True
    assert flt.should_drop_block("print('hi')", "python") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_filters.py -v`
Expected: FAIL — `ModuleNotFoundError: ... filters`.

- [ ] **Step 3: Add the protocol to ports**

In `src/dbs_vector/core/ports.py` add:

```python
class IContentFilter(Protocol):
    """Decides whether to skip a whole file or drop a single block."""

    name: str

    def should_skip_file(self, filepath: str, content: str) -> bool: ...

    def should_drop_block(self, text: str, info_string: str | None) -> bool: ...
```

- [ ] **Step 4: Implement the filter module**

Create `src/dbs_vector/infrastructure/chunking/filters.py`:

```python
from dbs_vector.core.ports import IContentFilter


class ExcalidrawFilter:
    name = "excalidraw"

    def should_skip_file(self, filepath: str, content: str) -> bool:
        if filepath.lower().endswith(".excalidraw.md"):
            return True
        head = content[:500]
        return "excalidraw-plugin" in head

    def should_drop_block(self, text: str, info_string: str | None) -> bool:
        if (info_string or "").strip().lower() == "json":
            return '"type": "excalidraw"' in text or '"type":"excalidraw"' in text
        return False


class CompressedJsonFilter:
    name = "compressed_json"

    def should_skip_file(self, filepath: str, content: str) -> bool:
        return False

    def should_drop_block(self, text: str, info_string: str | None) -> bool:
        return (info_string or "").strip().lower() == "compressed-json"


class FilterRegistry:
    """Open/closed registry of named content filters (cf. ModelRegistry)."""

    _filters: dict[str, IContentFilter] = {
        ExcalidrawFilter.name: ExcalidrawFilter(),
        CompressedJsonFilter.name: CompressedJsonFilter(),
    }

    @classmethod
    def register(cls, flt: IContentFilter) -> None:
        if flt.name in cls._filters:
            raise ValueError(f"Filter '{flt.name}' already registered")
        cls._filters[flt.name] = flt

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._filters)

    @classmethod
    def resolve(cls, names: list[str]) -> list[IContentFilter]:
        out: list[IContentFilter] = []
        for n in names:
            if n not in cls._filters:
                raise ValueError(
                    f"Unknown exclusion filter '{n}'. Known: {cls.keys()}"
                )
            out.append(cls._filters[n])
        return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_filters.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add src/dbs_vector/core/ports.py src/dbs_vector/infrastructure/chunking/filters.py tests/unit/test_filters.py
git commit -m "feat(chunking): add pluggable content-exclusion filter registry"
```

---

## Task 3: Config — new profile/engine knobs + validation

**Files:**
- Modify: `src/dbs_vector/config.py` (`TuningProfile` ~11-19, `EngineConfig` ~22-49, `chunker_kwargs` ~61-86, `_ALLOWED*` keys ~120-129, validation loop ~243-309)
- Test: `tests/unit/test_tuning_profile.py`, `tests/unit/test_config_validation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_tuning_profile.py`:

```python
from dbs_vector.config import TuningProfile


def test_token_budget_fields_default_to_zero():
    p = TuningProfile(max_token_length=2048, chunk_max_chars=0, batch_size=16)
    assert p.chunk_target_tokens == 0
    assert p.chunk_max_tokens == 0


def test_token_budget_fields_accepted():
    p = TuningProfile(
        max_token_length=2048, chunk_max_chars=0, batch_size=16,
        chunk_target_tokens=512, chunk_max_tokens=1024,
    )
    assert (p.chunk_target_tokens, p.chunk_max_tokens) == (512, 1024)
```

Add to `tests/unit/test_config_validation.py` (follow the file's existing helper for building a settings/engine; mirror the pattern already used there for invalid-profile cases):

```python
def test_chunk_max_tokens_below_target_is_rejected(write_config):
    # write_config: existing helper that writes a config.yaml and returns its path.
    cfg = write_config(profile_overrides={
        "chunk_target_tokens": 1024, "chunk_max_tokens": 512,
    })
    with pytest.raises(ValueError, match="chunk_max_tokens"):
        load_settings(cfg, validate=True)


def test_chunk_max_tokens_above_model_cap_is_rejected(write_config):
    cfg = write_config(profile_overrides={
        "max_token_length": 2048, "chunk_target_tokens": 512, "chunk_max_tokens": 4096,
    })
    with pytest.raises(ValueError, match="chunk_max_tokens"):
        load_settings(cfg, validate=True)


def test_unknown_exclusion_filter_is_rejected(write_config):
    cfg = write_config(engine_overrides={"exclusion_filters": ["bogus"]})
    with pytest.raises(ValueError, match="Unknown exclusion filter"):
        load_settings(cfg, validate=True)
```

> If `tests/unit/test_config_validation.py` has no `write_config` fixture, instead copy the nearest existing test in that file that builds an in-memory config dict and feeds it to the validator, and adapt the three assertions above to that style. Do not invent a fixture that does not exist — reuse the file's established pattern.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tuning_profile.py -v`
Expected: FAIL — `chunk_target_tokens` not a field / `extra="forbid"` rejects it.

- [ ] **Step 3: Add fields to `TuningProfile`**

In `src/dbs_vector/config.py`, extend `TuningProfile`:

```python
    max_token_length: int = Field(gt=0)
    chunk_max_chars: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    chunk_target_tokens: int = Field(default=0, ge=0)
    chunk_max_tokens: int = Field(default=0, ge=0)
```

- [ ] **Step 4: Add field to `EngineConfig`**

In `EngineConfig` (after `query_prefix`):

```python
    # Per-engine content exclusion (default: exclude nothing):
    exclusion_filters: list[str] = []
```

- [ ] **Step 5: Extend allowed profile keys**

Find the allowed-profile-key collection near `config.py:120-129` (the set listing `"max_token_length"`, `"chunk_max_chars"`, `"batch_size"`). Add `"chunk_target_tokens"` and `"chunk_max_tokens"` so the loader accepts them.

- [ ] **Step 6: Drop `chunk_max_chars` arg from `chunker_kwargs`**

Replace the signature/body of `EngineConfig.chunker_kwargs` (`config.py:61-86`) so it no longer takes or uses `chunk_max_chars` (the document chunker is now wired in bootstrap). Keep the `duckdb`/`api` branches **unchanged**, and make the fall-through return empty:

```python
    def chunker_kwargs(
        self,
        query_override: str | None = None,
        url_override: str | None = None,
    ) -> dict[str, object]:
        """Resolve chunker init kwargs for non-document chunkers. The document
        chunker is wired separately in bootstrap (token budgets / filters /
        length_fn)."""
        if self.chunker_type == "duckdb":
            return {"query": query_override or self.duckdb_query}
        if self.chunker_type == "api":
            kwargs: dict[str, object] = {
                "base_url": url_override or self.api_base_url,
                "api_key": self.api_key,
                "page_size": self.api_page_size,
                "since_days": self.api_since_days,
                "timeout_sec": self.api_timeout_sec,
                "min_execution_ms": self.api_min_execution_ms,
            }
            if self.api_database:
                kwargs["database"] = self.api_database
            if query_override:
                kwargs["custom_query"] = query_override
            return kwargs
        return {}
```

- [ ] **Step 7: Add validation rules**

In the per-engine validation loop, after the existing Rule 5 (`config.py:301-309`), add — importing `FilterRegistry` at the top of the file:

```python
        # Rule 6: token-budget coherence (only when token chunking is enabled)
        if profile.chunk_target_tokens > 0 or profile.chunk_max_tokens > 0:
            if profile.chunk_max_tokens < profile.chunk_target_tokens:
                raise ValueError(
                    f"Profile '{engine.tuning_profile}': chunk_max_tokens="
                    f"{profile.chunk_max_tokens} < chunk_target_tokens="
                    f"{profile.chunk_target_tokens}."
                )
            if profile.chunk_max_tokens > profile.max_token_length:
                raise ValueError(
                    f"Profile '{engine.tuning_profile}': chunk_max_tokens="
                    f"{profile.chunk_max_tokens} exceeds max_token_length="
                    f"{profile.max_token_length} (embedder truncation cap)."
                )

        # Rule 7: exclusion filters resolve
        if engine.exclusion_filters:
            FilterRegistry.resolve(engine.exclusion_filters)  # raises on unknown
```

Add near the other imports at the top of `config.py`:

```python
from dbs_vector.infrastructure.chunking.filters import FilterRegistry
```

> If this import creates a cycle (config ← filters ← ports), move the import *inside* the validation function body instead of module top-level.

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tuning_profile.py tests/unit/test_config_validation.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/dbs_vector/config.py tests/unit/test_tuning_profile.py tests/unit/test_config_validation.py
git commit -m "feat(config): token-budget profile knobs + per-engine exclusion_filters"
```

---

## Task 4: Rewrite `DocumentChunker` — section model + metadata

**Files:**
- Modify: `src/dbs_vector/infrastructure/chunking/document.py` (full rewrite of markdown path; keep `.txt`)
- Test: `tests/unit/test_chunker.py`

- [ ] **Step 1: Replace the chunker tests' markdown cases**

In `tests/unit/test_chunker.py`, **keep** `test_supported_extensions`, `test_chunk_ids_are_unique_across_different_paths`, and any `.txt` test. **Replace** `test_markdown_chunking_with_code_fences` (and other markdown-specific cases that assert the old greedy behavior) with:

```python
from dbs_vector.core.models import Document
from dbs_vector.infrastructure.chunking.document import DocumentChunker


def _chunks(content, **kw):
    # length_fn defaults to len (chars) -> deterministic, model-free.
    ch = DocumentChunker(target_tokens=120, max_tokens=240, min_tokens=8, **kw)
    return list(ch.process(Document(filepath="t.md", content=content, content_hash="h")))


def test_heading_is_prepended_not_emitted_alone():
    content = "# Top\n\n## Setup\n\nInstall the package and run it.\n"
    chunks = _chunks(content)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text.startswith("Top > Setup")
    assert "Install the package" in c.text
    assert c.parent_scope == "Top > Setup"
    assert c.node_type in {"section", "prose"}


def test_bare_heading_with_no_body_produces_no_chunk():
    content = "# Parent\n\n## Child\n\n### Grandchild\n\nReal content here.\n"
    chunks = _chunks(content)
    # Only the grandchild has body; one chunk, full heading path.
    assert len(chunks) == 1
    assert chunks[0].parent_scope == "Parent > Child > Grandchild"


def test_code_fence_under_budget_stays_atomic():
    content = "## Code\n\n```python\ndef f():\n    return 1\n```\n"
    chunks = _chunks(content)
    assert len(chunks) == 1
    assert "```python" in chunks[0].text
    assert chunks[0].node_type == "code"


def test_metadata_line_range_populated():
    content = "## A\n\nbody text that is long enough.\n"
    c = _chunks(content)[0]
    assert c.line_range and "-" in c.line_range
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_chunker.py -v`
Expected: FAIL (new constructor kwargs / new behavior not present).

- [ ] **Step 3: Rewrite `document.py`**

Replace the entire contents of `src/dbs_vector/infrastructure/chunking/document.py` with:

```python
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import markdown_it

from dbs_vector.core.models import Chunk, Document
from dbs_vector.core.ports import IContentFilter

_ATX_RE = re.compile(r"^#{1,6}\s*")
_LIST_MARKER = re.compile(r"^(\s*)([-*+]|\d+[.)])\s")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Block:
    node_type: str  # "heading" | "prose" | "code" | "list" | "table"
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
                blocks.append(_Block("prose", text, start, end))
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
        # 1) expand oversized blocks into <= max_tokens units
        units: list[tuple[str, str, int, int]] = []  # text, node_type, start, end
        for b in blocks:
            if self._len(b.text) <= self.max_tokens:
                units.append((b.text, b.node_type, b.start_line, b.end_line))
            else:
                for piece in self._split_block(b):
                    units.append((piece, b.node_type, b.start_line, b.end_line))

        # 2) greedy pack to target_tokens
        packed: list[list] = []  # [text, node_type, start, end]
        for text, ntype, start, end in units:
            if packed:
                cur = packed[-1]
                cand = cur[0] + "\n\n" + text
                if self._len(cand) <= self.target_tokens:
                    cur[0] = cand
                    cur[3] = end
                    if cur[1] != ntype:
                        cur[1] = "section"
                    continue
            packed.append([text, ntype, start, end])

        # 3) tiny-merge: a chunk below min_tokens folds into previous (same section)
        merged: list[list] = []
        for item in packed:
            if merged and self._len(item[0]) < self.min_tokens:
                p = merged[-1]
                p[0] = p[0] + "\n\n" + item[0]
                p[1] = "section"
                p[3] = item[3]
            else:
                merged.append(item)

        # 4) prepend heading path + build specs
        out: list[_Spec] = []
        for text, ntype, start, end in merged:
            body = f"{path}\n\n{text}" if path else text
            out.append(_Spec(body, ntype, path or None, f"{start}-{end}"))
        return out

    # ---- oversized-block splitting -------------------------------------

    def _split_block(self, b: _Block) -> list[str]:
        if b.node_type == "code":
            return self._split_code(b)
        if b.node_type == "table":
            return self._split_table(b)
        if b.node_type == "list":
            return self._pack_atoms(self._list_items(b.text), "\n")
        return self._pack_atoms(_SENTENCE.split(b.text), " ")

    def _split_code(self, b: _Block) -> list[str]:
        lines = b.text.split("\n")
        if len(lines) >= 2 and lines[0].lstrip().startswith("```"):
            inner = lines[1:-1]
        else:
            inner = lines
        parts = self._pack_atoms(inner, "\n")
        m = len(parts)
        if m <= 1:
            return [f"```{b.info}\n{parts[0] if parts else ''}\n```"]
        return [f"(code, part {k}/{m})\n```{b.info}\n{p}\n```" for k, p in enumerate(parts, 1)]

    def _split_table(self, b: _Block) -> list[str]:
        rows = [ln for ln in b.text.split("\n") if ln.strip()]
        if len(rows) <= 2:
            return [b.text]
        header = "\n".join(rows[:2])
        groups = self._pack_atoms(rows[2:], "\n")
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

    def _pack_atoms(self, atoms: list[str], joiner: str) -> list[str]:
        out: list[str] = []
        cur = ""
        for a in atoms:
            pieces = [a] if self._len(a) <= self.max_tokens else self._hard_window(a)
            for p in pieces:
                cand = p if not cur else cur + joiner + p
                if cur and self._len(cand) > self.target_tokens:
                    out.append(cur)
                    cur = p
                else:
                    cur = cand
        if cur:
            out.append(cur)
        return out

    def _hard_window(self, text: str) -> list[str]:
        """Split a single oversized atom by characters so each window is
        <= max_tokens. Last-resort guarantee against truncation."""
        windows: list[str] = []
        i, n = 0, len(text)
        step = max(1, self.max_tokens)
        while i < n:
            j = min(n, i + step)
            while j < n and self._len(text[i:j]) <= self.max_tokens:
                j = min(n, j + step)
            while j > i + 1 and self._len(text[i:j]) > self.max_tokens:
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
```

> Note: `_Block`/`_Spec` use `field` import only if defaults need it; the `field` import is harmless if unused, but remove it if Ruff flags F401. Keep `from dataclasses import dataclass` regardless.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_chunker.py -v`
Expected: PASS (kept tests + 4 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/infrastructure/chunking/document.py tests/unit/test_chunker.py
git commit -m "feat(chunking): heading-aware section chunker with token sizing + metadata"
```

---

## Task 5: Chunker — oversized splitting + filter behavior tests

**Files:**
- Test: `tests/unit/test_chunker.py` (add cases — implementation already exists from Task 4)

- [ ] **Step 1: Write the tests**

Append to `tests/unit/test_chunker.py`:

```python
from dbs_vector.infrastructure.chunking.filters import FilterRegistry


def test_oversized_code_fence_splits_by_lines_never_truncates():
    code = "\n".join(f"line_{i} = {i}" for i in range(400))  # ~ large
    content = f"## Big\n\n```python\n{code}\n```\n"
    chunks = _chunks(content, target_tokens=120, max_tokens=240)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 240 + 64  # heading path overhead tolerance
        assert "```python" in c.text
    # reassembled code retains all lines
    joined = "\n".join(c.text for c in chunks)
    assert "line_0 = 0" in joined and "line_399 = 399" in joined


def test_oversized_list_splits_at_item_boundaries():
    items = "\n".join(f"- item number {i} with some words" for i in range(80))
    content = f"## L\n\n{items}\n"
    chunks = _chunks(content, target_tokens=100, max_tokens=200)
    assert len(chunks) > 1
    # no chunk starts mid-item (every body line under heading begins with '- ')
    for c in chunks:
        body = c.text.split("\n\n", 1)[-1]
        first = body.splitlines()[0]
        assert first.startswith("- ")


def test_oversized_table_repeats_header_each_part():
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(60))
    content = f"## T\n\n| a | b |\n|---|---|\n{rows}\n"
    chunks = _chunks(content, target_tokens=100, max_tokens=200)
    assert len(chunks) > 1
    for c in chunks:
        assert "| a | b |" in c.text  # header repeated


def test_single_huge_line_is_hard_windowed():
    blob = "x" * 5000  # one line, no spaces
    content = f"## Blob\n\n```text\n{blob}\n```\n"
    chunks = _chunks(content, target_tokens=200, max_tokens=400)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 400 + 64


def test_excalidraw_file_yields_no_chunks():
    ch = DocumentChunker(filters=FilterRegistry.resolve(["excalidraw"]))
    out = list(ch.process(Document(
        filepath="d.excalidraw.md", content="# x\n\nbody\n", content_hash="h")))
    assert out == []


def test_compressed_json_block_is_dropped():
    content = "## Drawing\n\nIntro line that is long enough to keep.\n\n```compressed-json\nN4KAblob\n```\n"
    ch = DocumentChunker(
        target_tokens=120, max_tokens=240,
        filters=FilterRegistry.resolve(["compressed_json"]),
    )
    chunks = list(ch.process(Document(filepath="t.md", content=content, content_hash="h")))
    assert all("compressed-json" not in c.text and "N4KAblob" not in c.text for c in chunks)
    assert any("Intro line" in c.text for c in chunks)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/unit/test_chunker.py -v`
Expected: PASS. If a size assertion fails because the char `length_fn` overhead from the heading path is larger than the tolerance, raise the `+ 64` tolerance to account for the literal heading path length in that test (the heading paths there are short, so 64 is safe).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_chunker.py
git commit -m "test(chunking): cover oversized splitting + filters"
```

---

## Task 6: Wire token-aware document chunker in bootstrap

**Files:**
- Modify: `src/dbs_vector/services/bootstrap.py` (`build_dependencies`, ~54-72)
- Test: `tests/unit/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_bootstrap.py` (mirror the file's existing mocking of `MLXEmbedder`/`LanceDBStore`; if it stubs `MLXEmbedder` so the real model never loads, reuse that fixture):

```python
def test_document_chunker_receives_token_budgets_and_length_fn(monkeypatch, ...):
    # Arrange a config with an `md` engine whose profile sets
    # chunk_target_tokens=512, chunk_max_tokens=1024, and exclusion_filters=[].
    deps = build_dependencies("md")
    ch = deps.chunker
    assert ch.target_tokens == 512
    assert ch.max_tokens == 1024
    # length_fn is the embedder's count_tokens (not the default `len`)
    assert ch._len == deps.embedder.count_tokens
```

> Use the existing test's pattern for constructing settings/engines and for preventing real model load. If the file already has a helper that builds deps for `md`, extend it; do not introduce a new MLX mock style.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_bootstrap.py::test_document_chunker_receives_token_budgets_and_length_fn -v`
Expected: FAIL — chunker built without token budgets / `_len` is `len`.

- [ ] **Step 3: Update `build_dependencies`**

In `src/dbs_vector/services/bootstrap.py`, replace the chunker construction block:

```python
    MapperClass = ComponentRegistry.get_mapper(engine.mapper_type)
    ChunkerClass = ComponentRegistry.get_chunker(engine.chunker_type)

    mapper = MapperClass(vector_dimension=contract.vector_dimension)

    if engine.chunker_type == "document":
        from dbs_vector.infrastructure.chunking.filters import FilterRegistry

        chunker = ChunkerClass(
            max_chars=profile.chunk_max_chars or 1000,
            target_tokens=profile.chunk_target_tokens or 512,
            max_tokens=profile.chunk_max_tokens or 1024,
            length_fn=embedder.count_tokens,
            filters=FilterRegistry.resolve(engine.exclusion_filters),
        )
    else:
        chunker = ChunkerClass(
            **engine.chunker_kwargs(
                query_override=query_override,
                url_override=url_override,
            )
        )
```

(The `engine.chunker_kwargs(...)` call no longer passes `chunk_max_chars` — matches Task 3 Step 6.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_bootstrap.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dbs_vector/services/bootstrap.py tests/unit/test_bootstrap.py
git commit -m "feat(bootstrap): wire token budgets + filters into document chunker"
```

---

## Task 7: SQL-unaffected guard test

**Files:**
- Test: `tests/unit/test_bootstrap.py`

- [ ] **Step 1: Write the guard test**

Add to `tests/unit/test_bootstrap.py`:

```python
def test_sql_api_chunker_gets_no_chunking_kwargs(...):
    # build deps for the `sql-api` engine (api chunker). It must NOT receive
    # token budgets / filters / length_fn — proving the document-only wiring.
    deps = build_dependencies("sql-api")
    ch = deps.chunker
    assert not hasattr(ch, "target_tokens")
    assert not hasattr(ch, "_filters")
    # and the duckdb path likewise
    deps2 = build_dependencies("sql")
    assert not hasattr(deps2.chunker, "target_tokens")
```

> Reuse the file's existing config/mocking pattern for building deps. If `sql`/`sql-api` need a duckdb path or api_base_url, point them at harmless stubs the existing tests already use, or assert only on the `sql-api` engine if `sql` requires a real DuckDB file.

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_bootstrap.py -v`
Expected: PASS (the document-only `if` from Task 6 already guarantees this).

- [ ] **Step 3: Run the full SQL/chunker suite to confirm zero regressions**

Run: `uv run pytest tests/unit/test_sql_chunker.py tests/unit/test_duckdb_chunker.py tests/unit/test_api_chunker.py -v`
Expected: PASS, unchanged.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_bootstrap.py
git commit -m "test(bootstrap): guard that SQL engines get no document-chunking kwargs"
```

---

## Task 8: Update `config.yaml`

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Update the md profiles**

In `config.yaml` `profiles:`, set:

```yaml
  gemma-md:          {max_token_length: 2048, chunk_max_chars: 0, batch_size: 64, chunk_target_tokens: 512, chunk_max_tokens: 1024}
  granite-md-medium: {max_token_length: 2048, chunk_max_chars: 0, batch_size: 16, chunk_target_tokens: 768, chunk_max_tokens: 1536}
```

(Granite `max_token_length` drops 4096 → 2048 so `chunk_max_tokens` 1536 ≤ cap and the cap stays well within the model's ceiling. SQL profiles unchanged.)

- [ ] **Step 2: Add `exclusion_filters` to the md engines**

Under `engines: md:` and `engines: md-granite:` add (default empty; pre-populated here since this vault contains drawings — adjust to taste):

```yaml
    exclusion_filters: [excalidraw, compressed_json]
```

- [ ] **Step 3: Validate the config loads**

Run: `uv run dbs-vector --config-file config.yaml search "smoke" --type md --limit 1`
Expected: no config/validation error (search may return existing results from the current table; that's fine — this only proves the config validates and engines build).

- [ ] **Step 4: Commit**

```bash
git add config.yaml
git commit -m "config: token budgets (gemma 512/1024, granite 768/1536) + md exclusion filters"
```

---

## Task 9: Full validation suite

**Files:** none (verification)

- [ ] **Step 1: Run the whole suite**

Run: `uv run poe check`
Expected: format clean, lint clean, typecheck clean, all tests PASS.

- [ ] **Step 2: Fix any lint/type issues inline**

Common: remove unused `field` import in `document.py`; ensure `Callable` import; `IContentFilter` import not flagged. Re-run `uv run poe check` until green.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint/type fixes for section chunker"
```

---

## Task 10: Integration test on a fixture vault

**Files:**
- Test: `tests/integration/test_ingestion.py` (add a markdown-section case; reuse the file's tmp-LanceDB + real-chunker pattern)

- [ ] **Step 1: Write the integration test**

Add to `tests/integration/test_ingestion.py` (follow the existing fixture style — tmp dir, real `DocumentChunker`, a fake/stub embedder that returns deterministic vectors, real `LanceDBStore` on tmp path). If the file already ingests markdown via a helper, extend it; otherwise mirror its closest existing test:

```python
def test_section_chunking_no_noise_no_truncation(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "doc.md").write_text(
        "# Guide\n\n## Setup\n\nInstall and configure the service properly.\n\n"
        "## Big\n\n```python\n" + "\n".join(f"a{i}=1" for i in range(300)) + "\n```\n"
    )
    (vault / "draw.excalidraw.md").write_text("# d\n\n```compressed-json\nBLOB\n```\n")

    chunker = DocumentChunker(
        target_tokens=120, max_tokens=240,
        filters=FilterRegistry.resolve(["excalidraw", "compressed_json"]),
    )
    chunks = [c for f in vault.glob("*.md")
              for c in chunker.process(Document(
                  filepath=str(f), content=f.read_text(), content_hash="h"))]

    assert chunks, "expected chunks"
    assert all(c.source.endswith("doc.md") for c in chunks)  # excalidraw skipped
    assert all(len(c.text.strip()) >= 16 for c in chunks)    # no sub-16-char noise
    assert all("BLOB" not in c.text for c in chunks)          # compressed-json dropped
    assert all(c.parent_scope for c in chunks)                # heading context present
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/integration/test_ingestion.py::test_section_chunking_no_noise_no_truncation -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_ingestion.py
git commit -m "test(integration): section chunking removes noise + junk on a fixture vault"
```

---

## Task 11: Migration rebuild + before/after + comparison eval

**Files:** none (operational; record results in the spec or a NOTES file)

- [ ] **Step 1: Capture the BEFORE distribution**

Write a throwaway script `/tmp/dist.py` that loads `DocumentChunker` from the **current** install, chunks the real vault, and prints the median/p90 token sizes, % `<16` tokens, and count `>max_tokens`. (Reuse the analysis approach from the original investigation: iterate `*.md`, chunk, measure with the granite tokenizer.) Run it on `git stash`-ed old chunker OR note the known baseline: ~27% `<16` tok, 10 chunks `>2048`.

- [ ] **Step 2: Rebuild both md engines (real DB)**

```bash
uv run dbs-vector ingest "/Users/sinanalyuruk/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaspace" --type md --rebuild --force
uv run dbs-vector ingest "/Users/sinanalyuruk/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaspace" --type md-granite --rebuild --force
```

Expected: **no truncation warnings** in the logs (the alarm should not fire). Note chunk counts.

- [ ] **Step 3: Assert AFTER distribution**

Re-run `/tmp/dist.py` against the rebuilt tables (or re-chunk via the new `DocumentChunker`). Expected: `<16`-token share ≈ 0, `>max_tokens` count = 0.

- [ ] **Step 4: Compare gemma 512/1024 vs granite 768/1536**

Run ~10 representative vault queries through both engines and eyeball granularity/relevance:

```bash
for q in "btree index internals" "mysql connection pool tuning" "proxysql jmx metrics" "psql config shortcuts"; do
  echo "== $q (md) =="; uv run dbs-vector search "$q" --type md --limit 3
  echo "== $q (md-granite) =="; uv run dbs-vector search "$q" --type md-granite --limit 3
done
```

Record which token budget returns better-scoped passages. If gemma 512/1024 clearly wins, lower the granite profile to 512/1024 in `config.yaml` (and re-run Step 2 for `md-granite`).

- [ ] **Step 5: Record the outcome**

Append a short "Migration results" section to the spec doc with the before/after numbers and the gemma-vs-granite decision. Commit:

```bash
git add docs/superpowers/specs/2026-06-06-markdown-section-chunking-design.md
git commit -m "docs(spec): record section-chunking migration + comparison results"
```

---

## Task 12: Documentation

**Files:**
- Modify: `docs/README_PROFILES.md`
- Modify: `CLAUDE.md` (Tuning Profiles section) if it enumerates profile keys

- [ ] **Step 1: Document the new knobs + filters**

In `docs/README_PROFILES.md`, add a subsection describing:
- `chunk_target_tokens` / `chunk_max_tokens` (per-profile, document engines only; `0` = char-based `.txt` fallback only).
- `exclusion_filters` (per-engine list; built-ins `excalidraw`, `compressed_json`; default empty; add a filter via `FilterRegistry.register` + the name in config).
- That `max_token_length` is now a truncation safety net and `chunk_max_tokens ≤ max_token_length` is enforced.

- [ ] **Step 2: Sync CLAUDE.md if needed**

If `CLAUDE.md` lists the three profile knobs ("three numeric knobs per profile"), update it to mention the two token knobs and the per-engine `exclusion_filters`.

- [ ] **Step 3: Commit**

```bash
git add docs/README_PROFILES.md CLAUDE.md
git commit -m "docs: document token-budget knobs and exclusion filters"
```

---

## Self-review notes (author)

- **Spec coverage:** §3.1 algorithm → Task 4/5; §3.2 metadata → Task 4; §3.3 budgets → Task 3/8; §3.4 filters → Task 2; §3.5 config → Task 3/8; §3.6 wiring + `count_tokens` → Task 1/6; §5 migration → Task 11; §6 testing+comparison → Task 5/10/11; §6a SQL guarantee → Task 6 (gated wiring) + Task 7 (guard). All covered.
- **Type/name consistency:** `count_tokens`, `FilterRegistry.resolve`, `should_skip_file`/`should_drop_block`, `target_tokens`/`max_tokens`/`min_tokens`/`_len`/`_filters`, `chunk_target_tokens`/`chunk_max_tokens`, `exclusion_filters` used identically across tasks.
- **Known soft spots (flagged, not placeholders):** several tests say "reuse the file's existing fixture pattern" because the exact fixture helpers in `test_config_validation.py` / `test_bootstrap.py` / `test_ingestion.py` must be matched to what's there — the assertions are concrete; only the harness plumbing adapts.
