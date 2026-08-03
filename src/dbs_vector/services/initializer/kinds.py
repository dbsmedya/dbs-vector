"""Engine kinds: what to ask, and how to shape the engine block.

The interview is open/closed through this Protocol. Adding the SQL kinds is
a registration, not an edit to the wizard: `run_init` asks "what are you
indexing?" only when more than one kind is registered, so the branch already
exists with a single arm.
"""

from typing import Protocol

from dbs_vector.core.model_registry import ModelContract
from dbs_vector.core.profile_math import derive_document_profile
from dbs_vector.services.initializer.answers import KindAnswers
from dbs_vector.services.initializer.io import PromptIO

# Copied from EngineConfig.ignore_patterns. Setting ignore_patterns in YAML
# REPLACES this list rather than extending it, so the wizard always emits it
# in full - otherwise adding one pattern silently drops Emacs lock-file
# filtering (".#notes.md" has suffix ".md" and passes the extension gate).
DEFAULT_IGNORE_PATTERNS = [".#*", "*~", "*.tmp", ".DS_Store"]

DEFAULT_EXCLUSION_FILTERS = ["excalidraw", "compressed_json"]

PROMPT_PATH = "Directory to index (blank when done)"
PROMPT_IGNORE = "Additional ignore patterns (comma-separated)"
PROMPT_FILTERS = "Content exclusion filters"
PROMPT_WATCH = "Watch these paths for changes?"
PROMPT_DEBOUNCE = "Debounce seconds"


class EngineKind(Protocol):
    """One indexable content shape."""

    key: str
    label: str
    chunker_type: str
    mapper_type: str
    supports_watch: bool

    def ask(self, io: PromptIO) -> KindAnswers:
        """Collect this kind's engine-specific answers."""
        ...

    def build_profile(
        self, contract: ModelContract, tier_key: str, memory_budget_gb: float
    ) -> dict[str, int]:
        """Which derivation applies. The math itself lives in core."""
        ...

    def build_engine_block(self, base: dict, kind: KindAnswers) -> dict:
        """Add this kind's fields to the shared engine block."""
        ...


class DocumentKind:
    """Markdown and plain-text files on disk."""

    key = "document"
    label = "Markdown / text documents"
    chunker_type = "document"
    mapper_type = "document"
    supports_watch = True

    def ask(self, io: PromptIO) -> KindAnswers:
        from dbs_vector.infrastructure.chunking.filters import FilterRegistry

        paths: list[str] = []
        while True:
            raw = io.ask_text(PROMPT_PATH, default="").strip()
            if not raw:
                break
            if not raw.startswith("/"):
                raise ValueError(f"Path '{raw}' must be an absolute directory path.")
            paths.append(raw)

        extra = io.ask_text(PROMPT_IGNORE, default="").strip()
        extra_patterns = [p.strip() for p in extra.split(",") if p.strip()]

        filters = io.ask_multi(
            PROMPT_FILTERS, FilterRegistry.keys(), default=list(DEFAULT_EXCLUSION_FILTERS)
        )

        watch_enabled = io.ask_bool(PROMPT_WATCH, default=False)
        if watch_enabled and not paths:
            raise ValueError(
                "Watching requires at least one directory to index. "
                "Add a path, or answer no to watching."
            )
        debounce = 3.0
        if watch_enabled:
            debounce = float(io.ask_text(PROMPT_DEBOUNCE, default="3.0"))

        return KindAnswers(
            paths=paths,
            ignore_patterns=[*DEFAULT_IGNORE_PATTERNS, *extra_patterns],
            exclusion_filters=filters,
            watch_enabled=watch_enabled,
            watch_debounce_seconds=debounce,
        )

    def build_profile(
        self, contract: ModelContract, tier_key: str, memory_budget_gb: float
    ) -> dict[str, int]:
        return derive_document_profile(contract, tier_key, memory_budget_gb)

    def build_engine_block(self, base: dict, kind: KindAnswers) -> dict:
        block = dict(base)
        block["exclusion_filters"] = list(kind.exclusion_filters)
        block["paths"] = list(kind.paths)
        block["ignore_patterns"] = list(kind.ignore_patterns)
        if kind.watch_enabled:
            block["watch"] = {
                "enabled": True,
                "debounce_seconds": kind.watch_debounce_seconds,
            }
        return block


class EngineKindRegistry:
    """Open/closed registry of engine kinds (cf. ModelRegistry)."""

    _kinds: dict[str, EngineKind] = {}

    @classmethod
    def register(cls, kind: EngineKind) -> None:
        if kind.key in cls._kinds:
            raise ValueError(f"Engine kind '{kind.key}' already registered")
        cls._kinds[kind.key] = kind

    @classmethod
    def get(cls, key: str) -> EngineKind:
        if key not in cls._kinds:
            raise KeyError(f"Unknown engine kind '{key}'. Known: {cls.keys()}")
        return cls._kinds[key]

    @classmethod
    def keys(cls) -> list[str]:
        return list(cls._kinds)

    @classmethod
    def values(cls) -> list[EngineKind]:
        return list(cls._kinds.values())


EngineKindRegistry.register(DocumentKind())
