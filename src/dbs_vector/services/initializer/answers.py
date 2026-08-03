"""Plain data carried between the interview and the renderers."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KindAnswers:
    """Answers specific to one engine kind. Document-shaped this revision."""

    paths: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)
    exclusion_filters: list[str] = field(default_factory=list)
    watch_enabled: bool = False
    watch_debounce_seconds: float = 3.0


@dataclass
class InitAnswers:
    """Everything the interview collected. Pure data - no I/O, no defaults
    logic; the wizard resolves those before constructing this."""

    engine_name: str
    model_key: str
    kind_key: str
    tier_key: str
    passage_prefix: str
    query_prefix: str
    db_path: str
    # None when dbs-vector is installed (already on PATH) rather than a
    # source checkout - the published wheel ships no pyproject.toml, so
    # there is no directory for `uv --directory <dir>` to resolve.
    install_dir: str | None
    config_path: str
    mcp_path: str
    kind: KindAnswers


@dataclass
class InitResult:
    """What init actually wrote, for reporting to the user."""

    # Needed by the CLI's next-step hint: `ingest --type <engine_name>`.
    # `ingest` defaults to the `md` engine, so an arbitrary generated name
    # must be passed explicitly.
    engine_name: str
    config_path: Path
    mcp_path: Path
    config_backup: Path | None = None
    mcp_backup: Path | None = None
    notes: list[str] = field(default_factory=list)
    # False when dbs-vector was detected as an installed package rather than
    # a source checkout - `uv run` requires uv, which a `pip install` user
    # may not have, so the CLI's next-step hint must not assume it.
    used_checkout: bool = True
