import os
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dbs_vector.core.naming import ENGINE_NAME_PATTERN


class TuningProfile(BaseModel):
    """Numeric knobs validated against the engine's model contract and
    available memory at load time."""

    model_config = ConfigDict(extra="forbid")

    max_token_length: int = Field(gt=0)
    chunk_max_chars: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    chunk_target_tokens: int = Field(default=0, ge=0)
    chunk_max_tokens: int = Field(default=0, ge=0)


class WatchConfig(BaseModel):
    """Watch MECHANICS only. All filtering/scoping lives on EngineConfig."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    debounce_seconds: float = Field(default=3.0, ge=0)  # 0 = process immediately


class EngineConfig(BaseModel):
    """Per-deployment engine config. References model contract + tuning profile."""

    model_config = ConfigDict(extra="forbid")

    description: str
    model: str  # key into ModelRegistry
    mapper_type: str
    family: str | None = None  # optional override; defaults to mapper_type
    chunker_type: str
    table_name: str
    workflow: str
    tuning_profile: str  # key into Settings.profiles

    # Model-deployment-specific (kept on engine because they vary per workflow
    # using the same underlying model):
    passage_prefix: str = ""
    query_prefix: str = ""

    # Engine-level admission floor (policy, not a model property): minimum
    # exact cosine similarity for the semantic admission channel. None (the
    # baseline default for every engine) = no floor = today's behavior.
    # Default values ship with the calibration companion spec.
    similarity_floor: float | None = Field(default=None, ge=-1.0, le=1.0)

    # Per-engine content exclusion (default: exclude nothing):
    exclusion_filters: list[str] = []

    # Engine-owned ingestion scope (v1: document engines only; ignored elsewhere):
    paths: list[str] = []
    ignore_patterns: list[str] = [".#*", "*~", "*.tmp", ".DS_Store"]
    watch: WatchConfig = Field(default_factory=WatchConfig)

    # Chunker-specific (unchanged):
    duckdb_query: str | None = None
    api_base_url: str = ""
    api_key: str = ""
    api_page_size: int = 200
    api_since_days: int = 15
    api_timeout_sec: int = 30
    api_min_execution_ms: float = 0.0
    api_database: str = ""

    @field_validator("paths")
    @classmethod
    def _resolve_roots(cls, value: list[str]) -> list[str]:
        """Ingestion roots are absolute directories, resolved at config load.

        Existence is NOT checked here — an unmounted vault must not break
        `dbs-vector search`. A missing root is a use-time warning + skip.
        """
        resolved: list[str] = []
        for raw in value:
            if not os.path.isabs(raw):
                raise ValueError(f"paths entry '{raw}' must be an absolute directory path")
            resolved.append(str(Path(raw).resolve()))
        return resolved

    @property
    def resolved_family(self) -> str:
        """Family key for presentation-layer dispatch.

        Defaults to mapper_type for backwards compatibility; overridable via
        the `family:` config field when an engine's mapper differs from its
        intended presentation surface.
        """
        return self.family or self.mapper_type

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


class Settings(BaseSettings):
    """Global configuration for the dbs-vector application."""

    model_config = SettingsConfigDict(
        env_prefix="DBS_",
        env_file=".env",
        extra="ignore",  # env vars often include unrelated keys; ignore them
    )

    # General system
    db_path: str = "./lancedb_dbs_vector"
    nprobes: int = Field(default=20, gt=0)
    log_level: str = "INFO"
    log_serialize: bool = False

    # Profile-validation budget (None → auto-detect via MLX).
    memory_budget_gb: float | None = Field(default=None, gt=0)

    # Runtime MLX allocator limits. The total limit falls back to the resolved
    # memory budget; the cache limit falls back to the resolved total limit.
    mlx_memory_limit_gb: float | None = Field(default=None, gt=0)
    mlx_cache_limit_gb: float | None = Field(default=None, ge=0)

    # NEW: profile dict
    profiles: dict[str, TuningProfile] = {}

    # Engines dictionary (shape changes in Task 7)
    engines: dict[str, EngineConfig] = {}

    # REMOVED: batch_size (now per-profile)


_LEGACY_ENGINE_FIELDS = {
    "model_name",
    "vector_dimension",
    "max_token_length",
    "attention_mask_dtype",
    "chunk_max_chars",
    "batch_size",
}
_REQUIRED_ENGINE_FIELDS = {"model", "tuning_profile"}

_LEGACY_SYSTEM_KEYS = {"batch_size"}  # moved to TuningProfile in profiles: block
_KNOWN_SYSTEM_KEYS = {
    "db_path",
    "nprobes",
    "log_level",
    "log_serialize",
    "memory_budget_gb",
    "mlx_memory_limit_gb",
    "mlx_cache_limit_gb",
}


def _apply_system_config(system: dict[str, object], settings: Settings, config_file: str) -> None:
    """Apply system: keys onto the Settings instance with strict validation.

    - Legacy keys (e.g., batch_size) raise a migration hint.
    - Unknown keys raise with the allow-list to catch typos.
    - Known keys pass through to setattr().
    """
    legacy = sorted(set(system) & _LEGACY_SYSTEM_KEYS)
    unknown = sorted(set(system) - _KNOWN_SYSTEM_KEYS - _LEGACY_SYSTEM_KEYS)
    if legacy:
        raise ValueError(
            f"Config schema mismatch in {config_file} (system: block).\n"
            f"  Legacy keys found: {legacy}\n"
            f"  These moved to TuningProfile. Define profiles under "
            f"`profiles:` and reference them from each engine via "
            f"`tuning_profile:`. See "
            f"docs/README_PROFILES.md / docs/README_EMBEDDINGS.md."
        )
    if unknown:
        raise ValueError(
            f"Unknown keys in {config_file} system: block: {unknown}. "
            f"Allowed: {sorted(_KNOWN_SYSTEM_KEYS)}."
        )
    # Validate YAML values with the same Pydantic rules used at construction,
    # then copy only the explicitly configured fields onto this instance.
    validated = Settings.model_validate({**settings.model_dump(), **system})
    for key in system:
        setattr(settings, key, getattr(validated, key))


def _raise_migration_hint(err: ValidationError, config_file: str, where: str) -> None:
    """Detect old-schema fields in a Pydantic ValidationError and rewrap as a
    single migration message. If the error is unrelated to migration, propagate."""
    seen_legacy = {e["loc"][-1] for e in err.errors() if e["loc"][-1] in _LEGACY_ENGINE_FIELDS}
    missing_required = {
        e["loc"][-1]
        for e in err.errors()
        if e["type"] == "missing" and e["loc"][-1] in _REQUIRED_ENGINE_FIELDS
    }
    if seen_legacy or missing_required:
        raise ValueError(
            f"Config schema mismatch in {config_file} ({where}: block).\n"
            f"  Legacy per-engine fields found: {sorted(seen_legacy) or 'none'}\n"
            f"  Missing new required fields: {sorted(missing_required) or 'none'}\n"
            f"See docs/README_PROFILES.md or docs/README_EMBEDDINGS.md "
            f"for the new schema."
        ) from err
    raise err


_CALIBRATION_NOTE = (
    "Calibration reference: 2025-05 OOM measured 41 GB at "
    "batch=64, seq=16384, bf16; the 3.0× safety factor preserves headroom."
)


def _validate_config(settings: Settings, config_file: str) -> None:
    """Run the validation chain over every configured engine.

    Rules (each fails-fast with a remediation message), executed per engine:
      1. Engine.model exists in ModelRegistry.
      2. Engine.tuning_profile exists in settings.profiles.
      3. profile.max_token_length ≤ contract.model_max_token_length.
      4. estimate_peak_buffer_bytes ≤ memory_budget × 0.9.
      5. (warn) chunk_max_chars routinely exceeds max_token_length × 4.
      6. Engine name matches ^[a-z0-9][a-z0-9_-]*$ (MCP tool naming).
      7. resolved_family is a known FamilyKeyRegistry key.
      8. Document engines require chunk_target_tokens > 0 and chunk_max_tokens > 0;
         chunk_max_tokens ≥ chunk_target_tokens; chunk_max_tokens ≤ max_token_length.
      9. exclusion_filters (if set) must resolve to known FilterRegistry entries.
     10. watch.enabled requires non-empty paths and chunker_type == "document".
     11. A watched engine's table_name must not be shared with any other engine
         (prune is root-scoped; a shared table would cross-delete).

    Memory budget is resolved lazily (only when rule 4 actually runs) so a
    config with an unknown model/profile fails on rule 1/2 BEFORE we attempt
    Metal auto-detection — otherwise an MLX-unavailable environment would
    mask real config errors with "Could not auto-detect Metal memory budget."
    """
    from dbs_vector.core.families import FamilyKeyRegistry
    from dbs_vector.core.model_registry import ModelRegistry
    from dbs_vector.core.profile_math import (
        estimate_peak_buffer_bytes,
        recommend_profile,
    )
    from dbs_vector.infrastructure.chunking.filters import FilterRegistry
    from dbs_vector.infrastructure.hardware import resolve_memory_budget_gb

    if not settings.engines:
        return

    # Lazy: only resolved on first memory check (rule 4).
    budget_gb: float | None = None

    for engine_name, engine in settings.engines.items():
        # Rule 6: legal engine name (presentation layer requires this for
        # MCP tool naming and predictable URLs).
        if not ENGINE_NAME_PATTERN.match(engine_name):
            raise ValueError(
                f"Engine name '{engine_name}' must match "
                f"{ENGINE_NAME_PATTERN.pattern}. Allowed: lowercase letters, "
                f"digits, dash, underscore (must start with letter or digit). "
                f"Edit {config_file}."
            )

        # Rule 7: family resolves to a known FamilyKeyRegistry entry.
        if not FamilyKeyRegistry.is_valid(engine.resolved_family):
            raise ValueError(
                f"Engine '{engine_name}' references unknown family "
                f"'{engine.resolved_family}'. Known families: "
                f"{FamilyKeyRegistry.keys()}. Edit {config_file}."
            )

        # Rule 1: model contract exists
        try:
            contract = ModelRegistry.get(engine.model)
        except KeyError as e:
            raise ValueError(
                f"Engine '{engine_name}' references "
                + str(e).replace("Unknown ", "unknown ").strip("'\"")
            ) from e

        # Rule 2: profile exists
        if engine.tuning_profile not in settings.profiles:
            known = sorted(settings.profiles)
            raise ValueError(
                f"Engine '{engine_name}' references unknown tuning profile "
                f"'{engine.tuning_profile}'. Known: {known}"
            )
        profile = settings.profiles[engine.tuning_profile]

        # Rule 3: profile fits model cap
        if profile.max_token_length > contract.model_max_token_length:
            raise ValueError(
                f"Profile '{engine.tuning_profile}' requires "
                f"{profile.max_token_length} tokens but engine '{engine_name}' "
                f"uses model '{engine.model}' (cap {contract.model_max_token_length}). "
                f"Lower profile.max_token_length or pick a different model."
            )

        # Rule 4: profile fits memory budget — resolve budget lazily here.
        if budget_gb is None:
            budget_gb = resolve_memory_budget_gb(settings.memory_budget_gb)
        budget_bytes = int(budget_gb * 1024**3)
        peak = estimate_peak_buffer_bytes(profile, contract)
        cap = int(budget_bytes * 0.9)
        if peak > cap:
            raw_attention = (
                profile.batch_size * profile.max_token_length**2 * contract.compute_dtype_bytes
            )
            suggested = recommend_profile(
                contract,
                budget_gb,
                target_chunker=engine.chunker_type,
                target_seq_len=profile.max_token_length,
            )
            raise ValueError(
                f"Profile '{engine.tuning_profile}' on engine '{engine_name}' "
                f"fails the memory budget:\n"
                f"  conservative estimate: {peak / 1024**3:.1f} GB "
                f"(3.0× safety factor over raw attention buffer)\n"
                f"  raw attention buffer: {raw_attention / 1024**3:.1f} GB\n"
                f"  budget (Metal max buffer × 0.9): {cap / 1024**3:.1f} GB\n"
                f"  {_CALIBRATION_NOTE}\n"
                f"Suggested values: max_token_length={suggested['max_token_length']}, "
                f"batch_size={suggested['batch_size']}"
                f"{' (context length reduced)' if suggested['seq_len_reduced'] else ''}. "
                f"Edit {config_file}."
            )

        # Rule 5: chunk-vs-token sanity (warn only)
        if profile.chunk_max_chars > 0 and profile.chunk_max_chars > profile.max_token_length * 4:
            logger.warning(
                "Engine '{}' profile '{}': chunk_max_chars={} likely exceeds "
                "max_token_length={} (× 4 char/token heuristic). Chunks may truncate.",
                engine_name,
                engine.tuning_profile,
                profile.chunk_max_chars,
                profile.max_token_length,
            )

        # Rule 8: token budgets. Document engines REQUIRE explicit nonzero
        # budgets (0 is reserved for non-document profiles, where these fields
        # are unused). Coherence is then enforced.
        if engine.chunker_type == "document":
            if profile.chunk_target_tokens <= 0 or profile.chunk_max_tokens <= 0:
                raise ValueError(
                    f"Engine '{engine_name}' (document chunker) requires profile "
                    f"'{engine.tuning_profile}' to set chunk_target_tokens > 0 and "
                    f"chunk_max_tokens > 0 (got "
                    f"{profile.chunk_target_tokens}/{profile.chunk_max_tokens})."
                )
            if profile.chunk_max_tokens < profile.chunk_target_tokens:
                raise ValueError(
                    f"Engine '{engine_name}' profile '{engine.tuning_profile}': chunk_max_tokens="
                    f"{profile.chunk_max_tokens} < chunk_target_tokens="
                    f"{profile.chunk_target_tokens}."
                )
            if profile.chunk_max_tokens > profile.max_token_length:
                raise ValueError(
                    f"Engine '{engine_name}' profile '{engine.tuning_profile}': chunk_max_tokens="
                    f"{profile.chunk_max_tokens} exceeds max_token_length="
                    f"{profile.max_token_length} (embedder truncation cap)."
                )

        # Rule 9: exclusion filters resolve (any engine that sets them)
        if engine.exclusion_filters:
            FilterRegistry.resolve(engine.exclusion_filters)  # raises on unknown

        # Rule 10: watch prerequisites.
        if engine.watch.enabled:
            if not engine.paths:
                raise ValueError(
                    f"Engine '{engine_name}' sets watch.enabled but requires a "
                    f"non-empty `paths:` list (the watcher has nothing to watch). "
                    f"Edit {config_file}."
                )
            if engine.chunker_type != "document":
                raise ValueError(
                    f"Engine '{engine_name}' sets watch.enabled but "
                    f"chunker_type='{engine.chunker_type}'. v1 watch supports "
                    f'chunker_type: "document" only. Edit {config_file}.'
                )

            # Rule 11: exclusive table. Prune is root-scoped; a shared table
            # would let one engine delete another's rows.
            sharers = sorted(
                other
                for other, cfg in settings.engines.items()
                if other != engine_name and cfg.table_name == engine.table_name
            )
            if sharers:
                raise ValueError(
                    f"Watched engine '{engine_name}' shares table_name "
                    f"'{engine.table_name}' with {sharers}. A watched engine "
                    f"needs an exclusive table (reconciliation prunes rows under "
                    f"its roots). Give it its own table_name in {config_file}."
                )


def load_settings(config_file: str | None = None, validate: bool = False) -> Settings:
    """Load and (optionally) validate settings from a YAML file.

    Default validate=False: useful for tests and any caller that wants raw
    parsing. Runtime callers (CLI callback, API lifespan) pass validate=True.
    """
    base_settings = Settings()

    if config_file is None:
        config_file = os.getenv("DBS_CONFIG_FILE", "config.yaml")

    yaml_path = Path(config_file)
    if not yaml_path.exists():
        logger.warning("Configuration file '{}' not found, using defaults", yaml_path)
        return base_settings

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # System block — strict validation (legacy / unknown key rejection)
    if "system" in data and isinstance(data["system"], dict):
        _apply_system_config(data["system"], base_settings, config_file)

    # Profiles block
    if "profiles" in data and isinstance(data["profiles"], dict):
        try:
            base_settings.profiles = {k: TuningProfile(**v) for k, v in data["profiles"].items()}
        except ValidationError as e:
            _raise_migration_hint(e, config_file, where="profiles")

    # Engines block
    if "engines" in data and isinstance(data["engines"], dict):
        try:
            base_settings.engines = {k: EngineConfig(**v) for k, v in data["engines"].items()}
        except ValidationError as e:
            _raise_migration_hint(e, config_file, where="engines")

    if validate:
        _validate_config(base_settings, config_file)
    return base_settings


# Fields copied from a freshly-loaded Settings onto the module-level singleton.
_PROPAGATED_SETTINGS_FIELDS: set[str] = {
    "db_path",
    "nprobes",
    "engines",
    "profiles",
    "memory_budget_gb",
    "mlx_memory_limit_gb",
    "mlx_cache_limit_gb",
    "log_level",
    "log_serialize",
}
# Fields deliberately NOT propagated to the runtime singleton. Empty today; a
# future field that must keep its import-time default goes here EXPLICITLY.
_NOT_PROPAGATED_SETTINGS_FIELDS: set[str] = set()


def _populate_singleton_from(new_settings: "Settings") -> None:
    """Copy fields from a freshly-loaded Settings onto the module-level singleton.

    The CLI Typer callback (and the `mcp` subcommand's reload-when-set path)
    calls this with a Settings instance returned from
    `load_settings(config_file, validate=True)`. _PROPAGATED_SETTINGS_FIELDS
    is LOAD-BEARING: the loop below copies exactly that set, so adding a field
    name there (which the drift-guard test demands for every new Settings
    field) IS the propagation — there is no second list to forget.
    """
    for field in _PROPAGATED_SETTINGS_FIELDS:
        setattr(settings, field, getattr(new_settings, field))


# Module-level singleton: ZERO file I/O at import. We pass _env_file=None
# explicitly to disable pydantic-settings' .env file reading for this
# instance — otherwise BaseSettings will stat() the .env path and (if it
# exists) read it at import time, violating the import-safety contract.
# Runtime callers (cli.py callback, plus the mcp subcommand override path)
# call load_settings(config_file, validate=True), which constructs a fresh
# Settings() (without _env_file=None, so .env IS loaded then) and copies
# fields onto this singleton.
settings = Settings(_env_file=None)  # type: ignore[call-arg]
