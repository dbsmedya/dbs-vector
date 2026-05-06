import os
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TuningProfile(BaseModel):
    """Three numeric knobs validated against the engine's model contract and
    available memory at load time."""

    model_config = ConfigDict(extra="forbid")

    max_token_length: int = Field(gt=0)
    chunk_max_chars: int = Field(ge=0)
    batch_size: int = Field(gt=0)


class EngineConfig(BaseModel):
    """Configuration specific to a single AI engine/data source."""

    description: str
    model_name: str
    vector_dimension: int
    max_token_length: int
    table_name: str
    mapper_type: str
    chunker_type: str
    chunk_max_chars: int

    # Task Prefixes for models like embeddinggemma
    query_prefix: str = ""
    passage_prefix: str = ""
    workflow: str = "default"
    duckdb_query: str | None = None

    # API chunker fields
    api_base_url: str = ""
    api_key: str = ""
    api_page_size: int = 200
    api_since_days: int = 15
    api_timeout_sec: int = 30
    api_min_execution_ms: float = 0.0
    api_database: str = ""

    # Some MLX models (e.g. embeddinggemma-bf16) require the attention_mask cast
    # to a specific dtype to avoid type-promotion errors. Leave unset for models
    # that accept the default integer mask (e.g. ModernBERT / Granite).
    attention_mask_dtype: str | None = None  # accepted: None, "float16", "bfloat16", "float32"

    def chunker_kwargs(
        self, query_override: str | None = None, url_override: str | None = None
    ) -> dict[str, object]:
        """Resolve chunker initialization kwargs from engine config."""
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
        if self.chunk_max_chars > 0:
            return {"max_chars": self.chunk_max_chars}
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

    # NEW: memory budget (None → auto-detect via MLX in resolve_memory_budget_gb)
    memory_budget_gb: float | None = Field(default=None, gt=0)

    # NEW: profile dict
    profiles: dict[str, TuningProfile] = {}

    # Engines dictionary (shape changes in Task 7)
    engines: dict[str, EngineConfig] = {}

    # REMOVED: batch_size (now per-profile)


_LEGACY_SYSTEM_KEYS = {"batch_size"}  # moved to TuningProfile in profiles: block
_KNOWN_SYSTEM_KEYS = {
    "db_path",
    "nprobes",
    "log_level",
    "log_serialize",
    "memory_budget_gb",
}


def _apply_system_config(
    system: dict[str, object], settings: Settings, config_file: str
) -> None:
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
            f"docs/superpowers/specs/2026-05-06-tuning-profiles-design.md "
            f"§10 / docs/README_EMBEDDINGS.md."
        )
    if unknown:
        raise ValueError(
            f"Unknown keys in {config_file} system: block: {unknown}. "
            f"Allowed: {sorted(_KNOWN_SYSTEM_KEYS)}."
        )
    for key, value in system.items():
        setattr(settings, key, value)


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
        base_settings.profiles = {
            k: TuningProfile(**v) for k, v in data["profiles"].items()
        }

    # Engines block
    if "engines" in data and isinstance(data["engines"], dict):
        base_settings.engines = {k: EngineConfig(**v) for k, v in data["engines"].items()}

    # _validate_config wired up in Task 8
    return base_settings


# Module-level singleton: ZERO file I/O at import. We pass _env_file=None
# explicitly to disable pydantic-settings' .env file reading for this
# instance — otherwise BaseSettings will stat() the .env path and (if it
# exists) read it at import time, violating the import-safety contract.
# Runtime callers (cli.py callback, api/main.py lifespan) call
# load_settings(config_file, validate=True), which constructs a fresh
# Settings() (without _env_file=None, so .env IS loaded then) and copies
# fields onto this singleton.
settings = Settings(_env_file=None)  # type: ignore[call-arg]
