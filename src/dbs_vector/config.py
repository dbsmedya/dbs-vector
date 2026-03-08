import os
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # General System
    db_path: str = "./lancedb_dbs_vector"
    batch_size: int = 64
    nprobes: int = 20
    log_level: str = "INFO"
    log_serialize: bool = False

    # Engines dictionary
    engines: dict[str, EngineConfig] = {}

    model_config = SettingsConfigDict(env_prefix="DBS_", env_file=".env")


def load_settings(config_file: str | None = None) -> Settings:
    """Loads base settings and overrides them from config.yaml."""
    base_settings = Settings()

    if config_file is None:
        config_file = os.getenv("DBS_CONFIG_FILE", "config.yaml")

    yaml_path = Path(config_file)
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

            if not data:
                return base_settings

            # Override System configuration
            if "system" in data and isinstance(data["system"], dict):
                for key, value in data["system"].items():
                    if hasattr(base_settings, key):
                        setattr(base_settings, key, value)

            # Override Engine configuration
            if "engines" in data and isinstance(data["engines"], dict):
                engines = {k: EngineConfig(**v) for k, v in data["engines"].items()}
                base_settings.engines = engines
    else:
        logger.warning("Configuration file '{}' not found, using defaults", yaml_path)

    return base_settings


# Global singleton instance
settings = load_settings()
