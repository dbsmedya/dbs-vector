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
