from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "ci", "azure"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    postgres_dsn: str | None = None

    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None

    pinecone_api_key: str | None = None
    pinecone_index: str | None = None

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str | None = None

    groq_api_key: str | None = None

    langsmith_api_key: str | None = None
    langsmith_project: str = "rti-engine"


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton."""
    return Settings()
