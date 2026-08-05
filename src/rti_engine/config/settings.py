"""Application configuration, loaded from environment variables or .env.

Every credential defaults to None so the application imports cleanly before
a given service exists. Code that needs a value asks for it explicitly and
fails with a clear message if it is missing, rather than failing obscurely
at call time.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

AZURE_V1_PATH = "openai/v1/"
"""Path suffix for the Azure OpenAI v1 API, which needs no api-version."""


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
    azure_openai_chat_deployment: str | None = None
    azure_openai_mini_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    anthropic_api_key: str | None = None
    anthropic_model: str | None = None

    groq_api_key: str | None = None
    groq_model: str | None = None

    langsmith_api_key: str | None = None
    langsmith_project: str = "rti-engine"

    otel_endpoint: str | None = None
    otel_service_name: str = "rti-engine"

    applicationinsights_connection_string: str | None = None
    """Injected by the container app in deployment. Locally absent, so
    tracing falls back to the OTLP exporter pointed at Jaeger."""

    analytics_mcp_url: str | None = None
    knowledge_mcp_url: str | None = None
    """Where the MCP servers are reachable, when they are separate services.

    Unset locally, where they are spawned as subprocesses over stdio. Set
    in a deployment, where a pipe cannot cross a container boundary.
    """

    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_port: int = 8080

    @property
    def azure_openai_base_url(self) -> str | None:
        """The v1 API base URL, derived from the configured endpoint.

        The endpoint is stored as the bare resource URL; the v1 path is
        appended here so the API surface stays an implementation detail
        rather than something someone must remember to type into .env.
        """
        if not self.azure_openai_endpoint:
            return None
        return self.azure_openai_endpoint.rstrip("/") + "/" + AZURE_V1_PATH


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton."""
    return Settings()
