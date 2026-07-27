"""Chat model and embedding construction, with cross-vendor fallbacks.

Models are requested by *role*, not by name. A caller asks for the
reasoning model; which deployment that resolves to, and what it falls back
to, is decided here. Changing a model is then a change to one module
rather than a search through the agent code.

Fallbacks deliberately cross vendors. A chain from one Azure deployment to
another Azure deployment does not survive an Azure outage, so the backup
for an Azure model is Anthropic or Groq, reached over entirely separate
infrastructure.

Temperature is zero throughout. This system produces statutory documents;
identical input should produce identical output, and sampling variety has
no value here.
"""

import enum
from collections.abc import Sequence
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from rti_engine.config.settings import Settings, get_settings

TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 2

ChatRunnable = Runnable[LanguageModelInput, BaseMessage]
"""What a caller receives: a chat model, or a model with fallbacks attached."""


class ModelRole(enum.StrEnum):
    """What a model is being asked to do, rather than which model it is."""

    REASONING = "reasoning"
    """Regulatory analysis, interpretation, and drafting."""

    CLASSIFICATION = "classification"
    """Tier routing and other high-volume, low-difficulty decisions."""

    REVIEW = "review"
    """Independent check of a draft, deliberately a different vendor."""


class ModelConfigurationError(RuntimeError):
    """Raised when a model is requested but its credentials are absent."""


def _require(value: str | None, name: str) -> str:
    """Return a required setting, or fail with a message naming it."""
    if not value:
        raise ModelConfigurationError(f"{name} is not set; check your .env file")
    return value


def _require_secret(value: str | None, name: str) -> SecretStr:
    """Return a required credential wrapped so it cannot be logged by accident."""
    return SecretStr(_require(value, name))


def _azure_model(settings: Settings, deployment: str | None, field: str) -> BaseChatModel:
    """Build a chat model against an Azure OpenAI deployment.

    Uses the plain OpenAI client against the v1 endpoint rather than the
    Azure-specific client, so there is no dated api-version to maintain.
    The deployment name is passed as the model name.
    """
    return ChatOpenAI(
        model=_require(deployment, field),
        base_url=_require(settings.azure_openai_base_url, "AZURE_OPENAI_ENDPOINT"),
        api_key=_require_secret(settings.azure_openai_api_key, "AZURE_OPENAI_API_KEY"),
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def _anthropic_model(settings: Settings) -> BaseChatModel:
    """Build a chat model against the Anthropic API directly.

    No temperature is passed: newer Anthropic models reject the parameter
    as deprecated. Determinism for this provider therefore rests on the
    model's own default rather than an explicit zero.
    """
    return ChatAnthropic(
        model_name=_require(settings.anthropic_model, "ANTHROPIC_MODEL"),
        api_key=_require_secret(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        stop=None,
    )


def _groq_model(settings: Settings) -> BaseChatModel:
    """Build a chat model against Groq."""
    return ChatGroq(
        model=_require(settings.groq_model, "GROQ_MODEL"),
        api_key=_require_secret(settings.groq_api_key, "GROQ_API_KEY"),
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


@lru_cache
def _role_models(role: ModelRole) -> tuple[BaseChatModel, list[BaseChatModel]]:
    """Return the primary model for a role and its ordered fallbacks.

    Kept separate from fallback assembly so tools can be bound to every
    model in the chain. Binding after the chain is built would leave the
    fallbacks without tools, and an agent that fell back would quietly
    lose its capabilities rather than fail.
    """
    settings = get_settings()

    if role is ModelRole.REASONING:
        primary = _azure_model(
            settings, settings.azure_openai_chat_deployment, "AZURE_OPENAI_CHAT_DEPLOYMENT"
        )
        return primary, [_anthropic_model(settings), _groq_model(settings)]

    if role is ModelRole.CLASSIFICATION:
        primary = _azure_model(
            settings, settings.azure_openai_mini_deployment, "AZURE_OPENAI_MINI_DEPLOYMENT"
        )
        return primary, [_groq_model(settings)]

    # REVIEW: Anthropic first, so the reviewer is a different vendor from the
    # drafter. A reviewer sharing the drafter's family shares its blind spots.
    fallback = _azure_model(
        settings, settings.azure_openai_chat_deployment, "AZURE_OPENAI_CHAT_DEPLOYMENT"
    )
    return _anthropic_model(settings), [fallback]


@lru_cache
def get_chat_model(role: ModelRole) -> ChatRunnable:
    """Return the model for a role, with its fallback chain attached.

    Cached per role: constructing a client opens a connection pool, and one
    per role per process is enough.
    """
    primary, fallbacks = _role_models(role)
    return primary.with_fallbacks(fallbacks)


def get_tool_model(role: ModelRole, tools: Sequence[BaseTool]) -> ChatRunnable:
    """Return a model for a role with tools bound to every model in the chain.

    Not cached: the tool set varies by agent, and a list of tools is not
    hashable.
    """
    bound_tools = list(tools)
    primary, fallbacks = _role_models(role)

    return primary.bind_tools(bound_tools).with_fallbacks(
        [model.bind_tools(bound_tools) for model in fallbacks]
    )


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Return the embedding model used by the vector store.

    No fallback: embeddings written by one model are not comparable with
    those from another, so silently switching would corrupt the index.
    """
    settings = get_settings()
    return OpenAIEmbeddings(
        model=_require(
            settings.azure_openai_embedding_deployment, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ),
        base_url=_require(settings.azure_openai_base_url, "AZURE_OPENAI_ENDPOINT"),
        api_key=_require_secret(settings.azure_openai_api_key, "AZURE_OPENAI_API_KEY"),
    )
