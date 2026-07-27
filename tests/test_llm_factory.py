"""Tests for model configuration and role routing.

Deliberately offline. These assert that configuration is wired correctly
and that a missing credential fails loudly and by name — not that a
provider is reachable, which is a live-service concern rather than a
property of this code.
"""

import pytest
from pydantic import SecretStr

from rti_engine.config.settings import Settings
from rti_engine.llm.factory import (
    ModelConfigurationError,
    ModelRole,
    _require,
    _require_secret,
)

RESOURCE_URL = "https://example-resource.services.ai.azure.com"


def test_base_url_appends_the_v1_path() -> None:
    settings = Settings(azure_openai_endpoint=RESOURCE_URL)
    assert settings.azure_openai_base_url == f"{RESOURCE_URL}/openai/v1/"


def test_base_url_tolerates_a_trailing_slash() -> None:
    """A trailing slash in .env must not produce a doubled separator."""
    settings = Settings(azure_openai_endpoint=f"{RESOURCE_URL}/")
    assert settings.azure_openai_base_url == f"{RESOURCE_URL}/openai/v1/"


def test_base_url_is_none_without_an_endpoint() -> None:
    assert Settings(azure_openai_endpoint=None).azure_openai_base_url is None


def test_missing_value_fails_with_the_setting_name() -> None:
    """The error must name the variable, so the fix is obvious."""
    with pytest.raises(ModelConfigurationError, match="AZURE_OPENAI_API_KEY"):
        _require(None, "AZURE_OPENAI_API_KEY")

    with pytest.raises(ModelConfigurationError, match="GROQ_API_KEY"):
        _require("", "GROQ_API_KEY")


def test_present_value_is_returned_unchanged() -> None:
    assert _require("a-value", "SOME_SETTING") == "a-value"


def test_credentials_are_wrapped_so_they_cannot_be_logged() -> None:
    """A key must not appear in a repr, traceback, or log line."""
    secret = _require_secret("super-secret-key", "ANTHROPIC_API_KEY")

    assert isinstance(secret, SecretStr)
    assert "super-secret-key" not in repr(secret)
    assert "super-secret-key" not in str(secret)
    assert secret.get_secret_value() == "super-secret-key"


def test_missing_credential_fails_before_it_is_wrapped() -> None:
    with pytest.raises(ModelConfigurationError, match="ANTHROPIC_API_KEY"):
        _require_secret(None, "ANTHROPIC_API_KEY")


def test_every_role_is_distinct_and_named() -> None:
    """Roles are the stable interface agents use; names must not drift."""
    assert {role.value for role in ModelRole} == {
        "reasoning",
        "classification",
        "review",
    }
