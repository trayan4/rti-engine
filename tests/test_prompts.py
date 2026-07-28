"""Tests for the token utility and the prompt layer.

The prompt validation rules exist to catch a specific class of silent
failure: a template that renders with a blank where a value should be.
The model does not报 that as an error — it reasons over the gap and
produces something fluent. So the failure has to happen at construction.
"""

import pytest
from langchain_core.messages import HumanMessage

from rti_engine.agents.prompts import (
    CITATION_RULES,
    GROUNDING_RULES,
    JURISDICTION_RULES,
    TOOL_FAILURE_RULES,
    Prompt,
    PromptError,
    PromptRegistry,
)
from rti_engine.llm.tokens import (
    TRUNCATION_MARKER,
    count_tokens,
    fits_within,
    truncate_to_tokens,
)

# --- tokens ---


def test_token_counting_is_not_a_character_count() -> None:
    assert count_tokens("hello world") < len("hello world")
    assert count_tokens("") == 0


def test_budget_check_matches_the_count() -> None:
    text = "a moderately sized sentence for counting"
    size = count_tokens(text)

    assert fits_within(text, size)
    assert not fits_within(text, size - 1)


def test_short_text_is_returned_unchanged() -> None:
    assert truncate_to_tokens("short", 100) == "short"


def test_truncation_marks_itself_and_respects_the_budget() -> None:
    """Silently truncated input is reasoned over as though complete."""
    text = "word " * 500
    result = truncate_to_tokens(text, 50)

    assert result.endswith(TRUNCATION_MARKER)
    assert count_tokens(result) <= 50
    assert len(result) < len(text)


# --- prompt validation ---


def test_a_template_variable_must_be_declared() -> None:
    """An undeclared variable renders as a missing key at call time."""
    with pytest.raises(ValueError, match="undeclared inputs"):
        Prompt(
            name="p",
            version=1,
            description="d",
            template="Hello {name}",
            inputs=(),
        )


def test_a_declared_input_must_be_used() -> None:
    """Dead configuration suggests the template says something it does not."""
    with pytest.raises(ValueError, match="never used"):
        Prompt(
            name="p",
            version=1,
            description="d",
            template="Hello",
            inputs=("name",),
        )


def test_a_prompt_with_no_inputs_is_valid() -> None:
    prompt = Prompt(name="p", version=1, description="d", template="Fixed text.")
    assert prompt.render() == "Fixed text."


# --- rendering ---


@pytest.fixture
def greeting() -> Prompt:
    return Prompt(
        name="greeting",
        version=2,
        description="A greeting",
        template="Hello {name}, you are in {country}.",
        inputs=("name", "country"),
    )


def test_rendering_fills_every_placeholder(greeting: Prompt) -> None:
    rendered = greeting.render(name="Ana", country="ES")
    assert rendered == "Hello Ana, you are in ES."
    assert "{" not in rendered


def test_a_missing_value_is_refused(greeting: Prompt) -> None:
    """The failure that matters: a blank the model reasons over regardless."""
    with pytest.raises(PromptError, match="missing inputs: country"):
        greeting.render(name="Ana")


def test_an_unexpected_value_is_refused(greeting: Prompt) -> None:
    with pytest.raises(PromptError, match="unexpected inputs: tier"):
        greeting.render(name="Ana", country="ES", tier="T2")


def test_the_identifier_carries_the_version(greeting: Prompt) -> None:
    """Recorded per run, so a generated document is traceable to its prompt."""
    assert greeting.identifier == "greeting@v2"


# --- budgets ---


def test_a_prompt_reports_its_own_size(greeting: Prompt) -> None:
    assert greeting.token_count(name="Ana", country="ES") > 0
    assert greeting.fits(name="Ana", country="ES")


def test_an_oversized_prompt_reports_that_it_does_not_fit() -> None:
    prompt = Prompt(
        name="big",
        version=1,
        description="d",
        template="{body}",
        inputs=("body",),
        max_tokens=10,
    )
    assert not prompt.fits(body="word " * 200)


# --- the LangChain bridge ---


def test_a_chat_prompt_carries_the_rendered_system_message(greeting: Prompt) -> None:
    chat = greeting.to_chat_prompt(name="Ana", country="ES")
    messages = chat.format_messages(messages=[HumanMessage(content="hi")])

    assert "Hello Ana, you are in ES." in str(messages[0].content)
    assert messages[1].content == "hi"


def test_literal_braces_survive_the_bridge() -> None:
    """A JSON example in a prompt must not be read as a template variable."""
    prompt = Prompt(
        name="json",
        version=1,
        description="d",
        template='Return exactly: {{"verdict": "unexplained"}}',
    )

    chat = prompt.to_chat_prompt()
    messages = chat.format_messages(messages=[])

    assert '{"verdict": "unexplained"}' in str(messages[0].content)


# --- registry ---


def test_the_registry_refuses_duplicate_names() -> None:
    prompt = Prompt(name="same", version=1, description="d", template="t")
    with pytest.raises(ValueError, match="unique"):
        PromptRegistry(prompts=(prompt, prompt))


def test_the_registry_returns_a_prompt_by_name(greeting: Prompt) -> None:
    registry = PromptRegistry(prompts=(greeting,))
    assert registry.get("greeting") is greeting
    assert registry.names() == ["greeting"]


def test_an_unknown_prompt_is_refused(greeting: Prompt) -> None:
    registry = PromptRegistry(prompts=(greeting,))
    with pytest.raises(PromptError, match="unknown prompt"):
        registry.get("missing")


# --- shared rule blocks ---


def test_shared_blocks_state_the_rules_agents_depend_on() -> None:
    """Composed rather than copied, so no agent's copy can drift."""
    assert "You do not calculate" in GROUNDING_RULES
    assert "citation string" in CITATION_RULES
    assert "Error calling tool" in TOOL_FAILURE_RULES
    assert "not yet transposed" in JURISDICTION_RULES
