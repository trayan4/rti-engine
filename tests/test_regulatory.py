"""Tests for the regulatory position layer.

Offline by design. What the model concludes is measured against the
scenario catalog in the eval harness; what is asserted here is the code
around it — that a contradicted fact is refused, that a claim without a
source cannot be represented, and that retrieved passages reach the model
with their citations attached.
"""

import json
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from rti_engine.agents.regulatory import (
    REGULATORY_PROMPT,
    Citation,
    RegulatoryError,
    RegulatoryPosition,
    _call,
    _format_graph,
    _format_passages,
    check_transposition_agrees,
)


def position(transposed: bool = False, basis: str = "employer_policy_only") -> RegulatoryPosition:
    """A position with the fields tests do not care about filled in."""
    return RegulatoryPosition(
        jurisdiction="ES",
        transposed=transposed,
        legal_basis=basis,  # type: ignore[arg-type]
        obligation_summary="summary",
        national_position="national",
        policy_commitment="policy",
        caveats=[],
        citations=[Citation(citation="src", supports="claim")],
    )


def stub_tool(name: str, payload: Any) -> StructuredTool:
    async def _invoke(**_: Any) -> Any:
        return payload

    return StructuredTool(
        name=name, description=name, args_schema={"properties": {}}, coroutine=_invoke
    )


def content_block(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


# --- the transposition gate ---


def test_an_agreeing_position_passes() -> None:
    check_transposition_agrees(position(transposed=False), {"transposed": False}, "ES")


def test_a_contradicted_position_is_refused() -> None:
    """Everything the position concludes rests on this fact."""
    with pytest.raises(RegulatoryError, match="transposed"):
        check_transposition_agrees(position(transposed=True), {"transposed": False}, "ES")


def test_a_missing_status_is_read_as_not_transposed() -> None:
    """Absent evidence of transposition is not evidence of transposition."""
    with pytest.raises(RegulatoryError):
        check_transposition_agrees(position(transposed=True), {}, "DE")


# --- the position schema ---


def test_a_position_without_a_citation_cannot_be_built() -> None:
    """A legal claim with no source must be unrepresentable, not merely bad."""
    with pytest.raises(ValueError):
        RegulatoryPosition(
            jurisdiction="ES",
            transposed=False,
            legal_basis="employer_policy_only",
            obligation_summary="s",
            national_position="n",
            policy_commitment="p",
            citations=[],
        )


def test_an_unknown_legal_basis_is_refused() -> None:
    with pytest.raises(ValueError):
        position(basis="probably_fine")


def test_the_position_schema_is_closed() -> None:
    with pytest.raises(ValueError):
        RegulatoryPosition(
            jurisdiction="ES",
            transposed=False,
            legal_basis="employer_policy_only",
            obligation_summary="s",
            national_position="n",
            policy_commitment="p",
            citations=[Citation(citation="c", supports="s")],
            confidence=0.9,  # type: ignore[call-arg]
        )


# --- context formatting ---


def test_a_citation_precedes_the_passage_it_labels() -> None:
    """Read as part of the passage rather than as droppable metadata."""
    rendered = _format_passages(
        [{"citation": "Directive (EU) 2023/970, Article 7", "text": "Workers shall..."}]
    )

    assert rendered.index("Article 7") < rendered.index("Workers shall")


def test_an_empty_retrieval_says_so() -> None:
    """Silence would read as "the law is silent", which is a different claim."""
    assert "No relevant passages" in _format_passages([])


def test_graph_findings_are_passed_as_data() -> None:
    rendered = _format_graph({"transposed": False}, [], [{"article": 5}], [{"section": 8}])
    parsed = json.loads(rendered)

    assert parsed["jurisdiction_status"] == {"transposed": False}
    assert parsed["obligations_without_national_basis"] == [{"article": 5}]


# --- tool results ---


async def test_a_refusal_is_raised_not_summarised() -> None:
    refusal = "Error calling tool 'x': unknown jurisdiction 'XX'"
    tools = {"t": stub_tool("t", content_block(refusal))}

    with pytest.raises(RegulatoryError, match="unknown jurisdiction"):
        await _call(tools, "t")


async def test_unparseable_output_is_refused() -> None:
    tools = {"t": stub_tool("t", content_block("<html>error</html>"))}
    with pytest.raises(RegulatoryError, match="unparseable"):
        await _call(tools, "t")


async def test_a_missing_tool_is_refused() -> None:
    with pytest.raises(RegulatoryError, match="not available"):
        await _call({}, "absent")


# --- the prompt ---


def test_the_prompt_renders_within_its_budget() -> None:
    values = {
        "jurisdiction": "ES",
        "request_text": "What is the average pay by sex at my level?",
        "retrieved_context": "[Article 7]\nWorkers shall have the right...",
        "graph_context": '{"transposed": false}',
    }

    rendered = REGULATORY_PROMPT.render(**values)
    assert "ES" in rendered
    assert REGULATORY_PROMPT.fits(**values)
    assert REGULATORY_PROMPT.identifier == "regulatory_position@v1"


def test_the_prompt_carries_the_shared_rules() -> None:
    """Composed, not copied — no agent's copy of these can drift."""
    template = REGULATORY_PROMPT.template

    assert "You do not calculate" in template
    assert "Error calling tool" in template
    assert "not yet transposed" in template


def test_the_prompt_states_the_decision_rule() -> None:
    """The rule that stopped a subject-matter regime being read as a duty."""
    template = REGULATORY_PROMPT.template

    assert "could this\nrequester compel a response" in template
    assert "is not this basis" in template
