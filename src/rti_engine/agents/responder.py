"""Tier 0 and Tier 1 responses.

Two lighter paths than the disclosure pipeline. A Tier 0 request asks how
pay is set or what the law provides, and is answered from the corpus
alone. A Tier 1 request adds the requester's own record, and nothing
about anyone else.

Both return the same shape as the Tier 2 letter. Three response types
would mean three renderers, three validators and three sets of review
rules; one means every tier passes through the same checks.

Neither path can reach comparator data, and not because the prompt says
so: the tools refuse it at T0 and T1, and a request that needed it was
floored to T2 before reaching here.
"""

import json
from typing import Any

from rti_engine.agents.drafter import DraftLetter
from rti_engine.agents.prompts import (
    CITATION_RULES,
    GROUNDING_RULES,
    JURISDICTION_RULES,
    Prompt,
)
from rti_engine.agents.tools import ToolCallError, call_tool, format_passages
from rti_engine.llm.factory import ModelRole, get_structured_model
from rti_engine.mcp.client import ANALYTICS, KNOWLEDGE, tool_session

CURRENCY_PLACES = 2


class ResponseError(RuntimeError):
    """Raised when a response cannot be produced."""


INFORMATIONAL_PROMPT = Prompt(
    name="informational_response",
    version=1,
    description="Answer a general question about pay policy or pay transparency law.",
    inputs=("request_text", "jurisdiction", "retrieved_context"),
    max_tokens=8000,
    template=f"""\
You answer an employee's general question about how pay is set and what
pay transparency law provides. The reader is the employee, not a lawyer.

You have no access to any employee's pay data, including the reader's.
Answer the question that was asked from the sources given.

{CITATION_RULES}

{JURISDICTION_RULES}

{GROUNDING_RULES}

If the sources do not answer the question, say so and say what the
employee can ask for instead. Do not fill a gap with general knowledge
about pay or employment law.

If answering properly would require the reader's own pay or a comparison
against colleagues, say that such a request can be made and what it would
cover. Do not speculate about their situation.

Leave figures_used empty unless a figure appears in a retrieved source, in
which case cite it.

## Jurisdiction

{{jurisdiction}}

## The question

{{request_text}}

## Retrieved sources

{{retrieved_context}}""",
)


OWN_DATA_PROMPT = Prompt(
    name="own_data_response",
    version=1,
    description="Answer a request for the requester's own pay information.",
    inputs=("request_text", "jurisdiction", "own_record", "retrieved_context"),
    max_tokens=8000,
    template=f"""\
You answer an employee's request for their own pay information. The reader
is the employee.

{GROUNDING_RULES}

{CITATION_RULES}

## Declaring figures

Every number in the response must appear in the record below, and every
one must have an entry in figures_used giving the value as written, the
field it came from, and what it represents.

Write figures exactly as they appear. Do not round or approximate.

## What this response covers

The employee's own recorded pay, working pattern and length of service,
and the criteria used to determine pay and progression.

It does not cover what anyone else is paid. You have no comparator data,
and none was requested. If the response would be improved by a comparison,
say that a comparison against colleagues doing equal work can be requested
separately, and stop there.

## The request

{{request_text}}

## Jurisdiction

{{jurisdiction}}

## The employee's record

{{own_record}}

## Retrieved sources

{{retrieved_context}}""",
)


PAY_SETTING_QUERY = (
    "How is base pay determined, how does pay progress, and what criteria "
    "are used for pay-setting decisions?"
)


async def _retrieve(employee_id: str, tier: str, jurisdiction: str, query: str) -> str:
    """Retrieve passages relevant to a question, scoped to one country."""
    async with tool_session(employee_id, tier, servers=[KNOWLEDGE]) as tools:
        try:
            passages = await call_tool(
                tools,
                "search_regulatory_knowledge",
                query=query,
                jurisdiction=jurisdiction,
            )
        except ToolCallError as error:
            raise ResponseError(f"retrieval failed: {error}") from error

    if not isinstance(passages, list):
        raise ResponseError("retrieval did not return a list of passages")
    return format_passages(passages)


async def _own_record(employee_id: str) -> dict[str, Any]:
    """Fetch the requester's own pay record at Tier 1."""
    async with tool_session(employee_id, "T1", servers=[ANALYTICS]) as tools:
        try:
            record = await call_tool(tools, "get_own_pay_record")
        except ToolCallError as error:
            raise ResponseError(f"could not read own record: {error}") from error

    if not isinstance(record, dict) or not record.get("found"):
        raise ResponseError(f"no pay record found for {employee_id}")
    return record


def own_record_facts(record: dict[str, Any]) -> dict[str, Any]:
    """Shape the record into the fields a response may quote."""
    return {
        "country": record["country"],
        "job_family": record["job_family"],
        "level": record["level"],
        "working_pattern": record["working_pattern"],
        "fte": record["fte"],
        "tenure_years": round(float(record["tenure_years"]), 1),
        "base_salary_fte_eur": round(float(record["base_salary_fte_eur"]), CURRENCY_PLACES),
        "base_salary_actual_eur": round(float(record["base_salary_actual_eur"]), CURRENCY_PLACES),
        "bonus_actual_eur": round(float(record["bonus_actual_eur"]), CURRENCY_PLACES),
        "total_comp_actual_eur": round(float(record["total_comp_actual_eur"]), CURRENCY_PLACES),
        "currency": "EUR",
    }


async def _generate(prompt: Prompt, **values: Any) -> DraftLetter:
    """Render a prompt and produce a response from it."""
    model = get_structured_model(ModelRole.REASONING, DraftLetter)
    letter = await model.ainvoke(prompt.render(**values))

    if not isinstance(letter, DraftLetter):
        raise ResponseError(f"{prompt.name} did not return a response")
    return letter


async def answer_informational(
    employee_id: str, request_text: str, jurisdiction: str
) -> DraftLetter:
    """Answer a Tier 0 request from the corpus alone."""
    if not request_text.strip():
        raise ResponseError("request text is empty")

    retrieved = await _retrieve(employee_id, "T0", jurisdiction, request_text)

    return await _generate(
        INFORMATIONAL_PROMPT,
        request_text=request_text,
        jurisdiction=jurisdiction,
        retrieved_context=retrieved,
    )


async def answer_own_data(employee_id: str, request_text: str, jurisdiction: str) -> DraftLetter:
    """Answer a Tier 1 request from the requester's own record."""
    if not request_text.strip():
        raise ResponseError("request text is empty")

    record = await _own_record(employee_id)
    retrieved = await _retrieve(employee_id, "T1", jurisdiction, PAY_SETTING_QUERY)

    return await _generate(
        OWN_DATA_PROMPT,
        request_text=request_text,
        jurisdiction=jurisdiction,
        own_record=json.dumps(own_record_facts(record), indent=2),
        retrieved_context=retrieved,
    )
