"""Shared handling of MCP tool results.

Every agent calling a tool has to distinguish three outcomes: a result, a
refusal, and something unparseable. Refusals in particular arrive as
ordinary text rather than as exceptions, so an agent that does not check
will narrate an authorization failure as though it were a finding.

That check was written separately in each agent. Three copies of a rule
this consequential is three chances for one of them to be wrong, so it
lives here once.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool

from rti_engine.observability.otel import span

TOOL_ERROR_PREFIX = "Error calling tool"


class ToolCallError(RuntimeError):
    """Raised when a tool refuses, is absent, or returns unusable output."""


def result_text(result: Any) -> str:
    """Extract the payload from an MCP tool result.

    Results arrive as content blocks, and errors arrive the same way.
    """
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return str(result[0].get("text", ""))
    return str(result)


async def call_tool(tools: dict[str, BaseTool], name: str, **arguments: Any) -> Any:
    """Call one tool and return its parsed result.

    Raises on a refusal rather than returning it, so a caller cannot pass
    an error message on as though it were data.
    """
    tool = tools.get(name)
    if tool is None:
        raise ToolCallError(f"tool {name!r} is not available")

    # Traced from the caller rather than inside the server. The context is
    # propagated across the process boundary, so server-side spans would
    # join this trace if the servers were instrumented — but a call timed
    # from here already answers the question worth asking, which is where
    # a request spent its time.
    with span(f"tool.{name}", **{"rti.tool": name}) as current:
        text = result_text(await tool.ainvoke(arguments))

        if text.startswith(TOOL_ERROR_PREFIX):
            # Marked so a refusal is distinguishable from a fault. Both
            # end the call; only one means something is wrong.
            current.set_attribute("rti.refused", True)
            raise ToolCallError(text)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ToolCallError(f"{name} returned unparseable output: {text[:200]}") from error

        current.set_attribute("rti.result_bytes", len(text))
        return parsed


def format_passages(passages: list[dict[str, Any]]) -> str:
    """Render retrieved passages with their citations attached.

    The citation precedes the text so a model reads the attribution as
    part of the passage rather than as metadata it may drop.

    An empty result says so explicitly: silence would read as "the sources
    are silent on this", which is a much stronger claim than "nothing was
    retrieved".
    """
    if not passages:
        return "No relevant passages were retrieved."

    return "\n\n".join(
        f"[{item.get('citation', 'uncited')}]\n{item.get('text', '')}" for item in passages
    )
