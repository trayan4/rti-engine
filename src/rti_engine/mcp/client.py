"""Client that presents both MCP servers' tools to LangChain agents.

Servers run as separate processes and are reached over stdio. That
separation is the point: the analytics server holds the dataset and the
authorization rules, and an agent process cannot reach around the tool
surface to touch either.

Tools are loaded once and cached. Discovery spawns two subprocesses and
performs a handshake with each, which is not something to repeat per
agent invocation.
"""

import copy
import sys
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection, StdioConnection

ANALYTICS_SERVER = "rti_engine.mcp.analytics_server"
KNOWLEDGE_SERVER = "rti_engine.mcp.knowledge_server"

IDENTITY_FIELDS: tuple[str, ...] = ("requester_employee_id", "tier")
"""Arguments the application supplies, never the agent.

Left as ordinary tool parameters these are filled in by the model, which
means the authorization check runs against an identity the agent chose.
An injected agent would simply send a different employee id. Binding them
here restores the guarantee ADR-0004 states: identity comes from the
authenticated principal and nothing an agent produces can change it.
"""


def server_connections() -> dict[str, Connection]:
    """Describe how to launch each server.

    The current interpreter is used rather than a launcher command, so the
    servers run in the same virtual environment as their caller regardless
    of how that caller was started.
    """
    return {
        "analytics": StdioConnection(
            transport="stdio", command=sys.executable, args=["-m", ANALYTICS_SERVER]
        ),
        "knowledge": StdioConnection(
            transport="stdio", command=sys.executable, args=["-m", KNOWLEDGE_SERVER]
        ),
    }


def get_mcp_client() -> MultiServerMCPClient:
    """Return a client configured for both servers."""
    return MultiServerMCPClient(server_connections())


async def load_tools() -> list[BaseTool]:
    """Load every tool from both servers as LangChain tools.

    Returned in server order, so the set an agent is bound to is stable
    across runs rather than dependent on process scheduling.
    """
    return await get_mcp_client().get_tools()


async def load_tools_by_name() -> dict[str, BaseTool]:
    """Load tools keyed by name, for selective binding.

    Agents are bound to the subset their role needs rather than to
    everything: a drafter with a remediation tool in scope is a drafter
    that can call it.
    """
    return {tool.name: tool for tool in await load_tools()}


def _json_schema(tool: BaseTool) -> dict[str, Any]:
    """Return a tool's argument schema as a plain dictionary.

    LangChain permits three forms — a JSON-schema dict, a Pydantic v2
    model, or a v1 model — and the two model versions expose their schema
    under different method names.
    """
    schema = tool.args_schema
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema

    generate = getattr(schema, "model_json_schema", None) or getattr(schema, "schema", None)
    if generate is None:
        return {}

    produced: dict[str, Any] = dict(generate())
    return produced


def _takes_identity(tool: BaseTool) -> bool:
    """Report whether a tool expects the caller's identity."""
    properties = _json_schema(tool).get("properties", {})
    return any(field in properties for field in IDENTITY_FIELDS)


def _strip_identity_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove the identity arguments from a schema.

    The model is not asked for them and is not shown them, so there is
    nothing for it to get wrong or to be persuaded to change.
    """
    stripped = copy.deepcopy(schema)

    properties = stripped.get("properties")
    if isinstance(properties, dict):
        for field in IDENTITY_FIELDS:
            properties.pop(field, None)

    required = stripped.get("required")
    if isinstance(required, list):
        stripped["required"] = [name for name in required if name not in IDENTITY_FIELDS]

    return stripped


def bind_principal(tool: BaseTool, employee_id: str, tier: str) -> BaseTool:
    """Return the tool with the caller's identity fixed.

    Any identity arguments the agent supplies are discarded before
    dispatch and replaced with the authenticated values. Tools that do not
    take an identity are returned unchanged.
    """
    if not _takes_identity(tool):
        return tool

    async def _invoke(**kwargs: Any) -> Any:
        arguments = {name: value for name, value in kwargs.items() if name not in IDENTITY_FIELDS}
        arguments["requester_employee_id"] = employee_id
        arguments["tier"] = tier
        return await tool.ainvoke(arguments)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=_strip_identity_fields(_json_schema(tool)),
        coroutine=_invoke,
    )


async def load_tools_for(employee_id: str, tier: str) -> list[BaseTool]:
    """Load every tool, bound to one authenticated requester.

    This is what agents receive. The unbound `load_tools` remains for
    tests and for tools that carry no identity.
    """
    return [bind_principal(tool, employee_id, tier) for tool in await load_tools()]
