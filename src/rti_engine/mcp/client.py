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
import os
import sys
from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    Connection,
    StdioConnection,
    StreamableHttpConnection,
)
from langchain_mcp_adapters.tools import load_mcp_tools

from rti_engine.config.settings import get_settings
from rti_engine.observability.otel import propagation_env

ANALYTICS_SERVER = "rti_engine.mcp.analytics_server"
KNOWLEDGE_SERVER = "rti_engine.mcp.knowledge_server"

ANALYTICS = "analytics"
KNOWLEDGE = "knowledge"

IDENTITY_FIELDS: tuple[str, ...] = ("requester_employee_id", "tier")
"""Arguments the application supplies, never the agent.

Left as ordinary tool parameters these are filled in by the model, which
means the authorization check runs against an identity the agent chose.
An injected agent would simply send a different employee id. Binding them
here restores the guarantee ADR-0004 states: identity comes from the
authenticated principal and nothing an agent produces can change it.
"""


def server_connections() -> dict[str, Connection]:
    """Describe how to reach each server.

    Two modes for the same servers. Locally they are subprocesses spawned
    over stdio, which needs no ports and dies with its parent. In a
    deployment they are separate services reached over HTTP, because a
    pipe does not cross a container boundary.

    Which mode applies is decided by whether the URLs are configured, so
    nothing has to be told which environment it is in.
    """
    settings = get_settings()

    if settings.analytics_mcp_url and settings.knowledge_mcp_url:
        return {
            ANALYTICS: StreamableHttpConnection(
                transport="streamable_http", url=settings.analytics_mcp_url
            ),
            KNOWLEDGE: StreamableHttpConnection(
                transport="streamable_http", url=settings.knowledge_mcp_url
            ),
        }

    # The trace context travels as environment variables. A subprocess
    # started without it begins a fresh trace, and the request's picture
    # comes apart at exactly the boundary worth seeing across.
    env = {**os.environ, **propagation_env()}

    return {
        ANALYTICS: StdioConnection(
            transport="stdio",
            command=sys.executable,
            args=["-m", ANALYTICS_SERVER],
            env=env,
        ),
        KNOWLEDGE: StdioConnection(
            transport="stdio",
            command=sys.executable,
            args=["-m", KNOWLEDGE_SERVER],
            env=env,
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


def _unwrap_exception_group(error: BaseException) -> BaseException:
    """Return the single underlying exception from nested exception groups.

    The MCP stdio client runs its streams in an anyio task group, which
    repacks anything raised in the caller's body into a
    BaseExceptionGroup on the way out. A caller writing
    ``except AnalysisError`` would therefore never catch its own error.

    Groups holding more than one exception are left alone: there is no
    single cause to surface, and collapsing them would hide failures.
    """
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


@asynccontextmanager
async def tool_session(
    employee_id: str, tier: str, servers: Sequence[str] | None = None
) -> AsyncGenerator[dict[str, BaseTool]]:
    """Open one session per server and yield tools bound to a requester.

    Tools loaded outside a session open a fresh connection per call, which
    over stdio means spawning both server subprocesses, completing an MCP
    handshake and tearing them down again — roughly two seconds of process
    overhead for a call that takes milliseconds.

    Here the servers start once and stay up for the caller's whole unit of
    work. An analysis making five calls pays the cost once rather than
    five times, and the saving grows with the number of calls an agent
    makes.

    Tools are keyed by name, since callers select them rather than
    iterating.

    Callers that need one server should say so. Each server is a separate
    Python process importing its own heavy dependencies, so opening one
    that will not be used costs seconds for nothing.
    """
    client = get_mcp_client()

    try:
        async with AsyncExitStack() as stack:
            loaded: list[BaseTool] = []
            for server in servers if servers is not None else server_connections():
                session = await stack.enter_async_context(client.session(server))
                loaded.extend(await load_mcp_tools(session))

            yield {tool.name: bind_principal(tool, employee_id, tier) for tool in loaded}
    except BaseExceptionGroup as group:
        unwrapped = _unwrap_exception_group(group)
        if unwrapped is group:
            raise
        raise unwrapped from None
