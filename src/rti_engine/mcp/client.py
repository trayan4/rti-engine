"""Client that presents both MCP servers' tools to LangChain agents.

Servers run as separate processes and are reached over stdio. That
separation is the point: the analytics server holds the dataset and the
authorization rules, and an agent process cannot reach around the tool
surface to touch either.

Tools are loaded once and cached. Discovery spawns two subprocesses and
performs a handshake with each, which is not something to repeat per
agent invocation.
"""

import sys

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection, StdioConnection

ANALYTICS_SERVER = "rti_engine.mcp.analytics_server"
KNOWLEDGE_SERVER = "rti_engine.mcp.knowledge_server"


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
