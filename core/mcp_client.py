"""The one MCP stdio client, used by the Researcher and check_server.py."""

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

from core.config import MAX_SNIPPETS, MCP_SERVER_SCRIPT
from core.schemas import Snippet

SEARCH_TOOL = "search_corpus"


class MCPToolError(RuntimeError):
    """The MCP server reported an error for a tool call."""


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """Launch the corpus server over stdio and yield an initialised session."""
    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_SCRIPT)])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def _items(result: CallToolResult) -> list[dict]:
    """Tool output as dicts, from structured content when the server sends it."""
    if result.structuredContent is not None:
        return result.structuredContent.get("result", [])
    return [json.loads(c.text) for c in result.content if isinstance(c, TextContent)]


async def search(session: ClientSession, query: str, limit: int = MAX_SNIPPETS) -> list[Snippet]:
    """Call `search_corpus` and validate each result as a Snippet."""
    result = await session.call_tool(SEARCH_TOOL, {"query": query, "limit": limit})
    if result.isError:
        text = " ".join(c.text for c in result.content if isinstance(c, TextContent))
        raise MCPToolError(f"{SEARCH_TOOL} failed: {text}")
    return [Snippet.model_validate(item) for item in _items(result)]
