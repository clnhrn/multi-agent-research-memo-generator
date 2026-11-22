"""Demonstrate real MCP stdio discovery, resource reads and tool invocation."""

import asyncio

from core.mcp_client import mcp_session, search


async def main() -> None:
    """List the server's resources and tools, read each resource and run one search."""
    async with mcp_session() as session:
        resources = (await session.list_resources()).resources
        tools = (await session.list_tools()).tools
        print("Resources:", [f"{r.uri} ({r.mimeType})" for r in resources])
        print("Tools:", [t.name for t in tools])
        for resource in resources:
            result = await session.read_resource(resource.uri)
            print("Read", resource.uri, "items:", len(result.contents))
        for snippet in await search(session, "solar", limit=3):
            print(f"Search result: [{snippet.source}] {snippet.text[:70]}...")


if __name__ == "__main__":
    asyncio.run(main())
