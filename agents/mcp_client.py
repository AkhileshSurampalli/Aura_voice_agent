# agents/mcp_client.py
"""
Minimal MCP client used by ConversationAgent to talk to an MCP server over
stdio - by default our own mcp_server/travel_server.py, but any MCP server
(including an official one, e.g. Google's Maps MCP server) can be dropped in
by changing `command`/`args`.
"""

import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Project root (agents/.. ) - needed as the subprocess's cwd so
# `-m mcp_server.travel_server` reliably resolves as a package regardless of
# whatever directory the parent process happened to be launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MCPToolClient:
    """Launches an MCP server as a subprocess and exposes its tools."""

    def __init__(self, command: str = None, args: list[str] = None, cwd: str = None):
        # Default: run our travel-tools server as `python -m mcp_server.travel_server`
        # using the *same* interpreter this process is running under.
        self.command = command or sys.executable
        self.args = args if args is not None else ["-m", "mcp_server.travel_server"]
        self.cwd = cwd or _PROJECT_ROOT
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def connect(self):
        """Start the MCP server subprocess and open a session with it."""
        # stdio_client only passes a minimal default environment to the
        # child by default (a deliberate security default) - the server
        # needs OPENAI_API_KEY etc., so forward our own env explicitly.
        # cwd is pinned to the project root so `-m mcp_server.travel_server`
        # resolves correctly regardless of the parent process's cwd.
        params = StdioServerParameters(
            command=self.command, args=self.args, cwd=self.cwd, env=os.environ.copy()
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        logger.info(f"MCP client connected via `{self.command} {' '.join(self.args)}`")

    async def list_openai_tools(self) -> list[dict]:
        """Fetch the server's tools and convert them to OpenAI function-calling schema."""
        result = await self.session.list_tools()
        tools = []
        for t in result.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )
        logger.info(f"MCP server exposes tools: {[t.name for t in result.tools]}")
        return tools

    def make_handler(self, tool_name: str) -> Callable:
        """Build an async handler suitable for llm.register_function(name, handler)
        that proxies the call through to the MCP server."""

        async def handler(**kwargs) -> Any:
            result = await self.session.call_tool(tool_name, kwargs)
            return self._extract_result(result)

        return handler

    @staticmethod
    def _extract_result(result) -> Any:
        """MCP tool results come back as a list of content blocks. Our tools
        return plain dicts, which FastMCP serializes as a single JSON text
        block - unwrap that back into a Python object for the LLM."""
        if result.isError:
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return {"error": " ".join(texts) or "MCP tool call failed"}

        for block in result.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    return block.text
        return None

    async def close(self):
        """Shut down the MCP server subprocess and session.

        Note: anyio's stdio transport can raise a benign
        'cancel scope in a different task' RuntimeError during teardown on
        some platforms (observed on Windows) - it happens after the
        subprocess has already been signaled to exit, so it's swallowed
        here rather than propagated to callers (e.g. ConversationAgent.stop).
        """
        try:
            await self._stack.aclose()
        except RuntimeError as e:
            logger.warning(f"MCP client shutdown warning (non-fatal): {e}")
