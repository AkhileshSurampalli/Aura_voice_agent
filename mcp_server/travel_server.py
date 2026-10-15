# mcp_server/travel_server.py
"""
Minimal MCP server exposing the travel tools (worldwide random place picker +
itinerary planner) that already live in agents/tools.py.

This is a thin MCP wrapper, not a reimplementation - it reuses ToolHandlers
so there's exactly one place the actual tool logic lives.

Run standalone (for testing with any MCP client, e.g. `mcp dev`):
    python -m mcp_server.travel_server

The ConversationAgent launches this same module as a subprocess over stdio
(see agents/mcp_client.py) instead of calling ToolHandlers directly.

To swap in a real/official MCP server later (e.g. Google's Maps MCP server,
once you have a Google Maps Platform API key + billing set up), you don't
need to touch this file - just point MCPToolClient at that server's launch
command instead of this module.
"""

import logging
import os
from typing import Optional

from mcp.server.mcpserver import MCPServer  # mcp>=2.0 (FastMCP was renamed to MCPServer)

from agents.tools import ToolHandlers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _EnvConfig:
    """Minimal stand-in for AgentConfig - the MCP server runs as its own
    process, so it only needs the bits ToolHandlers actually reads."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY")


mcp = MCPServer("travel-tools")
_handlers = ToolHandlers(_EnvConfig())


@mcp.tool()
async def pick_random_place() -> dict:
    """Randomly select a travel destination anywhere in the world."""
    return await _handlers.pick_random_place()


@mcp.tool()
async def plan_itinerary(place: str, days: int = 3, interests: Optional[str] = None) -> dict:
    """Generate a concise day-by-day itinerary for a travel destination.

    Args:
        place: Destination to plan for, e.g. "Kyoto, Japan".
        days: Number of days the trip should cover.
        interests: Optional traveler interests, e.g. "food, hiking, history".
    """
    return await _handlers.plan_itinerary(place=place, days=days, interests=interests)


if __name__ == "__main__":
    logger.info("Starting travel-tools MCP server on stdio...")
    mcp.run(transport="stdio")
