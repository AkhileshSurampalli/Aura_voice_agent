# agents/specialists/travel.py
from openai import AsyncOpenAI

from ..tools import ToolHandlers, get_tool_definitions
from .base import SpecialistAgent

SYSTEM_PROMPT = """You are a travel-planning specialist.
Help the user pick a destination (randomly if they want a surprise) and
plan a short itinerary for it. Keep responses concise and conversational -
summarize itineraries rather than reading them verbatim. Stay focused on
travel; if the user asks about something unrelated, say so briefly."""


class TravelPlannerAgent(SpecialistAgent):
    name = "travel_agent"
    system_prompt = SYSTEM_PROMPT


def build_travel_agent(client: AsyncOpenAI, config) -> TravelPlannerAgent:
    handlers = ToolHandlers(config)
    return TravelPlannerAgent(
        client=client,
        tools=get_tool_definitions(),
        handlers={
            "pick_random_place": handlers.pick_random_place,
            "plan_itinerary": handlers.plan_itinerary,
        },
    )
