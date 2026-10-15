# eval/baseline_agent.py
"""
The "before" system to compare the router against: one agent, one giant
system prompt, every tool exposed every turn - which is what main.py's
ConversationAgent effectively did before the router existed. This file
exists purely as an eval fixture; it is not used by the running agent.
"""

from openai import AsyncOpenAI

from agents.specialists.base import SpecialistAgent
from agents.specialists.booking import BOOKING_TOOLS, check_booking_status
from agents.specialists.support import SUPPORT_TOOLS, search_faq
from agents.tools import ToolHandlers, get_tool_definitions

BASELINE_SYSTEM_PROMPT = """You are a general-purpose voice assistant.
Handle travel planning, booking status lookups, support/FAQ questions,
and casual conversation, using whichever tool (if any) fits the user's
request."""


class BaselineAgent(SpecialistAgent):
    name = "baseline_agent"
    system_prompt = BASELINE_SYSTEM_PROMPT


def build_baseline_agent(client: AsyncOpenAI, config) -> BaselineAgent:
    travel_handlers = ToolHandlers(config)
    return BaselineAgent(
        client=client,
        tools=[*get_tool_definitions(), *BOOKING_TOOLS, *SUPPORT_TOOLS],
        handlers={
            "pick_random_place": travel_handlers.pick_random_place,
            "plan_itinerary": travel_handlers.plan_itinerary,
            "check_booking_status": check_booking_status,
            "search_faq": search_faq,
        },
    )
