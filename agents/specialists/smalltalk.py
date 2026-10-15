# agents/specialists/smalltalk.py
from openai import AsyncOpenAI

from .base import SpecialistAgent

SYSTEM_PROMPT = """You are a friendly conversational voice assistant making
small talk. Keep replies short (<40 words), warm, and natural - no tools
are available to you, so never claim to look something up or take an
action; just converse."""


class SmallTalkAgent(SpecialistAgent):
    name = "smalltalk_agent"
    system_prompt = SYSTEM_PROMPT
    # No tools at all - this is the cheapest/fastest node in the graph.


def build_smalltalk_agent(client: AsyncOpenAI, config) -> SmallTalkAgent:
    return SmallTalkAgent(client=client, tools=[], handlers={})
