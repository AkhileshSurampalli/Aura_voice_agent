# agents/specialists/escalation.py
"""
Escalation specialist - deliberately no tools, no reasoning loop needed.
Routed to when the classifier is unsure, the user is frustrated, or the
other specialists can't help. Kept as a specialist (not a hardcoded string)
so it can still personalize the hand-off message.
"""

from openai import AsyncOpenAI

from .base import SpecialistAgent

SYSTEM_PROMPT = """You are handling an escalation. Acknowledge the user's
issue briefly, apologize once, and tell them you're connecting them to a
human agent. Keep it to one or two sentences. No tools are available."""


class EscalationAgent(SpecialistAgent):
    name = "escalation_agent"
    system_prompt = SYSTEM_PROMPT


def build_escalation_agent(client: AsyncOpenAI, config) -> EscalationAgent:
    return EscalationAgent(client=client, tools=[], handlers={})
