# agents/specialists/support.py
"""
Support/FAQ specialist - mock tool backed by a static FAQ table. Swap for
a real knowledge-base search (e.g. a vector store lookup) when one exists.
"""

from openai import AsyncOpenAI

from .base import SpecialistAgent

SYSTEM_PROMPT = """You are a customer-support specialist. Use the
search_faq tool to find relevant help-center answers before replying.
Be concise and point the user to a human agent if the FAQ doesn't cover
their question."""

_FAQ = [
    {"q": "cancel booking", "a": "You can cancel a booking up to 24 hours before departure for a full refund."},
    {"q": "change dates", "a": "Date changes are free once per booking; contact support for a second change."},
    {"q": "refund timeline", "a": "Refunds are processed within 5-7 business days back to the original payment method."},
]

SUPPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search the help-center FAQ for an answer to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords from the user's question."},
                },
                "required": ["query"],
            },
        },
    }
]


async def search_faq(query: str = "") -> dict:
    query_lower = query.lower()
    matches = [entry for entry in _FAQ if any(word in entry["q"] for word in query_lower.split())]
    if not matches:
        return {"results": [], "note": "No FAQ match - suggest escalating to a human agent."}
    return {"results": matches}


class SupportAgent(SpecialistAgent):
    name = "support_agent"
    system_prompt = SYSTEM_PROMPT


def build_support_agent(client: AsyncOpenAI, config) -> SupportAgent:
    return SupportAgent(
        client=client,
        tools=SUPPORT_TOOLS,
        handlers={"search_faq": search_faq},
    )
