# agents/router.py
"""
Multi-agent router: classifies each user turn's intent with one cheap LLM
call, then dispatches to exactly one of 5 specialist agents (each with its
own system prompt and scoped toolset - see agents/specialists/).

This is the piece that makes the "reduced unnecessary LLM calls" and
"task-specific toolsets improve tool-selection reliability" resume claims
checkable: every call through here is counted (agents/metrics.py), and the
baseline to compare against is a single agent with every tool exposed
every turn (see eval/run_baseline_vs_router.py).
"""

import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

from .config import AgentConfig
from .metrics import metrics
from .specialists import (
    build_booking_agent,
    build_escalation_agent,
    build_smalltalk_agent,
    build_support_agent,
    build_travel_agent,
)

logger = logging.getLogger(__name__)

INTENTS = ["travel", "booking", "support", "smalltalk", "escalation"]

ROUTER_SYSTEM_PROMPT = f"""Classify the user's latest message into exactly
one of these intents: {", ".join(INTENTS)}.
- travel: wants a destination suggestion or an itinerary
- booking: asking about an existing booking's status
- support: general help/FAQ question (cancellation, refunds, policies)
- smalltalk: casual conversation, greetings, chit-chat
- escalation: frustrated, asking for a human, or doesn't fit the above
Respond with ONLY the intent word, nothing else."""


class RouterState(TypedDict):
    messages: list[dict]
    session_id: str
    intent: str
    response: str


class AgentRouter:
    """Builds and runs the LangGraph graph: classify -> dispatch -> specialist."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.specialists = {
            "travel": build_travel_agent(self.client, config),
            "booking": build_booking_agent(self.client, config),
            "support": build_support_agent(self.client, config),
            "smalltalk": build_smalltalk_agent(self.client, config),
            "escalation": build_escalation_agent(self.client, config),
        }
        self.graph = self._build_graph()

    async def _classify(self, state: RouterState) -> RouterState:
        metrics.record_call("router", session_id=state["session_id"])
        last_user_msg = next(
            (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), ""
        )
        response = await self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": last_user_msg},
            ],
            max_tokens=10,
            temperature=0,
        )
        intent = (response.choices[0].message.content or "").strip().lower()
        if intent not in INTENTS:
            logger.warning(f"Router produced unknown intent '{intent}', falling back to escalation")
            intent = "escalation"
        state["intent"] = intent
        return state

    def _route(self, state: RouterState) -> str:
        return state["intent"]

    def _make_specialist_node(self, intent: str):
        async def node(state: RouterState) -> RouterState:
            agent = self.specialists[intent]
            state["response"] = await agent.run(state["messages"], session_id=state["session_id"])
            return state

        return node

    def _build_graph(self):
        graph = StateGraph(RouterState)
        graph.add_node("classify", self._classify)
        for intent in INTENTS:
            graph.add_node(intent, self._make_specialist_node(intent))
            graph.add_edge(intent, END)

        graph.set_entry_point("classify")
        graph.add_conditional_edges("classify", self._route, {i: i for i in INTENTS})
        return graph.compile()

    async def handle_turn(self, messages: list[dict], session_id: str = "default") -> dict:
        """Run one user turn through the graph. `messages` is the running
        conversation history (OpenAI-style role/content dicts), most recent
        user message last."""
        result = await self.graph.ainvoke(
            {"messages": messages, "session_id": session_id, "intent": "", "response": ""}
        )
        return {"intent": result["intent"], "response": result["response"]}
