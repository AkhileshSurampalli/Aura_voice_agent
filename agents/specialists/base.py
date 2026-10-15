# agents/specialists/base.py
"""
Base class for a task-specific specialist agent.

Each specialist deliberately sees only its own system prompt and its own
tool subset - the point is to keep tool-selection surface small per turn
(fewer irrelevant tools in context -> fewer spurious/hallucinated tool
calls, and a tighter prompt -> fewer re-asks), which is what the router
exists to make possible.
"""

import json
import logging
from typing import Any, Callable, Optional

from openai import AsyncOpenAI

from ..metrics import metrics

logger = logging.getLogger(__name__)


class SpecialistAgent:
    name: str = "base"
    system_prompt: str = "You are a helpful assistant."
    model: str = "gpt-4.1"
    max_tool_hops: int = 3  # guard against runaway tool-call loops

    def __init__(self, client: AsyncOpenAI, tools: Optional[list[dict]] = None,
                 handlers: Optional[dict[str, Callable]] = None):
        self.client = client
        self.tools = tools or []
        self.handlers = handlers or {}

    async def run(self, messages: list[dict], session_id: str = "default") -> str:
        """Run this specialist's reasoning loop (including any tool calls)
        over the conversation so far, and return its final text reply."""
        convo = [{"role": "system", "content": self.system_prompt}, *messages]

        for _ in range(self.max_tool_hops):
            metrics.record_call(self.name, session_id=session_id)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=convo,
                tools=self.tools or None,
                max_tokens=300,
                temperature=0.7,
            )
            choice = response.choices[0].message

            if not choice.tool_calls:
                return choice.content or ""

            # Execute each requested tool call, feed results back, loop again.
            convo.append(choice.model_dump(exclude_none=True))
            for call in choice.tool_calls:
                handler = self.handlers.get(call.function.name)
                if handler is None:
                    result = {"error": f"Unknown tool '{call.function.name}'"}
                else:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await handler(**args)

                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )

        logger.warning(f"[{self.name}] hit max_tool_hops without a final reply")
        return "Sorry, I'm having trouble with that request right now."
