# eval/run_baseline_vs_router.py
"""
Produces the actual number behind the "reduced unnecessary LLM calls by X%"
claim. Replays eval/conversations.py through:
  (a) BaselineAgent - one agent, every tool, every turn
  (b) AgentRouter    - classify + dispatch to a scoped specialist
counting LLM calls via agents/metrics.py in both, then reports the diff.

Run from the project root (needs a real OPENAI_API_KEY - this makes live
API calls and costs a small amount of money):
    python -m eval.run_baseline_vs_router
"""

import asyncio
import json

from openai import AsyncOpenAI

from agents.config import AgentConfig
from agents.metrics import metrics
from agents.router import AgentRouter
from eval.baseline_agent import build_baseline_agent
from eval.conversations import CONVERSATIONS


async def run_baseline(config: AgentConfig) -> dict[str, int]:
    metrics.reset()
    client = AsyncOpenAI(api_key=config.openai_api_key)
    agent = build_baseline_agent(client, config)

    for session_id, turns in CONVERSATIONS.items():
        history = []
        for turn in turns:
            history.append({"role": "user", "content": turn})
            reply = await agent.run(history, session_id=session_id)
            history.append({"role": "assistant", "content": reply})

    return metrics.per_session()


async def run_router(config: AgentConfig) -> dict[str, int]:
    metrics.reset()
    router = AgentRouter(config)

    for session_id, turns in CONVERSATIONS.items():
        history = []
        for turn in turns:
            history.append({"role": "user", "content": turn})
            result = await router.handle_turn(history, session_id=session_id)
            history.append({"role": "assistant", "content": result["response"]})

    return metrics.per_session()


async def main():
    config = AgentConfig()

    print("Running baseline (single agent, all tools)...")
    baseline_counts = await run_baseline(config)

    print("Running router (classify + 5 scoped specialists)...")
    router_counts = await run_router(config)

    baseline_total = sum(baseline_counts.values())
    router_total = sum(router_counts.values())
    reduction_pct = (
        100 * (baseline_total - router_total) / baseline_total if baseline_total else 0
    )

    report = {
        "baseline_calls_per_session": baseline_counts,
        "router_calls_per_session": router_counts,
        "baseline_total": baseline_total,
        "router_total": router_total,
        "reduction_pct": round(reduction_pct, 1),
    }
    print(json.dumps(report, indent=2))
    print(f"\n==> LLM call reduction: {report['reduction_pct']}%")


if __name__ == "__main__":
    asyncio.run(main())
