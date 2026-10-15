# agents/metrics.py
"""
Minimal in-memory LLM-call counter.

This exists to make the "reduced unnecessary LLM calls by X%" claim
checkable rather than asserted: instrument both the old single-agent path
and the new routed path with the same counter, replay the same test
conversations through each, and diff the totals.

Usage:
    from .metrics import metrics

    metrics.record_call("router", session_id="s1")
    metrics.record_call("travel_agent", session_id="s1")
    ...
    metrics.summary()        # {"router": 12, "travel_agent": 5, ...}
    metrics.total()          # 17
    metrics.per_session()    # {"s1": 2, ...}
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _CallRecord:
    node: str
    session_id: str


class LLMCallMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[_CallRecord] = []

    def record_call(self, node: str, session_id: str = "default"):
        """Call this exactly once per LLM API call (not per turn - a turn
        that retries or chains calls should record each one)."""
        with self._lock:
            self._records.append(_CallRecord(node=node, session_id=session_id))
        logger.debug(f"[metrics] LLM call recorded: node={node} session={session_id}")

    def total(self) -> int:
        with self._lock:
            return len(self._records)

    def summary(self) -> dict[str, int]:
        """Call count per node (router, travel_agent, etc.)."""
        counts = defaultdict(int)
        with self._lock:
            for r in self._records:
                counts[r.node] += 1
        return dict(counts)

    def per_session(self) -> dict[str, int]:
        """Call count per session id - use this to compare conversation-by-
        conversation against a baseline run."""
        counts = defaultdict(int)
        with self._lock:
            for r in self._records:
                counts[r.session_id] += 1
        return dict(counts)

    def reset(self):
        with self._lock:
            self._records.clear()


# Module-level singleton - import and share across router/specialists/baseline.
metrics = LLMCallMetrics()
