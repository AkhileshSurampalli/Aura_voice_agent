# agents/specialists/__init__.py
from .base import SpecialistAgent
from .travel import build_travel_agent
from .smalltalk import build_smalltalk_agent
from .booking import build_booking_agent
from .support import build_support_agent
from .escalation import build_escalation_agent

__all__ = [
    "SpecialistAgent",
    "build_travel_agent",
    "build_smalltalk_agent",
    "build_booking_agent",
    "build_support_agent",
    "build_escalation_agent",
]
