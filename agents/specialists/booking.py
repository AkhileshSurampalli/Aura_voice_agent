# agents/specialists/booking.py
"""
Booking specialist - mock tool backed by an in-memory table so the router
and tool-calling loop can be exercised end-to-end without a real booking
system. Swap `_MOCK_BOOKINGS` / `check_booking_status` for a real backend
call when one exists.
"""

from openai import AsyncOpenAI

from .base import SpecialistAgent

SYSTEM_PROMPT = """You are a booking-status specialist. Use the
check_booking_status tool to look up a booking by its reference code and
report the status conversationally. If the user doesn't have a reference
code, ask for one - don't guess."""

_MOCK_BOOKINGS = {
    "AB123": {"status": "confirmed", "destination": "Lisbon, Portugal", "date": "2026-11-02"},
    "CD456": {"status": "pending payment", "destination": "Kyoto, Japan", "date": "2026-12-10"},
}

BOOKING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_booking_status",
            "description": "Look up the status of a travel booking by its reference code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string", "description": "Booking reference code, e.g. 'AB123'."},
                },
                "required": ["reference"],
            },
        },
    }
]


async def check_booking_status(reference: str = "") -> dict:
    booking = _MOCK_BOOKINGS.get(reference.upper())
    if not booking:
        return {"error": f"No booking found for reference '{reference}'."}
    return {"reference": reference.upper(), **booking}


class BookingAgent(SpecialistAgent):
    name = "booking_agent"
    system_prompt = SYSTEM_PROMPT


def build_booking_agent(client: AsyncOpenAI, config) -> BookingAgent:
    return BookingAgent(
        client=client,
        tools=BOOKING_TOOLS,
        handlers={"check_booking_status": check_booking_status},
    )
