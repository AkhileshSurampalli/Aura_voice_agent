# agents/tools.py
"""
Multi-tool support for the conversational agent.

Exposes two tools to the LLM via OpenAI-style function calling:
  1. pick_random_place  - randomly selects a travel destination
  2. plan_itinerary     - generates a day-by-day itinerary for a given place

Usage (wired up in ConversationAgent.setup):
    from .tools import get_tool_definitions, ToolHandlers

    handlers = ToolHandlers(config)
    llm.register_function("pick_random_place", handlers.pick_random_place)
    llm.register_function("plan_itinerary", handlers.plan_itinerary)
    self.llm_context = OpenAILLMContext(..., tools=get_tool_definitions())
"""

import logging
import random
from typing import Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Free, keyless reverse-geocoding endpoint (OpenStreetMap Nominatim).
# Usage policy requires a descriptive User-Agent and ~1 req/sec.
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "Aura-voice-agent/1.0 (travel-planner tool)"}

# Fallback pool, used only if worldwide reverse-geocoding is unavailable
# (network issue, rate limit, etc.) so the tool never hard-fails.
FALLBACK_DESTINATIONS = [
    "Kyoto, Japan",
    "Reykjavik, Iceland",
    "Marrakech, Morocco",
    "Queenstown, New Zealand",
    "Lisbon, Portugal",
    "Cape Town, South Africa",
    "Cusco, Peru",
    "Ubud, Bali, Indonesia",
    "Prague, Czech Republic",
    "Jaipur, India",
    "Vancouver, Canada",
    "Santorini, Greece",
]


def _random_coordinate() -> tuple[float, float]:
    """Uniform random lat/lon, avoiding the poles where reverse geocoding is useless."""
    lat = random.uniform(-60.0, 75.0)
    lon = random.uniform(-180.0, 180.0)
    return lat, lon


def _extract_place_name(address: dict) -> Optional[str]:
    """Pull a human-friendly 'City, Country' (or best available) out of a
    Nominatim address block."""
    if not address:
        return None

    locality = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or address.get("state")
    )
    country = address.get("country")

    if locality and country:
        return f"{locality}, {country}"
    return country or locality


async def _reverse_geocode_random_place(max_attempts: int = 6) -> Optional[str]:
    """Pick random worldwide coordinates and reverse-geocode until we land on
    an actual named place (retries on ocean/uninhabited hits)."""
    async with httpx.AsyncClient(timeout=5.0, headers=NOMINATIM_HEADERS) as client:
        for attempt in range(max_attempts):
            lat, lon = _random_coordinate()
            try:
                resp = await client.get(
                    NOMINATIM_REVERSE_URL,
                    params={
                        "lat": lat,
                        "lon": lon,
                        "format": "json",
                        "zoom": 10,
                        "addressdetails": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.warning(f"[tool] reverse geocode failed (attempt {attempt}): {e}")
                continue

            place = _extract_place_name(data.get("address", {}))
            if place:
                logger.info(f"[tool] worldwide pick: ({lat:.2f}, {lon:.2f}) -> {place}")
                return place
            # Likely hit open ocean/unnamed area - try another coordinate.

    logger.warning("[tool] worldwide reverse geocoding exhausted attempts, falling back")
    return None


def get_tool_definitions() -> list[dict]:
    """OpenAI-style function/tool schemas the LLM can call."""
    return [
        {
            "type": "function",
            "function": {
                "name": "pick_random_place",
                "description": (
                    "Randomly select a travel destination anywhere in the world "
                    "to talk about. Call this when the user wants a surprise "
                    "place, doesn't know where to go, or asks 'pick somewhere "
                    "for me'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plan_itinerary",
                "description": (
                    "Generate a day-by-day travel itinerary for a specific place. "
                    "Call this after a destination is known (either picked randomly "
                    "or named by the user) and the user wants a plan."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "place": {
                            "type": "string",
                            "description": "The destination to plan the itinerary for, e.g. 'Kyoto, Japan'.",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days the trip should cover.",
                            "default": 3,
                        },
                        "interests": {
                            "type": "string",
                            "description": "Optional traveler interests, e.g. 'food, hiking, history'.",
                        },
                    },
                    "required": ["place"],
                },
            },
        },
    ]


class ToolHandlers:
    """Holds the actual tool implementations, bound to agent config (API keys)."""

    def __init__(self, config):
        self.config = config
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        self.last_place: Optional[str] = None

    async def pick_random_place(self, **_args) -> dict:
        """Tool: randomly select a destination anywhere in the world.

        Picks a random lat/lon and reverse-geocodes it so the result isn't
        limited to a fixed shortlist. Falls back to a small curated pool if
        geocoding is unreachable (offline, rate-limited, etc.).
        """
        place = await _reverse_geocode_random_place()
        if not place:
            place = random.choice(FALLBACK_DESTINATIONS)
        self.last_place = place
        logger.info(f"[tool] pick_random_place -> {place}")
        return {"place": place}

    async def plan_itinerary(
        self,
        place: Optional[str] = None,
        days: int = 3,
        interests: Optional[str] = None,
        **_args,
    ) -> dict:
        """Tool: generate a short day-by-day itinerary for a place."""
        place = place or self.last_place
        if not place:
            return {"error": "No place specified yet. Pick or name a destination first."}

        days = max(1, min(int(days or 3), 14))
        logger.info(f"[tool] plan_itinerary -> place={place} days={days} interests={interests}")

        prompt = (
            f"Create a concise {days}-day travel itinerary for {place}. "
            f"{'Focus on: ' + interests + '.' if interests else ''} "
            "For each day give a short title and 2-3 bullet activities "
            "(morning/afternoon/evening). Keep it tight and speakable out loud "
            "since it will be read by a voice assistant - avoid markdown headers, "
            "just plain short lines."
        )

        response = await self._client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise, practical travel planner.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.6,
        )

        itinerary = response.choices[0].message.content
        self.last_place = place
        return {"place": place, "days": days, "itinerary": itinerary}
