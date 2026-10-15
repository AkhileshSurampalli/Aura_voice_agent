# eval/conversations.py
"""
Fixed test set used by run_baseline_vs_router.py to produce a real,
reproducible number for the "reduced unnecessary LLM calls" claim.

Each conversation is a list of user turns (strings). Keep this set
representative of real usage - skewed toward one intent and the numbers
mean nothing. Grow this list as real usage patterns become clearer.
"""

CONVERSATIONS: dict[str, list[str]] = {
    "travel_1": [
        "Surprise me with somewhere to go",
        "Can you plan a 3 day trip there?",
    ],
    "travel_2": [
        "I want to visit Kyoto, plan me 5 days focused on food and history",
    ],
    "booking_1": [
        "What's the status of my booking AB123?",
    ],
    "booking_2": [
        "Can you check booking CD456 for me",
        "when is that trip?",
    ],
    "support_1": [
        "How do I cancel my booking?",
        "how long do refunds take",
    ],
    "smalltalk_1": [
        "Hey, how's it going?",
        "Nice, what can you help me with?",
    ],
    "escalation_1": [
        "This is ridiculous, I've been trying to get help for an hour, let me talk to a person",
    ],
    "mixed_1": [
        "Hi there!",
        "Actually, can you suggest a random travel destination?",
        "Plan a quick 2 day itinerary for it",
        "Also, what's your refund policy?",
    ],
}
