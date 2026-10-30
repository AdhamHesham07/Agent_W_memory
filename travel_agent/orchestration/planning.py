"""Fast, deterministic planning so the user sees the plan immediately."""

LABELS = {
    "load_preferences": "Read saved preferences",
    "search_flights": "Search flights",
    "search_hotels": "Search hotels",
    "find_attractions": "Find attractions",
    "search_restaurants": "Search restaurants",
    "get_weather": "Check weather",
    "final_answer": "Prepare the answer",
}
TOOL_STEPS = {
    "search_flights", "search_hotels", "find_attractions", "search_restaurants", "get_weather"
}


def make_plan(user_text: str) -> list[dict]:
    """Choose visible tool steps from the request without spending an LLM call to plan."""
    text = user_text.lower()
    trip_request = any(term in text for term in (
        "plan", "itinerary", "weekend in", "trip to", "travel to", "visit ",
    ))
    preference_statement = text.startswith((
        "i like", "i love", "i enjoy", "i prefer", "i hate", "i avoid",
        "i don't like", "i do not like", "my budget", "our budget",
    ))

    selected = []
    if trip_request:
        selected = [
            "search_flights", "search_hotels", "find_attractions",
            "search_restaurants", "get_weather",
        ]
    elif not preference_statement:
        rules = (
            ("search_flights", ("flight", "airfare", "fly to", "flying to")),
            ("search_hotels", ("hotel", "accommodation", "where to stay", "place to stay")),
            ("find_attractions", ("attraction", "museum", "sightseeing", "things to do", "activities")),
            ("search_restaurants", ("restaurant", "food", "where to eat", "dining", "cuisine")),
            ("get_weather", ("weather", "forecast")),
        )
        selected = [kind for kind, terms in rules if any(term in text for term in terms)]

    kinds = ["load_preferences", *selected, "final_answer"]
    return [{"kind": kind, "label": LABELS[kind], "status": "pending"} for kind in kinds]