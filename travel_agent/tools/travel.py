"""Deterministic mock travel tools. All returned data is fictional."""

import json
import time

from langchain_core.tools import tool

from ..config import DEMO_DELAY
from ..memory import MemoryStore, avoids_early_morning, liked_interest, preference_values


HOTELS = [
    {"name": "Hostel Santa Maria", "price_per_night_eur": 70, "stars": 2, "area": "Centro"},
    {"name": "Hotel Davanzati", "price_per_night_eur": 145, "stars": 3, "area": "Duomo"},
    {"name": "Grand Palazzo Firenze", "price_per_night_eur": 320, "stars": 5, "area": "Arno riverside"},
]
ATTRACTIONS = [
    {"name": "Uffizi Gallery", "type": "museum", "start_time": "09:00", "price_eur": 25},
    {"name": "Galleria dell'Accademia (David)", "type": "museum", "start_time": "10:00", "price_eur": 16},
    {"name": "Palazzo Pitti & Boboli Gardens", "type": "museum/park", "start_time": "10:00", "price_eur": 18},
    {"name": "Sunrise hike to Piazzale Michelangelo", "type": "outdoor", "start_time": "05:45", "price_eur": 0},
    {"name": "Duomo & Brunelleschi's Dome climb", "type": "landmark", "start_time": "09:30", "price_eur": 30},
    {"name": "Ponte Vecchio evening walk", "type": "outdoor", "start_time": "18:00", "price_eur": 0},
]
RESTAURANTS = [
    {"name": "Trattoria Mario", "cuisine": "Italian", "price": "€€", "open": "12:00-15:00"},
    {"name": "Osteria Santo Spirito", "cuisine": "Italian", "price": "€€", "open": "19:00-23:00"},
    {"name": "Il Latini", "cuisine": "Italian", "price": "€€€", "open": "19:30-22:30"},
    {"name": "Sushi Zen", "cuisine": "Japanese", "price": "€€", "open": "19:00-22:30"},
]


def _pause():
    if DEMO_DELAY > 0:
        time.sleep(DEMO_DELAY)


def build_tools(memory: MemoryStore, user_id: str):
    """Bind the current user's saved preferences into tools for deterministic filtering."""

    @tool
    def search_flights(origin: str, destination: str, date: str = "this weekend") -> str:
        """Search fictional flights. Use origin='HOME' if the user has not said where they fly from."""
        _pause()
        return json.dumps({
            "origin": origin, "destination": destination, "date": date,
            "flights": [
                {"flight": "AZ1593", "depart": "06:10", "arrive": "08:05", "price_eur": 120},
                {"flight": "LH1832", "depart": "11:40", "arrive": "13:35", "price_eur": 165},
                {"flight": "BA0610", "depart": "17:25", "arrive": "19:20", "price_eur": 140},
            ],
        }, ensure_ascii=False)

    @tool
    def search_hotels(city: str, max_price_per_night_eur: int = 0) -> str:
        """Search fictional hotels; saved budget and the requested cap are applied as hard limits."""
        _pause()
        saved_cap = memory.budget_ceiling(user_id)
        caps = [cap for cap in (saved_cap, max_price_per_night_eur or None) if cap is not None]
        effective_cap = min(caps) if caps else None
        matches = [
            hotel for hotel in HOTELS
            if effective_cap is None or hotel["price_per_night_eur"] <= effective_cap
        ]
        return json.dumps({
            "city": city, "applied_saved_budget_eur": saved_cap,
            "max_price_per_night_eur": effective_cap, "hotels": matches,
        }, ensure_ascii=False)

    @tool
    def find_attractions(city: str, interest: str = "") -> str:
        """Find fictional attractions. Saved interests and early-morning avoidance are applied."""
        _pause()
        likes_museums = liked_interest(memory, user_id, "museum") or liked_interest(memory, user_id, "museums")
        filter_by_museum = "museum" in interest.lower() or likes_museums
        avoid_early = avoids_early_morning(memory, user_id)
        matches = ATTRACTIONS
        if filter_by_museum:
            matches = [item for item in matches if "museum" in item["type"].lower()]
        if avoid_early:
            matches = [item for item in matches if item["start_time"] >= "09:00"]
        return json.dumps({
            "city": city, "interest_filter": "museum" if filter_by_museum else interest or None,
            "early_morning_activities_removed": avoid_early, "attractions": matches,
        }, ensure_ascii=False)

    @tool
    def search_restaurants(city: str, cuisine: str = "") -> str:
        """Search fictional restaurants, using the user's saved cuisine preference when available."""
        _pause()
        saved_likes = " ".join(preference_values(memory, user_id, "likes")).lower()
        selected_cuisine = cuisine or ("Italian" if "italian" in saved_likes else "")
        matches = [
            restaurant for restaurant in RESTAURANTS
            if not selected_cuisine or selected_cuisine.lower() in restaurant["cuisine"].lower()
        ]
        return json.dumps({"city": city, "cuisine_filter": selected_cuisine or None, "restaurants": matches}, ensure_ascii=False)

    @tool
    def get_weather(city: str, days: int = 3) -> str:
        """Get a fictional short weather forecast for a city."""
        _pause()
        forecast = [
            {"day": "Saturday", "condition": "Sunny", "high_c": 24, "low_c": 14},
            {"day": "Sunday", "condition": "Light rain in the afternoon", "high_c": 21, "low_c": 13},
        ][:max(days, 1)]
        return json.dumps({"city": city, "forecast": forecast}, ensure_ascii=False)

    return [search_flights, search_hotels, find_attractions, search_restaurants, get_weather]
