"""Small SQLite long-term memory store for preferences, episodes, and rules."""

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


DEFAULT_RULES = (
    "Present itineraries day by day with approximate times.",
    "Apply saved preferences automatically and mention which ones were applied.",
    "Do not ask again for information that is already saved.",
)


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    value TEXT NOT NULL COLLATE NOCASE,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, category, value)
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS procedures (
                    user_id TEXT NOT NULL,
                    rule TEXT NOT NULL,
                    PRIMARY KEY (user_id, rule)
                );
                """
            )

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def save_preference(self, user_id: str, category: str, value: str) -> None:
        value = value.strip()
        if not value:
            return
        with self._db() as db:
            # A budget is a current setting; likes and avoids can contain several values.
            if category == "budget":
                db.execute(
                    "DELETE FROM preferences WHERE user_id=? AND category=?",
                    (user_id, category),
                )
            db.execute(
                "INSERT OR REPLACE INTO preferences VALUES (?, ?, ?, ?)",
                (user_id, category, value, datetime.now().isoformat(timespec="seconds")),
            )

    def add_episode(self, user_id: str, summary: str) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO episodes(user_id, summary, created_at) VALUES (?, ?, ?)",
                (user_id, summary, datetime.now().strftime("%Y-%m-%d %H:%M")),
            )

    def add_rule(self, user_id: str, rule: str) -> None:
        rule = rule.strip()
        if rule:
            with self._db() as db:
                db.execute("INSERT OR IGNORE INTO procedures VALUES (?, ?)", (user_id, rule))

    def get_all(self, user_id: str, n_episodes: int = 5) -> dict:
        with self._db() as db:
            preferences = db.execute(
                "SELECT category, value FROM preferences WHERE user_id=? ORDER BY category, value",
                (user_id,),
            ).fetchall()
            episodes = db.execute(
                "SELECT created_at, summary FROM episodes WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, n_episodes),
            ).fetchall()
            custom_rules = [
                row[0]
                for row in db.execute(
                    "SELECT rule FROM procedures WHERE user_id=? ORDER BY rule", (user_id,)
                )
            ]
        return {
            "preferences": preferences,
            "episodes": episodes,
            "rules": list(DEFAULT_RULES) + custom_rules,
        }

    def budget_ceiling(self, user_id: str) -> int | None:
        """Return the highest amount in a saved EUR budget, for deterministic filtering."""
        values = [value for category, value in self.get_all(user_id)["preferences"] if category == "budget"]
        if not values:
            return None
        # Supports common forms such as "100-150 EUR per night" and "about 120 euros".
        amounts = [int(float(number)) for number in re.findall(r"\d+(?:\.\d+)?", values[-1])]
        return max(amounts) if amounts else None

    def reset(self, user_id: str) -> None:
        with self._db() as db:
            for table in ("preferences", "episodes", "procedures"):
                db.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))


def preference_values(memory: MemoryStore, user_id: str, category: str) -> list[str]:
    return [
        value
        for saved_category, value in memory.get_all(user_id)["preferences"]
        if saved_category == category
    ]


def avoids_early_morning(memory: MemoryStore, user_id: str) -> bool:
    avoids = " ".join(preference_values(memory, user_id, "avoids")).lower()
    return "early" in avoids and ("morning" in avoids or "activities" in avoids)


def liked_interest(memory: MemoryStore, user_id: str, interest: str) -> bool:
    likes = " ".join(preference_values(memory, user_id, "likes")).lower()
    return interest.lower() in likes
