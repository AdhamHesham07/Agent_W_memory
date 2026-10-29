import os
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=value entries without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")


def _remove_unavailable_local_proxy() -> None:
    """Ignore a dead localhost:9 proxy inherited by this process.

    Some managed/dev environments inject localhost:9 as a placeholder proxy.
    Port 9 is the discard service, not an HTTP proxy; requests then fail before
    they can reach the configured model API. Only this known unusable endpoint
    is removed, and only from this Python process.
    """
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        value = os.environ.get(name)
        if not value:
            continue
        parsed = urlparse(value if "://" in value else f"//{value}")
        host = (parsed.hostname or "").lower()
        try:
            port = parsed.port
        except ValueError:
            continue
        if host in {"127.0.0.1", "localhost", "::1"} and port == 9:
            os.environ.pop(name, None)


_remove_unavailable_local_proxy()
DEFAULT_USER_ID = "demo_user"
DATA_DIR = Path(os.environ.get("TRAVEL_AGENT_DATA_DIR", BASE_DIR / "data")).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DB = DATA_DIR / "long_term_memory.sqlite"
CHECKPOINT_DB = DATA_DIR / "conversation_checkpoints.sqlite"
MODEL_NAME = os.environ.get("TRAVEL_AGENT_MODEL", "google_genai:gemini-3.5-flash-lite")
DEMO_DELAY = float(os.environ.get("TRAVEL_AGENT_DEMO_DELAY", "0.35"))


def require_api_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is required. Set it in your environment before starting the app."
        )
