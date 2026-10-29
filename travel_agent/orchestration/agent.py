"""LangChain agent orchestration with a user-visible execution trace."""

import copy
import json
import sqlite3
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "The SQLite checkpointer is missing from this Python environment. "
        "Install this project's dependencies into the same environment used to run app.py: "
        "python -m pip install -r requirements.txt"
    ) from error

from ..config import CHECKPOINT_DB, MEMORY_DB, MODEL_NAME, require_api_key
from ..memory import MemoryStore
from ..tools import build_tools
from .planning import LABELS, TOOL_STEPS, make_plan


class Preference(BaseModel):
    category: Literal["likes", "avoids", "budget", "other"]
    value: str


class ExtractedMemory(BaseModel):
    preferences: list[Preference] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)


EXTRACT_PROMPT = """Extract only durable travel preferences and answer-style rules explicitly stated in this message.
Use likes for enjoyed things, avoids for things the user wants to avoid, and budget for a preferred price range.
Do not infer or invent preferences. A direct trip request with no stated preference should return empty lists."""
PREFERENCE_CUES = (
    "i like", "i love", "i enjoy", "i prefer", "i hate", "i avoid", "i dislike",
    "i don't like", "i do not like", "my budget", "our budget", "budget is",
    "per night", "please remember", "keep in mind",
)


def create_runtime():
    require_api_key()
    # Gemini 3.8 supports low thinking; omit temperature (unsupported on Gemini 3 models).
    model = init_chat_model(MODEL_NAME, thinking_level="low", max_retries=1)
    return {
        "model": model,
        "memory": MemoryStore(MEMORY_DB),
        "checkpointer_connection": sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False),
    }


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _extract_and_save(model, memory: MemoryStore, user_id: str, text: str) -> list[str]:
    result = model.with_structured_output(ExtractedMemory).invoke(
        [("system", EXTRACT_PROMPT), ("user", text)]
    )
    saved = []
    for pref in result.preferences:
        memory.save_preference(user_id, pref.category, pref.value)
        saved.append(f"{pref.category}: {pref.value}")
    for rule in result.style_rules:
        memory.add_rule(user_id, rule)
        saved.append(f"rule: {rule}")
    return saved


def _system_prompt(memory: MemoryStore, user_id: str, plan: list[dict]) -> str:
    saved = memory.get_all(user_id)
    preferences = "\n".join(f"- {category}: {value}" for category, value in saved["preferences"])
    episodes = "\n".join(f"- {date}: {summary}" for date, summary in saved["episodes"])
    rules = "\n".join(f"- {rule}" for rule in saved["rules"])
    steps = "\n".join(f"{index + 1}. {step['label']}" for index, step in enumerate(plan))
    return f"""You are a friendly travel-planning assistant. All travel tools return fictional mock data.

Semantic memory (saved preferences):
{preferences or '- none saved'}

Episodic memory (past planning sessions):
{episodes or '- none saved'}

Procedural memory (answer rules):
{rules}

Visible plan for this request:
{steps}

Call each planned tool that is relevant before answering. The tools apply saved budget, cuisine, interest, and
early-morning constraints as hard filters. Say which saved preferences you applied. For a preference-only message,
acknowledge it briefly. Never claim to have booked anything. If a planned tool fails or returns no results, say so."""


def _preference_signal(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in PREFERENCE_CUES)


def _snapshot(state: dict) -> dict:
    return copy.deepcopy(state)


def _record(state: dict, stage: str, detail: str, status: str = "running") -> None:
    state["stage"] = stage
    state["trace"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "stage": stage,
        "detail": detail,
        "status": status,
    })


def _episode_summary(user_text: str, tool_names: list[str]) -> str:
    short_request = " ".join(user_text.split())[:130]
    return f"Request: {short_request}. Tools used: {', '.join(tool_names)}."


def run_agent(runtime: dict, user_id: str, conversation_id: str, user_text: str):
    """Yield a visible state before and after each meaningful agent action."""
    model = runtime["model"]
    memory: MemoryStore = runtime["memory"]
    saver = SqliteSaver(runtime["checkpointer_connection"])
    state = {"plan": [], "log": [], "answer": None, "stage": "Starting", "trace": []}

    def mark(kind: str, status: str):
        for item in state["plan"]:
            if item["kind"] == kind:
                item["status"] = status
                return
        if kind in LABELS:
            state["plan"].insert(
                max(1, len(state["plan"]) - 1),
                {"kind": kind, "label": LABELS[kind], "status": status},
            )

    try:
        _record(state, "Request received", "The agent has the user's request and is preparing its work plan.", "done")
        yield _snapshot(state)

        state["plan"] = make_plan(user_text)
        _record(
            state, "Plan ready",
            f"Selected {sum(step['kind'] in TOOL_STEPS for step in state['plan'])} tool step(s) from the request.",
            "done",
        )
        yield _snapshot(state)

        if _preference_signal(user_text):
            _record(state, "Extracting preferences", "Checking this message for durable preferences to save.")
            yield _snapshot(state)
            try:
                saved_now = _extract_and_save(model, memory, user_id, user_text)
                detail = f"Saved {len(saved_now)} preference/rule item(s)." if saved_now else "No new durable preferences found."
                _record(state, "Preferences updated", detail, "done")
                if saved_now:
                    state["log"].append({
                        "name": "save_memory", "args": {"saved": saved_now},
                        "result": "Stored in persistent SQLite memory.", "id": "memory",
                    })
            except Exception as error:
                state["trace"].append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "stage": "Preference extraction warning",
                    "detail": f"Could not extract preferences ({type(error).__name__}: {error}). Continuing without saving this message.",
                    "status": "warning",
                })
            yield _snapshot(state)
        else:
            _record(state, "Preference check", "No preference statement detected; skipped the extra model call.", "done")
            yield _snapshot(state)

        mark("load_preferences", "running")
        _record(state, "Loading memory", "Reading saved preferences, prior trip summaries, and behavior rules.")
        yield _snapshot(state)
        loaded = memory.get_all(user_id)
        state["log"].append({
            "name": "load_user_memory",
            "args": {"scope": "current user"},
            "result": json.dumps({
                "preferences": loaded["preferences"],
                "recent_episodes": len(loaded["episodes"]),
                "behavior_rules": len(loaded["rules"]),
            }, ensure_ascii=False),
            "id": "load",
        })
        mark("load_preferences", "done")
        _record(state, "Memory loaded", f"Loaded {len(loaded['preferences'])} preference(s) and {len(loaded['episodes'])} past episode(s).", "done")
        yield _snapshot(state)

        agent = create_agent(
            model,
            build_tools(memory, user_id),
            system_prompt=_system_prompt(memory, user_id, state["plan"]),
            checkpointer=saver,
        )
        final = ""
        config = {"configurable": {"thread_id": conversation_id}}
        _record(state, "Agent reasoning", "Gemini is deciding whether to call tools, using the visible plan and saved memory.")
        yield _snapshot(state)

        for chunk in agent.stream({"messages": [HumanMessage(user_text)]}, config, stream_mode="updates"):
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages") or []:
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for call in message.tool_calls:
                            mark(call["name"], "running")
                            state["log"].append({
                                "name": call["name"], "args": call["args"], "result": None,
                                "id": call["id"],
                            })
                            _record(state, "Tool selected", f"Agent called {call['name']} with the shown arguments.")
                    elif isinstance(message, AIMessage):
                        final = _text(message.content)
                    elif isinstance(message, ToolMessage):
                        mark(message.name, "done")
                        for entry in state["log"]:
                            if entry["id"] == message.tool_call_id:
                                entry["result"] = _text(message.content)
                        _record(state, "Tool completed", f"{message.name} returned a result.", "done")
            yield _snapshot(state)

        incomplete = [step for step in state["plan"] if step["kind"] in TOOL_STEPS and step["status"] != "done"]
        for step in incomplete:
            step["status"] = "skipped"
            state["log"].append({
                "name": "plan_status", "args": {"step": step["label"]},
                "result": "This planned tool was not called by the agent.",
                "id": f"skipped-{step['kind']}",
            })
        state["plan"][-1]["status"] = "done" if not incomplete else "needs_attention"
        state["answer"] = final or "The agent returned no answer. Check the activity trace and try again."
        tool_names = [entry["name"] for entry in state["log"] if entry["name"] in TOOL_STEPS]
        if tool_names:
            memory.add_episode(user_id, _episode_summary(user_text, tool_names))
        _record(state, "Answer ready", "The assistant combined the tool results and saved memory into its response.", "done")
        yield _snapshot(state)
    except Exception as error:
        for step in state["plan"]:
            if step["status"] == "running":
                step["status"] = "failed"
            if step["kind"] == "final_answer":
                step["status"] = "needs_attention"
        state["answer"] = f"Error: {type(error).__name__}: {error}"
        _record(state, "Run needs attention", f"{type(error).__name__}: {error}", "error")
        yield _snapshot(state)


def conversation_messages(runtime: dict, conversation_id: str) -> list[dict]:
    if not conversation_id:
        return []
    saver = SqliteSaver(runtime["checkpointer_connection"])
    checkpoint = saver.get_tuple({"configurable": {"thread_id": conversation_id}})
    if not checkpoint:
        return []
    history = []
    for message in checkpoint.checkpoint["channel_values"].get("messages", []):
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": _text(message.content)})
        elif isinstance(message, AIMessage) and not message.tool_calls and _text(message.content).strip():
            history.append({"role": "assistant", "content": _text(message.content)})
    return history
