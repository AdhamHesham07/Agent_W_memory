"""Human-readable Gradio dashboard for the memory-enabled travel agent."""

import json
import uuid

import gradio as gr

from ..config import DEFAULT_USER_ID, MODEL_NAME
from ..orchestration.agent import conversation_messages, create_runtime, run_agent


ICONS = {
    "pending": "◯", "running": "⟳", "done": "✓", "skipped": "↷",
    "failed": "✗", "needs_attention": "⚠", "warning": "⚠", "error": "✗",
}


def render_plan(plan: list[dict]) -> str:
    if not plan:
        return "_The plan will appear as soon as you send a message._"
    complete = sum(step["status"] in ("done", "skipped") for step in plan)
    bar = "▰" * complete + "▱" * (len(plan) - complete)
    rows = []
    for step in plan:
        status = step["status"]
        hint = {
            "running": " · in progress",
            "skipped": " · not called",
            "failed": " · failed",
            "needs_attention": " · check trace",
        }.get(status, "")
        rows.append(f"- {ICONS.get(status, '◯')} **{step['label']}**{hint}")
    return f"**{complete} of {len(plan)} steps complete**  `{bar}`\n\n" + "\n".join(rows)


def render_trace(trace: list[dict]) -> str:
    if not trace:
        return "_Waiting for a request._"
    rows = []
    for item in trace[-14:]:
        icon = ICONS.get(item["status"], "•")
        detail = item["detail"].replace("\n", " ")
        rows.append(f"- `{item['time']}` {icon} **{item['stage']}** — {detail}")
    return "\n".join(rows)


def render_tools(log: list[dict]) -> str:
    if not log:
        return "_No tools have run yet._"
    blocks = []
    for entry in log:
        result = entry["result"] if entry["result"] is not None else "⟳ Running…"
        blocks.append(
            f"**{entry['name']}**\n\n"
            f"Arguments: `{json.dumps(entry['args'], ensure_ascii=False)}`\n\n"
            f"```json\n{result[:900]}\n```"
        )
    return "\n\n".join(blocks)


def render_memory(runtime: dict, conversation_id: str) -> str:
    saved = runtime["memory"].get_all(DEFAULT_USER_ID)
    preferences = "\n".join(f"- **{category}** — {value}" for category, value in saved["preferences"])
    episodes = "\n".join(f"- `{date}` {summary}" for date, summary in saved["episodes"])
    rules = "\n".join(f"- {rule}" for rule in saved["rules"])
    message_count = len(conversation_messages(runtime, conversation_id))
    return (
        f"**Saved preferences**\n{preferences or '_None saved yet._'}\n\n"
        f"**Past trip notes**\n{episodes or '_None saved yet._'}\n\n"
        f"**Assistant behavior rules**\n{rules or '_No custom rules._'}\n\n"
        f"**This conversation**\n{message_count} messages saved in the LangGraph checkpointer."
    )


def build_app():
    runtime = create_runtime()

    def on_send(message, history, conversation_id):
        message = (message or "").strip()
        history = list(history or [])
        if not message:
            yield history, "", gr.skip(), gr.skip(), gr.skip(), gr.skip(), render_memory(runtime, conversation_id)
            return

        history.append({"role": "user", "content": message})
        waiting = history + [{"role": "assistant", "content": "⏳ Working through the steps shown at right…"}]
        yield (
            waiting, "", "**Starting…**", render_plan([]), "_Preparing the activity trace…_",
            "_No tools have run yet._", render_memory(runtime, conversation_id),
        )

        for state in run_agent(runtime, DEFAULT_USER_ID, conversation_id, message):
            shown = history + [{"role": "assistant", "content": state["answer"]}] if state["answer"] else waiting
            yield (
                shown,
                "",
                f"### {state['stage']}\n\nModel: `{MODEL_NAME}`",
                render_plan(state["plan"]),
                render_trace(state["trace"]),
                render_tools(state["log"]),
                render_memory(runtime, conversation_id),
            )

    def start_new_conversation():
        conversation_id = uuid.uuid4().hex
        return (
            conversation_id, [], "**Ready for a new request.**", render_plan([]),
            "_Waiting for a request._", render_tools([]), render_memory(runtime, conversation_id),
        )

    with gr.Blocks(title="Travel Agent · Memory in Action", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🧭 Travel Agent · Memory in Action\n"
            "Watch the agent read memory, prepare a plan, choose mock tools, and assemble an answer. "
            "The activity trace shows observable actions, not hidden chain-of-thought."
        )
        conversation_id = gr.State(value=lambda: uuid.uuid4().hex)
        with gr.Row():
            with gr.Column(scale=6):
                chat = gr.Chatbot(type="messages", height=540, label="Conversation")
                message = gr.Textbox(
                    placeholder="Try: I like museums and Italian food, and avoid early morning activities.",
                    show_label=False,
                    lines=2,
                )
                with gr.Row():
                    send_hint = gr.Markdown("Press **Enter** to send. Start by sharing preferences, then request a trip.")
                    new_conversation = gr.Button("New conversation", size="sm")
                gr.Examples(
                    [
                        "I like museums and Italian food, and I don't like early morning activities.",
                        "My hotel budget is 100-150 EUR per night.",
                        "Plan me a weekend in Florence.",
                        "What preferences are saved for me?",
                    ], inputs=message,
                )
            with gr.Column(scale=5):
                stage_panel = gr.Markdown("**Ready.**")
                with gr.Accordion("① Plan and progress", open=True):
                    plan_panel = gr.Markdown(render_plan([]))
                with gr.Accordion("② Agent activity trace", open=True):
                    trace_panel = gr.Markdown("_Waiting for a request._")
                with gr.Accordion("③ Tools called and results", open=True):
                    tools_panel = gr.Markdown("_No tools have run yet._")
                with gr.Accordion("④ Memory in use", open=True):
                    memory_panel = gr.Markdown(render_memory(runtime, ""))
                gr.Markdown(
                    "**Memory key:** one local demo profile\n\n"
                    "**Persistence:** preferences and trip notes in SQLite; conversation messages in the LangGraph checkpointer.\n\n"
                    "**Tools:** flights · hotels · attractions · restaurants · weather (all mock results)."
                )

        message.submit(
            on_send,
            [message, chat, conversation_id],
            [chat, message, stage_panel, plan_panel, trace_panel, tools_panel, memory_panel],
        )
        new_conversation.click(
            start_new_conversation,
            [],
            [conversation_id, chat, stage_panel, plan_panel, trace_panel, tools_panel, memory_panel],
        )
    return app
