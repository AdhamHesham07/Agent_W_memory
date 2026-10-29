"""Run a two-process demo to show preferences survive an actual process restart."""

import argparse
import os
import subprocess
import sys
import uuid

from travel_agent.orchestration.agent import create_runtime, run_agent
from travel_agent.config import DATA_DIR
from travel_agent.interface.gradio_app import ICONS, render_memory


USER_ID = "demo_user"


def phase_one():
    runtime = create_runtime()
    text = "I like museums and Italian food, avoid early morning activities, and prefer hotels around 100-150 EUR per night."
    print("Conversation 1:", text)
    for state in run_agent(runtime, USER_ID, f"demo-{uuid.uuid4().hex}", text):
        pass
    print(render_memory(runtime, USER_ID, ""))
    runtime["checkpointer_connection"].close()


def phase_two():
    runtime = create_runtime()
    text = "Plan me a weekend in Florence."
    print("Conversation 2:", text)
    last_plan = None
    final_state = None
    for state in run_agent(runtime, USER_ID, f"demo-{uuid.uuid4().hex}", text):
        final_state = state
        plan = " | ".join(f"{ICONS[state['status']]} {state['label']}" for state in state["plan"])
        if plan != last_plan:
            print("Plan:", plan)
            last_plan = plan
    print("Assistant:", final_state["answer"] if final_state else "No response")
    print("Persistent memories loaded by the second process:")
    print(render_memory(runtime, USER_ID, ""))
    runtime["checkpointer_connection"].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", nargs="?", choices=("remember", "plan", "run"), default="run")
    args = parser.parse_args()
    if args.phase == "remember":
        phase_one()
    elif args.phase == "plan":
        phase_two()
    else:
        # Each phase runs in a fresh OS process against the same SQLite files.
        env = os.environ.copy()
        env["TRAVEL_AGENT_DATA_DIR"] = str(DATA_DIR)
        for phase in ("remember", "plan"):
            subprocess.run([sys.executable, __file__, phase], env=env, check=True)


if __name__ == "__main__":
    main()
